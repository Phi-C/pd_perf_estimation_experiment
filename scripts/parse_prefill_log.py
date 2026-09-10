#!/usr/bin/env python3
"""
ATOM prefill node CI log  ->  prefill_forwards.csv + gap_events.json + idle_gaps.json + prefill_stat.md

用法:
    gh api repos/ROCm/ATOM/actions/jobs/<JOB_ID>/logs > job.log
    python3 parse_prefill_log.py job.log -o <outdir> [--label "kimi-k3 c32"]

只读一遍 log。所有正则都是"匹配不到就产出 0", 换用例不会崩。

时间模型
--------
引擎自己打的 `[atom HH:MM:SS]` 是**调度时刻**(秒级)。GitHub runner 行首的
`2026-08-29T12:21:02.6193014Z` 是**日志转发时刻**, 比引擎晚 12~30s 且批量刷新, 不能当时间用,
只能用来定顺序。因此:
    t[k+1] - t[k]  ==  第 k 个 engine step 的完整延迟 (forward 计算 + KV 收尾 + 下一次调度)
两次 forward 之间**没有**天然空洞; 只有引擎 running+waiting 全空、无请求可调度时才是真空转。
"""
import re, sys, csv, json, gzip, os, argparse, collections, statistics

AP = argparse.ArgumentParser()
AP.add_argument("log")
AP.add_argument("-o", "--outdir", default=".")
AP.add_argument("--label", default="", help="写进 stat.md / trace 标题的用例名")
ARGS = AP.parse_args()
os.makedirs(ARGS.outdir, exist_ok=True)
out = lambda n: os.path.join(ARGS.outdir, n)

# ---------------------------------------------------------------- 正则
RE_ATOM  = re.compile(r"\[atom (\d\d):(\d\d):(\d\d)\]")
RE_SCHED = re.compile(r"Scheduled prefill batch: (\d+) reqs, (\d+) new tokens "
                      r"\(done: \[([^\]]*)\], new: \[([^\]]*)\]\), req_ids: \(([^)]*)\)")
RE_ARR   = re.compile(r"Request (\d+) arrived, input tokens: (\d+), pending requests: (\d+)")
# —— step 内可能出现的"这一步为什么慢"的证据 ——
RE_AUTOT = re.compile(r"Triton autotuning for function (\w+)")
RE_AUTOD = re.compile(r"finished after ([\d.]+)s")
RE_AITER = re.compile(r"\[aiter\] type hints mismatch, override to --> (\w+)")
RE_ALLOC = re.compile(r"Failed to allocate memory block")
RE_RDMA  = re.compile(r"\[PRODUCER\] block RDMA: req=(\d+),.*total_bytes=(\d+)")
RE_STORE = re.compile(r"\[req_id=(\d+)\] Stored (\d+) out of total \d+ tokens\..*cost ([\d.]+) ms")
RE_PDTR  = re.compile(r"\[PD-TRANSITION\] seq (\d+):")
# —— 配置 ——
RE_CFG   = {k: re.compile(r'"?%s"?\s*[:=]\s*(\d+)' % k) for k in
            ("max_num_batched_tokens", "block_size", "max_num_seqs",
             "long_prefill_token_threshold")}
RE_LMCHUNK = re.compile(r"'chunk_size':\s*(\d+)")   # 只从 LMCacheEngine config 那行取

# ---------------------------------------------------------------- 工作量等价
# 见 ragged_packed_prefill_workload_equivalence.md
#   A(q,c) = q*c + q(q+1)/2          一条请求的有效 query-key 对数
#   q_eq   = Q / n                   匹配"新 token 总量"
#   c_eq   = A_total/Q - (q_eq+1)/2  匹配"Attention 总工作量"
# 一个 ragged/packed forward -> 同构性能表里的一个点 (q_eq, c_eq, b=n)。
# 对本数据: q_i = chunk_tokens(本步新算的 token), c_i = done_tokens(已在 KV cache 里的上下文)。
def equivalent_point(reqs):
    """reqs = [(q, c), ...] -> (q_eq, c_eq, b, A_total, r_q, r_L)"""
    n = len(reqs)
    Q = sum(q for q, _ in reqs)
    A = sum(q * c + q * (q + 1) / 2.0 for q, c in reqs)
    if n == 0 or Q <= 0:
        return 0.0, 0.0, n, A, 0.0, 0.0
    q_eq = Q / n
    c_eq = A / Q - (q_eq + 1.0) / 2.0
    r_q  = max(q for q, _ in reqs) / q_eq                       # query 长度不均衡
    L    = [q + c for q, c in reqs]
    r_L  = max(L) / (sum(L) / n) if sum(L) else 0.0             # 总长度不均衡
    return q_eq, c_eq, n, A, r_q, r_L


def blank():
    return {"autotune": 0, "autotune_done": 0, "autotune_sec": 0.0, "kernels": [], "aiter": 0,
            "alloc_fail": 0, "rdma": 0, "rdma_bytes": 0, "lm_store": 0, "lm_store_ms": 0.0,
            "lm_store_tokens": 0, "pd_trans": 0}

forwards, arrivals, cfg = [], [], {}
gapev = collections.defaultdict(blank)
cur = -1                      # 当前 step 序号: 落在 sched[k] 与 sched[k+1] 之间的证据行归给 k
progress, last_sched_sec, dup = {}, None, 0
pending_kernel = None

with open(ARGS.log, "r", errors="replace") as f:
    for line in f:
        if "Scheduled prefill batch" in line:
            m = RE_SCHED.search(line); a = RE_ATOM.search(line)
            if not (m and a): continue
            done = [int(x) for x in m.group(3).split(",") if x.strip()]
            new  = [int(x) for x in m.group(4).split(",") if x.strip()]
            ids  = [int(x) for x in m.group(5).split(",") if x.strip()]
            sec  = int(a.group(1))*3600 + int(a.group(2))*60 + int(a.group(3))
            # CI log 偶尔会把一小段行重放一遍。判据: 同一秒内, 该 batch 里**每一条**请求的
            # done 都比我们已经处理过的进度还小 -> 是重放, 丢掉(否则会凭空多出 forward,
            # 并把一条请求错误地劈成两个实例)。真正的 req_id 复用发生在几分钟后, 不会误伤。
            if all(rid in progress and dn < progress[rid] for rid, dn in zip(ids, done)) \
               and sec == last_sched_sec:
                dup += 1; continue
            for rid, dn, ck in zip(ids, done, new): progress[rid] = dn + ck
            last_sched_sec = sec
            cur += 1
            forwards.append({"t": sec,
                             "wall": f"{a.group(1)}:{a.group(2)}:{a.group(3)}",
                             "batch_reqs": int(m.group(1)), "batch_tokens": int(m.group(2)),
                             "ids": ids, "done": done, "new": new})
            continue
        if "arrived, input tokens" in line:
            m = RE_ARR.search(line); a = RE_ATOM.search(line)
            if m: arrivals.append({"req_id": int(m.group(1)), "ISL": int(m.group(2)),
                                   "pending": int(m.group(3)),
                                   "t": (int(a.group(1))*3600+int(a.group(2))*60+int(a.group(3))) if a else None})
            continue
        if len(cfg) < len(RE_CFG) + 1:
            for k, r in RE_CFG.items():
                if k in cfg: continue
                mm = r.search(line)
                if mm: cfg[k] = int(mm.group(1))
            if "LMCACHE_CHUNK_SIZE" not in cfg and "LMCacheEngine with config" in line:
                mm = RE_LMCHUNK.search(line)
                if mm: cfg["LMCACHE_CHUNK_SIZE"] = int(mm.group(1))
        if cur < 0: continue
        e = None
        m = RE_AUTOT.search(line)
        if m:
            e = gapev[cur]; e["autotune"] += 1; pending_kernel = m.group(1)
            if m.group(1) not in e["kernels"]: e["kernels"].append(m.group(1))
        elif RE_AUTOD.search(line) and pending_kernel:
            e = gapev[cur]; e["autotune_done"] += 1
            e["autotune_sec"] += float(RE_AUTOD.search(line).group(1))
        elif RE_AITER.search(line):
            e = gapev[cur]; e["aiter"] += 1
            k = RE_AITER.search(line).group(1)
            if k not in e["kernels"]: e["kernels"].append(k)
        elif RE_ALLOC.search(line):
            gapev[cur]["alloc_fail"] += 1
        else:
            m = RE_RDMA.search(line)
            if m:
                e = gapev[cur]; e["rdma"] += 1; e["rdma_bytes"] += int(m.group(2))
            else:
                m = RE_STORE.search(line)
                if m:
                    e = gapev[cur]; e["lm_store"] += 1
                    e["lm_store_tokens"] += int(m.group(2)); e["lm_store_ms"] += float(m.group(3))
                elif RE_PDTR.search(line):
                    gapev[cur]["pd_trans"] += 1

if not forwards:
    sys.exit("没有匹配到任何 'Scheduled prefill batch' 行 —— 确认这是 prefill node 的 log")

# ------------------------------------------------- request 实例化 + ISL
# req_id 会跨阶段(warmup->profiling)复用, 所以按 "done_tokens==0 开启新实例" 切分。
# 注意 log 里偶有重复行(同一条 scheduled 行打两遍), 所以"再来一个 done==0"只有在该实例
# 已经推进过(见过 done>0)时才算新实例, 否则视为重复。
inst_of, insts, prog = {}, [], {}
for k, f in enumerate(forwards):
    for rid, dn, ck in zip(f["ids"], f["done"], f["new"]):
        # done 相对该 req 当前实例的进度**回退** => 这是同一个 req_id 的新一轮请求。
        # 注意不能简单用 `dn == 0`: LMCache 前缀命中时, 请求的第一块 done 就等于命中长度。
        if rid not in inst_of or dn < prog[rid]:
            inst_of[rid] = len(insts)
            insts.append({"req_id": rid, "first": k, "last": k, "ISL": dn + ck, "prefix_hit": dn})
        it = insts[inst_of[rid]]
        it["last"] = k; it["ISL"] = max(it["ISL"], dn + ck)   # 末块的 done+new 即 ISL
        prog[rid] = dn + ck
        f.setdefault("inst", []).append(inst_of[rid])

# arrival 行同样会重复打印, 先折叠连续重复; 再按 req_id 的出现顺序配到实例上。
# arrival 行的 input tokens 才是权威 ISL —— 少数请求(被抢占/日志截断)不会跑完全部 chunk,
# 只靠 max(done+new) 会低估。
dedup, last = [], None
for a in arrivals:
    key = (a["req_id"], a["ISL"], a["t"])
    if key != last: dedup.append(a)
    last = key
by_id = collections.defaultdict(collections.deque)
for a in dedup: by_id[a["req_id"]].append(a)
matched = 0
for it in sorted(insts, key=lambda x: (x["first"], x["req_id"])):
    q = by_id.get(it["req_id"])
    if q:
        a = q.popleft(); it["arrive_t"] = a["t"]; matched += 1
        if a["ISL"] >= it["ISL"]: it["ISL"] = a["ISL"]     # 以 arrival 为准
    else:
        it["arrive_t"] = None

# ------------------------------------------------- 写 CSV
with open(out("prefill_forwards.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["forward_idx","time","req_id","done_tokens","chunk_tokens",
                "ISL","progress_pct","batch_reqs","batch_tokens",
                "eq_batch_size","eq_query_length","eq_context_length",
                "attn_work_pairs","imbalance_rq","imbalance_rL"])
    for k, f in enumerate(forwards):
        # 该 forward 的等价同构点(整个 batch 一个值, 每行重复, 和 batch_reqs/batch_tokens 一样)
        qeq, ceq, b, aw, rq, rL = equivalent_point(list(zip(f["new"], f["done"])))
        for j, (rid, dn, ck) in enumerate(zip(f["ids"], f["done"], f["new"])):
            isl = insts[f["inst"][j]]["ISL"]
            w.writerow([k, f["wall"], rid, dn, ck, isl,
                        round((dn+ck)*100.0/isl, 2) if isl else 0, f["batch_reqs"], f["batch_tokens"],
                        b, round(qeq, 1), round(ceq, 1), int(aw), round(rq, 3), round(rL, 3)])

# ------------------------------------------------- 空转(starvation) step
# 判据: step k 结束后 (a) 没有任何实例还有剩余 tokens  且 (b) 下一个 step 的请求是 t_k 之后才到的
alive = collections.Counter()
for it in insts:
    for k in range(it["first"], it["last"]+1): alive[k] += 1
end_at = collections.Counter()
for it in insts: end_at[it["last"]] += 1
running_after = {}
open_n = 0
for k in range(len(forwards)):
    open_n += sum(1 for it in insts if it["first"] == k)
    open_n -= end_at[k]
    running_after[k] = open_n

idle = {}
for k in range(len(forwards)-1):
    if running_after[k] != 0: continue
    nxt = forwards[k+1]
    gap = nxt["t"] - forwards[k]["t"]
    if gap <= 0: continue
    nr = []
    for j in set(nxt["inst"]):
        it = insts[j]
        if it["first"] != k+1 or it["arrive_t"] is None: continue
        if it["arrive_t"] > forwards[k]["t"]:
            nr.append({"req_id": it["req_id"], "ISL": it["ISL"],
                       "arrived_after_s": it["arrive_t"] - forwards[k]["t"]})
    if nr: idle[k] = {"gap_s": gap, "next_reqs": sorted(nr, key=lambda x: x["arrived_after_s"])}

json.dump(idle, open(out("idle_gaps.json"), "w"), indent=0)
ge = {}
for k, v in gapev.items():
    v = dict(v); v["rdma_GB"] = round(v.pop("rdma_bytes")/2**30, 3)
    v["lm_store_ms"] = round(v["lm_store_ms"], 1); v["autotune_sec"] = round(v["autotune_sec"], 2)
    ge[k] = {a: b for a, b in v.items() if b}
json.dump(ge, open(out("gap_events.json"), "w"), indent=0)

# ------------------------------------------------- stat.md
BUD = cfg.get("max_num_batched_tokens", 8192)
span = forwards[-1]["t"] - forwards[0]["t"]
br = collections.Counter(f["batch_reqs"] for f in forwards)
full = sum(1 for f in forwards if f["batch_tokens"] == BUD)
pairs = sum(f["batch_reqs"] for f in forwards)
step = {k: forwards[k+1]["t"]-forwards[k]["t"] for k in range(len(forwards)-1)}
idle_s = sum(v["gap_s"] for v in idle.values())
L = [f"# Prefill forward 统计 {ARGS.label}", "", f"log: `{os.path.abspath(ARGS.log)}`", "",
     "| 参数 | 值 |", "|---|---|"]
for k in ("max_num_batched_tokens","block_size","max_num_seqs","long_prefill_token_threshold","LMCACHE_CHUNK_SIZE"):
    if k in cfg: L.append(f"| `{k}` | {cfg[k]} |")
L += ["", f"- forward: **{len(forwards)}**, (forward,request) pairs: **{pairs}**, "
          f"request 实例: **{len(insts)}** (unique req_id {len(inst_of)}, arrival 匹配 {matched}/{len(insts)})",
      f"- 时间跨度: {span}s ({forwards[0]['wall']} -> {forwards[-1]['wall']})",
      f"- 打满 budget 的 forward: {full} ({full*100.0/len(forwards):.1f}%)",
      f"- LMCache 前缀命中的请求: {sum(1 for i in insts if i['prefix_hit'])} "
      f"(命中 tokens 合计 {sum(i['prefix_hit'] for i in insts):,})",
      f"- 空转 step: {len(idle)}, 空转 step 总延迟 {idle_s}s ({idle_s*100.0/max(span,1):.1f}% 的墙钟)",
      "", "## batch_reqs 分布", "", "| batch_reqs | forward 数 | 占比 |", "|---:|---:|---:|"]
for n in sorted(br): L.append(f"| {n} | {br[n]} | {br[n]*100.0/len(forwards):.2f}% |")

eqs = collections.defaultdict(list)
for f in forwards:
    qeq, ceq, b, _aw, rq, rL = equivalent_point(list(zip(f["new"], f["done"])))
    eqs[b].append((qeq, ceq, rq, rL))
L += ["", "## 等价同构 batch (工作量等价法, 见 ragged_packed_prefill_workload_equivalence.md)", "",
      "对每个 forward 取 q_i=chunk_tokens, c_i=done_tokens, 映射到同构性能表里的点 (q_eq, c_eq, b)。",
      "", "| b | forward 数 | q_eq 中位 | c_eq 中位 | c_eq p90 | r_q 中位 | r_L 中位 |", "|---:|---:|---:|---:|---:|---:|---:|"]
for b in sorted(eqs):
    v = eqs[b]; med = lambda i: sorted(x[i] for x in v)[len(v)//2]
    p90 = sorted(x[1] for x in v)[min(len(v)-1, int(len(v)*0.9))]
    L.append(f"| {b} | {len(v)} | {med(0):,.0f} | {med(1):,.0f} | {p90:,.0f} | {med(2):.2f} | {med(3):.2f} |")
L += ["", "`r_q = max q_i / q_eq`, `r_L = max(q_i+c_i) / mean(q_i+c_i)`: 长度不均衡度。",
      "两者越接近 1, 等价映射越可信; 明显大于 1 时该点需要异构样本校正 (原文 §10)。"]
open(out("prefill_stat.md"), "w").write("\n".join(L) + "\n")

print(f"dup_sched_lines_dropped={dup}")
print(f"forwards={len(forwards)} pairs={pairs} instances={len(insts)} "
      f"arrival_matched={matched}/{len(insts)} idle_steps={len(idle)} span={span}s")
print("wrote:", ", ".join(out(x) for x in
      ("prefill_forwards.csv","gap_events.json","idle_gaps.json","prefill_stat.md")))
