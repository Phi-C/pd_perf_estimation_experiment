#!/usr/bin/env python3
"""
ATOM decode node CI log  ->  decode_forwards.csv + decode_requests.csv
                             + decode_gap_events.json + decode_stat.md

用法:
    gh api repos/ROCm/ATOM/actions/jobs/<JOB_ID>/logs --allow-escape-sequences > job.log
    python3 parse_decode_log.py job.log -o <outdir> [--label "kimi-k3 c32"]

只读一遍 log。所有正则都是"匹配不到就产出 0", 换用例不会崩。

decode node 的 log 长什么样
---------------------------
1p1d 的 CI job 把 prefill node 和 decode node 的 stdout **混在同一条 slurm.out 流里**,
没有 node 前缀。靠 tag 区分:

    prefill node :  "Scheduled prefill batch"  / [PRODUCER] / [SCHEDULER] ... save
    decode  node :  [SCHEDULER-CONSUMER] / [PD-TRANSITION] / [PD-FIRST-DECODE] /
                    [PD-DECODE] / [CONSUMER] / [SCHEDULER] ... recv

一条请求在 decode 侧的完整生命周期能看到 4 类行:

    [SCHEDULER-CONSUMER] Queued req 1971 for remote KV recv (488 blocks, dst_slot=19,
                         src_slot=15), transfer_id=1968, remote_host=..., ...
        -> 全局 req_id(和 prefill node 同一套编号) + KV 传输块数 + 落到哪个 batch slot
    [PD-TRANSITION]  seq 19: num_tokens=..., num_prompt=..., blocks=..., first_token=...
        -> num_prompt 就是 ISL; first_token 是 prefill node 产出的那个 token
    [PD-FIRST-DECODE] seq 19: num_tokens=..., blocks=..., injected_t0=True, context_will_be=...
    [PD-DECODE]      seq 19: comp_tokens=k, new_token=(...), num_tokens=..., blocks=...
        -> 真正的 decode forward 证据

⚠️ 最重要的一条限制: [PD-DECODE] 只打前 3 步
--------------------------------------------
本 job 8252 条 PD-DECODE 行 = 2861(comp_tokens=1) + 2875(=2) + 2516(=3), 一条 comp_tokens>3
的都没有。也就是说引擎**只在每条请求的前 3 个 decode step 打印它**, 稳态 decode 完全不可见。

推论(决定了这个脚本能产出什么、不能产出什么):

  * 一个 decode forward 的**真实 batch 组成不可知** —— 任一时刻 batch 里绝大多数请求早已
    过了第 3 步, 不再打印。所以 CSV 里的 `batch_reqs_observed` 是**下界**, 不是 batch size。
  * 因此 `eq_batch_size / eq_query_length / eq_context_length`(工作量等价点)**算不出来**:
    等价映射要求知道 batch 里**每一条**请求的 (q_i, c_i), 而我们只看得到其中一小撮。
    这三列按需求**不写进 CSV**。等 decode 侧补上 per-step 埋点再加。
    (decode 的 q_i 恒等于 1 —— new_token 元组本 job 8252 行全是 1 个 token, 无投机解码 ——
     所以真正缺的只有 c_i 的完整集合。)

⚠️ 第二条限制: PD-DECODE 行**不是按 forward 排好序的**
------------------------------------------------------
prefill 那边一行 "Scheduled prefill batch" 就是一个 forward, 天然对齐。decode 这边没有这种行,
只能把 PD-DECODE 行**聚成组**。而实测数据表明 log 里这些行的顺序**违反 forward-major**:

    719 slot355 c1 (req A)   720 slot356 c1 (req B)  ... 724 slot362 c1 (req F)
    725 slot357 c2 (req A)   726 slot357 c3 (req A)     <- A 的第 2、3 步紧挨着打
    727 slot358 c2 (req F)   728 slot359 c2 (req C) ...

若 A 与 B 在 c=1 时同处一个 forward, 则 c=2 时也必须同处一个 forward(decode 是齐步走的);
但 A@2、A@3 在 B@2 之前就打出来了。这个矛盾无解 —— 说明打印来自 worker 侧、相对 scheduler
被缓冲/重排过。所以:

  * 本脚本用的分组规则是 **"同一秒内、连续、请求不重复的最长一段" = 一个 observed forward**。
    这样得到的每一组都是某个真实 forward 的**子集**(不会把两条同一请求的行塞进一组),
    但**可能把一个真实 forward 拆成几组**。
  * `forward_idx` 因此是 **observed forward 的序号**, 不是引擎 step 序号。两个相邻
    forward_idx 之间通常隔着几十上百个没有打印的真实 step。
  * **不要用 t[k+1]-t[k] 当 step latency** —— 这是 prefill 脚本的时间模型, 在 decode 侧不成立。

时间戳
------
和 prefill 一样: 行首 `2026-08-29T12:31:04.2497087Z` 是 GitHub runner 的**日志转发时刻**
(比引擎晚 10~30s 且批量刷新), 只能定顺序。引擎自己打的 `[atom HH:MM:SS]` 是秒级真实时刻。
"""
import re, sys, csv, json, os, argparse, collections, statistics

AP = argparse.ArgumentParser()
AP.add_argument("log")
AP.add_argument("-o", "--outdir", default=".")
AP.add_argument("--label", default="", help="写进 stat.md / trace 标题的用例名")
ARGS = AP.parse_args()
os.makedirs(ARGS.outdir, exist_ok=True)
out = lambda n: os.path.join(ARGS.outdir, n)

# ---------------------------------------------------------------- 正则
# 换 engine 基本只需要改这一段。
RE_ATOM   = re.compile(r"\[atom (\d\d):(\d\d):(\d\d)\]")
RE_DEC    = re.compile(r"\[PD-DECODE\] seq (\d+): comp_tokens=(\d+), "
                       r"new_token=\(([^)]*)\), num_tokens=(\d+), blocks=(\d+)")
RE_TRANS  = re.compile(r"\[PD-TRANSITION\] seq (\d+): num_tokens=(\d+), num_prompt=(\d+), "
                       r"blocks=(\d+), first_token=(\d+)")
RE_FIRST  = re.compile(r"\[PD-FIRST-DECODE\] seq (\d+): num_tokens=(\d+), blocks=(\d+), "
                       r"injected_t0=(\w+), last_block_num=(\d+), context_will_be=(\d+)")
RE_QUEUED = re.compile(r"\[SCHEDULER-CONSUMER\] Queued req (\d+) for remote KV recv "
                       r"\((\d+) blocks, dst_slot=(\d+), src_slot=(\d+)\), transfer_id=(\d+)")
RE_BP     = re.compile(r"PD backpressure: parked=(\d+), waiting=(\d+), running=(\d+), "
                       r"resident=(\d+)/(\d+), kv_usage=([\d.]+)")
# —— step 窗口内"这一步为什么慢"的证据(和 prefill 脚本同一套) ——
RE_RECVQ  = re.compile(r"\[CONSUMER\] start_load_kv: (\d+) reqs_to_recv")
RE_WRDONE = re.compile(r"\[CONSUMER\] Write-done received for req (\d+)")
RE_ALLOC  = re.compile(r"Failed to allocate memory block")
RE_AUTOT  = re.compile(r"Triton autotuning for function (\w+)")
RE_AUTOD  = re.compile(r"finished after ([\d.]+)s")
RE_AITER  = re.compile(r"\[aiter\] type hints mismatch, override to --> (\w+)")
RE_OSL    = re.compile(r"OSL mismatch: got (\d+) tokens \(requested (\d+)")
# —— 配置 ——
RE_CFG    = {k: re.compile(r'"?%s"?\s*[:=]\s*(\d+)' % k) for k in
             ("max_num_batched_tokens", "block_size", "max_num_seqs")}
RE_DCFG   = {k: re.compile(r"^\s*(?:\[slurm\.out\]\s*)?%s=(\S+)" % k, re.M) for k in
             ("DECODE_TP", "DECODE_DCP_SIZE", "DECODE_MAX_NUM_SEQS",
              "DECODE_MAX_NUM_BATCHED_TOKENS", "DECODE_CUDAGRAPH", "DECODE_WORKERS")}

sec_of = lambda a: int(a.group(1))*3600 + int(a.group(2))*60 + int(a.group(3))
hms    = lambda a: f"{a.group(1)}:{a.group(2)}:{a.group(3)}"

dec_lines, trans, first, queued, bp, osl = [], [], [], [], [], []
cfg = {}
gapev = collections.defaultdict(lambda: collections.Counter())
cur = -1                     # 当前 observed forward 序号(证据行归给它)

with open(ARGS.log, "r", errors="replace") as fh:
    for line in fh:
        a = RE_ATOM.search(line)
        if "[PD-DECODE]" in line:
            m = RE_DEC.search(line)
            if m and a:
                toks = [t for t in m.group(3).split(",") if t.strip()]
                dec_lines.append({"t": sec_of(a), "wall": hms(a), "slot": int(m.group(1)),
                                  "step": int(m.group(2)), "new_tokens": len(toks),
                                  "new_token": toks[0].strip() if toks else "",
                                  "nt": int(m.group(4)), "blocks": int(m.group(5))})
            continue
        if "[PD-TRANSITION]" in line:
            m = RE_TRANS.search(line)
            if m and a:
                trans.append({"t": sec_of(a), "wall": hms(a), "slot": int(m.group(1)),
                              "nt": int(m.group(2)), "ISL": int(m.group(3)),
                              "blocks": int(m.group(4)), "first_token": int(m.group(5))})
            continue
        if "[PD-FIRST-DECODE]" in line:
            m = RE_FIRST.search(line)
            if m and a:
                first.append({"t": sec_of(a), "slot": int(m.group(1)), "nt": int(m.group(2)),
                              "blocks": int(m.group(3)), "injected_t0": m.group(4),
                              "context_will_be": int(m.group(6))})
            continue
        if "[SCHEDULER-CONSUMER]" in line:
            m = RE_QUEUED.search(line)
            if m and a:
                queued.append({"t": sec_of(a), "wall": hms(a), "req_id": int(m.group(1)),
                               "blocks": int(m.group(2)), "dst_slot": int(m.group(3)),
                               "src_slot": int(m.group(4)), "transfer_id": int(m.group(5))})
            continue
        if "PD backpressure" in line:
            m = RE_BP.search(line)
            if m and a:
                bp.append({"t": sec_of(a), "wall": hms(a), "parked": int(m.group(1)),
                           "waiting": int(m.group(2)), "running": int(m.group(3)),
                           "resident": int(m.group(4)), "max_seqs": int(m.group(5)),
                           "kv_usage": float(m.group(6))})
            continue
        if "OSL mismatch" in line:
            m = RE_OSL.search(line)
            if m: osl.append((int(m.group(1)), int(m.group(2))))
        if len(cfg) < len(RE_CFG) + len(RE_DCFG):
            for k, r in RE_CFG.items():
                if k not in cfg:
                    mm = r.search(line)
                    if mm: cfg[k] = mm.group(1)
            for k, r in RE_DCFG.items():
                if k not in cfg and k in line:
                    mm = r.search(line)
                    if mm and mm.group(1): cfg[k] = mm.group(1)
        # 证据行归到当前 observed forward(分组在下面做, 这里先按 PD-DECODE 行数推进)
        if cur < 0: continue
        if RE_ALLOC.search(line):            gapev[cur]["alloc_fail"] += 1
        elif RE_RECVQ.search(line):          gapev[cur]["kv_recv_start"] += 1
        elif RE_WRDONE.search(line):         gapev[cur]["kv_recv_done"] += 1
        elif RE_AUTOT.search(line):          gapev[cur]["autotune"] += 1
        elif RE_AITER.search(line):          gapev[cur]["aiter"] += 1

if not dec_lines:
    sys.exit("没有匹配到任何 '[PD-DECODE]' 行 —— 确认这份 log 里有 decode node 的输出")

# ------------------------------------------------- ① 去掉 log 重放行
# CI log 会把一小段整体重放一遍(prefill 侧同样的坑)。decode 侧的判据更简单:
# (slot, step, num_tokens, 秒) 完全相同且在很近的位置 -> 重放。
dedup, seen_recent = [], {}
dups = 0
for d in dec_lines:
    key = (d["slot"], d["step"], d["nt"], d["t"])
    if key in seen_recent and len(dedup) - seen_recent[key] <= 64:
        dups += 1; continue
    seen_recent[key] = len(dedup); dedup.append(d)
dec_lines = dedup

# ------------------------------------------------- ② request 实例化
# 一条请求在第 k 步有 num_tokens = ISL + k, 所以 ISL = num_tokens - comp_tokens 是它的不变量。
# 但短 ISL 会撞车(本 job 有 ISL=483 的两条请求几乎同时在跑), 所以还要求:
#   - 只接在 last_step == step-1 的 open 实例后面
#   - 撞车时优先挑 slot 相同的, 其次挑最近一次出现的
open_by_isl = collections.defaultdict(list)          # ISL -> [inst_idx, ...]
insts = []
for i, d in enumerate(dec_lines):
    isl = d["nt"] - d["step"]
    cand = [j for j in open_by_isl[isl] if insts[j]["last_step"] == d["step"] - 1]
    if cand:
        same = [j for j in cand if insts[j]["slot_last"] == d["slot"]]
        j = (same or cand)[-1]
    else:
        j = len(insts)
        insts.append({"uid": j, "ISL": isl, "first_i": i, "steps": [], "slots": set(),
                      "last_step": 0, "slot_last": d["slot"], "t_first": d["t"],
                      "partial_start": d["step"] != 1})
        open_by_isl[isl].append(j)
    it = insts[j]
    it["steps"].append(i); it["slots"].add(d["slot"])
    it["last_step"] = d["step"]; it["slot_last"] = d["slot"]; it["t_last"] = d["t"]
    d["uid"] = j

# ------------------------------------------------- ③ observed forward 分组
# 规则: 同一秒内、连续、req 不重复的最长一段 = 一个 observed forward。
# 得到的组保证是某个真实 forward 的子集(见文件头), batch_reqs_observed 是下界。
groups, g = [], []
for d in dec_lines:
    if g and (d["t"] != g[0]["t"] or any(x["uid"] == d["uid"] for x in g)):
        groups.append(g); g = []
    g.append(d)
if g: groups.append(g)
for k, grp in enumerate(groups):
    for d in grp: d["fwd"] = k

# 自检: decode 是齐步走的, 同组的两条请求下一次出现也应该同组。统计违反率。
pos = {}
for k, grp in enumerate(groups):
    for d in grp: pos[(d["uid"], d["step"])] = k
chain_ok = chain_bad = 0
for it in insts:
    ks = [pos[(it["uid"], dec_lines[i]["step"])] for i in it["steps"]]
    if all(ks[i+1] - ks[i] >= 1 for i in range(len(ks)-1)): chain_ok += 1
    else: chain_bad += 1

# ------------------------------------------------- ④ 把 req_id 接到实例上
# ⚠️ 三个 tag 里的 "seq" 不是一个东西:
#   [PD-TRANSITION] seq / [PD-FIRST-DECODE] seq / [SCHEDULER-CONSUMER] req
#       -> decode node 的**全局 request id**(本 job 0..2826, 2827 个, 三者一一对应,
#          blocks 字段 2827/2827 完全吻合)
#   [PD-DECODE] seq
#       -> running batch 里的**槽位下标**, 每步都会重排, 不是 request 身份
# 所以 req_id 直接取 PD-TRANSITION 的 seq, 再用它去接 queued / first-decode。
#
# ⚠️ decode node 的 req_id 是它自己的一套编号(按 KV 到达 decode 的顺序), **和 prefill node
#    的 req_id 不是同一套**。本 job 两边的 ISL 多重集完全相等(2827/2827), 但编号顺序不同
#    (prefill req 0 的 ISL 是 61545, decode req 0 的是 58763)。要跨节点对齐请按 ISL 匹配。
q_by_id = {q["req_id"]: q for q in queued}
f_by_id = {f["slot"]: f for f in first}          # 这里的 slot 字段装的其实是 req_id
for tr in trans:
    tr["req_id"] = tr["slot"]                    # PD-TRANSITION 的 seq 就是 req_id
    q = q_by_id.get(tr["req_id"]); f = f_by_id.get(tr["req_id"])
    tr["kv_recv_blocks"] = q["blocks"] if q else tr["blocks"]
    tr["dst_slot"]  = q["dst_slot"] if q else -1
    tr["src_slot"]  = q["src_slot"] if q else -1
    tr["queued_t"]  = q["t"] if q else None
    tr["queued_wall"] = q["wall"] if q else ""
    tr["injected_t0"] = f["injected_t0"] if f else ""

# 实例(由 PD-DECODE 链条切出来) <-> transition 记录, 按 ISL 做 FIFO 匹配。
# ISL 是两边共有的不变量; 少数短 ISL 会撞车, 用到达顺序消歧。
isl_pool = collections.defaultdict(collections.deque)
for tr in sorted(trans, key=lambda x: x["t"]): isl_pool[tr["ISL"]].append(tr)

matched = 0
for it in sorted(insts, key=lambda x: x["first_i"]):
    pool = isl_pool.get(it["ISL"])
    tr = pool.popleft() if pool else None
    it["req_id"]      = tr["req_id"] if tr else -1
    it["arrive_t"]    = tr["t"] if tr else None
    it["arrive_wall"] = tr["wall"] if tr else ""
    it["queued_t"]    = tr["queued_t"] if tr else None
    it["queued_wall"] = tr["queued_wall"] if tr else ""
    it["blocks"]      = tr["kv_recv_blocks"] if tr else 0
    it["dst_slot"]    = tr["dst_slot"] if tr else -1
    it["src_slot"]    = tr["src_slot"] if tr else -1
    it["first_token"] = tr["first_token"] if tr else -1
    it["injected_t0"] = tr["injected_t0"] if tr else ""
    matched += tr is not None

# ------------------------------------------------- ⑤ 写 decode_forwards.csv
# 一行 = 一个 (observed forward, request) 对。
# 注意: 没有 eq_batch_size / eq_query_length / eq_context_length 三列 —— 见文件头的限制说明。
with open(out("decode_forwards.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["forward_idx","time","req_uid","req_id","slot","decode_step",
                "context_length","query_length","ISL","gen_tokens_so_far","blocks",
                "new_token","batch_reqs_observed","batch_tokens_observed",
                "kv_recv_blocks","kv_queued_time","arrive_time","wait_to_first_decode_s"])
    for k, grp in enumerate(groups):
        b = len(grp)
        for d in grp:
            it = insts[d["uid"]]
            wait = (d["t"] - it["arrive_t"]) if (it["arrive_t"] is not None and d["step"] == 1) else ""
            w.writerow([k, d["wall"], d["uid"], it["req_id"], d["slot"], d["step"],
                        d["nt"], d["new_tokens"], it["ISL"], d["step"], d["blocks"],
                        d["new_token"], b, b * d["new_tokens"],
                        it["blocks"], it["queued_wall"], it["arrive_wall"], wait])

# ------------------------------------------------- ⑥ 写 decode_requests.csv
with open(out("decode_requests.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["req_uid","req_id","ISL","kv_recv_blocks","dst_slot","src_slot",
                "first_token","injected_t0","kv_queued_time","arrive_time",
                "kv_recv_seconds","first_decode_time","last_observed_time",
                "steps_observed","first_step","last_step","batch_slots"])
    for it in sorted(insts, key=lambda x: x["first_i"]):
        st = [dec_lines[i]["step"] for i in it["steps"]]
        recv = (it["arrive_t"] - it["queued_t"]) if (it["arrive_t"] is not None
                and it["queued_t"] is not None) else ""
        w.writerow([it["uid"], it["req_id"], it["ISL"], it["blocks"], it["dst_slot"],
                    it["src_slot"], it["first_token"], it["injected_t0"], it["queued_wall"],
                    it["arrive_wall"], recv, dec_lines[it["steps"][0]]["wall"],
                    dec_lines[it["steps"][-1]]["wall"], len(st), min(st), max(st),
                    "|".join(str(s) for s in sorted(it["slots"]))])

json.dump({str(k): dict(v) for k, v in gapev.items() if v},
          open(out("decode_gap_events.json"), "w"), indent=0)
json.dump(bp, open(out("decode_backpressure.json"), "w"), indent=0)

# ------------------------------------------------- ⑦ stat.md
span = dec_lines[-1]["t"] - dec_lines[0]["t"]
bsz  = collections.Counter(len(g) for g in groups)
stepc = collections.Counter(d["step"] for d in dec_lines)
ctx  = sorted(it["ISL"] for it in insts)
pct  = lambda v, p: v[min(len(v)-1, int(len(v)*p))] if v else 0
per_sec = collections.Counter(g[0]["t"] for g in groups)

L = [f"# Decode forward 统计 {ARGS.label}", "", f"log: `{os.path.abspath(ARGS.log)}`", "",
     "| 参数 | 值 |", "|---|---|"]
for k in ("DECODE_TP","DECODE_DCP_SIZE","DECODE_WORKERS","DECODE_CUDAGRAPH",
          "DECODE_MAX_NUM_SEQS","DECODE_MAX_NUM_BATCHED_TOKENS",
          "max_num_batched_tokens","block_size","max_num_seqs"):
    if cfg.get(k): L.append(f"| `{k}` | {cfg[k]} |")
L += ["",
      f"- PD-DECODE 行: **{len(dec_lines)}** (丢弃重放 {dups} 行)",
      f"- observed forward: **{len(groups)}**",
      f"- request 实例: **{len(insts)}**, 其中 {sum(1 for i in insts if i['partial_start'])} "
      f"条的第 1 步没打到 log",
      f"- [PD-TRANSITION] {len(trans)} 条 / [PD-FIRST-DECODE] {len(first)} 条 / "
      f"[SCHEDULER-CONSUMER] Queued {len(queued)} 条; 实例接上 req_id: {matched}/{len(insts)}",
      f"- 时间跨度: {span}s ({dec_lines[0]['wall']} -> {dec_lines[-1]['wall']})",
      f"- 齐步走自检: 链条单调 {chain_ok} / 违反 {chain_bad}",
      "",
      "## ⚠️ 这份 log 能给到哪一步", "",
      "`[PD-DECODE]` **只在每条请求的前 3 个 decode step 打印**(本 job: "
      f"step1={stepc.get(1,0)}, step2={stepc.get(2,0)}, step3={stepc.get(3,0)}, "
      f"step>3 = {sum(v for s,v in stepc.items() if s>3)} 条)。稳态 decode 完全不可见。", "",
      "因此:", "",
      "- `batch_reqs_observed` 是**真实 batch size 的下界**, 不是 batch size;",
      "- **`eq_batch_size` / `eq_query_length` / `eq_context_length` 算不出来**, 没有写进 CSV ——",
      "  工作量等价映射要求知道 batch 里每一条请求的 `(q_i, c_i)`, 而我们只看得到其中一小撮。",
      f"  (decode 的 `q_i` 恒为 1: 本 job {len(dec_lines)} 行 new_token 全是单 token, 无投机解码。)",
      "- `forward_idx` 是 **observed forward 的序号**, 不是引擎 step 序号; 相邻两个之间通常隔着",
      "  几十上百个没有打印的真实 step。**不要**用 `t[k+1]-t[k]` 当 step latency",
      "  (那是 prefill 脚本的时间模型, decode 侧不成立)。", "",
      "## observed forward 的 batch_reqs 分布(下界)", "",
      "| batch_reqs_observed | forward 数 | 占比 |", "|---:|---:|---:|"]
for n in sorted(bsz): L.append(f"| {n} | {bsz[n]} | {bsz[n]*100.0/len(groups):.2f}% |")

L += ["", "## 进入 decode 的请求 ISL 分布", "",
      "| 分位 | ISL |", "|---|---:|"]
for p, nm in ((0.0,"min"),(0.25,"p25"),(0.5,"p50"),(0.75,"p75"),(0.9,"p90"),(0.99,"p99"),(1.0,"max")):
    L.append(f"| {nm} | {pct(ctx, p) if p < 1 else ctx[-1]:,} |")
L += ["", f"decode 侧 context length 在第 k 步 = `ISL + k`, query length 恒为 1。",
      f"整段时间里 observed forward 的密度: 中位 {statistics.median(per_sec.values()):.0f} 个/秒, "
      f"最大 {max(per_sec.values())} 个/秒 (只统计有打印的秒, 共 {len(per_sec)} 秒)。"]

if bp:
    L += ["", "## PD backpressure 采样", "",
          f"共 {len(bp)} 条采样。resident 上限 {bp[0]['max_seqs']}; ",
          f"resident 中位 {statistics.median(b['resident'] for b in bp):.0f}, "
          f"最大 {max(b['resident'] for b in bp)}; "
          f"kv_usage 中位 {statistics.median(b['kv_usage'] for b in bp):.3f}, "
          f"最大 {max(b['kv_usage'] for b in bp):.3f}。",
          "", "这是目前唯一能看到**真实并发度**的行 —— 但采样太稀, 只能做量级参考,",
          "不能用来补 `batch_reqs_observed`。"]
if osl:
    bad = sum(1 for g, r in osl if g != r)
    L += ["", f"## OSL mismatch", "", f"{len(osl)} 条告警, 其中 got != requested 的 {bad} 条。"]
open(out("decode_stat.md"), "w").write("\n".join(L) + "\n")

print(f"dup_pd_decode_lines_dropped={dups}")
print(f"pd_decode_lines={len(dec_lines)} observed_forwards={len(groups)} instances={len(insts)} "
      f"req_id_matched={matched}/{len(insts)} chain_ok={chain_ok} chain_bad={chain_bad} span={span}s")
print(f"steps_logged={dict(sorted(stepc.items()))}  <-- 只有前 3 步, 见 decode_stat.md")
print("wrote:", ", ".join(out(x) for x in
      ("decode_forwards.csv","decode_requests.csv","decode_gap_events.json",
       "decode_backpressure.json","decode_stat.md")))
