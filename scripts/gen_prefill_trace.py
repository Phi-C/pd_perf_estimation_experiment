#!/usr/bin/env python3
"""
prefill_forwards.csv (+ gap_events.json / idle_gaps.json) -> Perfetto trace (.json / .json.gz)

打开方式: https://ui.perfetto.dev -> Open trace file

时间模型(关键, 别改错)
----------------------
CSV 的 `time` 是**调度时刻**, 所以 t[k+1]-t[k] = 第 k 个 engine step 的**完整延迟**
(forward 计算 + KV offload/RDMA 收尾 + 下一次调度)。引擎这段时间是连续忙的。

  busy step        -> forward slice 铺满 [t_k, t_{k+1}), 不画任何 gap, 利用率 counter 保持
  starvation step  -> forward slice 只占 D̂ (同 ctx 桶、同样收尾的 busy step 均值),
                      剩下的才是**真正的 GPU 空转**, 单独画在 idle 轨上, counter 掉到 0

常见错误: 把 forward 画成固定宽度、把剩下的都标成 "gap"。那样一个跑了 3s 的 long-context
forward 会被画成 "1s forward + 2s 空档", 于是 "长 context 计算" 被错误地归因成空档原因。
长 context 只会让 slice 变宽, 不会在两次 forward 之间造出洞。

Track 布局 (pid=1)
  Counter "Token Budget Util (%)" / "batch_tokens" / "batch_reqs"
  tid=1     Forward (step 执行)     每 forward 一条, 宽度 = 该 step 执行时长, 颜色 = budget 利用率
  tid=2     GPU idle (等新请求到达)  只画真正的空转
  tid=10+j  req slot j              该 forward 内第 j 条 request 的 chunk 明细
"""
import csv, json, gzip, os, collections, statistics, argparse

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--dir", default=".", help="parse_prefill_log.py 的输出目录")
AP.add_argument("-o", "--out", default=None, help="输出 .json (同时写 .json.gz)")
AP.add_argument("--budget", type=int, default=0, help="max_num_batched_tokens; 0=从 stat.md 读")
AP.add_argument("--label", default="ATOM Prefill Engine")
AP.add_argument("--no-gz", action="store_true")
AP.add_argument("--phase-split", type=float, default=0.0,
                help="profiling 阶段起点, 单位: 距第一个 forward 的秒数。"
                     "aiperf log 里 `Credit phase start: profiling` 的时刻减去 t0。0=不画分界线")
AP.add_argument("--phase-labels", default="warmup,profiling")
A = AP.parse_args()
P = lambda n: os.path.join(A.dir, n)
OUT = A.out or P("prefill_forwards_trace_full.json")
SEC = 1_000_000

BUDGET = A.budget
if not BUDGET and os.path.exists(P("prefill_stat.md")):
    for ln in open(P("prefill_stat.md")):
        if "max_num_batched_tokens" in ln:
            BUDGET = int(ln.rsplit("|", 2)[1].strip()); break
BUDGET = BUDGET or 8192

rows = list(csv.DictReader(open(P("prefill_forwards.csv"))))
fwd = collections.OrderedDict()
for r in rows: fwd.setdefault(int(r["forward_idx"]), []).append(r)
order = sorted(fwd)
load = lambda n: {int(k): v for k, v in json.load(open(P(n))).items()} if os.path.exists(P(n)) else {}
gapev, idle = load("gap_events.json"), load("idle_gaps.json")

def sec_of(t):
    h, m, s = t.split(":"); return int(h)*3600 + int(m)*60 + int(s)

# 秒级时间戳单调化(log 里偶有 1s 回退, Perfetto 不接受乱序), 同一秒内的 forward 均匀平铺
t_abs, prev = {}, None
for k in order:
    x = sec_of(fwd[k][0]["time"])
    if prev is not None and x < prev: x = prev
    t_abs[k] = x; prev = x
t0 = t_abs[order[0]]
per_sec, seen = collections.Counter(t_abs.values()), collections.Counter()
start, tile = {}, {}
for k in order:
    s = t_abs[k]; i = seen[s]; seen[s] += 1
    tile[k] = SEC // per_sec[s]
    start[k] = (s - t0) * SEC + i * tile[k]

# ---- 工作量等价点 (ragged_packed_prefill_workload_equivalence.md) ----
# A(q,c)=q*c+q(q+1)/2 ; q_eq=Q/n ; c_eq=A_total/Q-(q_eq+1)/2
# q_i=chunk_tokens(本步新算), c_i=done_tokens(已在 KV cache 的上下文)。
# CSV 里已经算好就直接用, 没有(老版 CSV)就现算, 保证向后兼容。
def eq_point(g):
    if "eq_query_length" in g[0]:
        return (float(g[0]["eq_query_length"]), float(g[0]["eq_context_length"]),
                int(g[0]["eq_batch_size"]), float(g[0]["attn_work_pairs"]),
                float(g[0]["imbalance_rq"]), float(g[0]["imbalance_rL"]))
    reqs = [(int(r["chunk_tokens"]), int(r["done_tokens"])) for r in g]
    n = len(reqs); Q = sum(q for q, _ in reqs)
    A = sum(q*c + q*(q+1)/2.0 for q, c in reqs)
    if not n or Q <= 0: return 0.0, 0.0, n, A, 0.0, 0.0
    q_eq = Q / n; L = [q + c for q, c in reqs]
    return (q_eq, A/Q - (q_eq+1)/2.0, n, A,
            max(q for q, _ in reqs)/q_eq, max(L)/(sum(L)/n) if sum(L) else 0.0)

ctx_of    = lambda k: max(int(r["done_tokens"]) + int(r["chunk_tokens"]) for r in fwd[k])
completes = lambda k: any(float(r["progress_pct"]) >= 100 for r in fwd[k])
BUCKETS = [(0,131072,"<128K"), (131072,262144,"128-256K"), (262144,524288,"256-512K"), (524288,1<<40,">512K")]
def bucket(c):
    for lo, hi, n in BUCKETS:
        if lo <= c < hi: return n
    return ">512K"

# D̂: starvation step 自己的执行时长, 用同 ctx 桶、同样"跑完了 request"的 busy step 均值折算。
# 这是保守估计(这类 step 偏慢), 所以推算出的空转时间是**下界**。
step_s = {k: t_abs[order[i+1]] - t_abs[k] for i, k in enumerate(order[:-1])}
base = collections.defaultdict(list)
for k in step_s:
    if k not in idle and completes(k): base[bucket(ctx_of(k))].append(step_s[k])
Dhat = {b: statistics.mean(v) for b, v in base.items()}
fallback = statistics.mean([v for k, v in step_s.items() if k not in idle]) if step_s else 0.5

events = [{"ph": "M", "pid": 1, "name": "process_name",
           "args": {"name": f"{A.label} (budget={BUDGET})"}}]
if A.phase_split > 0:
    events += [{"ph":"M","pid":1,"tid":0,"name":"thread_name","args":{"name":"aiperf Phase"}},
               {"ph":"M","pid":1,"tid":0,"name":"thread_sort_index","args":{"sort_index":-1}}]
for tid, nm, si in [(1, "Forward (step 执行)", 0), (2, "GPU idle (等新请求到达)", 1)]:
    events += [{"ph":"M","pid":1,"tid":tid,"name":"thread_name","args":{"name":nm}},
               {"ph":"M","pid":1,"tid":tid,"name":"thread_sort_index","args":{"sort_index":si}}]
for j in range(max(len(g) for g in fwd.values())):
    events += [{"ph":"M","pid":1,"tid":10+j,"name":"thread_name","args":{"name":f"req slot {j}"}},
               {"ph":"M","pid":1,"tid":10+j,"name":"thread_sort_index","args":{"sort_index":2+j}}]

color = lambda p: "good" if p >= 99.9 else "yellow" if p >= 75 else "bad" if p >= 25 else "terrible"
stat = collections.defaultdict(lambda: [0, 0.0])

for n, k in enumerate(order):
    g, ts = fwd[k], start[k]
    bt, br = int(g[0]["batch_tokens"]), int(g[0]["batch_reqs"])
    pct = round(bt * 100.0 / BUDGET, 2)
    e, c = gapev.get(k, {}), ctx_of(k)
    qeq, ceq, beq, aw, rq, rL = eq_point(g)

    if n + 1 < len(order):
        span = start[order[n+1]] - ts
        d = max(1, min(int(Dhat.get(bucket(c), fallback) * SEC), span)) if k in idle else span
    else:
        d = span = tile[k]

    tags, notes = [], []
    if e.get("autotune_done"):
        tags.append("JIT"); notes.append(f"首次 Triton autotune {'/'.join(e.get('kernels') or [])} "
                                         f"({e['autotune_done']} 次, 各 rank 合计 {e.get('autotune_sec',0)}s)")
    if e.get("aiter"):
        tags.append("JIT"); notes.append(f"aiter {'/'.join(e.get('kernels') or [])} 首次 dispatch ({e['aiter']} ranks)")
    if e.get("rdma") or e.get("lm_store"):
        tags.append("KV→D"); notes.append(
            f"收尾: LMCache offload {e.get('lm_store_tokens',0)} tok / {e.get('lm_store_ms',0)} ms"
            f" + Mooncake RDMA {e.get('rdma',0)} 笔 / {e.get('rdma_GB',0)} GB 送往 decode")
    if c >= 262144:
        tags.append(f"ctx {c//1024}K"); notes.append(f"chunked-prefill 要 attend {c} tok 的已有 KV, 计算本身更慢")
    if e.get("alloc_fail"):
        notes.append(f"(同窗口 {e['alloc_fail']} 条 LMCache 内存告警; 控制 ctx 与是否收尾后这类 step 并不更慢, 不作耗时归因)")

    events += [{"ph":"C","pid":1,"ts":ts,"name":"Token Budget Util (%)","args":{"util":pct}},
               {"ph":"C","pid":1,"ts":ts,"name":"batch_tokens","args":{"tokens":bt}},
               {"ph":"C","pid":1,"ts":ts,"name":"batch_reqs","args":{"reqs":br}},
               {"ph":"C","pid":1,"ts":ts,"name":"eq_query_length (q_eq)","args":{"q_eq":round(qeq,1)}},
               {"ph":"C","pid":1,"ts":ts,"name":"eq_context_length (c_eq)","args":{"c_eq":round(ceq,1)}},
               {"ph":"C","pid":1,"ts":ts,"name":"attn_work (G query-key pairs)","args":{"A_total":round(aw/1e9,3)}}]
    events.append({"ph":"X","pid":1,"tid":1,"ts":ts,"dur":d, "cname":color(pct),
        "name": (f"F{k} · {br}req · {bt}tok · {pct:g}% ≙ (q={qeq:,.0f}, c={ceq:,.0f}, b={beq})"
                 + (" [" + " ".join(tags) + "]" if tags else "")),
        "args": {"forward_idx":k, "wall_clock":g[0]["time"], "batch_reqs":br, "batch_tokens":bt,
                 "budget":BUDGET, "budget_util_pct":pct, "budget_idle_tokens":BUDGET-bt,
                 "max_context_tokens":c, "step_latency_s":span/SEC,
                 # 工作量等价法: 把这个 ragged/packed batch 映射到同构性能表里的一个点。
                 # 拿 (q_eq, c_eq) 在 batch-size=b 的曲面上做二维插值即可估 prefill 时间。
                 "eq_batch_size":beq, "eq_query_length":round(qeq,1), "eq_context_length":round(ceq,1),
                 "eq_note":f"等价同构点 (q_eq={qeq:,.1f}, c_eq={ceq:,.1f}, b={beq}); "
                           f"注意 c_eq != 各请求 context 的算术平均",
                 "attn_work_pairs":int(aw), "attn_work_Gpairs":round(aw/1e9,3),
                 "imbalance_rq":round(rq,3), "imbalance_rL":round(rL,3),
                 "eq_confidence":("高 (近同构)" if rq <= 1.2 and rL <= 1.2 else
                                  "中" if rq <= 2.5 and rL <= 1.8 else "低 (长度严重不均衡, 需异构样本校正)"),
                 "slice_width_is_estimated": k in idle,
                 "why_this_long": "; ".join(notes) or "常规 step",
                 "log_evidence": e or "(该 step 窗口内 log 无特征事件)",
                 "req_ids": ",".join(r["req_id"] for r in g),
                 "requests": [{"req_id":int(r["req_id"]), "chunk_tokens":int(r["chunk_tokens"]),
                               "done_tokens":int(r["done_tokens"]), "ISL":int(r["ISL"]),
                               "progress_pct":float(r["progress_pct"]),
                               "share_of_batch_%":round(int(r["chunk_tokens"])*100.0/bt,2) if bt else 0}
                              for r in g]}})
    for j, r in enumerate(g):
        ck, isl, dn = int(r["chunk_tokens"]), int(r["ISL"]), int(r["done_tokens"])
        events.append({"ph":"X","pid":1,"tid":10+j,"ts":ts,"dur":d,
            "name": f"req{r['req_id']} · {ck}tok · {dn}->{dn+ck}/{isl} ({r['progress_pct']}%)",
            "args": {"forward_idx":k, "req_id":int(r["req_id"]), "chunk_tokens":ck,
                     "done_tokens_before":dn, "done_tokens_after":dn+ck, "ISL":isl,
                     "progress_pct":float(r["progress_pct"]), "remaining_tokens":isl-dn-ck,
                     "share_of_batch_%":round(ck*100.0/bt,2) if bt else 0}})

    if k in idle and span - d > 0:
        who = ", ".join(f'req{x["req_id"]}(ISL={x["ISL"]}) 在 {x["arrived_after_s"]}s 后到达'
                        for x in (idle[k].get("next_reqs") or [])) or "?"
        events.append({"ph":"X","pid":1,"tid":2,"ts":ts+d,"dur":span-d,"cname":"white",
            "name": f"IDLE {(span-d)/SEC:.2f}s · 等新请求 (after F{k})",
            "args": {"after_forward":k, "idle_seconds_est":round((span-d)/SEC,2),
                     "step_latency_s":span/SEC, "exec_part_estimated_s":round(d/SEC,2),
                     "cause":"引擎 running 与 waiting 均为空, 没有任何 request 可以调度",
                     "next_forward_runs":who,
                     "why":"客户端并发有限, agentic replay 同一条 lane 要等上一轮 decode 结束才发下一轮, "
                           "prefill 节点因此周期性空转",
                     "note":"执行部分按同 ctx 桶、同样收尾的 busy step 均值折算(保守), CSV 时间戳仅秒级分辨率"}})
        stat["真正空转 (等新请求到达)"][0] += 1
        stat["真正空转 (等新请求到达)"][1] += (span-d)/SEC
    if n + 1 < len(order):
        key = ("step 执行: 首次 JIT" if "JIT" in tags else
               "step 执行: 收尾 KV offload+RDMA" if "KV→D" in tags else
               "step 执行: 长 context 计算" if c >= 262144 else "step 执行: 常规 forward")
        stat[key][0] += 1; stat[key][1] += d/SEC

# ---- aiperf 阶段分界 ----
# warmup 跑在冷 prefix cache 上, 每个 token 都要真算; profiling 段命中率 95.7%。
# 两段混在一起统计吞吐会把结果拉偏一倍, 所以画一条全局竖线把它们分开。
if A.phase_split > 0:
    SPLIT = int(A.phase_split * SEC)
    END   = start[order[-1]] + tile[order[-1]]
    L1, L2 = (A.phase_labels.split(",") + ["warmup", "profiling"])[:2]
    events.append({"ph":"i","pid":1,"tid":0,"ts":SPLIT,"s":"g",
        "name": f"▼ {L2} 开始 (aiperf: Credit phase start)",
        "args": {"t_offset_s": A.phase_split,
                 "source": "aiperf log: `Credit phase start: profiling`",
                 "note": "左侧 warmup 冷缓存, 右侧 profiling 才是被计入 CI 指标的 3600s 窗口"}})
    for lbl, a, b, cn in [(L1, 0, SPLIT, "thread_state_iowait"),
                          (L2, SPLIT, END, "thread_state_running")]:
        events.append({"ph":"X","pid":1,"tid":0,"ts":a,"dur":b-a,"cname":cn,
            "name": f"{lbl}  ({(b-a)/SEC:.0f}s)",
            "args": {"phase": lbl, "duration_s": round((b-a)/SEC, 1),
                     "split_at_s": A.phase_split}})

doc = {"traceEvents": events, "displayTimeUnit": "ms",
       "otherData": {"label": A.label, "max_num_batched_tokens": str(BUDGET),
                     "forwards": str(len(fwd)), "pairs": str(len(rows)),
                     "time_model": "t[k+1]-t[k] = engine step 延迟; 只有 starvation step 才有真正空档",
                     "profiling_starts_at_s": str(A.phase_split) if A.phase_split > 0 else "(未标注)"}}
json.dump(doc, open(OUT, "w"))
if not A.no_gz:
    with open(OUT, "rb") as i, gzip.open(OUT + ".gz", "wb") as o: o.writelines(i)

tot = sum(v[1] for v in stat.values())
print(f"{OUT}{'(.gz)' if not A.no_gz else ''}: {len(events)} events, {len(fwd)} forwards, D̂={ {b:round(v,2) for b,v in Dhat.items()} }\n")
print(f"{'':38}{'次数':>7}{'秒':>9}{'占比':>8}{'平均':>9}")
for c_, (n_, s_) in sorted(stat.items(), key=lambda x: -x[1][1]):
    print(f"{c_:38}{n_:7d}{s_:9.0f}{s_/tot*100:7.1f}%{s_/n_:8.2f}s")
print(f"{'合计':38}{'':7}{tot:9.0f}")
