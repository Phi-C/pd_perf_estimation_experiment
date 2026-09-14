#!/usr/bin/env python3
"""
decode_forwards.csv (+ decode_requests.csv / decode_backpressure.json)
    -> Perfetto trace (.json / .json.gz)

打开方式: https://ui.perfetto.dev -> Open trace file

    python3 gen_decode_trace.py -d <outdir> --label "kimi-k3 c32"

⚠️ 时间模型: 和 prefill 的 trace **完全不一样**, 别照搬
------------------------------------------------------
prefill 那边一行 "Scheduled prefill batch" = 一个 forward, 行是**连续**的, 所以
`t[k+1]-t[k]` 就是第 k 个 engine step 的完整延迟, slice 可以铺满两次调度之间。

decode 这边不成立。`[PD-DECODE]` 只在每条请求的**前 3 个 step** 打印, 稳态 decode 一行都没有,
所以两个相邻 `forward_idx` 之间通常隔着几十上百个**没有打印**的真实 step。把 slice 铺满
`[t_k, t_{k+1})` 会把这些看不见的 step 全算到某一个 forward 头上, 得到彻底错误的 step latency。

本脚本因此:

  * 每个 observed forward 画成**等宽的窄 slice**, 在它那一秒里均匀平铺。
    **slice 宽度只表示顺序和密度, 不表示任何耗时**(args 里也写了 `width_is_meaningless: true`)。
  * **不画 idle / gap 轨**。两个 observed forward 之间的空白 = "这段时间没有请求处于前 3 步",
    引擎大概率忙得很, 那不是空转。prefill trace 里的 starvation 归因在这里无从谈起。
  * 真实并发度只能看 `PD backpressure` 那几条 counter(采样很稀, 见 decode_stat.md)。

Track 布局 (pid=2)
  Counter "batch_reqs_observed(≥)" / "context_length (max in fwd)" / "attn_work_observed"
  Counter "backpressure: resident" / "running" / "waiting" / "kv_usage(%)"
  tid=1     Decode forward (observed)   每个 observed forward 一条
  tid=2000+ KV recv -> first decode     每条请求从 KV 排队到第一次 decode(区间重叠, 分泳道)
  tid=10+j  req slot j                  该 forward 内第 j 条 request 的明细
"""
import csv, json, gzip, os, collections, argparse

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--dir", default=".", help="parse_decode_log.py 的输出目录")
AP.add_argument("-o", "--out", default=None, help="输出 .json (同时写 .json.gz)")
AP.add_argument("--label", default="ATOM Decode Engine")
AP.add_argument("--slice-us", type=int, default=0,
                help="observed forward 的固定 slice 宽度(微秒); 0=按所在秒内的个数自动平铺")
AP.add_argument("--no-gz", action="store_true")
A = AP.parse_args()
P = lambda n: os.path.join(A.dir, n)
OUT = A.out or P("decode_forwards_trace_full.json")
SEC = 1_000_000

rows = list(csv.DictReader(open(P("decode_forwards.csv"))))
if not rows:
    raise SystemExit("decode_forwards.csv 是空的 —— 先跑 parse_decode_log.py")
fwd = collections.OrderedDict()
for r in rows: fwd.setdefault(int(r["forward_idx"]), []).append(r)
order = sorted(fwd)

reqs = {}
if os.path.exists(P("decode_requests.csv")):
    for r in csv.DictReader(open(P("decode_requests.csv"))): reqs[int(r["req_uid"])] = r
bp = json.load(open(P("decode_backpressure.json"))) if os.path.exists(P("decode_backpressure.json")) else []

sec_of = lambda t: (lambda h, m, s: int(h)*3600 + int(m)*60 + int(s))(*t.split(":"))

# 秒级时间戳单调化(log 里偶有 1s 回退, Perfetto 不接受乱序), 同一秒内的 forward 均匀平铺
t_abs, prev = {}, None
for k in order:
    x = sec_of(fwd[k][0]["time"])
    if prev is not None and x < prev: x = prev
    t_abs[k] = x; prev = x
t0 = min(t_abs.values())
if bp: t0 = min(t0, min(b["t"] for b in bp))
per_sec, seen = collections.Counter(t_abs.values()), collections.Counter()
start, width = {}, {}
for k in order:
    s = t_abs[k]; i = seen[s]; seen[s] += 1
    tile = SEC // per_sec[s]
    width[k] = A.slice_us or max(1, int(tile * 0.6))     # 窄 slice: 明确表示"宽度无意义"
    start[k] = (s - t0) * SEC + i * tile

events = [{"ph": "M", "pid": 2, "name": "process_name",
           "args": {"name": f"{A.label} (PD-DECODE 只覆盖每条请求的前 3 步)"}}]
events += [{"ph":"M","pid":2,"tid":1,"name":"thread_name",
            "args":{"name":"Decode forward (observed)"}},
           {"ph":"M","pid":2,"tid":1,"name":"thread_sort_index","args":{"sort_index":0}}]
nslot = max(len(g) for g in fwd.values())
for j in range(nslot):
    events += [{"ph":"M","pid":2,"tid":10+j,"name":"thread_name","args":{"name":f"req slot {j}"}},
               {"ph":"M","pid":2,"tid":10+j,"name":"thread_sort_index","args":{"sort_index":1+j}}]

# context length 决定 decode 的 attention 成本, 用它上色
BUCKETS = [(0,131072,"<128K","good"), (131072,262144,"128-256K","yellow"),
           (262144,524288,"256-512K","bad"), (524288,1<<40,">512K","terrible")]
def bucket(c):
    for lo, hi, n, col in BUCKETS:
        if lo <= c < hi: return n, col
    return ">512K", "terrible"

stat = collections.Counter()
for k in order:
    g, ts = fwd[k], start[k]
    b = len(g)
    ctxs = [int(r["context_length"]) for r in g]
    cmax, cmin = max(ctxs), min(ctxs)
    bname, col = bucket(cmax)
    steps = collections.Counter(int(r["decode_step"]) for r in g)
    stat[bname] += 1

    events += [{"ph":"C","pid":2,"ts":ts,"name":"batch_reqs_observed (>=)","args":{"reqs":b}},
               {"ph":"C","pid":2,"ts":ts,"name":"context_length (max in fwd)","args":{"ctx":cmax}},
               {"ph":"C","pid":2,"ts":ts,"name":"attn_work_observed (K query-key pairs)",
                "args":{"pairs_K": round(sum(ctxs)/1000.0, 1)}}]

    events.append({"ph":"X","pid":2,"tid":1,"ts":ts,"dur":width[k],"cname":col,
        "name": f"D{k} · {b}req(≥) · ctx {cmin:,}-{cmax:,} · q=1 · step{'/'.join(str(s) for s in sorted(steps))}",
        "args": {
            "forward_idx": k, "wall_clock": g[0]["time"],
            "batch_reqs_observed": b,
            "batch_reqs_note": "这是**下界**: PD-DECODE 只打前 3 步, batch 里过了第 3 步的请求全部看不见",
            "query_length_per_req": 1,
            "context_length_min": cmin, "context_length_max": cmax,
            "context_length_sum": sum(ctxs),
            "context_bucket": bname,
            "decode_steps_in_fwd": dict(sorted(steps.items())),
            "width_is_meaningless": True,
            "width_note": "slice 宽度只表示顺序/密度。相邻 forward_idx 之间隔着几十上百个未打印的真实 step, "
                          "所以 t[k+1]-t[k] **不是** step latency(prefill trace 才是那个模型)",
            "eq_point_note": "eq_batch_size / eq_query_length / eq_context_length 无法计算 —— "
                             "工作量等价映射需要 batch 里每一条请求的 (q_i, c_i), 这里只看得到一小撮。"
                             "decode 的 q_i 恒为 1, 缺的是完整的 c_i 集合",
            "req_ids": ",".join(r["req_id"] for r in g),
            "requests": [{"req_uid": int(r["req_uid"]), "req_id": int(r["req_id"]),
                          "batch_slot": int(r["slot"]), "decode_step": int(r["decode_step"]),
                          "context_length": int(r["context_length"]),
                          "ISL": int(r["ISL"]), "new_token": r["new_token"]} for r in g]}})

    for j, r in enumerate(g):
        c = int(r["context_length"])
        events.append({"ph":"X","pid":2,"tid":10+j,"ts":ts,"dur":width[k],
            "cname": bucket(c)[1],
            "name": f"req{r['req_id']} · step{r['decode_step']} · ctx {c:,} (ISL {int(r['ISL']):,})",
            "args": {"forward_idx": k, "req_uid": int(r["req_uid"]), "req_id": int(r["req_id"]),
                     "batch_slot": int(r["slot"]), "decode_step": int(r["decode_step"]),
                     "context_length": c, "query_length": 1, "ISL": int(r["ISL"]),
                     "gen_tokens_so_far": int(r["gen_tokens_so_far"]),
                     "new_token": r["new_token"], "kv_recv_blocks": r["kv_recv_blocks"],
                     "kv_queued_time": r["kv_queued_time"], "arrive_time": r["arrive_time"]}})

# ---- KV recv -> first decode: 每条请求进入 decode 节点的那一段 ----
# 这些区间大量重叠(并发到达), 而 Perfetto 的同轨 ph:X 必须严格嵌套, 所以做泳道分配:
# 每条 slice 放进第一条"上一条已经结束"的泳道, 泳道号 = tid 2000+lane。
kv = []
for uid, r in reqs.items():
    qw = r["kv_queued_time"] or r["arrive_time"]
    if not qw: continue
    ts = (sec_of(qw) - t0) * SEC
    dur = max(SEC // 4, (sec_of(r["first_decode_time"]) - sec_of(qw)) * SEC) if r["first_decode_time"] else SEC // 4
    kv.append((ts, dur, uid, r))
kv.sort(key=lambda x: (x[0], x[1]))
lane_end = []
for ts, dur, uid, r in kv:
    lane = next((i for i, e in enumerate(lane_end) if e <= ts), len(lane_end))
    if lane == len(lane_end): lane_end.append(0)
    lane_end[lane] = ts + dur
    events.append({"ph":"X","pid":2,"tid":2000+lane,"ts":ts,"dur":dur,"cname":bucket(int(r["ISL"]))[1],
        "name": f"req{r['req_id']} 进入 decode · ISL {int(r['ISL']):,} · {r['kv_recv_blocks']} blocks",
        "args": {"req_uid": uid, "req_id": int(r["req_id"]), "ISL": int(r["ISL"]),
                 "kv_recv_blocks": int(r["kv_recv_blocks"]),
                 "dst_slot": r["dst_slot"], "src_slot": r["src_slot"],
                 "first_token_from_prefill": r["first_token"], "injected_t0": r["injected_t0"],
                 "kv_queued_time": r["kv_queued_time"], "arrive_time": r["arrive_time"],
                 "first_decode_time": r["first_decode_time"],
                 "steps_observed": int(r["steps_observed"]),
                 "note": "[atom] 只有秒级分辨率, 所以这条 slice 的宽度下界被钳到 0.25s",
                 "lane": lane}})
for i in range(len(lane_end)):
    events += [{"ph":"M","pid":2,"tid":2000+i,"name":"thread_name",
                "args":{"name": f"KV recv -> first decode [lane {i}]"}},
               {"ph":"M","pid":2,"tid":2000+i,"name":"thread_sort_index","args":{"sort_index":100+i}}]

for x in bp:
    ts = (x["t"] - t0) * SEC
    events += [{"ph":"C","pid":2,"ts":ts,"name":"backpressure: resident","args":{"resident":x["resident"]}},
               {"ph":"C","pid":2,"ts":ts,"name":"backpressure: running","args":{"running":x["running"]}},
               {"ph":"C","pid":2,"ts":ts,"name":"backpressure: waiting/parked",
                "args":{"waiting":x["waiting"],"parked":x["parked"]}},
               {"ph":"C","pid":2,"ts":ts,"name":"backpressure: kv_usage (%)",
                "args":{"kv_usage":round(x["kv_usage"]*100,2)}}]

doc = {"traceEvents": events, "displayTimeUnit": "ms",
       "otherData": {"label": A.label,
                     "observed_forwards": str(len(fwd)), "pairs": str(len(rows)),
                     "requests": str(len(reqs)),
                     "coverage": "PD-DECODE 只覆盖每条请求的前 3 个 decode step",
                     "time_model": "slice 宽度无意义; t[k+1]-t[k] 不是 step latency; 不画 idle 轨"}}
json.dump(doc, open(OUT, "w"))
if not A.no_gz:
    with open(OUT, "rb") as i, gzip.open(OUT + ".gz", "wb") as o: o.writelines(i)

print(f"{OUT}{'' if A.no_gz else ' (+ .gz)'}: {len(events)} events, "
      f"{len(fwd)} observed forwards, {len(rows)} (forward,request) pairs, {len(reqs)} requests")
print(f"{'context 桶':16}{'forward 数':>10}{'占比':>9}")
tot = sum(stat.values())
for lo, hi, nm, _ in BUCKETS:
    if stat.get(nm): print(f"{nm:16}{stat[nm]:10d}{stat[nm]/tot*100:8.1f}%")
print("\n注意: 没有 idle 轨、没有 step latency —— 见脚本头部的时间模型说明。")
