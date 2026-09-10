#!/usr/bin/env python3
"""用 aiperf 实测的性能点估 profiling 段的理论吞吐。

步骤(按需求):
  1. profiling 段每个 forward batch 按 (c, q, b) 去对实测点表; 能对上的留下
  2. N = 对上的 forward 数, T = Σ 实测单次 GPU 时间
  3. 吞吐 = Σ_i w_i * tput_i, 权重 w_i = t_i / T

  ⚠️ 第 3 步的加权平均在代数上恒等于 Σtokens/T:
       Σ (t_i/T)*(tok_i/t_i) = Σ tok_i / T
     所以它就是聚合吞吐, 不是"平均每 forward 的吞吐"。
     后者要按**次数**加权(Σ tput_i / N), 会被大量廉价小 batch 拉高, 是另一个量。
     两个都算出来对照, 免得混用。

匹配键用 (final_context_length, final_query_length, eq_batch_size) —— 和当初选点、
以及 RESULTS 表里的 (c,q,b) 是同一套等价同构映射后的坐标。
分子给两种 token 口径: fresh(真过 GPU 的 chunk_tokens) 和 完整 ISL。
"""
import csv, collections

MEAS = {}
for r in csv.DictReader(open("bench_run/RESULTS_223_points.csv")):
    if r["status"] == "measured":
        MEAS[(int(r["c"]), int(r["q"]), int(r["b"]))] = float(r["target_fwd_gpu_ms"])

# prefill_forwards_profiling.csv 是 (forward, request) 对, 要先按 forward_idx 去重
# ⚠️ ISL 不能按 (forward,request) 行直接累加: 被 chunk 切开的请求会在它跨的每个
# forward 里各记一遍完整 ISL, 全段加起来 484M, 而真实 ISL 只有 320.76M(虚高 1.5x)。
# 正确做法: 每个请求的 ISL 只归到它**最后一个** forward(prefill 完成的那次)。
rows = list(csv.DictReader(open("prefill_forwards_profiling.csv")))
last = {}
for r in rows:
    k = r["req_id"]
    if k not in last or int(r["forward_idx"]) > last[k][0]:
        last[k] = (int(r["forward_idx"]), int(r["ISL"]))

fwd = {}
for r in rows:
    i = int(r["forward_idx"])
    if i not in fwd:
        fwd[i] = dict(c=int(r["final_context_length"]), q=int(r["final_query_length"]),
                      b=int(r["eq_batch_size"]), fresh=int(r["batch_tokens"]), isl=0)
for i, isl in last.values():
    fwd[i]["isl"] += isl

tot_f = len(fwd)
tot_fresh = sum(f["fresh"] for f in fwd.values())
tot_isl = sum(f["isl"] for f in fwd.values())

hit, miss = [], collections.Counter()
for f in fwd.values():
    ms = MEAS.get((f["c"], f["q"], f["b"]))
    if ms is None:
        miss[f["b"]] += 1
    else:
        hit.append((f, ms / 1000.0))

N = len(hit)
T = sum(t for _, t in hit)
H_fresh = sum(f["fresh"] for f, _ in hit)
H_isl = sum(f["isl"] for f, _ in hit)

print(f"① 匹配: profiling 段 {tot_f} 个 forward, 对上实测点的 N = {N} ({100*N/tot_f:.1f}%)")
print(f"   未匹配 {tot_f-N} 个, 按 b 分布: {dict(sorted(miss.items()))}")
print(f"② T = {T:.1f} s   (这 N 个 forward 的实测 GPU 时间合计)")
print(f"   分子: fresh {H_fresh/1e6:.2f} M tok / 完整 ISL {H_isl/1e6:.2f} M tok")
print(f"   覆盖: forward {100*N/tot_f:.1f}%, fresh token {100*H_fresh/tot_fresh:.1f}%, "
      f"ISL {100*H_isl/tot_isl:.1f}%")

print("\n③ 吞吐 (权重 = 时间占比)")
for nm, key in (("fresh token 口径", "fresh"), ("完整 ISL 口径", "isl")):
    # 显式按权重求和, 验证它等于 Σtok/T
    wsum = sum((t / T) * (f[key] / t) for f, t in hit)
    num = sum(f[key] for f, _ in hit)
    assert abs(wsum - num / T) < 1e-6 * (num / T)
    print(f"   {nm:16s} {wsum:11,.0f} tok/s")

# 对照: 按次数加权的"平均每 forward 吞吐" —— 不是同一个量
for nm, key in (("fresh", "fresh"), ("ISL", "isl")):
    avg = sum(f[key] / t for f, t in hit) / N
    print(f"   [对照] 按次数加权的平均每-forward {nm:5s} 吞吐 {avg:11,.0f} tok/s"
          f"  ({avg/(sum(f[key] for f,_ in hit)/T):.2f}x 于上面)")

print("\n④ 外推到整个 profiling 段 (假设未匹配部分同吞吐)")
print(f"   T_all = T / 时间覆盖率 -> ", end="")
# 未匹配的用旧模型补, 得到全段总时间
import json
P = json.load(open("bench_run/model_params.json"))
tm = lambda c, b, q: (P["a0"]+P["ar"]*b+P["a1"]*c+P["a2"]*b*c+P["b0"]*b*q+P["b1"]*b*c*q)/1000
T_all = T + sum(tm(f["c"], f["b"], f["q"]) for f in fwd.values()
                if (f["c"], f["q"], f["b"]) not in MEAS)
print(f"{T_all:.1f} s  (匹配部分用实测 {T:.1f}s + 未匹配部分用模型 {T_all-T:.1f}s)")
print(f"   fresh 口径 {tot_fresh/T_all:11,.0f} tok/s")
print(f"   ISL   口径 {tot_isl/T_all:11,.0f} tok/s")
