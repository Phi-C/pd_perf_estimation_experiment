#!/usr/bin/env python3
"""用探针实测的**纯模型 forward** GPU 时间估 profiling 段的理论吞吐。

与 throughput_from_measured.py 唯一的区别: 每个 forward batch 的耗时不再取
TTFT 推算的 target_fwd_gpu_ms, 而取 HIP event 直接量到的 fwd_gpu_ms
(只覆盖 run_model = 模型 forward + compute_logits)。

两条路线共用同一套 (c,q,b) 匹配键和同一份 profiling 段 forward 清单, 所以
差异只来自"每个 batch 值多少时间"这一项。
"""
import argparse, csv, collections, json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--forwards", type=Path, default=BASE / "prefill/prefill_forwards_profiling.csv")
ap.add_argument("--probe", type=Path, default=BASE / "bench_run/RESULTS_probe_vs_bench.csv")
ap.add_argument("--measured", type=Path, default=BASE / "bench_run/RESULTS_223_points.csv")
ap.add_argument("--model", type=Path, default=BASE / "bench_run/model_params.json")
ap.add_argument("--extrapolate", action="store_true",
                help="另报用 TTFT 差分模型补缺的敏感性估计，不增加实测覆盖率")
a = ap.parse_args()
for path in (a.forwards, a.probe, a.measured):
    if not path.is_file():
        ap.error(f"input not found: {path}; see experiments/exp1_kimi_k3/Reproduce.md")

# ---- 口径 A: TTFT 推算 (原口径)
MEAS = {}
for r in csv.DictReader(open(a.measured)):
    if r["status"] == "measured":
        MEAS[(int(r["c"]), int(r["q"]), int(r["b"]))] = float(r["target_fwd_gpu_ms"])

# ---- 口径 B: 探针实测纯 forward
# 同一个参考点可能被多个探针桶匹配(q<1024 的 ±16 容差), 按 forward 次数加权合并
acc = collections.defaultdict(lambda: [0.0, 0])
for r in csv.DictReader(open(a.probe)):
    if r["status"] != "measured" or not r["fwd_gpu_ms"]:
        continue
    k = (int(r["c"]), int(r["q"]), int(r["b"]))
    n = int(r["probe_n"])
    acc[k][0] += float(r["fwd_gpu_ms"]) * n
    acc[k][1] += n
PROBE = {k: s / n for k, (s, n) in acc.items()}

print(f"参考点表: TTFT 推算 {len(MEAS)} 点 / 探针实测 {len(PROBE)} 点 "
      f"/ 两者都有 {len(set(MEAS) & set(PROBE))} 点\n")

# ---- profiling 段 forward 清单 (与原脚本完全一致)
rows = list(csv.DictReader(open(a.forwards)))
if not rows:
    ap.error("forward CSV is empty")
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

tot_f = len(fwd); tot_fresh = sum(f["fresh"] for f in fwd.values())
tot_isl = sum(f["isl"] for f in fwd.values())
print(f"profiling 段: {tot_f} forward, fresh {tot_fresh/1e6:.2f} M tok, "
      f"ISL {tot_isl/1e6:.2f} M tok\n")

def run(tbl, name):
    hit = [(f, tbl[(f['c'],f['q'],f['b'])]/1000.0) for f in fwd.values()
           if (f['c'],f['q'],f['b']) in tbl]
    N = len(hit); T = sum(t for _, t in hit)
    if not hit or T <= 0:
        ap.error(f"{name}: no matched forwards with positive total time")
    H = {k: sum(f[k] for f, _ in hit) for k in ("fresh", "isl")}
    print(f"=== {name} ===")
    print(f"  ① 匹配 N = {N}/{tot_f} ({100*N/tot_f:.1f}%)")
    print(f"  ② T = {T:.1f} s   覆盖 fresh {100*H['fresh']/tot_fresh:.1f}% / "
          f"ISL {100*H['isl']/tot_isl:.1f}%")
    print(f"  ③ 时间加权吞吐(= Σtok/T):  fresh {H['fresh']/T:11,.0f} tok/s"
          f"   ISL {H['isl']/T:11,.0f} tok/s")
    return hit, N, T, H

hitA, NA, TA, HA = run(MEAS, "口径 A: TTFT 推算 target_fwd_gpu_ms (原口径)")
print()
hitB, NB, TB, HB = run(PROBE, "口径 B: 探针实测纯 forward fwd_gpu_ms (本次)")

# ---- 同一批 forward 上的严格对照 (两个表都能匹配的)
both = set(MEAS) & set(PROBE)
hb = [(f, MEAS[(f['c'],f['q'],f['b'])]/1000, PROBE[(f['c'],f['q'],f['b'])]/1000)
      for f in fwd.values() if (f['c'],f['q'],f['b']) in both]
Ta = sum(a for _, a, _ in hb); Tb = sum(b for _, _, b in hb)
if not hb or Ta <= 0 or Tb <= 0:
    ap.error("no common forwards with positive timing for the two tables")
Hf = sum(f["fresh"] for f, _, _ in hb); Hi = sum(f["isl"] for f, _, _ in hb)
print(f"\n=== 严格对照: 同一批 {len(hb)} 个 forward, 只换单次耗时 ===")
print(f"  T(TTFT 推算) = {Ta:.1f} s   ->  fresh {Hf/Ta:11,.0f}   ISL {Hi/Ta:11,.0f} tok/s")
print(f"  T(探针实测)  = {Tb:.1f} s   ->  fresh {Hf/Tb:11,.0f}   ISL {Hi/Tb:11,.0f} tok/s")
print(f"  时间比 {Tb/Ta:.4f}  ->  吞吐提升 {Ta/Tb:.4f}x  (+{100*(Ta/Tb-1):.1f}%)")

# ---- 可选的全段模型补值：与子集实测分开报告
if not a.extrapolate:
    raise SystemExit(0)
P = json.load(open(a.model))
tm = lambda c, b, q: (P["a0"]+P["ar"]*b+P["a1"]*c+P["a2"]*b*c+P["b0"]*b*q+P["b1"]*b*c*q)/1000
ratio = Tb / Ta   # 未匹配段按同一比例折算成纯 forward
print(f"\n=== 敏感性估计：用 TTFT 差分模型补缺，非全量实测 ===")
for nm, tbl, sc in (("A: TTFT 推算", MEAS, 1.0), ("B: 探针实测", PROBE, ratio)):
    T = sum(tbl[(f['c'],f['q'],f['b'])]/1000 for f in fwd.values()
            if (f['c'],f['q'],f['b']) in tbl)
    Tu = sum(tm(f["c"], f["b"], f["q"]) for f in fwd.values()
             if (f['c'],f['q'],f['b']) not in tbl) * sc
    Tall = T + Tu
    print(f"  {nm}: T_all = {Tall:7.1f} s (实测 {T:.1f} + 未匹配补 {Tu:.1f}, 补的部分 x{sc:.4f})")
    print(f"     fresh {tot_fresh/Tall:11,.0f} tok/s     ISL {tot_isl/Tall:11,.0f} tok/s")
