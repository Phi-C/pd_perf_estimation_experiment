#!/usr/bin/env python3
"""Prefill_throughput_statistical_estimation.md §3-4：把入选格代表点投影到可测坐标，
从实测点表取单次 forward 时间，再按 w_ij 汇总。

投影 (§2 "投影到可测坐标")：
  q_tgt = j==0 时保留 q_rep（短 chunk 不吸附），否则 1024*j
  c_tgt = 离 c_rep 最近的 8192 倍数（点表的 c 网格 / cache hash block 对齐）

定价：在**同一 q 家族**内按 c 线性插值实测时间。
  exact  : 家族里正好有 c_tgt
  interp : c_tgt 落在家族两个实测 c 之间
  extrap : c_tgt 在家族实测 c 范围外，用最近两点外推
  single : 家族只有一个实测点，直接取用（dc 可能很大）
j==0 的短 chunk 没有独立 q 网格，统一归到 "q<qbin" 家族，家族内 q 本身有抖动。

时间口径为点表的 target_fwd_gpu_ms（TTFT 差分估计，非 GPU event 直测）。
"""
import argparse, csv, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--selected", required=True)
ap.add_argument("-m", "--measured", default="bench_run/RESULTS_223_points.csv")
ap.add_argument("-b", "--batch", type=int, default=1)
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--qbin", type=int, default=1024)
a = ap.parse_args()

fam = defaultdict(list)          # q_family -> [(c, q_actual, t_ms)]
for r in csv.DictReader(open(a.measured)):
    if r["status"] != "measured" or int(r["b"]) != a.batch:
        continue
    q = int(r["q"])
    fam["small" if q < a.qbin else q].append(
        (int(r["c"]), float(r["q_actual"]), float(r["target_fwd_gpu_ms"])))
for k in fam:
    fam[k].sort()

def price(key, c_tgt):
    P = fam.get(key)
    if not P:
        return None
    if len(P) == 1:
        c, q, t = P[0]
        return t, c, q, "single"
    for c, q, t in P:
        if c == c_tgt:
            return t, c, q, "exact"
    lo = [p for p in P if p[0] < c_tgt]
    hi = [p for p in P if p[0] > c_tgt]
    if lo and hi:
        (c0, q0, t0), (c1, q1, t1) = lo[-1], hi[0]
        f = (c_tgt - c0) / (c1 - c0)
        return t0 + f * (t1 - t0), c_tgt, q0 + f * (q1 - q0), "interp"
    (c0, q0, t0), (c1, q1, t1) = (P[0], P[1]) if hi else (P[-2], P[-1])
    f = (c_tgt - c0) / (c1 - c0)
    return t0 + f * (t1 - t0), c_tgt, q0 + f * (q1 - q0), "extrap"

rows = list(csv.DictReader(open(a.selected)))
out, miss = [], []
for r in rows:
    j = int(r["j"]); c_rep = int(r["c_rep"]); q_rep = int(r["q_rep"])
    c_tgt = round(c_rep / 8192) * 8192
    key = "small" if j == 0 else a.qbin * j
    q_tgt = q_rep if j == 0 else a.qbin * j
    p = price(key, c_tgt)
    if p is None:
        miss.append(r); continue
    t_ms, c_used, q_used, how = p
    t = t_ms / 1000.0
    tok_full = c_used + q_used
    out.append(dict(rank=int(r["rank"]), i=int(r["i"]), j=j, n=int(r["n"]), w=float(r["w"]),
                    c_rep=c_rep, q_rep=q_rep, c_tgt=c_tgt, q_tgt=q_tgt, method=how,
                    c_used=round(c_used), q_used=round(q_used),
                    dc=round(c_used - c_rep), dq=round(q_used - q_rep),
                    t_ms=round(t_ms, 2), tok_full=round(tok_full), tok_fresh=round(q_used),
                    T_full=tok_full / t, T_fresh=q_used / t))

W = sum(r["w"] for r in out)
for r in out:
    r["w_renorm"] = r["w"] / W

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    wtr = csv.DictWriter(f, list(out[0].keys())); wtr.writeheader()
    for r in out:
        wtr.writerow(r)

wsum = lambda k: sum(r["w_renorm"] * r[k] for r in out)
T_full, T_fresh = wsum("T_full"), wsum("T_fresh")
t_bar = wsum("t_ms") / 1000
mix_full, mix_fresh = wsum("tok_full") / t_bar, wsum("tok_fresh") / t_bar

by_m = defaultdict(float)
for r in out:
    by_m[r["method"]] += r["w_renorm"]
print(f"入选格 {len(rows)}，定价成功 {len(out)}，缺 q 家族 {len(miss)}")
print(f"权重覆盖(入选集合内) = {W*100:.2f}%   定价方式权重: "
      + ", ".join(f"{k} {v*100:.1f}%" for k, v in sorted(by_m.items(), key=lambda x: -x[1])))
ad = sorted(abs(r["dc"]) for r in out)
print(f"|dc| 中位 {ad[len(ad)//2]}, p90 {ad[int(len(ad)*0.9)]}, max {ad[-1]}; "
      f"|dq| max {max(abs(r['dq']) for r in out)}")
print()
print(f"=== b = {a.batch} ===")
print(f"  T_shape (chunk 的 c+q，非去重 ISL) = {T_full:,.0f} tok/s")
print(f"  T_shape (fresh 口径 q) = {T_fresh:,.0f} tok/s")
print(f"  T_mix   (chunk 的 c+q，非去重 ISL) = {mix_full:,.0f} tok/s")
print(f"  T_mix   (fresh 口径 q) = {mix_fresh:,.0f} tok/s")
print(f"  加权平均单次 forward = {t_bar*1000:.1f} ms, "
      f"加权平均 c+q = {wsum('tok_full'):,.0f}, fresh = {wsum('tok_fresh'):,.0f}")
