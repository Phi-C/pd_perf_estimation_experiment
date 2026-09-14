#!/usr/bin/env python3
"""Decode_throughput_statistical_estimation.md §3-4：给入选格的代表点定价并按 w_ij 汇总。

假定 running batch 稳定为 b = C，单步吞吐用 §3 表格的 b/D 口径：
    T_ij = C / D(L_ctx, C) * 1000      (output tok/s)
另报纯模型上界 C / fwd_gpu。
客户端并发 C 本身不能保证实际 b = C；该脚本只查表，不运行新的测量。

D 的来源，按优先级：
  exact : 实测点表里有 (L_test 最近一格, b=C)，且 |L_rep - L_test| <= --l-tol
  model : 同一波 (wave = ceil(b/16)) 内对实测点做 D = a + sL*L + sb*b 最小二乘，
          再代入 (L_rep, C)。波外 / 超出实测 L 范围的点另行标记 extrap。
"""
import argparse, csv, math, os, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--selected", required=True)
ap.add_argument("-m", "--measured", default="decode/decode_points_measured.csv")
ap.add_argument("-C", "--concurrency", type=int, default=32)
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--l-tol", type=int, default=8192)
a = ap.parse_args()

M = [dict(L=int(r["L_test"]), b=int(r["b"]), D=float(r["D_ms"]), F=float(r["fwd_gpu_ms"]))
     for r in csv.DictReader(open(a.measured)) if r["accept"] == "OK"]
wave = lambda b: math.ceil(b / 16)
W = wave(a.concurrency)
fam = [m for m in M if wave(m["b"]) == W]
if not fam:
    ap.error(f"实测点表里没有 wave {W} (b≈{a.concurrency}) 的样本")
Lmin, Lmax = min(m["L"] for m in fam), max(m["L"] for m in fam)

def fit(key):
    n = len(fam)
    X = [[1.0, m["L"], m["b"]] for m in fam]
    y = [m[key] for m in fam]
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    v = [sum(X[k][i] * y[k] for k in range(n)) for i in range(3)]
    G = [A[i][:] + [v[i]] for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(G[r][i])); G[i], G[p] = G[p], G[i]
        for r in range(3):
            if r != i:
                f = G[r][i] / G[i][i]
                for c in range(i, 4):
                    G[r][c] -= f * G[i][c]
    sol = [G[i][3] / G[i][i] for i in range(3)]
    rmse = statistics.pstdev([y[k] - sum(sol[i] * X[k][i] for i in range(3)) for k in range(n)])
    return sol, rmse

(fD, rD), (fF, rF) = fit("D"), fit("F")
ev = lambda s, L: s[0] + s[1] * L + s[2] * a.concurrency
exact = {m["L"]: m for m in fam if m["b"] == a.concurrency}

rows = list(csv.DictReader(open(a.selected)))
out = []
for r in rows:
    L = int(r["L_rep"]); O = int(r["O_rep"]); I = int(r["I_rep"])
    hit = min(exact, key=lambda x: abs(x - L)) if exact else None
    if hit is not None and abs(hit - L) <= a.l_tol:
        D, F, how = exact[hit]["D"], exact[hit]["F"], "exact"
    else:
        D, F = ev(fD, L), ev(fF, L)
        how = "model" if Lmin <= L <= Lmax else "extrap"
    out.append(dict(rank=int(r["rank"]), i=int(r["i"]), j=int(r["j"]), n=int(r["n"]),
                    w=float(r["w"]), I_rep=I, L_rep=L, O_rep=O, b=a.concurrency, method=how,
                    D_ms=round(D, 3), fwd_gpu_ms=round(F, 3),
                    T=a.concurrency / D * 1000, T_pure=a.concurrency / F * 1000))

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, list(out[0].keys())); w.writeheader()
    for r in out:
        w.writerow(r)

ws = lambda k: sum(r["w"] * r[k] for r in out)
T_shape, T_pure = ws("T"), ws("T_pure")
T_mix = ws("O_rep") / sum(r["w"] * r["O_rep"] / r["T"] for r in out)
bym = defaultdict(float)
for r in out:
    bym[r["method"]] += r["w"]
print(f"入选格 {len(out)}，假定稳定 b = C = {a.concurrency} (wave {W}，历史点表估计)")
print(f"wave {W} 实测样本 {len(fam)} 点，L 实测范围 {Lmin:,}..{Lmax:,}")
print(f"  D  = {fD[0]:.4f} + {fD[1]:.4e}*L + {fD[2]:.4f}*b   rmse {rD:.3f} ms")
print(f"  F  = {fF[0]:.4f} + {fF[1]:.4e}*L + {fF[2]:.4f}*b   rmse {rF:.3f} ms")
print("定价方式权重: " + ", ".join(f"{k} {v*100:.1f}%" for k, v in sorted(bym.items(), key=lambda x: -x[1])))
print()
print(f"  T_shape (b/D, output tok/s)      = {T_shape:,.0f}")
print(f"  T_shape (b/fwd_gpu, 纯模型上界)  = {T_pure:,.0f}")
print(f"  T_mix   (按每请求 OSL 摊时间)    = {T_mix:,.0f}")
print(f"  加权平均 D = {ws('D_ms'):.3f} ms, 加权平均 L_ctx = {ws('L_rep'):,.0f}, "
      f"加权平均 OSL = {ws('O_rep'):,.0f}")
