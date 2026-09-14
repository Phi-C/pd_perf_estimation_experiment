#!/usr/bin/env python3
"""Decode_throughput_statistical_estimation.md §1-2：请求级二维分箱 -> 代表点 -> 按覆盖率选格赋权。

一行 = 一条请求：
    I     = ISL
    L_ctx = ISL + OSL/2      (整条生成路径的代表 KV 位置)
网格：I 每 IBIN=8192 一格，L_ctx 每 LBIN=1024 一格（照原文 §2）。
代表点 = 格内 (I, L_ctx) 的取整均值；代表输出长度 O = 2*(L_ctx_rep - I_rep)。

与 scripts/prefill_grid_alpha.py 是同一套 §1-3 流程，只是样本粒度从
(forward, request) 换成 request，坐标从 (c, q) 换成 (I, L_ctx)。
"""
import argparse, csv, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="decode/decode_requests.csv")
ap.add_argument("-o", "--out", default="decode/decode_grid_alpha")
ap.add_argument("--alpha", type=float, default=0.8)
ap.add_argument("--ibin", type=int, default=8192)
ap.add_argument("--lbin", type=int, default=1024)
ap.add_argument("--first-step", type=int, default=None,
                help="只保留 first_step >= 该值的请求（profiling 段边界）")
ap.add_argument("--isl-col", default="ISL")
ap.add_argument("--osl-col", default="decode_steps")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.input)))
if a.first_step is not None:
    rows = [r for r in rows if int(r["first_step"]) >= a.first_step]

pts = []
for r in rows:
    I = int(r[a.isl_col]); O = int(r[a.osl_col])
    pts.append((I, I + O / 2.0, O))
N = len(pts)

cells = defaultdict(list)
for I, L, O in pts:
    cells[(I // a.ibin, int(L) // a.lbin)].append((I, L, O))

rec = []
for (i, j), P in cells.items():
    n = len(P)
    I_rep = round(sum(p[0] for p in P) / n)
    L_rep = round(sum(p[1] for p in P) / n)
    rec.append(dict(i=i, j=j, n=n, I_rep=I_rep, L_rep=L_rep,
                    O_rep=2 * (L_rep - I_rep),
                    O_mean=round(sum(p[2] for p in P) / n),
                    O_min=min(p[2] for p in P), O_max=max(p[2] for p in P),
                    I_lo=a.ibin * i, I_hi=a.ibin * (i + 1),
                    L_lo=a.lbin * j, L_hi=a.lbin * (j + 1),
                    I_min=min(p[0] for p in P), I_max=max(p[0] for p in P),
                    n_osl1=sum(1 for p in P if p[2] <= 1)))

rec.sort(key=lambda r: (-r["n"], r["i"], r["j"]))
cum = 0; S = []
for r in rec:
    S.append(r); cum += r["n"]
    if cum >= a.alpha * N:
        break
NS = sum(r["n"] for r in S)
for r in S:
    r["w"] = r["n"] / NS

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
cols = ["rank", "i", "j", "n", "w", "I_rep", "L_rep", "O_rep", "O_mean", "O_min", "O_max",
        "I_lo", "I_hi", "L_lo", "L_hi", "I_min", "I_max", "n_osl1"]
with open(a.out + "_selected.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader()
    for k, r in enumerate(S, 1):
        w.writerow({**{c: r.get(c) for c in cols}, "rank": k})
with open(a.out + "_allcells.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["i", "j", "n", "I_rep", "L_rep", "O_rep"]); w.writeheader()
    for r in rec:
        w.writerow({k: r[k] for k in ("i", "j", "n", "I_rep", "L_rep", "O_rep")})

bad = [r for r in S if r["O_rep"] <= 0]
print(f"N = {N:,} 条请求")
print(f"非空格子 = {len(rec)}")
print(f"alpha = {a.alpha}  ->  选中 {len(S)} 格，覆盖 {NS:,} ({NS/N*100:.2f}%)")
print(f"未选中 {len(rec)-len(S)} 格，占 {(N-NS)/N*100:.2f}%")
print(f"代表输出长度非法 (O_rep <= 0) 的入选格: {len(bad)}")
print(f"入选格里 OSL<=1 的请求数: {sum(r['n_osl1'] for r in S)}")
