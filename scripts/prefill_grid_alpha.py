#!/usr/bin/env python3
"""Prefill_throughput_statistical_estimation.md §1-3：分箱 -> 代表点 -> 按覆盖率选格赋权。

一行 = 一个 (forward, request) 对，即"这一步这条请求"的形状：
    c = done_tokens   (已在 KV cache 里的上下文)
    q = chunk_tokens  (本步新算的 token)
网格：c 每 CBIN=8192 一格，q 每 QBIN=1024 一格（照原文 §1）。
"""
import argparse, csv, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="prefill/prefill_forwards_profiling.csv")
ap.add_argument("-o", "--out", default="prefill/prefill_grid_alpha")
ap.add_argument("--alpha", type=float, default=0.8)
ap.add_argument("--cbin", type=int, default=8192)
ap.add_argument("--qbin", type=int, default=1024)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.input)))
pts = [(int(r["done_tokens"]), int(r["chunk_tokens"])) for r in rows]
N = len(pts)

# §1 分箱
cells = defaultdict(list)
for c, q in pts:
    cells[(c // a.cbin, q // a.qbin)].append((c, q))

# §2 代表点 = 格内取整均值
rec = []
for (i, j), P in cells.items():
    n = len(P)
    rec.append(dict(i=i, j=j, n=n,
                    c_rep=round(sum(p[0] for p in P) / n),
                    q_rep=round(sum(p[1] for p in P) / n),
                    c_lo=a.cbin * i, c_hi=a.cbin * (i + 1),
                    q_lo=a.qbin * j, q_hi=a.qbin * (j + 1),
                    c_min=min(p[0] for p in P), c_max=max(p[0] for p in P),
                    q_min=min(p[1] for p in P), q_max=max(p[1] for p in P)))

# §3 按 n 降序选格到覆盖率 alpha
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
cols = ["rank", "i", "j", "n", "w", "c_rep", "q_rep", "c_lo", "c_hi", "q_lo", "q_hi",
        "c_min", "c_max", "q_min", "q_max"]
with open(a.out + "_selected.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader()
    for k, r in enumerate(S, 1):
        w.writerow({**{c: r.get(c) for c in cols}, "rank": k})
with open(a.out + "_allcells.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["i", "j", "n", "c_rep", "q_rep"]); w.writeheader()
    for r in rec:
        w.writerow({k: r[k] for k in ("i", "j", "n", "c_rep", "q_rep")})

print(f"N = {N:,} 个 (forward, request) 对")
print(f"非空格子 = {len(rec)}")
print(f"alpha = {a.alpha}  ->  选中 {len(S)} 格，覆盖 {NS:,} ({NS/N*100:.2f}%)")
print(f"未选中 {len(rec)-len(S)} 格，占 {(N-NS)/N*100:.2f}%")
print()
print("| rank | c 格 | q 格 | N_ij | w_ij | 代表点 (c,q) |")
print("|---:|---|---|---:|---:|---|")
for k, r in enumerate(S, 1):
    print(f"| {k} | [{r['c_lo']:,},{r['c_hi']:,}) | [{r['q_lo']:,},{r['q_hi']:,}) | "
          f"{r['n']:,} | {r['w']:.4f} | ({r['c_rep']:,}, {r['q_rep']:,}) |")
