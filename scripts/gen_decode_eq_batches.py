#!/usr/bin/env python3
"""每个 decode forward 的等价同构 batch —— 逐 forward，精确展开。

依据 `ragged_packed_decode_workload_equivalence.md`：
    L_i   = ISL_i + O_i
    L_eq  = mean(L_i)  →  等价点 (L_eq, b=n)
    TPS   = n / D_n(L_eq)，D 为选定口径的单步时间

与 `decode_steps.csv` 的区别：那里一行是一个 membership **区间**（--max-span 切段，
用区间中点近似）；这里一行是**一个 forward**。区间内 b 恒定、每个 L_i 每步 +1，
所以 L_eq 每步精确 +1、std(L_i) 恒定 —— 逐 forward 展开是无损的，不是插值。

同时保留两个敏感性场景的坐标：
    L_eq  → 工作量等价估计   D_work      = D_n(mean L_i)
    L_max → 最大长度场景     D_max_shape = D_n(max  L_i)
    rho_L = L_max / L_eq，接近 1 时工作量等价更可信。

⚠️ step 是 scheduler 的 `_schedule_tick`（含 prefill 轮/空转轮），区间内被无差别
当成 decode 步；本例重放 token 比 AIPerf 多 1.85%，需和客户端逐次对账。
"""
import argparse, csv, math, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default="decode")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--page", type=int, default=1024,
                help="KV page = block_size * dcp_world_size (128*8)")
ap.add_argument("--l-bin", type=int, default=8192,
                help="L_test grid: snap L_eq to the nearest multiple of this")
ap.add_argument("--since", default=None,
                help="profiling 起点 HH:MM:SS，用于标 phase 列")
a = ap.parse_args()
dst = a.out or os.path.join(a.dir, "decode_eq_batches.csv")

# 按 membership 区间聚合：区间起点 step、span、成员在起点处的 L_i
segs = defaultdict(lambda: dict(L=[]))
for r in csv.DictReader(open(os.path.join(a.dir, "decode_members.csv"))):
    s = segs[int(r["row_idx"])]
    s["step"] = int(r["step"]); s["span"] = int(r["span_steps"]); s["time"] = r["time"]
    s["L"].append(int(r["context_length"]))          # 区间起点那一步的 KV length

out = []
for rid in sorted(segs):
    s = segs[rid]
    L0 = np.asarray(s["L"], dtype=np.int64)
    b = L0.size
    d = np.arange(s["span"], dtype=np.int64)          # 区间内第 0..span-1 步
    sd = float(L0.std())                              # 加同一常数，std 不变
    mn, mx, sm = int(L0.min()), int(L0.max()), int(L0.sum())
    mean0 = L0.mean()
    for k in d:
        k = int(k)
        L_eq = mean0 + k
        L_max = mx + k
        nb = np.ceil((L0 + k) / a.page)               # §10.2 block 级特征
        cvb = float(nb.std() / nb.mean()) if nb.mean() else 0.0
        r_max = L_max / L_eq
        out.append(dict(
            step=s["step"] + k, time=s["time"], seg_row_idx=rid,
            b=b,
            L_eq=round(L_eq, 2),
            # L_test: the performance-test point this forward gets priced at --
            # nearest grid endpoint, not floor.  0 is not testable, so clamp up.
            L_test=max(a.l_bin, int((L_eq + a.l_bin / 2) // a.l_bin) * a.l_bin),
            L_min=mn + k, L_max=L_max, L_sum=sm + b * k,
            CV_L=round(sd / L_eq, 4), r_max=round(r_max, 4), CV_block=round(cvb, 4),
            rho_L=round(r_max, 4),
            eq_confidence="high" if r_max <= 1.2 else ("mid" if r_max <= 2.5 else "low"),
            phase=("" if not a.since else
                   ("profiling" if s["time"] >= a.since else "warmup")),
        ))

out.sort(key=lambda r: r["step"])
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)

tok = sum(r["b"] for r in out)
print(f"{dst}: {len(out):,} forwards")
print(f"  Σb (= decode token 总数): {tok:,}")
print(f"  step 加权平均 b: {tok/len(out):.2f}")
from collections import Counter
c = Counter(r["eq_confidence"] for r in out)
for k in ("high", "mid", "low"):
    print(f"  eq_confidence={k:4s}: {c[k]:>7,} ({c[k]/len(out)*100:5.2f}%)")
