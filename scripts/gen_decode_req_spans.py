#!/usr/bin/env python3
"""每条 request 的 "在哪些 step 上被调度进了 forward batch" —— 区间形式。

输入 : decode_members.csv (parse_decode_stepmembers.py 的产物)
输出 : decode_req_spans.csv，一行 = 一条 request 的一段**连续在批**区间。

为什么是区间而不是逐 step 展开：本例 123,717 个 step × 平均 batch size 展开会是
~220 万行。请求在批与否只在 join/leave 时变化，区间是无损压缩。

⚠️ `step` 是 scheduler 的 `_schedule_tick`，计的是调度轮次（含 prefill 轮和空转轮），
不是纯 decode forward 序号。所以区间长度 / total_decode_steps 是 output length 的
**上界**（本用例对 aiperf 残差 ≤2.2%，见 decode/SKILL.md 坑 2）。
"""
import argparse, csv, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default="decode", help="含 decode_members.csv 的目录")
ap.add_argument("-o", "--out", default=None)
a = ap.parse_args()
src = os.path.join(a.dir, "decode_members.csv")
dst = a.out or os.path.join(a.dir, "decode_req_spans.csv")

per = defaultdict(list)
for r in csv.DictReader(open(src)):
    per[int(r["req_id"])].append(r)

out, n_seg_multi = [], 0
for rid in sorted(per):
    rows = sorted(per[rid], key=lambda x: int(x["step"]))
    isl = rows[0]["ISL"]
    total = int(rows[0]["total_decode_steps"])
    segs = []                                   # 合并 step 上首尾相接的行
    for r in rows:
        s, d = int(r["step"]), int(r["span_steps"])
        g = int(r["gen_tokens"])
        if segs and segs[-1][1] == s:           # 前一段的 end+1 == 本行 start
            segs[-1][1] = s + d
            segs[-1][3] = g + d
        else:
            segs.append([s, s + d, g, g + d])
    if len(segs) > 1:
        n_seg_multi += 1
    for k, (s0, s1, g0, g1) in enumerate(segs):
        out.append(dict(
            req_id=rid, ISL=isl,
            seg_idx=k, n_segs=len(segs),
            step_start=s0, step_end=s1 - 1, seg_steps=s1 - s0,
            gen_start=g0, gen_end=g1,
            total_decode_steps=total,
        ))

with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, list(out[0]))
    w.writeheader()
    w.writerows(out)

print(f"{dst}: {len(out)} 行 / {len(per)} requests")
print(f"  被中断过(离开又回批)的 request: {n_seg_multi}")
print(f"  seg_steps 求和: {sum(r['seg_steps'] for r in out):,}"
      f"  (== Σ total_decode_steps: {sum(int(rows[0]['total_decode_steps']) for rows in per.values()):,})")
