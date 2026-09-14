#!/usr/bin/env python3
"""decode_steps.csv -> decode_points_unique.csv (待测同构点清单).

Decode 的等价映射是线性的: L_eq = mean(L_i), b = n.  所以这一步只做分桶/吸附/
去重/排序, 没有 prefill 那个二次项。

排序权重默认 = forward 数 (span_steps)。decode 单步耗时随 L 变化很平缓, 步数就是
最接近"墙钟占比"的免费代理; 想按 token 或 attention 工作量排用 --rank-by。
"""
import argparse, csv, os, statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default="decode")
    ap.add_argument("--l-bin", type=int, default=8192)
    ap.add_argument("--rank-by", choices=["steps", "tokens", "work"], default="steps")
    ap.add_argument("--since", default="", metavar="HH:MM:SS",
                    help="只保留该时刻之后的 forward (aiperf profiling 段起点), 切掉 warmup")
    ap.add_argument("--out", default="decode_points_unique.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.dir, "decode_steps.csv"))))
    if a.since:
        sec = lambda t: (lambda h, m, s: h * 3600 + m * 60 + s)(*map(int, t.split(":")))
        cut, n0 = sec(a.since), len(rows)
        rows = [r for r in rows if sec(r["time"]) >= cut]
        print(f"--since {a.since}: 保留 {len(rows)}/{n0} 行")
    cells = defaultdict(list)
    for r in rows:
        L, b, w = float(r["L_eq"]), int(r["b"]), int(r["span_steps"])
        cells[(int(L // a.l_bin), b)].append((L, w, float(r["r_max"]), float(r["CV_L"]), int(r["L_sum"])))

    pts = defaultdict(lambda: dict(steps=0, tokens=0, work=0, rmax=[], cvl=[], cells=0))
    for (lb, b), vs in cells.items():
        med = statistics.median([v[0] for v in vs])
        lo, hi = lb * a.l_bin, (lb + 1) * a.l_bin
        # 各自独立取最近角点; L 维不存在 prefill 的 "q=0 测不了" 问题, 但 L=0 也测不了,
        # 所以最低那格不吸附到 0, 直接取中位数。
        snap = med if lb == 0 else (hi if med - lo >= a.l_bin / 2 else lo)
        p = pts[(int(round(snap)), b)]
        p["cells"] += 1
        for L, w, rm, cv, ls in vs:
            p["steps"] += w
            p["tokens"] += w * b
            p["work"] += w * ls
            p["rmax"].append(rm)
            p["cvl"].append(cv)

    key = a.rank_by
    out = sorted(pts.items(), key=lambda kv: -kv[1][key])
    tot = {k: sum(v[k] for v in pts.values()) for k in ("steps", "tokens", "work")}
    cum = dict(steps=0, tokens=0, work=0)
    dst = os.path.join(a.dir, a.out)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["priority", "context_length", "batch_size", "n_forwards", "decode_tokens",
                    "attn_work_pairs", "n_cells", "r_max_med", "CV_L_med", "eq_confidence",
                    "cum_forward_pct", "cum_token_pct", "cum_work_pct"])
        for i, ((L, b), v) in enumerate(out, 1):
            for k in cum:
                cum[k] += v[k]
            rm = statistics.median(v["rmax"])
            w.writerow([i, L, b, v["steps"], v["tokens"], v["work"], v["cells"],
                        round(rm, 3), round(statistics.median(v["cvl"]), 3),
                        "high" if rm <= 1.2 else ("mid" if rm <= 2.5 else "low"),
                        f"{100*cum['steps']/tot['steps']:.2f}",
                        f"{100*cum['tokens']/tot['tokens']:.2f}",
                        f"{100*cum['work']/tot['work']:.2f}"])
    n50 = n80 = n90 = 0
    c = 0
    for i, (_, v) in enumerate(out, 1):
        c += v[key]
        if not n50 and c >= .5 * tot[key]: n50 = i
        if not n80 and c >= .8 * tot[key]: n80 = i
        if not n90 and c >= .9 * tot[key]: n90 = i
    print(f"{len(rows)} rows -> {len(cells)} cells -> {len(pts)} points  (rank-by {key})")
    print(f"覆盖 50%/80%/90% 需要 {n50}/{n80}/{n90} 个点  -> {dst}")


if __name__ == "__main__":
    main()
