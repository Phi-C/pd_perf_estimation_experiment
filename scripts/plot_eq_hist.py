#!/usr/bin/env python3
"""Histogram of L_eq (decode_eq_batches.csv) -> a standalone SVG.

No matplotlib in this env, so the SVG is emitted by hand (same convention as
scripts/plot_eq_grid512.py).  Two faceted panels share one x domain so the
warmup segment can be compared against profiling by eye.
"""
import argparse, csv, math, os
from collections import Counter

SANS = "IBM Plex Sans,Helvetica Neue,Arial,sans-serif"
MONO = "IBM Plex Mono,DejaVu Sans Mono,Menlo,monospace"
INK, INK2, INK3 = "#14161a", "#4e545e", "#7c828d"
RULE, GRID, PANEL, GROUND = "#e2e4e9", "#eef0f3", "#ffffff", "#fbfbfa"
SERIES = "#2a78d6"
QS = [("min", 0.0), ("p1", .01), ("p5", .05), ("p10", .10), ("p25", .25),
      ("p50", .50), ("p75", .75), ("p90", .90), ("p95", .95), ("p99", .99), ("max", 1.0)]


def q(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1)))]


def wq(pairs, p):                      # pairs: sorted (value, weight)
    tot = sum(w for _, w in pairs)
    acc = 0
    for v, w in pairs:
        acc += w
        if acc >= p * tot:
            return v
    return pairs[-1][0]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")


def txt(x, y, s, size=12, fill=INK, anchor="start", family=SANS, weight=400, ls=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    l = f' letter-spacing="{ls}"' if ls else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{a}{l}>{esc(s)}</text>')


def fmt(n):
    return f"{int(round(n)):,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--csv", default="decode_eq_batches.csv")
    ap.add_argument("-o", "--out", default="decode_eq_hist.svg")
    ap.add_argument("--bin", type=int, default=8192)
    ap.add_argument("--label", default="kimi-k3-mxfp4 · 1P1D · TP8+DCP8 · agentic-lmcache-1m · c32 · 20260910")
    a = ap.parse_args()

    segs = {"profiling": [], "warmup": []}
    wpairs = []
    for r in csv.DictReader(open(a.csv)):
        L = float(r["L_eq"])
        ph = r["phase"] or "profiling"
        segs[ph].append(L)
        if ph == "profiling":
            wpairs.append((L, int(r["b"])))
    for v in segs.values():
        v.sort()
    wpairs.sort()

    xmax = a.bin * (int(max(segs["profiling"][-1], segs["warmup"][-1]) // a.bin) + 1)
    nbin = xmax // a.bin
    hist = {k: Counter() for k in segs}
    for k, v in segs.items():
        for L in v:
            hist[k][min(nbin - 1, int(L // a.bin))] += 1

    W, PL, PR = 1100, 62, 24
    pw = W - PL - PR
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="__H__" '
         f'viewBox="0 0 {W} __H__" font-family="{SANS}">',
         f'<rect width="{W}" height="__H__" fill="{GROUND}"/>']

    y = 52
    o.append(txt(PL, y, "L_eq DISTRIBUTION · decode_eq_batches.csv", 11.5, INK3,
                 family=MONO, weight=500, ls="1.4"))
    y += 34
    o.append(txt(PL, y, "每个 decode forward 的等价长度落在哪里", 27, INK, weight=600))
    y += 26
    o.append(txt(PL, y, f"一行一个 forward，L_eq = mean(L_i)，L_i = ISL_i + O_i 是该请求此刻的 KV 长度。"
                        f"每根柱子 = {a.bin:,} token 一格（与 gen_decode_points --l-bin 同刻度），"
                        f"柱高 = 落在该格的 forward 个数。", 13, INK2))
    y += 22
    o.append(txt(PL, y, a.label + f"  |  {len(segs['profiling']) + len(segs['warmup']):,} forwards",
                 11.5, INK3, family=MONO))

    # ---- stat tiles -------------------------------------------------------
    y += 22
    P = segs["profiling"]
    tiles = [(fmt(len(P)), "profiling forwards"),
             (fmt(q(P, .5)), "p50 · 中位 L_eq"),
             (fmt(sum(P) / len(P)), "mean L_eq"),
             (f"{fmt(P[0])} – {fmt(P[-1])}", "profiling 全域")]
    tw, th = pw / len(tiles), 68
    o.append(f'<rect x="{PL}" y="{y}" width="{pw}" height="{th}" fill="{PANEL}" '
             f'stroke="{RULE}" rx="3"/>')
    for i, (v, u) in enumerate(tiles):
        x = PL + i * tw
        if i:
            o.append(f'<line x1="{x}" x2="{x}" y1="{y}" y2="{y+th}" stroke="{RULE}"/>')
        o.append(txt(x + 18, y + 32, v, 21, INK, family=MONO, weight=600))
        o.append(txt(x + 18, y + 52, u, 11.5, INK3))
    y += th

    # ---- panels -----------------------------------------------------------
    def panel(top, key, title, note, h, marks):
        out, ymax = [], max(hist[key].values())
        out.append(txt(PL, top + 4, title, 16, INK, weight=600))
        out.append(txt(PL, top + 25, note, 12.5, INK2))
        py = top + 40
        out.append(f'<rect x="{PL}" y="{py}" width="{pw}" height="{h}" fill="{PANEL}" '
                   f'stroke="{RULE}" rx="3"/>')
        iy, ih = py + 14, h - 42
        X = lambda v: PL + 14 + v / xmax * (pw - 28)
        Y = lambda v: iy + ih - v / ymax * ih
        mag = 10 ** int(math.log10(ymax))
        stp = mag * (2 if ymax / mag > 5 else 1)
        n = 0
        while n <= ymax:
            out.append(f'<line x1="{PL+14}" x2="{PL+pw-14}" y1="{Y(n):.1f}" y2="{Y(n):.1f}" '
                       f'stroke="{RULE if n==0 else GRID}"/>')
            out.append(txt(PL + 8, Y(n) + 3.5, fmt(n), 10.5, INK3, "end", MONO))
            n += stp
        bw = (pw - 28) / nbin - 2
        for i, c in sorted(hist[key].items()):
            bh = max(1.5, ih * c / ymax)
            out.append(f'<rect x="{X(i*a.bin)+1:.1f}" y="{Y(c):.1f}" width="{bw:.1f}" '
                       f'height="{bh:.1f}" fill="{SERIES}" rx="1.5"/>')
        for v, lab in marks:
            out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{iy}" y2="{iy+ih}" '
                       f'stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3"/>')
            out.append(txt(X(v) + 5, iy + 11, lab, 10.5, INK2, family=MONO, weight=500))
        for xv in range(0, xmax + 1, 100_000):
            out.append(txt(X(xv), py + h - 10, "0" if not xv else f"{xv//1000}K",
                           10.5, INK3, "middle", MONO))
        return out, top + 40 + h

    y += 46
    blk, y = panel(y, "profiling", f"Profiling 段 — {len(P):,} 个 forward",
                   "主体是集中在 150K–260K 的单峰，外加一小簇 65K–95K 的次峰；426K 处有个孤立尖峰。",
                   300, [(q(P, .5), f"p50 {fmt(q(P,.5))}"), (q(P, .9), f"p90 {fmt(q(P,.9))}")])
    o += blk
    y += 40
    Wm = segs["warmup"]
    blk, y = panel(y, "warmup", f"Warmup 段 — {len(Wm):,} 个 forward",
                   "形状完全不同：右峰压在 650K–712K，是 1M-ISL 的 warmup 请求（b 平均仅 1.13）。这就是必须切 warmup 的理由。",
                   170, [(q(Wm, .5), f"p50 {fmt(q(Wm,.5))}")])
    o += blk

    # ---- percentile table -------------------------------------------------
    y += 46
    o.append(txt(PL, y, "分位数", 16, INK, weight=600))
    y += 14
    rows = [("profiling · forward 加权", [q(P, p) for _, p in QS]),
            ("profiling · token 加权", [wq(wpairs, p) for _, p in QS]),
            ("warmup · forward 加权", [q(Wm, p) for _, p in QS])]
    rh, hh = 26, 26
    th2 = hh + rh * len(rows)
    o.append(f'<rect x="{PL}" y="{y}" width="{pw}" height="{th2}" fill="{PANEL}" '
             f'stroke="{RULE}" rx="3"/>')
    lab_w = 210
    cw = (pw - lab_w - 16) / len(QS)
    o.append(txt(PL + 12, y + 17, "口径", 10.5, INK3, family=MONO, ls="0.8"))
    for i, (nm, _) in enumerate(QS):
        o.append(txt(PL + lab_w + cw * (i + 1) - 6, y + 17, nm.upper(), 10.5, INK3, "end", MONO, ls="0.8"))
    for j, (lab, vals) in enumerate(rows):
        ry = y + hh + rh * j
        o.append(f'<line x1="{PL}" x2="{PL+pw}" y1="{ry}" y2="{ry}" stroke="{GRID}"/>')
        o.append(txt(PL + 12, ry + 17, lab, 12, INK2, family=MONO))
        for i, v in enumerate(vals):
            o.append(txt(PL + lab_w + cw * (i + 1) - 6, ry + 17, fmt(v), 12, INK, "end", MONO))
    y += th2 + 20
    o.append(txt(PL, y, "token 加权 = 按每个 forward 产出的 b 个 token 加权；它比 forward 加权更窄，"
                        "因为低 L_eq 的 forward 往往也是小 b。做吞吐归因时用这一行。", 12, INK2))
    y += 20
    o.append(txt(PL, y, "注意：本图只画 L_eq 的位置，未画 batch 内部离散度 —— profiling 段 r_max 中位数 3.58，"
                        "75.5% 的 forward 是 eq_confidence=low。窄的是均值分布，宽的是每个 batch 内部。", 12, INK2))
    H = int(y + 34)
    o.append("</svg>")
    open(a.out, "w").write("\n".join(o).replace("__H__", str(H)) + "\n")
    print(f"{a.out}  {H}px  bins={nbin} bin={a.bin} "
          f"profiling={len(P)} warmup={len(Wm)}")


if __name__ == "__main__":
    main()
