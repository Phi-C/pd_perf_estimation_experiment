#!/usr/bin/env python3
"""Rank the (L_test, b) test points of decode_eq_batches.csv and keep the top N.

Weight = output tokens produced at that point (sum of b over its forwards), so
the ranking answers "measuring these points prices what fraction of the run".
Each row also carries the imbalance spread *inside* the point, because a single
(L_test, b) coordinate is not one workload: r_max there can run from 1 to 5+,
which is exactly what the §10 heterogeneous calibration needs to know.
"""
import argparse, csv, os, statistics as st
from collections import defaultdict


def pct(v, p):
    return v[min(len(v) - 1, int(p * (len(v) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--csv", default="decode_eq_batches.csv")
    ap.add_argument("-o", "--out", default="decode_points_top150.csv")
    ap.add_argument("-n", "--top", type=int, default=150)
    ap.add_argument("--by", choices=["tokens", "forwards"], default="tokens")
    ap.add_argument("--phase", default="profiling", help="'' = 不过滤")
    a = ap.parse_args()

    g = defaultdict(lambda: dict(f=0, tok=0, leq=[], rmax=[], cv=[], cvb=[], lmax=[], conf=[]))
    for r in csv.DictReader(open(a.csv)):
        if a.phase and r["phase"] != a.phase:
            continue
        k = (int(r["L_test"]), int(r["b"]))
        d = g[k]
        d["f"] += 1
        d["tok"] += k[1]
        d["leq"].append(float(r["L_eq"]))
        d["rmax"].append(float(r["r_max"]))
        d["cv"].append(float(r["CV_L"]))
        d["cvb"].append(float(r["CV_block"]))
        d["lmax"].append(int(r["L_max"]))
        d["conf"].append(r["eq_confidence"])

    tot_t = sum(d["tok"] for d in g.values())
    tot_f = sum(d["f"] for d in g.values())
    key = (lambda kv: -kv[1]["tok"]) if a.by == "tokens" else (lambda kv: -kv[1]["f"])
    rank = sorted(g.items(), key=key)[: a.top]

    rows, ct, cf = [], 0, 0
    for i, ((L, b), d) in enumerate(rank, 1):
        ct += d["tok"]
        cf += d["f"]
        rm = sorted(d["rmax"])
        lm = sorted(d["lmax"])
        rows.append(dict(
            rank=i, L_test=L, b=b,
            forwards=d["f"], tokens=d["tok"],
            tok_pct=round(100 * d["tok"] / tot_t, 4),
            cum_tok_pct=round(100 * ct / tot_t, 3),
            cum_fwd_pct=round(100 * cf / tot_f, 3),
            # 这个点上真实 L_eq 的落点（吸附前），确认吸附没走样
            L_eq_mean=round(sum(d["leq"]) / d["f"], 1),
            L_eq_min=round(min(d["leq"]), 1), L_eq_max=round(max(d["leq"]), 1),
            # 点内部的不均衡分布 —— §10 校准要成对造的就是这个跨度
            r_max_p50=round(pct(rm, .5), 3), r_max_p90=round(pct(rm, .9), 3),
            r_max_max=round(rm[-1], 3),
            CV_L_p50=round(pct(sorted(d["cv"]), .5), 4),
            CV_block_p50=round(pct(sorted(d["cvb"]), .5), 4),
            # §11 max-shape 场景的平行坐标
            L_max_p50=pct(lm, .5), L_max_p90=pct(lm, .9),
            conf_low_pct=round(100 * d["conf"].count("low") / d["f"], 1),
            eq_confidence=st.mode(d["conf"]),
        ))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.out}: top-{a.top}/{len(g)} points by {a.by} | "
          f"tokens {rows[-1]['cum_tok_pct']}% forwards {rows[-1]['cum_fwd_pct']}% | "
          f"L {min(r['L_test'] for r in rows):,}..{max(r['L_test'] for r in rows):,} "
          f"b {min(r['b'] for r in rows)}..{max(r['b'] for r in rows)}")


if __name__ == "__main__":
    main()
