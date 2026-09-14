#!/usr/bin/env python3
"""Join uniform Prefill probe buckets to a reference point table.

fwd_gpu_ms is the GPU-event time around run_model. The historical reference
field target_fwd_gpu_ms is inferred from TTFT, not directly measured GPU time.
Host durations and return-to-return periods are diagnostic only: the Prefill
harness interleaves primer, blocker and target shapes.

See docs/forward_probe.md for the measurement boundaries and matching rule.
"""
import argparse, json, csv, statistics as st
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--probe", type=Path, required=True)
ap.add_argument("--reference", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
PROBE, REF, OUT = a.probe, a.reference, a.output
OUT.parent.mkdir(parents=True, exist_ok=True)

def med(v): return st.median(v) if v else float("nan")
def pct(v, p):
    if not v: return float("nan")
    s = sorted(v); return s[min(len(s) - 1, int(p / 100 * len(s)))]

# ---- probe: uniform-shape, non-dummy, pure-prefill forwards
buckets = defaultdict(lambda: defaultdict(list))
n_raw = n_kept = 0
for line in open(PROBE):
    line = line.strip()
    if not line: continue
    try: r = json.loads(line)
    except json.JSONDecodeError: continue
    n_raw += 1
    if r.get("dummy") or r.get("decode", 0) or "ctx" not in r: continue
    ctx, q = r["ctx"], r["q"]
    if len(set(ctx)) != 1 or len(set(q)) != 1: continue
    key = (ctx[0], q[0], r["b"])
    n_kept += 1
    for src, dst in (("fwd_gpu_ms", "gpu"), ("step_wall_ms", "host"),
                     ("step_period_ms", "period")):
        if src in r: buckets[key][dst].append(r[src])

# ---- reference
ref, refrow = {}, {}
I = lambda x: int(round(float(x)))
for r in csv.DictReader(open(REF)):
    if r.get("status") != "measured": continue
    qa = r.get("q_actual") or r["q"]
    try: k = (I(r["c"]), I(qa), I(r["b"]))
    except (ValueError, TypeError): continue
    ref[k] = float(r["target_fwd_gpu_ms"]); refrow[k] = r

REFCOLS = ["priority","c","q","q_actual","b","stage","measured_ms","metric",
           "client_ms","tail_fwd_ms","target_fwd_gpu_ms","n_samples","status","round"]
PROBECOLS = ["probe_c","probe_q","probe_b","probe_n","fwd_gpu_ms","fwd_gpu_p10",
             "fwd_gpu_p90","host_ms","period_ms","gpu_over_target","gpu_minus_target_ms"]

print(f"probe: {n_raw} records, {n_kept} usable uniform-prefill forwards, "
      f"{len(buckets)} distinct (c,q,b)\n")
hdr = (f"{'c':>8} {'q':>6} {'b':>3} {'n':>4} | {'fwd_gpu':>9} {'host':>8} "
       f"{'period':>8} | {'target':>9} {'gpu/tgt':>8}")
print(hdr); print("-" * len(hdr))

out, ratios, unmatched = [], [], 0
for k in sorted(buckets):
    c, q, b = k
    v = buckets[k]
    g, h, p = med(v["gpu"]), med(v["host"]), med(v["period"])
    n = len(v["gpu"])
    # q<1024 has no ISL anchor, so aiperf's synthetic length jitters a few
    # tokens (see run_stage_b1.sh header); match within tolerance or every
    # small-q point silently fails to join.
    rk = k if k in ref else None
    if rk is None:
        tol = 0 if q >= 1024 else 16
        cand = [j for j in ref if j[0] == c and j[2] == b and abs(j[1] - q) <= tol]
        if cand: rk = min(cand, key=lambda j: abs(j[1] - q))
    row = {cc: "" for cc in REFCOLS}
    row.update(probe_c=c, probe_q=q, probe_b=b, probe_n=n,
               fwd_gpu_ms=round(g, 2), fwd_gpu_p10=round(pct(v["gpu"], 10), 2),
               fwd_gpu_p90=round(pct(v["gpu"], 90), 2),
               host_ms=round(h, 2), period_ms=round(p, 2),
               gpu_over_target="", gpu_minus_target_ms="")
    if rk is not None:
        tgt = ref[rk]
        row.update({cc: refrow[rk].get(cc, "") for cc in REFCOLS})
        row["gpu_over_target"] = round(g / tgt, 4) if tgt else ""
        row["gpu_minus_target_ms"] = round(g - tgt, 2)
        ratios.append((k, g, tgt))
        rs = f"{tgt:9.1f} {g / tgt:8.3f}"
    else:
        unmatched += 1
        rs = f"{'-':>9} {'-':>8}"
    out.append(row)
    print(f"{c:8d} {q:6d} {b:3d} {n:4d} | {g:9.2f} {h:8.2f} {p:8.2f} | {rs}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=REFCOLS + PROBECOLS)
    w.writeheader(); w.writerows(out)

print(f"\n-> {OUT}   ({len(out)} rows, {len(ratios)} matched, {unmatched} probe-only)")

if ratios:
    r = [g / t for _, g, t in ratios if t]
    big = [g / t for (c, q, b), g, t in ratios if t and q >= 1024]
    small = [g / t for (c, q, b), g, t in ratios if t and q < 1024]
    print(f"\n=== fwd_gpu / target_fwd_gpu over {len(r)} matched points ===")
    print("  NOTE: fwd_gpu covers run_model (model forward + compute_logits) only,")
    print("        while target_fwd_gpu_ms was inferred from TTFT and therefore also")
    print("        carries prepare_inputs GPU ops, KV offload and sampling. A ratio")
    print("        below 1 compares two different measurement boundaries; it is not")
    print("        an independent measurement of each non-model overhead component.")
    print(f"  all      n={len(r):3d}  median {med(r):.3f}  min {min(r):.3f}  max {max(r):.3f}")
    if big:   print(f"  q>=1024  n={len(big):3d}  median {med(big):.3f}   <- tail_fwd_ms (model-derived) applied here")
    if small: print(f"  q< 1024  n={len(small):3d}  median {med(small):.3f}   <- tail=0, tests the client_ms=92 constant alone")
    solid = [(k, g, t) for k, g, t in ratios
             if t and len(buckets[k]["gpu"]) >= 5]
    if solid:
        rs_ = [g / t for _, g, t in solid]
        sb = [g / t for (c, q, b), g, t in solid if q >= 1024]
        ss = [g / t for (c, q, b), g, t in solid if q < 1024]
        print(f"\n  --- restricted to buckets with n>=5 forwards (noise-robust) ---")
        print(f"  all      n={len(rs_):3d}  median {med(rs_):.3f}  min {min(rs_):.3f}  max {max(rs_):.3f}")
        if sb: print(f"  q>=1024  n={len(sb):3d}  median {med(sb):.3f}")
        if ss: print(f"  q< 1024  n={len(ss):3d}  median {med(ss):.3f}")

    gb = sum(1 for k in buckets for a, b_ in
             zip(buckets[k]["host"], buckets[k]["gpu"]) if a < b_)
    tot = sum(len(buckets[k]["gpu"]) for k in buckets)
    print(f"\nForwards with host duration < GPU-event duration: {gb}/{tot}")
