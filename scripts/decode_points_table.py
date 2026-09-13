#!/usr/bin/env python3
"""把 decode 主表的逐组结果整理成性能点表。

一行 = 一个 `(L, b)` 等价点，给三个量：
    D_ms        稳定窗口的引擎侧步长，`TPS = b/D`
    fwd_gpu_ms  模型 forward（HIP event），与 D 的差仅作周期差额诊断
    itl_ms      客户端可见 ITL

同一 tag 保留最后一个 OK 结果；没有 OK 时保留最后一次失败记录。
"""
import argparse, csv, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--results", default="decode/bench_run/traces_top150_results.csv")
ap.add_argument("-p", "--points", default="decode/decode_points_top150.csv")
ap.add_argument("-o", "--out-prefix", default="decode/decode_points_measured")
a = ap.parse_args()

res = {}
for line in open(a.results):
    if not line.strip():
        continue
    record = json.loads(line)
    tag = record["tag"]
    if record.get("accept") == "OK" or res.get(tag, {}).get("accept") != "OK":
        res[tag] = record
meta = {}
for r in csv.DictReader(open(a.points)):
    meta[f"p{r['rank']}_L{r['L_test']}_b{r['b']}"] = r

rows = []
for d in res.values():
    m = meta.get(d["tag"], {})
    rows.append(dict(
        rank=int(m.get("rank", 0) or 0), tag=d["tag"], accept=d.get("accept"),
        L_test=int(m.get("L_test", 0) or 0), b=d.get("b"),
        L_eq_meas=d.get("L_eq_mid"),
        D_ms=d.get("D_ms"), fwd_gpu_ms=d.get("fwd_gpu_ms_p50"),
        step_wall_ms=d.get("step_wall_ms_p50"), itl_ms=d.get("itl_ms_p50"),
        tps_tok_s=d.get("tps_tok_s"),
        overhead_ms=d.get("overhead_ms"), gpu_busy=d.get("gpu_busy_frac"),
        n_steps=d.get("n_probe_used"), reason=d.get("reason", ""),
        forwards=m.get("forwards", ""), tokens=m.get("tokens", ""),
        tok_pct=m.get("tok_pct", ""),
    ))
rows.sort(key=lambda r: r["rank"])
if not rows:
    sys.exit("结果文件是空的")

with open(a.out_prefix + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["accept"] == "OK"]
L = ["# decode 性能点实测（主表）", "",
     f"{len(rows)} 个点（按 tag 去重），accept OK **{len(ok)}**，BAD {len(rows)-len(ok)}。", "",
     "`D` = 引擎侧 step_period；`fwd` = 纯模型 forward；`ITL` = 客户端可见。",
     "`TPS = b/D`（整个 batch 的 decode token 吞吐）。", "",
     "| rank | L | b | D (ms) | fwd (ms) | ITL (ms) | TPS (tok/s) | D − fwd (ms) | fwd / D | 步数 | accept |",
     "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
def f(v, n=2):
    return f"{v:.{n}f}" if isinstance(v, (int, float)) else "—"
for r in rows:
    L.append(f"| {r['rank']} | {r['L_test']:,} | {r['b']} | {f(r['D_ms'])} | {f(r['fwd_gpu_ms'])} | "
             f"{f(r['itl_ms'])} | {f(r['tps_tok_s'],1)} | {f(r['overhead_ms'])} | "
             f"{f(r['gpu_busy'],3)} | {r['n_steps']} | {r['accept']}"
             + (f" ({r['reason']})" if r["accept"] != "OK" else "") + " |")
if ok:
    gb = [r["gpu_busy"] for r in ok if r["gpu_busy"]]
    dv = [r["itl_ms"] - r["D_ms"] for r in ok if r["itl_ms"] and r["D_ms"]]
    L += ["", "## 两路口径的差", "",
          f"- GPU busy `fwd/D`：{min(gb):.3f} .. {max(gb):.3f}（中位 {sorted(gb)[len(gb)//2]:.3f}）",
          f"- `ITL − D`：{min(dv):+.2f} .. {max(dv):+.2f} ms（中位 {sorted(dv)[len(dv)//2]:+.2f}）"]
open(a.out_prefix + ".md", "w").write("\n".join(L) + "\n")
print(f"{a.out_prefix}.csv / .md: {len(rows)} 行，OK {len(ok)}")
