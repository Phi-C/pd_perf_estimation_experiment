#!/usr/bin/env python3
"""把新一轮的待测点清单和上一轮已实测的点对齐, 标出哪些真的要重测。

背景: 换了口径(比如从全 trace 切到 profiling 段)重跑 gen_final_points.py 之后, 点集会变。
但"新点"不等于"要测的点" —— 拟合好的 7 参数模型在**已测点张成的包络内**是插值, 精度可信;
只有掉到包络外的才是外推, 必须真去机器上测。这个脚本就是做这个三分类。

包络按 b 分别算: 对每个 b, 取已测点的 c 范围和 q 范围, 做矩形包络。
不用最近邻距离 —— 相对距离会把 q=1024 vs 2048 这种"恰好差一个网格步长"的点误判成远点。

产物 <dir>/points_todo.csv, status 三类:
  measured     上轮已测, 直接查表
  interp       包络内, 模型插值, 不用测
  extrap       包络外, **需要补测**
"""
import argparse, csv, collections, os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default=".", help="含 final_points_unique.csv 的目录")
ap.add_argument("--measured", default="bench_run/RESULTS_214_points.csv",
                help="上一轮结果, 需含 c,q,b,status 列")
ap.add_argument("--cover", type=float, default=80.0,
                help="只关心累计时间覆盖到这个百分比的前 N 个点; 100 = 全部")
a = ap.parse_args()

M = {(int(r["c"]), int(r["q"]), int(r["b"]))
     for r in csv.DictReader(open(a.measured)) if r["status"] == "measured"}
env = collections.defaultdict(lambda: [10**18, -1, 10**18, -1])
for c, q, b in M:
    e = env[b]
    e[0], e[1], e[2], e[3] = min(e[0], c), max(e[1], c), min(e[2], q), max(e[3], q)

rows = list(csv.DictReader(open(os.path.join(a.dir, "final_points_unique.csv"))))
total = sum(float(r["exec_seconds"]) for r in rows)
# 含跨过阈值的那一行: 累计 <= cover 会停在 79.x%, 少一个点
keep = []
for r in rows:
    keep.append(r)
    if float(r["cum_time_pct"]) >= a.cover:
        break

out, agg = [], collections.Counter()
for r in keep:
    c, q, b = (int(r["final_context_length"]), int(r["final_query_length"]),
               int(r["eq_batch_size"]))
    s, e = float(r["exec_seconds"]), env.get(b)
    if (c, q, b) in M:
        stt, why = "measured", ""
    elif e and e[0] <= c <= e[1] and e[2] <= q <= e[3]:
        stt, why = "interp", f"b={b} 包络 c[{e[0]},{e[1]}] q[{e[2]},{e[3]}] 内"
    else:
        d = ([] if not e else
             [f"c {c} {'<' if c < e[0] else '>'} [{e[0]},{e[1]}]"] * (not e[0] <= c <= e[1]) +
             [f"q {q} {'<' if q < e[2] else '>'} [{e[2]},{e[3]}]"] * (not e[2] <= q <= e[3]))
        stt, why = "extrap", "; ".join(d) or f"b={b} 从未测过"
    agg[stt] += s
    out.append({**r, "status": stt, "reason": why,
                "time_pct_of_all": round(100 * s / total, 3)})

P = os.path.join(a.dir, "points_todo.csv")
with open(P, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    # extrap 排最前: 这份表是拿去排测试计划的
    w.writerows(sorted(out, key=lambda r: ({"extrap": 0, "interp": 1, "measured": 2}[r["status"]],
                                           -float(r["exec_seconds"]))))
print(f"{P}: {len(keep)}/{len(rows)} 点 (覆盖 {a.cover}% 时间), 全量 {total:.1f}s")
for k in ("measured", "interp", "extrap"):
    n = sum(1 for r in out if r["status"] == k)
    print(f"  {k:9s} {n:4d} 点  {agg[k]:7.1f}s  {100*agg[k]/total:5.1f}% 时间")
