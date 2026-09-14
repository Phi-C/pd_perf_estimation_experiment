#!/usr/bin/env python3
"""用成对校准批拟合 Decode 工作量等价法中的异构修正系数。

    log(D_real / D_base) = b1 * CV_L^2 + b2 * log r_max        (+ b3 * CV_block^2)

`D_base` 取同一个点的 A_homog（同 L_eq、同 b、CV_L=0、r_max=1），所以每个点自带
基准，`L` 和 `b` 的主效应被差分掉了 —— 剩下的只有形状。

β3 在本 setup 下不可识别（align == page == 1024 ⇒ CV_block ≡ CV_L），
默认只拟两项；--with-cvb 强行加第三项只是为了把病态条件数打出来看。

系数不假设为正：Split-K / 分区 / 负载均衡完全可能让某些异构形状更快。
"""
import argparse, csv, json, math, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--results", default="decode/bench_run/traces_calib_results.csv")
ap.add_argument("-p", "--points", default="decode/decode_calib_points.csv")
ap.add_argument("--with-cvb", action="store_true")
ap.add_argument("--metric", default="D_ms", choices=["D_ms", "fwd_gpu_ms_p50", "itl_ms_p50"])
ap.add_argument("-o", "--out", default="decode/decode_imbalance_fit.md")
a = ap.parse_args()

res = {}
for ln in open(a.results):
    ln = ln.strip()
    if not ln:
        continue
    d = json.loads(ln)
    res[d["tag"]] = d
pts = {f"{r['point_id']}__{r['shape']}": r for r in csv.DictReader(open(a.points))}

rows, dropped = [], []
for tag, d in sorted(res.items()):
    if d.get("accept") != "OK":
        dropped.append((tag, d.get("reason", "accept != OK"))); continue
    pid = tag.split("__")[0]
    base = res.get(f"{pid}__A_homog")
    if not base or base.get("accept") != "OK":
        dropped.append((tag, "同点的 A_homog 基准不可用")); continue
    if tag.endswith("__A_homog"):
        continue
    y = math.log(d[a.metric] / base[a.metric])
    # 特征用**实测** ctx 还原出来的，不是设计值 —— 落地可能有偏差
    cv, rm = d["CV_L_meas"], d["r_max_meas"]
    rows.append(dict(tag=tag, point=pid, shape=tag.split("__")[1],
                     b=d["b"], L_eq=d["L_eq_mid"], CV_L=cv, r_max=rm,
                     D=d[a.metric], D_base=base[a.metric], ratio=d[a.metric]/base[a.metric],
                     y=y, x_cv2=cv*cv, x_logr=math.log(rm),
                     x_cvb2=float(pts.get(tag, {}).get("CV_block", cv))**2))

if len(rows) < 4:
    sys.exit(f"可用样本只有 {len(rows)} 条，先把校准批跑完（drop {len(dropped)}）")

cols = ["x_cv2", "x_logr"] + (["x_cvb2"] if a.with_cvb else [])
X = np.array([[r[c] for c in cols] for r in rows])
y = np.array([r["y"] for r in rows])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
cond = float(np.linalg.cond(X))
mape = float(np.mean(np.abs(np.expm1(pred - y)))) * 100

L = ["# Decode 异构修正系数", "",
     f"指标 `{a.metric}`，样本 {len(rows)} 条（B/C/D 形状），基准 = 同点 A_homog。", "",
     "| 项 | 系数 |", "|---|---:|"]
for c, v in zip(cols, beta):
    L.append(f"| `{c}` | {v:+.5f} |")
L += ["", f"- 样本内 R² = **{r2:.4f}**", f"- cond(X) = **{cond:.2f}**（较大时检查共线性）",
      f"- 样本内时间预测的平均相对误差 = **{mape:.2f}%**；未做独立验证", "",
      "定价公式：`D = D_n(L_eq) · exp(" +
      " ".join(f"{v:+.5f}·{c.replace('x_cv2','CV_L²').replace('x_logr','log r_max').replace('x_cvb2','CV_block²')}"
                 for c, v in zip(cols, beta)) + ")`", "",
      "## 逐样本", "",
      "| tag | b | L_eq | CV_L | r_max | D_base | D | 实测比 | 拟合比 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r, p in zip(rows, pred):
    L.append(f"| {r['tag']} | {r['b']} | {r['L_eq']:.0f} | {r['CV_L']:.4f} | {r['r_max']:.4f} | "
             f"{r['D_base']:.2f} | {r['D']:.2f} | {r['ratio']:.4f} | {math.exp(p):.4f} |")
if dropped:
    L += ["", f"## 丢弃 {len(dropped)} 条", ""] + [f"- `{t}`: {w}" for t, w in dropped]
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L[:14]))
print(f"\n-> {a.out}")
