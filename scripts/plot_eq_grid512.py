#!/usr/bin/env python3
"""
prefill_forwards.csv -> (q_eq, c_eq) 在 512-token 等距网格上的点数分布图 + 每格中位数中心表

    python3 plot_eq_grid512.py -d <outdir> [--bin 512]

产出:
    eq_grid512_distribution.svg   网格点数热力图 + 两条边缘直方图 + 计数图例
    eq_grid512_cells.csv          每个非空网格一行: 区间边界 / 点数 / 中位数中心 / 出现过的 batch_size
    prefill_forwards.csv          原地补 4 列 grid_*(每行所属格号 + 该格中位数中心), --no-write-back 可关
    eq_point_distribution.md      追加一节 "附: 512-token 等距网格上的点数分布"(幂等, 重跑会覆盖该节)

注意运行顺序: 本脚本会往 plot_eq_distribution.py 产出的 eq_point_distribution.md 末尾追加章节,
所以两个都要重跑时, 先跑 plot_eq_distribution.py, 再跑本脚本。

统计粒度和 plot_eq_distribution.py 一致: 一个 forward 一个点(先按 forward_idx 去重)。
网格定义: q_bin = floor(q_eq / 512), c_bin = floor(c_eq / 512), 区间左闭右开。
q_eq == 8192(=max_num_batched_tokens) 落在最后一格 [8192, 8704)。

纯标准库, 不依赖 matplotlib/numpy。
"""
import csv, os, math, statistics, collections, argparse

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--dir", default=".")
AP.add_argument("--bin", type=int, default=512)
AP.add_argument("--label", default="")
AP.add_argument("--no-write-back", action="store_true",
                help="不要往 prefill_forwards.csv 回写 grid_* 列")
ARGS = AP.parse_args()
B = ARGS.bin
P_ = lambda n: os.path.join(ARGS.dir, n)

rows = list(csv.DictReader(open(P_("prefill_forwards.csv"))))
one = {}
for x in rows: one[int(x["forward_idx"])] = x          # 一个 forward 一个点
PTS = [(float(v["eq_query_length"]), float(v["eq_context_length"]), int(v["eq_batch_size"]))
       for v in one.values()]
N = len(PTS)

cell = collections.defaultdict(list)                   # (c_bin, q_bin) -> [(q, c, b), ...]
for q, c, b in PTS:
    cell[(int(c // B), int(q // B))].append((q, c, b))
NQ = max(k[1] for k in cell) + 1                       # 列数(q 方向)
NC = max(k[0] for k in cell) + 1                       # 行数(c 方向)
CNT = {k: len(v) for k, v in cell.items()}
MAXN = max(CNT.values())

# ------------------------------------------------------------------ CSV: 每格中位数中心
out = []
for (ci, qi), pts in sorted(cell.items()):
    qs = [p[0] for p in pts]; cs = [p[1] for p in pts]
    bs = sorted({p[2] for p in pts})
    out.append({
        "q_bin": qi, "c_bin": ci,
        "q_lo": qi * B, "q_hi": (qi + 1) * B,
        "c_lo": ci * B, "c_hi": (ci + 1) * B,
        "n_points": len(pts),
        "median_eq_query_length": round(statistics.median(qs), 1),
        "median_eq_context_length": round(statistics.median(cs), 1),
        "n_batch_sizes": len(bs),
        "batch_sizes": ";".join(str(x) for x in bs),
    })
FIELDS = ["q_bin", "c_bin", "q_lo", "q_hi", "c_lo", "c_hi", "n_points",
          "median_eq_query_length", "median_eq_context_length", "n_batch_sizes", "batch_sizes"]
with open(P_("eq_grid512_cells.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(out)

# ------------------------------------------------------------------ 回写网格列到 prefill_forwards.csv
# 每一行(每个 forward 里的每条请求)都按它所属的 forward 的 (q_eq, c_eq) 落格,
# 补上格号和该格的中位数中心。幂等: 已存在的这几列先删掉再写。
GRID_COLS = ["grid_q_bin", "grid_c_bin", "grid_median_query_length", "grid_median_context_length"]
if not ARGS.no_write_back:
    MED = {(r["c_bin"], r["q_bin"]): (r["median_eq_query_length"], r["median_eq_context_length"])
           for r in out}
    src = P_("prefill_forwards.csv")
    base_fields = [f for f in csv.DictReader(open(src)).fieldnames if f not in GRID_COLS]
    for x in rows:
        qi, ci = int(float(x["eq_query_length"]) // B), int(float(x["eq_context_length"]) // B)
        mq, mc = MED[(ci, qi)]
        x["grid_q_bin"], x["grid_c_bin"] = qi, ci
        x["grid_median_query_length"], x["grid_median_context_length"] = mq, mc
    with open(src, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + GRID_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("wrote {} rows x {} cols back into {}".format(len(rows), len(base_fields) + len(GRID_COLS), src))

# ------------------------------------------------------------------ 条纹效应(c_eq 每步 +8192)
ROW = collections.Counter(k[0] for k, v in cell.items() for _ in v)
_m16 = [ROW.get(i, 0) for i in range(NC) if i % 16 == 0]
_oth = [ROW.get(i, 0) for i in range(NC) if i % 16]
STRIPE_RATIO = (sum(_m16) / len(_m16)) / (sum(_oth) / len(_oth))
N1 = sum(1 for v in CNT.values() if v == 1)
N5 = sum(1 for v in CNT.values() if v >= 5)
NBS = collections.Counter(len({p[2] for p in v}) for v in cell.values())
QCOL = sum(1 for k in cell if k[1] == NQ - 1)
_multi = [k for k, v in cell.items() if len({p[2] for p in v}) > 1]
_solo  = [k for k, v in cell.items() if len({p[2] for p in v}) == 1]
MQ_MULTI = statistics.median(k[1] for k in _multi) * B; MQ_SOLO = statistics.median(k[1] for k in _solo) * B
MC_MULTI = statistics.median(k[0] for k in _multi) * B; MC_SOLO = statistics.median(k[0] for k in _solo) * B
BCELL = collections.Counter(b for v in cell.values() for b in {p[2] for p in v})
OCCR = len({k[0] for k in cell})

# ------------------------------------------------------------------ 追加到 eq_point_distribution.md
MARK = "<!-- BEGIN eq_grid512 (由 scripts/plot_eq_grid512.py 生成, 勿手工编辑) -->"
M = [MARK, "",
     "## 附: {}-token 等距网格上的点数分布".format(B), "",
     "前面的分桶是 log 尺度的。这里换成 **等距网格**: `q_bin = floor(q_eq/{0})`, `c_bin = floor(c_eq/{0})`, "
     "区间左闭右开, `q_eq = 8192` 落在最后一格 `[8192, 8704)`。".format(B), "",
     "- 网格规模: {} x {} = {:,} 格, **非空 {:,} 格 (占用率 {:.1f}%)**"
     .format(NQ, NC, NQ * NC, len(cell), len(cell) / (NQ * NC) * 100),
     "- 其中只有 1 个点的 {:,} 格 ({:.0f}%), >=5 个点的只有 {:,} 格; 单格最多 {} 个点"
     .format(N1, N1 / len(cell) * 100, N5, MAXN),
     "- c 方向 {:,} 行里有 {:,} 行被占用 ({:.0f}%)".format(NC, OCCR, OCCR / NC * 100), "",
     "点数最多的几个格 (全部落在 `q_eq = 8192` 那一列, 且 `c_lo` 都是 8192 的整数倍):", "",
     "| q 区间 | c 区间 | 点数 | 中位数中心 (q, c) | batch_size |", "|---|---|---:|---|---|"]
for r in sorted(out, key=lambda r: -r["n_points"])[:5]:
    M.append("| `[{:,}, {:,})` | `[{:,}, {:,})` | {} | ({:,.0f}, {:,.0f}) | {} |".format(
        r["q_lo"], r["q_hi"], r["c_lo"], r["c_hi"], r["n_points"],
        r["median_eq_query_length"], r["median_eq_context_length"], r["batch_sizes"]))
M += ["",
      "### 等距分箱暴露出来的结构: c_eq 的 8192 条纹", "",
      "单请求 chunked prefill 每步 `c_eq` 正好 +8192(= `max_num_batched_tokens`), 所以点会周期性地"
      "落在 `c_bin` 为 16 的倍数上。量化一下: 这些行平均 {:.1f} 个点, 其余行平均 {:.1f} 个, **相差 {:.1f}x**。"
      .format(sum(_m16) / len(_m16), sum(_oth) / len(_oth), STRIPE_RATIO),
      "这是个真实但温和的效应 —— 不是\"只有 16 倍数行有点\": 条纹外的行合计仍占 {:.0f}% 的 forward"
      "(多请求 batch 和不同 ISL 会把 c_eq 打散), 而且条纹主要集中在 c 较小的前几十个格。"
      .format(sum(_oth) / N * 100),
      "log 分桶的图看不到这个周期性, 等距网格才看得见。", "",
      "### batch_size 在网格里的分布", "",
      "| 一个格里出现的不同 b 数 | 格数 |", "|---|---:|"]
for k in sorted(NBS):
    M.append("| {} | {:,} |".format(k, NBS[k]))
M += ["",
      "各 b 值覆盖的格数: " + ", ".join("`b={}`: {:,}".format(b, BCELL[b]) for b in sorted(BCELL)) + "。",
      "绝大多数格 ({:.0f}%) 只出现过一种 b, 所以\"网格 -> b\"基本是单值的。混合了多个 b 的 {:,} 个格明显偏向 "
      "**q 小的一侧**(这些格 `q_lo` 中位数 {:,.0f}, 单 b 格是 {:,.0f}) —— q 小才有余量拼进第二条请求; "
      "c 方向没有明显偏好({:,.0f} vs {:,.0f})。".format(
          NBS[1] / len(cell) * 100, len(cell) - NBS[1], MQ_MULTI, MQ_SOLO, MC_MULTI, MC_SOLO), "",
      "### 对采样的意思", "",
      "1. **等距网格太稀**: {:.0f}% 的非空格只有 1 个点, 直接按格拟合会过拟合噪声。"
      "把等距网格当\"查询索引\"用(找某个 (q,c) 附近实测到什么), 拟合还是按 log 网格来。"
      .format(N1 / len(cell) * 100),
      "2. **`q_eq = 8192` 那一列是唯一稠密的一列**: 它占了 {:,} 个非空格, 覆盖 {:.0f}% 的被占用 c 行, "
      "适合先沿它把 c 方向的曲线拟出来。".format(QCOL, QCOL / OCCR * 100),
      "3. **条纹告诉你实测点的 c 是有系统性偏好的**: 想覆盖 c_bin 非 16 倍数的区域, 得靠多请求 batch 或"
      "不同 ISL 的 trace, 单请求 trace 采不到。", "",
      "配套产物:", "",
      "- `eq_grid512_distribution.svg` —— 网格点数热力图 + 两条边缘分布",
      "- `eq_grid512_cells.csv` —— 每个非空格一行: `q_bin, c_bin, q_lo, q_hi, c_lo, c_hi, n_points, "
      "median_eq_query_length, median_eq_context_length, n_batch_sizes, batch_sizes`",
      "  (`batch_sizes` = 该格内所有不重复 b, 升序, `;` 分隔; 中位数中心 = 格内所有点 q/c 的中位数)",
      "- `prefill_forwards.csv` —— 本脚本会原地补 4 列: `grid_q_bin`, `grid_c_bin`, "
      "`grid_median_query_length`, `grid_median_context_length`, 即每一行所属的格号和该格的中位数中心。",
      "  注意这 4 列是本脚本加的, `parse_prefill_log.py` 重跑会把它们冲掉, 重跑后再跑一次本脚本即可。"]

MDP = P_("eq_point_distribution.md")
base = open(MDP).read() if os.path.exists(MDP) else ""
base = base.split(MARK)[0].rstrip()          # 幂等: 重跑先摘掉旧章节
open(MDP, "w").write(base + "\n\n" + "\n".join(M) + "\n")

# ------------------------------------------------------------------ SVG
qm = [sum(CNT.get((ci, qi), 0) for ci in range(NC)) for qi in range(NQ)]   # 列边缘(q)
cm = [sum(CNT.get((ci, qi), 0) for qi in range(NQ)) for ci in range(NC)]   # 行边缘(c)

CW  = 34                       # 每列宽
RH  = 1.0                      # 每行高(c 方向 1487 行, 1px 一行)
ML, MT = 108, 150              # 左/上边距
TH  = 62                       # 上边缘直方图高
RW  = 130                      # 右边缘直方图宽
GW, GH = NQ * CW, NC * RH
PX0, PY0 = ML, MT
PX1, PY1 = PX0 + GW, PY0 + GH
W = PX1 + 24 + RW + 190
H = PY1 + 78

def col(n):                    # 计数 -> 颜色(log 标度)
    t = math.log(n) / math.log(MAXN) if MAXN > 1 else 1.0
    stops = [(0.00, (233, 240, 247)), (0.35, (141, 178, 211)),
             (0.70, (49, 110, 165)),  (1.00, (16, 45, 84))]
    for i in range(len(stops) - 1):
        a, b_ = stops[i], stops[i + 1]
        if t <= b_[0] or i == len(stops) - 2:
            u = (t - a[0]) / (b_[0] - a[0]) if b_[0] > a[0] else 0
            u = min(max(u, 0), 1)
            return "#%02x%02x%02x" % tuple(round(a[1][k] + (b_[1][k] - a[1][k]) * u) for k in range(3))

esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
S = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" '
     'font-family="DejaVu Sans, Helvetica, Arial, sans-serif">'.format(W, H, W, H),
     '<rect width="{}" height="{}" fill="#ffffff"/>'.format(W, H)]
def T(x, y, t, sz=11, fill="#333", anc="start", w="normal"):
    S.append('<text x="{:.1f}" y="{:.1f}" font-size="{}" fill="{}" text-anchor="{}" '
             'font-weight="{}">{}</text>'.format(x, y, sz, fill, anc, w, esc(t)))

T(ML, 30, "等价同构点在 {}-token 等距网格上的点数分布".format(B), 18, "#111", w="bold")
T(ML, 51, "{:,} 个 prefill forward / {:,} 个非空网格 (共 {} x {} = {:,} 格, 占用率 {:.1f}%)   {}"
  .format(N, len(cell), NQ, NC, NQ * NC, len(cell) / (NQ * NC) * 100, ARGS.label), 12, "#666")
T(ML, 68, "q_bin = floor(q_eq/{0}), c_bin = floor(c_eq/{0}); 每格颜色 = 落在该格的 forward 数(log 标度)"
  .format(B), 11, "#888")
T(ML, 84, "注意 c 轴每行只有 {:.0f}px。图上的水平亮条是真实结构: 单请求 chunked prefill 每步 c_eq 正好 +8192, "
  "c_bin 为 16 倍数的行平均点数是其余行的 {:.1f}x。".format(RH, STRIPE_RATIO), 11, "#a05a2c")

S.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#fafbfc" stroke="#d5d8dd"/>'
         .format(PX0, PY0, GW, GH))
for (ci, qi), n in CNT.items():
    S.append('<rect x="{:.1f}" y="{:.1f}" width="{}" height="{:.1f}" fill="{}"/>'
             .format(PX0 + qi * CW, PY0 + ci * RH, CW, max(RH, 1.0), col(n)))
for qi in range(NQ + 1):       # 列分隔线
    S.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#ffffff" stroke-width="0.6" '
             'stroke-opacity="0.7"/>'.format(PX0 + qi * CW, PY0, PY1))

# --- x 轴: q 区间
for qi in range(NQ):
    x = PX0 + (qi + .5) * CW
    lab = "8192" if qi * B == 8192 else "{:.0f}K".format(qi * B / 1024) if qi * B >= 1024 else str(qi * B)
    T(x, PY1 + 16, lab, 9.5, "#555", "middle", "bold" if qi * B == 8192 else "normal")
    T(x, PY1 + 28, "{:,}".format(qm[qi]), 9, "#8a8f98", "middle")
T(PX0 + GW / 2, PY1 + 52, "eq_query_length 区间下界 (每格宽 {} tokens; 灰色数字 = 该列 forward 数)"
  .format(B), 12, "#222", "middle", "bold")
# --- y 轴: c 区间
for v in range(0, NC * B + 1, 65536):
    y = PY0 + v / B * RH
    if y > PY1: break
    S.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" stroke="#888"/>'.format(PX0 - 6, y, PX0))
    T(PX0 - 10, y + 3, "{:.0f}K".format(v / 1024) if v else "0", 9.5, "#555", "end")
S.append('<text x="30" y="{0:.1f}" font-size="12" fill="#222" text-anchor="middle" font-weight="bold" '
         'transform="rotate(-90 30 {0:.1f})">eq_context_length 区间下界 (每格高 {1} tokens)</text>'
         .format((PY0 + PY1) / 2, B))

# --- 上边缘直方图 (q)
mx = max(qm)
for qi, n in enumerate(qm):
    h = n / mx * TH
    S.append('<rect x="{:.1f}" y="{:.1f}" width="{}" height="{:.1f}" fill="#7d9dbe" fill-opacity="0.9" '
             'stroke="#fff"/>'.format(PX0 + qi * CW + 1, PY0 - 8 - h, CW - 2, h))
    if n / N > .05:
        T(PX0 + (qi + .5) * CW, PY0 - 12 - h, "{:.0f}%".format(n / N * 100), 9, "#5b7fa6", "middle", "bold")
T(PX0, PY0 - 8 - TH - 8, "q_eq 边缘分布", 11, "#5b7fa6", w="bold")

# --- 右边缘直方图 (c), 按 16 格(=8192 tokens)聚合一根, 否则 1px 高的条看不见
GRP = 16
mx = max(sum(cm[i:i + GRP]) for i in range(0, NC, GRP))
for i in range(0, NC, GRP):
    n = sum(cm[i:i + GRP])
    if not n: continue
    w = n / mx * RW
    S.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="#7d9dbe" '
             'fill-opacity="0.9"/>'.format(PX1 + 24, PY0 + i * RH, w, GRP * RH))
T(PX1 + 24, PY0 - 14, "c_eq 边缘分布 (每 {} 格 = {:,} tokens 聚合)".format(GRP, GRP * B), 10.5,
  "#5b7fa6", w="bold")

# --- 图例
lx, ly = PX1 + 24 + RW + 28, PY0 + 6
T(lx, ly, "每格 forward 数", 11, "#333", w="bold")
LEG = [1, 2, 3, 5, 8, 13, 21, 34, MAXN]
for i, n in enumerate(LEG):
    y = ly + 14 + i * 19
    S.append('<rect x="{}" y="{}" width="22" height="13" fill="{}" stroke="#c9ccd2"/>'.format(lx, y, col(n)))
    T(lx + 29, y + 11, "{}{}".format(n, "  (max)" if n == MAXN else ""), 10, "#444")
y = ly + 14 + len(LEG) * 19 + 16
for i, t in enumerate([
        "空白 = 该网格没有点", "",
        "非空网格 {:,} 个, 其中".format(len(cell)),
        "n=1 的 {:,} 个 ({:.0f}%),".format(sum(1 for v in CNT.values() if v == 1),
                                          sum(1 for v in CNT.values() if v == 1) / len(cell) * 100),
        "n>=5 的 {:,} 个".format(sum(1 for v in CNT.values() if v >= 5)), "",
        "每格的中位数中心 (q,c) 和", "出现过的 batch_size 见", "eq_grid512_cells.csv"]):
    T(lx, y + i * 14, t, 9.5, "#777")
S.append("</svg>")
open(P_("eq_grid512_distribution.svg"), "w").write("\n".join(S))

print("wrote {} ({:,} cells) and {}".format(P_("eq_grid512_cells.csv"), len(out),
                                            P_("eq_grid512_distribution.svg")))
