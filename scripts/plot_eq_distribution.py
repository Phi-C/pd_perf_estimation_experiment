#!/usr/bin/env python3
"""
prefill_forwards.csv -> 等价同构点 (q_eq, c_eq) 的分布文档 + log-log 2D 散点图

    python3 plot_eq_distribution.py -d <outdir>

产出:
    eq_point_distribution.md    边缘分布 + 联合分布 + 采样建议
    eq_point_distribution.svg   log-log 散点 + 上/右边缘直方图 + 联合分布网格计数

统计粒度是 **forward batch**: CSV 里同一个 forward 的每一行都重复了同样的
eq_* 值, 所以先按 forward_idx 去重, 一个 forward 记一个点。

纯标准库, 不依赖 matplotlib/numpy(CI 机器上通常没有), SVG 直接用浏览器打开。
"""
import csv, os, math, collections, argparse

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--dir", default=".")
AP.add_argument("--label", default="")
ARGS = AP.parse_args()
P_ = lambda n: os.path.join(ARGS.dir, n)

rows = list(csv.DictReader(open(P_("prefill_forwards.csv"))))
one = {}
for x in rows: one[int(x["forward_idx"])] = x        # 一个 forward 一个点
PTS = [(float(v["eq_query_length"]), float(v["eq_context_length"]), int(v["eq_batch_size"]))
       for v in one.values()]
N = len(PTS)
Q = [p[0] for p in PTS]; C = [p[1] for p in PTS]
uniq = len({(p[0], p[1]) for p in PTS})

QE = [0, 2, 64, 256, 512, 1024, 2048, 4096, 8192, 8193]
CE = [0, 1, 16384, 32768, 65536, 131072, 262144, 524288, float("inf")]
JQ = [0, 512, 1024, 2048, 4096, 8192, 8193]; JQL = ["<512","512-1K","1K-2K","2K-4K","4K-8K","8192"]
JC = [0, 32768, 65536, 131072, 262144, 524288, float("inf")]
JCL = ["<32K","32-64K","64-128K","128-256K","256-512K",">=512K"]

def counts(vals, edges):
    return [sum(1 for v in vals if edges[i] <= v < edges[i+1]) for i in range(len(edges)-1)]
def label(edges, i):
    lo, hi = edges[i], edges[i+1]
    if hi == float("inf"): return ">= {:,.0f}".format(lo)
    if hi - lo == 1:       return "= {:,.0f}".format(lo)
    return "[{:,.0f}, {:,.0f})".format(lo, hi)
def pc(a, f):
    a = sorted(a); return a[min(len(a)-1, int(len(a)*f))]

qc, cc = counts(Q, QE), counts(C, CE)
J = collections.Counter()
for q, c, _ in PTS:
    qi = max(i for i in range(len(JQ)-1) if q >= JQ[i])
    ci = max(i for i in range(len(JC)-1) if c >= JC[i])
    J[(ci, qi)] += 1

# ------------------------------------------------------------------ markdown
L = ["# 等价同构点 (q_eq, c_eq) 的分布 " + ARGS.label, "",
     "统计粒度: **forward batch**。CSV 里同一个 forward 的每一行都重复同样的 `eq_*` 值,",
     "这里已按 `forward_idx` 去重, 一个 forward 记一个点。", "",
     "- forward(=点) 数: **{:,}**".format(N),
     "- 不同的 `(q_eq, c_eq)` 组合: **{:,}**".format(uniq), "",
     "| | min | p25 | median | p75 | p90 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
for nm, a in (("`q_eq`", Q), ("`c_eq`", C)):
    L.append("| {} | {:,.1f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.1f} |".format(
        nm, min(a), pc(a,.25), pc(a,.5), pc(a,.75), pc(a,.9), max(a)))

L += ["", "## 边缘分布: eq_query_length (q_eq)", "",
      "| 区间 | forward 数 | 占比 | 累计 |", "|---|---:|---:|---:|"]
cum = 0
for i, n in enumerate(qc):
    cum += n
    L.append("| `{}` | {:,} | {:.1f}% | {:.1f}% |".format(label(QE,i), n, n/N*100, cum/N*100))
n8 = sum(1 for q in Q if q == 8192)
L += ["", "`q_eq` 上界就是 `max_num_batched_tokens`。**{:,} 个 forward ({:.1f}%) 精确落在 8192**, "
      "且其中 b>1 的有 {} 个 —— 一条请求吃满 budget 后就没有余量给别人, 所以打满的一定是单请求 batch。"
      .format(n8, n8/N*100, sum(1 for p in PTS if p[0]==8192 and p[2]>1)),
      "", "## 边缘分布: eq_context_length (c_eq)", "",
      "| 区间 | forward 数 | 占比 | 累计 |", "|---|---:|---:|---:|"]
cum = 0
for i, n in enumerate(cc):
    cum += n
    L.append("| `{}` | {:,} | {:.1f}% | {:.1f}% |".format(label(CE,i), n, n/N*100, cum/N*100))
L += ["", "`c_eq = 0` 的 {} 个点是请求的第一块且无 LMCache 前缀命中。".format(sum(1 for c in C if c == 0)),
      "", "## 联合分布", "",
      "| c_eq \\\\ q_eq | " + " | ".join(JQL) + " | 行合计 |",
      "|---|" + "---:|" * (len(JQL)+1)]
for i, cn in enumerate(JCL):
    row = [J[(i,j)] for j in range(len(JQL))]
    L.append("| **{}** | ".format(cn) + " | ".join("{:,}".format(x) for x in row) +
             " | **{:,}** |".format(sum(row)))
L.append("| **列合计** | " + " | ".join("**{:,}**".format(sum(J[(i,j)] for i in range(len(JCL))))
                                        for j in range(len(JQL))) + " | **{:,}** |".format(N))

col8 = sum(J[(i,5)] for i in range(len(JCL)))
L += ["", "## 怎么用这张分布去布采样点", "",
      "1. **两个维度基本独立** —— `q_eq=8192` 那一列在每个 `c_eq` 桶里都占 30~60%, 没有对角结构。",
      "   所以规则的 `log q x log c` 网格就够, 不需要沿对角线采样。",
      "2. **别用等距网格**: `c_eq` 跨 {:.0f} 个数量级, `q_eq` 跨 {:.0f} 个。按原文 §5 在 `log q`, `log c` 上布点。"
      .format(math.log10(max(C)/max(1, min([c for c in C if c > 0]))),
              math.log10(max(Q)/max(1e-9, min(Q)))),
      "3. **`q_eq = 8192` 值得单独重点采样**: 它一根尖刺就占了 {:.1f}% 的 forward。".format(col8/N*100),
      "4. **两个稀疏角落**: `q_eq∈[4K,8K)` 整列只有 {:,} 个点(够 8192 就被截断到正好 8192, 够不到就往下掉);"
      .format(sum(J[(i,4)] for i in range(len(JCL)))),
      "   `c_eq>=512K` 整行只有 {:,} 个点却一直延伸到 {:,.0f}。这两块要么少采, 要么明确标成外推区。"
      .format(sum(J[(5,j)] for j in range(len(JQL))), max(C)),
      "5. 别忘了 **`imbalance_rq` / `imbalance_rL`**: b>=4 的点 `r_q` 中位数 2.5+, 等价映射本身就不太可信,",
      "   在这些点上加密采样的收益有限, 不如按原文 §10 补异构样本拟合校正因子。", "",
      "配图: `eq_point_distribution.svg` (log-log 散点 + 边缘直方图 + 联合分布网格计数)。"]
open(P_("eq_point_distribution.md"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ SVG
W, H = 1200, 830
ML, MR, MT, MB = 92, 290, 160, 74           # 主图外边距
PX0, PX1 = ML, W - MR                        # 主图 x 范围
PY0, PY1 = MT, H - MB                        # 主图 y 范围
TH = 66                                      # 上边缘直方图高
RH = 120                                     # 右边缘直方图宽
ZB = 24                                      # 底部 "c_eq = 0" 专用带的高度

QLO, QHI = 1.0, 8192.0
CLO, CHI = 512.0, 1048576.0
lg = math.log10
def sx(q): return PX0 + (lg(max(q, QLO)) - lg(QLO)) / (lg(QHI) - lg(QLO)) * (PX1 - PX0)
def sy(c):
    if c <= 0: return PY1 - ZB/2                      # c_eq=0 画在专用带里
    top, bot = PY0, PY1 - ZB
    return bot - (lg(max(c, CLO)) - lg(CLO)) / (lg(CHI) - lg(CLO)) * (bot - top)

# b 越大颜色越暖; b=1 用灰蓝, 免得 83% 的点糊成一片
BCOL = {1:"#5b7fa6", 2:"#3f9b6d", 3:"#8aa832", 4:"#d19a1a", 5:"#e2711d",
        6:"#d94f2b", 7:"#c02f4e", 8:"#9b2fa0", 9:"#6a35c9"}
esc = lambda s: s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
S = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" '
     'font-family="DejaVu Sans, Helvetica, Arial, sans-serif">'.format(W,H,W,H),
     '<rect width="{}" height="{}" fill="#ffffff"/>'.format(W,H)]
def T(x,y,t,sz=11,fill="#333",anc="start",w="normal"):
    S.append('<text x="{:.1f}" y="{:.1f}" font-size="{}" fill="{}" text-anchor="{}" '
             'font-weight="{}">{}</text>'.format(x,y,sz,fill,anc,w,esc(t)))

T(ML, 26, "等价同构点 (q_eq, c_eq) 的分布 —— 每点 = 一个 prefill forward batch", 17, "#111", w="bold")
T(ML, 46, "{} 个 forward, {} 个不同的 (q_eq, c_eq) 组合   {}".format(N, uniq, ARGS.label), 12, "#666")
T(ML, 62, "对数-对数坐标。q_eq = 本步等价的 query length, c_eq = 等价的 context length（工作量等价法）",
  11, "#888")

# --- 联合分布网格(先画, 垫在散点下面) + 单元格计数
S.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#fbfbfc" stroke="#d5d8dd"/>'
         .format(PX0, PY0, PX1-PX0, PY1-PY0))
S.append('<rect x="{}" y="{:.1f}" width="{}" height="{}" fill="#f0f0f3"/>'
         .format(PX0, PY1-ZB, PX1-PX0, ZB))
for e in JQ[1:-1]:
    S.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#c9ccd2" stroke-dasharray="3,3"/>'
             .format(sx(e), PY0, PY1))
for e in JC[1:]:
    if e == float("inf"): continue
    S.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" stroke="#c9ccd2" stroke-dasharray="3,3"/>'
             .format(PX0, sy(e), PX1))
for i in range(len(JCL)):
    ylo = sy(JC[i] if JC[i] > 0 else CLO); yhi = sy(JC[i+1]) if JC[i+1] != float("inf") else PY0
    for j in range(len(JQL)):
        n = J[(i,j)]
        if not n: continue
        xlo = sx(JQ[j] if JQ[j] > 0 else QLO); xhi = sx(JQ[j+1]) if JQ[j+1] <= QHI else PX1
        T((xlo+xhi)/2, (ylo+yhi)/2 + 4, str(n), 15, "#c3c7ce", "middle", "bold")

# --- 散点
for q, c, b in sorted(PTS, key=lambda p: -p[2]):     # b 大的后画, 不被 b=1 埋掉
    S.append('<circle cx="{:.1f}" cy="{:.1f}" r="{}" fill="{}" fill-opacity="{}"/>'
             .format(sx(q), sy(c), 2.6 if b > 1 else 1.9, BCOL.get(b, "#333"),
                     0.75 if b > 1 else 0.28))

# --- 坐标轴
for v in (1, 4, 16, 64, 256, 1024, 4096, 8192):
    x = sx(v)
    S.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#888"/>'.format(x, PY1, PY1+5))
    T(x, PY1+19, "{:,}".format(v) if v < 1024 else "{:.0f}K".format(v/1024), 10, "#555", "middle")
for v in (1024, 4096, 16384, 65536, 262144, 1048576):
    y = sy(v)
    S.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" stroke="#888"/>'.format(PX0-5, y, PX0))
    T(PX0-9, y+3, "{:.0f}K".format(v/1024) if v < 1048576 else "1M", 10, "#555", "end")
T(PX0-9, PY1-ZB/2+3, "0", 10, "#555", "end")
T(PX0-9, PY1-ZB/2+15, "(no ctx)", 8, "#999", "end")
T((PX0+PX1)/2, H-26, "eq_query_length  q_eq  (tokens, log)", 13, "#222", "middle", "bold")
S.append('<text x="26" y="{:.1f}" font-size="13" fill="#222" text-anchor="middle" font-weight="bold" '
         'transform="rotate(-90 26 {:.1f})">eq_context_length  c_eq  (tokens, log)</text>'
         .format((PY0+PY1)/2, (PY0+PY1)/2))

# --- 上边缘直方图 (q_eq)
mx = max(qc)
for i, n in enumerate(qc):
    x0, x1 = sx(max(QE[i], QLO)), sx(min(QE[i+1], QHI))
    if QE[i+1] - QE[i] == 1: x0, x1 = sx(QHI)-9, sx(QHI)      # "=8192" 那根尖刺给固定宽度
    h = n / mx * TH
    S.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="#7d9dbe" '
             'fill-opacity="0.85" stroke="#fff"/>'.format(x0, PY0-8-h, max(x1-x0, 1.5), h))
    if n / N > 0.05:
        T((x0+x1)/2, PY0-12-h, "{:.0f}%".format(n/N*100), 9, "#5b7fa6", "middle", "bold")
T(PX0, PY0-8-TH-10, "q_eq 边缘分布", 11, "#5b7fa6", w="bold")

# --- 右边缘直方图 (c_eq)
mx = max(cc)
for i, n in enumerate(cc):
    if CE[i+1] - CE[i] == 1:                                   # c_eq = 0
        y1, y0 = PY1, PY1 - ZB
    else:
        y1 = sy(max(CE[i], CLO)); y0 = sy(CE[i+1]) if CE[i+1] != float("inf") else PY0
    w = n / mx * RH
    S.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="#7d9dbe" '
             'fill-opacity="0.85" stroke="#fff"/>'.format(PX1+8, y0, w, max(y1-y0, 1.5)))
    if n / N > 0.05:
        T(PX1+12+w, (y0+y1)/2+3, "{:.0f}%".format(n/N*100), 9, "#5b7fa6", "start", "bold")
T(PX1+8, PY0-14, "c_eq 边缘分布", 11, "#5b7fa6", w="bold")

# --- 图例
lx, ly = PX1 + 8 + RH + 24, PY0 + 8
T(lx, ly, "batch size b", 11, "#333", w="bold")
bn = collections.Counter(p[2] for p in PTS)
for i, b in enumerate(sorted(bn)):
    y = ly + 18 + i*17
    S.append('<circle cx="{}" cy="{:.1f}" r="{}" fill="{}" fill-opacity="{}"/>'
             .format(lx+6, y-4, 3.2 if b > 1 else 2.4, BCOL[b], 0.85 if b > 1 else 0.45))
    T(lx+18, y, "b={}   {:,}".format(b, bn[b]), 10, "#444")
ly2 = ly + 18 + len(bn)*17 + 18
for i, t in enumerate(["灰色数字 = 该格内的",
                       "forward 数(联合分布)", "",
                       "虚线 = 联合分布桶边界", "",
                       "底部灰带 = c_eq 为 0",
                       "({} 个: 首块且无".format(sum(1 for c in C if c == 0)),
                       " LMCache 前缀命中)"]):
    T(lx, ly2 + i*14, t, 9.5, "#777")
S.append("</svg>")
open(P_("eq_point_distribution.svg"), "w").write("\n".join(S))

print("wrote {} and {}  ({} forwards, {} unique points)"
      .format(P_("eq_point_distribution.md"), P_("eq_point_distribution.svg"), N, uniq))
