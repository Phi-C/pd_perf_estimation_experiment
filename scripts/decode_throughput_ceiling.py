#!/usr/bin/env python3
"""给定 Decode batch 分布下的纯模型与引擎吞吐估计。

给 profiling 段每个 forward 按实测点表查 `fwd_gpu_ms` 定价，
Σ 得到纯 forward 总时间，再用 token 数去除。decode 侧完全同构，只是
    单价键   (final_context_length, final_query_length, eq_batch_size)  ->  (L_test, b)
    分子     ISL / fresh token                                          ->  output token = Σb

未命中点采用 K3 实验的 ceil(b/16) 波均值；换配置后须重新验证。
结果文件虽使用历史 .csv 后缀，实际内容是逐行 JSON。
"""
import csv, json, collections, statistics as st, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="decode/bench_run/traces_top150_results.csv")
ap.add_argument("--eq", default="decode/decode_eq_batches.csv")
ap.add_argument("--aiperf-tokens", type=int, default=2_206_436)
ap.add_argument("--window-s", type=float, default=3684.0)
a = ap.parse_args()

# ── 实测点表 (L_test, b) -> (fwd_gpu_ms, D_ms)
pts = {}
for l in open(a.results):
    r = json.loads(l)
    if r.get("accept") != "OK":
        continue
    t = r["tag"]
    L = int(t.split("_L")[1].split("_b")[0]); b = int(t.split("_b")[1])
    pts[(L, b)] = (r["fwd_gpu_ms_p50"], r["D_ms"])

# ── 波模型单价（未命中形状的外推依据：D ≈ f(ceil(b/16))）
wf, wd = collections.defaultdict(list), collections.defaultdict(list)
for (L, b), (f, d) in pts.items():
    w = -(-b // 16); wf[w].append(f); wd[w].append(d)
WF = {k: st.mean(v) for k, v in wf.items()}
WD = {k: st.mean(v) for k, v in wd.items()}

fw = [r for r in csv.DictReader(open(a.eq)) if r["phase"] == "profiling"]
tok_all = sum(int(r["b"]) for r in fw)

sub_f = sub_d = sub_tok = 0.0   # 子集口径：只累加命中的
all_f = all_d = 0.0             # 全覆盖：未命中用波模型
n_hit = 0
for r in fw:
    L, b = int(r["L_test"]), int(r["b"])
    if (L, b) in pts:
        f, d = pts[(L, b)]; n_hit += 1; sub_f += f; sub_d += d; sub_tok += b
    else:
        w = -(-b // 16); f, d = WF[w], WD[w]
    all_f += f; all_d += d

print(f"实测 OK 点 {len(pts)}   profiling forward {len(fw):,}   Σoutput token {tok_all:,}")
print(f"命中 {n_hit:,} ({n_hit/len(fw)*100:.1f}%)  token 覆盖 {sub_tok/tok_all*100:.1f}%")
print()
print(f"#1 实际交付   aiperf {a.aiperf_tokens:,} / {a.window_s:.0f}s = {a.aiperf_tokens/a.window_s:,.0f} tok/s")
print(f"              重放   {tok_all:,} / {a.window_s:.0f}s = {tok_all/a.window_s:,.0f} tok/s")
print(f"#3 子集口径   fwd_gpu Σ={sub_f/1000:,.1f}s -> {sub_tok/(sub_f/1000):,.0f} tok/s"
      f"   |  D_ms Σ={sub_d/1000:,.1f}s -> {sub_tok/(sub_d/1000):,.0f} tok/s")
print(f"#3 全段估计   fwd_gpu Σ={all_f/1000:,.1f}s -> {tok_all/(all_f/1000):,.0f} tok/s"
      f"   |  D_ms Σ={all_d/1000:,.1f}s -> {tok_all/(all_d/1000):,.0f} tok/s")
print()
print(f"纯模型估计 / 窗口吞吐(同为重放口径) = {(tok_all/(all_f/1000))/(tok_all/a.window_s):.3f}x")
print(f"周期模型 − GPU 模型 = {(all_d-all_f)/1000:.1f}s; "
      f"窗口 − 周期模型 = {a.window_s-all_d/1000:.1f}s (残差，非空转实测)")
