---
name: pd-prefill-trace
description: 从 ATOM Prefill 日志提取 forward 形状，筛选 profiling 阶段，生成同构测点，并结合 GPU 探针表估计吞吐。适用于 Prefill trace 分析、等价点选取和测量结果回代。
---

# Prefill 日志到吞吐估计

从仓库根目录运行以下命令。流程为：解析完整日志 → 选择 profiling 段 → 按 forward 选点 → 实测 → 查表累加时间。公式见[工作量等价法](ragged_packed_prefill_workload_equivalence.md)，测量细节见[测点方法](measure_prefill_performance_point.md)。

## 1. 解析完整日志

```bash
PREFILL_LOG=/path/to/prefill.log
python3 scripts/parse_prefill_log.py "$PREFILL_LOG" \
  -o out/prefill/all --label '目标用例'
```

输出 `prefill_forwards.csv`、`gap_events.json`、`idle_gaps.json` 和 `prefill_stat.md`。

**CSV 一行是一个 `(forward, request)` 对**。按 forward 统计时先用 `forward_idx` 去重；统计请求总 ISL 时按解析后的请求实例去重。

| 字段 | 用途 |
|---|---|
| `forward_idx, time` | forward 编号、引擎秒级时间 |
| `req_id, ISL` | 请求实例、完整输入长度 |
| `done_tokens, chunk_tokens` | 本步的 c、q |
| `batch_reqs, batch_tokens` | 整个 batch 的规模，同一 forward 的多行重复 |
| `eq_batch_size, eq_query_length, eq_context_length` | 工作量映射后的同构坐标 |
| `attn_work_pairs, imbalance_rq, imbalance_rL` | 工作量及不均衡特征 |

解析器处理日志重放、原始 request ID 复用和 arrival 匹配。换引擎版本时，检查文件顶部的 `RE_SCHED`、`RE_ARR`，并核对解析器输出的去重数与匹配数。

## 2. 选择 profiling 段

从驱动端日志读取 `Credit phase start: profiling`。完整解析之后再筛选，避免丢失请求身份及前序状态。

```bash
rg -n -a 'Credit phase start|Initialized [0-9]+ phase' "$PREFILL_LOG"
python3 scripts/split_prefill_phase.py \
  out/prefill/all/prefill_forwards.csv --since 12:43:45 \
  -o out/prefill/profiling/prefill_forwards.csv
```

`12:43:45` 是 K3 CI 示例的边界，换用例必须替换。脚本以 `HH:MM:SS` 筛选同一自然日的数据；跨日日志应先按日期拆分。

用 AIPerf 的请求数、prompt / fresh token 和 cache hit 核对阶段。边界请求可能跨阶段，应报告差额和处理规则，不能靠调切点强行对齐总数。

## 3. 生成 trace 并选择测点

需要时间排序时，先从完整解析结果生成 Perfetto 文件，再让 profiling 点表使用同一套 forward ID 的时间估计：

```bash
python3 scripts/gen_prefill_trace.py -d out/prefill/all \
  --phase-split 1374.97 --label '目标用例'
cp out/prefill/all/prefill_forwards_trace_full.json.gz \
  out/prefill/profiling/
python3 scripts/gen_final_points.py -d out/prefill/profiling \
  --c-bin 8192 --q-bin 1024 --rank-by time
```

`1374.97` 是示例中 profiling marker 相对首个引擎 forward 的秒数。Perfetto 文件可在 [ui.perfetto.dev](https://ui.perfetto.dev) 打开。只按次数选点时，可直接用 `--rank-by count`，不依赖 trace。

选点脚本对每个 `(c, q)` 格子取中位数、吸附到最近网格角点，再按 `(final_c, final_q, b)` 去重。默认不吸附 `q < q-bin` 的格子，避免把短尾 chunk 拉大或吸到 0。

| 输出 | 含义 |
|---|---|
| `prefill_forwards.csv` | 原地追加 / 更新 `final_context_length, final_query_length` |
| `final_grid_cells.csv` | 格子、代表点及出现过的 b |
| `final_points_unique.csv` | 按优先级排列的 `(c, q, b)` 测点与累计覆盖率 |

`time` 使用 forward slice 的估算时长；`count` 使用次数；`work` 使用 attention 工作量。三种排序对应不同覆盖口径。秒内均摊时间只适合粗粒度分析和选点，不能当作单次 GPU 测时。

可选的分布图与旧测点对照：

```bash
python3 scripts/plot_eq_distribution.py -d out/prefill/profiling
python3 scripts/diff_points.py -d out/prefill/profiling \
  --measured bench_run/RESULTS_223_points.csv --cover 80
```

`diff_points.py` 的 `interp` 仅表示落在同 b 的测点矩形包络内，不保证误差；优先检查未覆盖点的工作量占比及其实际预测误差。

## 4. 测量并回代

按[测点方法](measure_prefill_performance_point.md)运行目标 batch，用[forward 探针](../docs/forward_probe.md)得到时间表，再计算：

```bash
python3 scripts/throughput_from_probe.py \
  --forwards out/prefill/profiling/prefill_forwards.csv \
  --probe out/prefill/bench/RESULTS_probe_vs_bench.csv \
  --measured bench_run/RESULTS_223_points.csv
```

匹配键为 `(final_c, final_q, b)`，token 数仍取原始 batch。每条请求的完整 ISL 归到它最后一个 forward。默认报告命中子集、覆盖率及与 TTFT 差分表的对照；额外模型外推需显式启用并单独解释。

## 5. 检查结果

- 请求 `done + chunk` 进度与 arrival ISL 一致；有截断或缺行时列出数量。
- Forward 数与 `unique forward_idx` 一致；`b × q_eq` 与原始 `batch_tokens` 在 CSV 舍入误差内相等。
- 测点 `q > 0`，实际 batch 满足服务端 token budget；超过预算的吸附点不能通过更改服务配置冒充原工况。
- 时间排序的 trace 与 CSV 来自同一日志、同一 ID 序列，没有缺失映射。
- GPU 时间表与 TTFT 差分表明确区分，子集分子与分母范围一致。

`idle_gaps.json` 记录队列耗尽的证据；Perfetto 中的 idle 时长还使用同类 step 的估算值。它不能证明精确的 GPU 空闲时长，更不能仅凭相邻秒级调度时间归因某个 kernel 或通信事件。
