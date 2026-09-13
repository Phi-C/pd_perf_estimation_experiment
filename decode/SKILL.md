---
name: pd-decode-trace
description: 从 ATOM StepMembers 日志重放 Decode batch，生成等价同构测点及异构校准负载，并用实测时间表估计给定 batch 分布下的吞吐。适用于 Decode 形状重建、性能采样和吞吐上限分析。
---

# Decode 日志到吞吐估计

从仓库根目录运行。主流程为：完整重放成员增量 → 标记 profiling → 展开逐步形状 → 选同构点及校准组 → 测量 → 逐步累加时间。

公式见[工作量等价法](ragged_packed_decode_workload_equivalence.md)，上机操作见[测点方法](measure_decode_performance_point.md)。

## 1. 确认日志能力

完整流程需要 `ATOM_LOG_STEP_MEMBERSHIP=1` 产生的两类日志：

| 日志 | 信息 |
|---|---|
| `[StepMembers] step=S decode n=N tokens=T +[...] -[...]` | 成员增量和 batch 自检数 |
| `[PD-TRANSITION] seq R: num_tokens=ISL, ...` | 请求身份、初始输入长度 |

`StepMembers` 只在成员变化时打印；两行之间保留上一集合。`step` 是 scheduler tick，可能包含未输出 token 的轮次，因此展开得到的是重放估计，需和 AIPerf completion tokens 对账。

只有 `[PD-DECODE]` 的旧日志通常只打印每请求前 3 步，应使用[旧日志诊断流程](legacy_log_trace.md)，无法据此估计稳态 batch 吞吐。

## 2. 完整解析与阶段标记

```bash
DECODE_LOG=/path/to/decode.log
python3 scripts/parse_decode_stepmembers.py "$DECODE_LOG" \
  -o out/decode --page 1024 --label '目标用例'
python3 scripts/gen_decode_eq_batches.py -d out/decode \
  --since 04:42:08 --l-bin 8192
```

`04:42:08` 是 K3 本地重跑的 profiling 示例边界，应从对应驱动日志的 phase marker 获取。先解析完整日志，后筛选阶段，才能保留 warmup 期间已累积的 KV 长度。当前阶段判定继承区间的秒级时间；跨边界区间及跨日数据需额外处理。

| 输出 | 行粒度 |
|---|---|
| `decode_steps.csv` | 一个成员稳定区间，权重为 `span_steps` |
| `decode_members.csv` | 一个区间中的一条请求，保存起点和中点 KV 长度 |
| `decode_requests.csv` | 一个请求及其重放步数 |
| `decode_eq_batches.csv` | 一个重放 step，包含 `L_eq, L_test, b, phase` 与不均衡特征 |
| `decode_stat.md` | 完整日志统计，包含 warmup |

区间内成员不变、每步每条请求增加 1 token，因而可按该重放假设精确展开长度。`gen_decode_eq_batches.py` 的 `--since` **标记 phase，不删除 warmup 行**；后续统计必须筛选 `phase == profiling`。

`L_test` 是 `L_eq` 就近吸附到 `l-bin` 的坐标，至少为一个 bin。低长度点的吸附误差可能较大，应单列检查。`L_eq` 保留原始映射结果。

## 3. 选点与异构校准

逐步定价使用同一 `L_test` 网格选点：

```bash
python3 scripts/gen_decode_top_points.py \
  -i out/decode/decode_eq_batches.csv \
  -o out/decode/decode_points_top150.csv -n 150 --phase profiling
python3 scripts/gen_decode_calib_workloads.py \
  -d out/decode -n 16 --strata 5
```

主表按 `(L_test, b)` 聚合，默认用 output token 数排序；校准组在不同不均衡程度中分层选点，包含同构基准与三类异构形状。

`eq_confidence` 由 `r_max` 阈值生成，只是选校准样本的启发式。报告它时说明按 step 还是 token 加权，以及是否包含 warmup。大量 low 样本应检查校准误差；没有校准时可报告 mean / max-length 两个敏感性场景。

另一路 [gen_decode_points.py](../scripts/gen_decode_points.py) 按稳定区间分格，再吸附代表点。它可用于探索选点策略，但与逐步 `L_test` 的点集不同，不能直接混用。

## 4. 测量与吞吐回代

按[测点方法](measure_decode_performance_point.md)生成并运行 trace。主表给同构单价，校准组验证真实 batch 能否用该单价代表，两者分别保存。

```bash
python3 scripts/decode_points_table.py \
  -r out/decode/bench/points_results.jsonl \
  -p out/decode/decode_points_top150.csv \
  -o out/decode/decode_points_measured
python3 scripts/fit_decode_imbalance.py \
  -r out/decode/bench/calib_results.jsonl \
  -p out/decode/decode_calib_points.csv \
  -o out/decode/decode_imbalance_fit.md
python3 scripts/decode_throughput_ceiling.py \
  --results out/decode/bench/points_results.jsonl \
  --eq out/decode/decode_eq_batches.csv \
  --aiperf-tokens 2206436 --window-s 3684
```

最后一条命令的窗口和 token 数是 K3 示例值，必须替换为目标运行的数据。脚本对命中点使用实测 `fwd_gpu_ms` / `D_ms`，对未命中点使用 K3 的 `ceil(b / 16)` 波均值模型。

**波宽 16 是本实验的经验模型。** 换硬件、模型或 kernel 后，先验证台阶边界与波内变化，再使用该补值规则。若校准发现非零异构修正，也要先把修正应用到逐步定价，现有吞吐脚本不会自动读取拟合系数。

## 5. 检查结果与限制

- 解析输出 `n_mismatch == 0`、`missing_isl == 0`；非零时停止后续定价并检查日志。
- 实际记录满足 `tokens_reported == b`；MTP 或混合步需要不同模型。
- 展开后的 step 不重复，`Σ b` 与 `Σ(b × span_steps)` 相等，`b × L_eq ≈ L_sum`。
- 对 `join_events > 1` 的请求核查抢占 / 重入，不把它们当作连续稳定 decode。
- 与 AIPerf 对账重放 token 数，保留差额；请求级 OSL 不用成员 CSV 行数代替。
- 测量结果按 tag 去重，保留最新有效结果与失败记录；报告命中点数、覆盖率、补值比例。
- 从窗口减去预测 step 总时间得到的是残差，没有同次运行的空队列证据时不称为空转实测。

Decode 与 Prefill 使用各自的 request ID。ISL 多重集可用于检查总体一致性；长度重复时，仅凭 ISL 也无法唯一匹配跨节点请求。
