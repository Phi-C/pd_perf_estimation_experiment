# 脚本索引

从仓库根目录运行脚本。完整流程见 [Prefill](../prefill/SKILL.md)、[Decode](../decode/SKILL.md)；使用本地数据复算见 [K3 复现说明](../experiments/exp1_kimi_k3/Reproduce.md)。多数 Python 脚本支持 `--help`。

## 1. 依赖与输入约定

| 场景 | 依赖 |
|---|---|
| 离线解析、分箱、查表、SVG 绘图 | Python 3.10+；大部分脚本只用标准库 |
| Decode 逐步展开、校准生成、探针分析、拟合 | 另需 NumPy：`python3 -m pip install numpy` |
| 上机测量 | Bash、Docker、已启动的 ATOM 服务端和 AIPerf client；Prefill runner 另用 `bc` |
| 引擎内探针 | 目标引擎的 PyTorch，以及与 runner 匹配的挂点 |

日志、模型、测量 CSV / JSONL 和 AIPerf artifact 是外部输入，不随源码分发。脚本中的默认路径对应 K3 本地目录；换实验时显式指定输入、输出、模型和容器。SVG 脚本直接生成文件，不依赖 Matplotlib。

## 2. Prefill：日志到吞吐

| 脚本 | 输入 → 输出 | 关键约定 |
|---|---|---|
| [parse_prefill_log.py](parse_prefill_log.py) | 引擎日志 → `prefill_forwards.csv`、事件 JSON、统计 Markdown | 一行是 `(forward, request)`；时间戳只有秒级精度 |
| [split_prefill_phase.py](split_prefill_phase.py) | 完整 CSV → 阶段 CSV | `--since` 含起点，`--until` 不含终点；只处理同一自然日 |
| [gen_prefill_trace.py](gen_prefill_trace.py) | CSV + 事件 JSON → Perfetto JSON / gzip | slice 时长由日志估算，用于观察顺序和粗粒度选点 |
| [gen_final_points.py](gen_final_points.py) | 带 `eq_*` 的 CSV → 格子表、唯一测点表 | 原地更新 CSV 的 `final_*`，删除旧 `grid_*`；`--rank-by time` 需要同 ID 的 trace |
| [diff_points.py](diff_points.py) | 唯一测点表 + 已测表 → 待补测 / 外推点清单 | `interp` 只表示在同 b 的矩形包络内，不验证插值误差 |
| [join_prefill_probe.py](join_prefill_probe.py) | 探针 JSONL + 参考表 → 探针测时表 | 按同构 `(c,q,b)` 匹配，短 q 允许 ±16 token |
| [throughput_from_probe.py](throughput_from_probe.py) | 阶段 CSV + 探针表 + 参考表 → 吞吐与覆盖率 | 分母用 GPU event；`--extrapolate` 另报模型补缺的敏感性估计 |
| [throughput_from_measured.py](throughput_from_measured.py) | 阶段 CSV + 旧测点表 + 模型参数 → 吞吐与覆盖率 | `target_fwd_gpu_ms` 是 TTFT 差分估计；最后一段用模型补缺 |
| [plot_eq_distribution.py](plot_eq_distribution.py) | CSV → 分布 Markdown / SVG | 按 forward 去重，展示等价点分布 |
| [plot_eq_grid512.py](plot_eq_grid512.py) | CSV → 等距网格 CSV / SVG | 默认回写 `grid_*`；只画图时加 `--no-write-back` |

吞吐脚本用 `(final_context_length, final_query_length, eq_batch_size)` 查表，用原始 `batch_tokens` 统计 fresh token。完整 ISL 归到每条请求的最后一个 forward。当前实现按 `req_id` 去重，要求所选窗口内没有 ID 复用；完整日志跨阶段时应先筛选并核对请求身份。

## 3. Decode：重放、选点、测量与回代

| 脚本 | 输入 → 输出 | 关键约定 |
|---|---|---|
| [parse_decode_stepmembers.py](parse_decode_stepmembers.py) | StepMembers 日志 → `decode_steps.csv`、成员表、请求表、统计 Markdown | 一行 step 表表示成员稳定区间，权重为 `span_steps` |
| [gen_decode_eq_batches.py](gen_decode_eq_batches.py) | 成员表 → 每步 `decode_eq_batches.csv` | `--since` 标记 phase，保留 warmup；scheduler tick 重放需和客户端 token 对账 |
| [gen_decode_top_points.py](gen_decode_top_points.py) | 每步等价点 → Top-N `(L_test,b)` | 默认按 profiling 的 output token 数排序 |
| [gen_decode_calib_workloads.py](gen_decode_calib_workloads.py) | 成员 / 等价点 / Top-N 表 → 校准点表、校准请求表 | 同一均值与 b 下构造 A/B/C/D 成对形状；需 NumPy |
| [gen_decode_bench_traces.py](gen_decode_bench_traces.py) | 主测点或校准请求 → Mooncake JSONL + `index.csv` | `--mode points` / `calib`；prime 与 meas 共用同一 trace |
| [run_decode_bench.sh](run_decode_bench.sh) | trace 目录 → 每组结果 JSONL、AIPerf artifact | 用环境变量 `TR/ART/CSV/PROBE` 指定路径；已有 OK tag 跳过 |
| [analyze_decode_group.py](analyze_decode_group.py) | probe 字节区间 + artifact → 一条验收 JSON | 筛选纯 Decode、目标 b、q=1，再丢弃暖机；需 NumPy |
| [decode_points_table.py](decode_points_table.py) | 每组 JSONL + 主测点 → 性能 CSV / Markdown | 同 tag 保留最后一个 OK；没有 OK 时保留最后一次失败 |
| [fit_decode_imbalance.py](fit_decode_imbalance.py) | 校准结果 + 校准点表 → 拟合 Markdown | 默认拟合 `D_ms`；可选 GPU 时间；报告的是样本内拟合，需 NumPy |
| [decode_throughput_ceiling.py](decode_throughput_ceiling.py) | 每组 JSONL + profiling 等价点 → 子集与全段吞吐 | 缺失点按 K3 的 `ceil(b/16)` 波均值补值，换配置须验证 |
| [gen_decode_req_spans.py](gen_decode_req_spans.py) | 成员表 → 每请求的调度区间表 | 用于核对重入、区间及 token 计数 |
| [gen_decode_points.py](gen_decode_points.py) | 稳定区间表 → `decode_points_unique.csv` | 探索用的区间分箱路线，点集与逐步 `L_test` 不同 |
| [plot_eq_hist.py](plot_eq_hist.py) | 等价点 CSV → 长度分布 SVG | 读取所有输入行；只看 profiling 时先过滤 CSV |

历史 `traces_top150_results.csv`、`traces_calib_results.csv` 的内容是 **JSONL**，不能用 CSV reader 读取。`D_ms` 是引擎返回间隔，`fwd_gpu_ms_p50` 是 GPU event 时间，`itl_ms_p50` 是客户端 ITL。字段 `overhead_ms` / `gpu_busy_frac` 只是 `D − fwd` / `fwd / D`，不代表逐项开销或硬件利用率测量。

## 4. 形状统计路线

| 脚本 | 输入 → 输出 | 关键约定 |
|---|---|---|
| [prefill_grid_alpha.py](prefill_grid_alpha.py) | Prefill chunk CSV → 全部格子、入选格子与权重 | 样本为 `(forward, request)`，按记录数达到 α |
| [decode_grid_alpha.py](decode_grid_alpha.py) | 请求 CSV → `(ISL, ISL+OSL/2)` 格子与权重 | `--first-step` 选择阶段；默认 OSL 列为重放 `decode_steps` |
| [stat_price_prefill.py](stat_price_prefill.py) | 入选格 + 历史点表 → 估计时间和加权指标 | 默认 b=1、TTFT 差分口径；`c+q` 按 chunk 计，不是去重 ISL |
| [stat_price_decode.py](stat_price_decode.py) | 入选格 + 历史点表 → 波模型估计和加权指标 | 假定稳定 `b=C`；只查表，不执行固定并发实测 |

上述两个 `stat_price_*` 脚本复现早期估计。后来新增的真实测量结果由下列实验脚本处理，结果状态见 [统计实验报告](../experiments/exp1_kimi_k3/stat_alpha0.8/Report.md)。

| 实验脚本 | 用途 |
|---|---|
| [price_prefill_measured.py](../experiments/exp1_kimi_k3/stat_alpha0.8/measure/prefill/price_prefill_measured.py) | 连接 Prefill 代表点、b=1 验收表与探针，分别汇总 GPU 时间和 TTFT 派生比率 |
| [run_stat_decode.sh](../experiments/exp1_kimi_k3/stat_alpha0.8/measure/decode/run_stat_decode.sh) | 对请求级 `(I,O,C)` 点执行两遍 cache-replay；`ONLY` 按 rank 筛选 |
| [parse_stat_decode.py](../experiments/exp1_kimi_k3/stat_alpha0.8/measure/decode/parse_stat_decode.py) | 汇总客户端、ITL、引擎周期和模型时间；30 步门槛仅供短窗口诊断 |
| [aggregate_decode.py](../experiments/exp1_kimi_k3/stat_alpha0.8/measure/decode/aggregate_decode.py) | 同 tag 取最后一条结果，再筛 OK；报告计划点数、完成覆盖率与四种加权指标 |

## 5. 测量辅助脚本

| 脚本 | 用途 |
|---|---|
| [_fwdprobe.py](_fwdprobe.py) | 安装到 ATOM runner 内的探针；不是独立 CLI，挂点见 [探针说明](../docs/forward_probe.md) |
| [cache_read.py](cache_read.py) | `python3 scripts/cache_read.py ARTIFACT_DIR` 读取平均 cache-read；缺指标时返回 0 |
| [run_stage_b1.sh](../bench_run/run_stage_b1.sh) | Prefill b=1：`MAXPRIO 点表 ART目录 输出CSV` |
| [run_stage_bn.sh](../bench_run/run_stage_bn.sh) | Prefill b≥2：`TOPN PAIRS [ONLY]`；用 `SRC/ART/CSV/TR` 指定路径 |
| [gen_b2_trace.py](../bench_run/gen_b2_trace.py) | 为 b≥2 生成 primer、占位请求、七个 blocker 和目标请求；`--members` 指定 b |
| [analyze_bn_fs.py](../bench_run/analyze_bn_fs.py) | 按目标形状及请求时间邻近关系连接日志和 AIPerf，估算扣排队 TTFT |

Prefill runner 按已有 priority 跳过记录，包括失败记录。重新测量或更换点表时使用新的结果文件。Decode runner 按已有 OK tag 跳过；改变配置后也应换文件，避免沿用旧测时。服务端与 client 容器必须能访问相同绝对路径，配置与验收见 [Prefill 测点](../prefill/measure_prefill_performance_point.md)和 [Decode 测点](../decode/measure_decode_performance_point.md)。

## 6. 旧日志兼容

[parse_decode_log.py](parse_decode_log.py) 与 [gen_decode_trace.py](gen_decode_trace.py) 处理只含前几步 `[PD-DECODE]` 的旧日志。它们输出的是观测到的请求进入过程，不能用于稳态 Decode 定价。输出放到独立目录，避免覆盖 StepMembers 流程的同名请求表；见 [旧日志说明](../decode/legacy_log_trace.md)。

`*_stat.md`、`*_measured.md` 和分布图是自动生成的数据快照，更新输入后应重新生成。其它本地 sweep、重试脚本、HTML 草稿和 `archive/` 保留为过程资料。当前入口以本文和两侧操作流程为准，输出建议写入 `out/`。
