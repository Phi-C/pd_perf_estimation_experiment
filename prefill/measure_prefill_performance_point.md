# Prefill 同构测点：构造、测量与验收

目标是测定给定 `(c, q, b)` 的 forward 时间：每条请求已有 c 个 KV token，本步新算 q 个 token，共 b 条请求。AIPerf 用于构造负载和记录客户端指标，纯模型时间来自[引擎探针](../docs/forward_probe.md)。

## 1. 固定测量配置

沿用目标运行的模型、硬件、TP / DCP、KV dtype、prefix cache、checkpoint 和调度配置。K3 示例为：

| 参数 | 值 |
|---|---:|
| TP / DCP | 8 / 8 |
| `max_num_batched_tokens` | 8192 |
| `max_num_seqs` | 64 |
| `block_size` | 128 |
| `hash_block_size = block_size × DCP` | 1024 |

在这个预算下，目标 batch 必须满足 `b × q ≤ 8192`。分桶可能产生超过预算的代表点，应重新选点或标记缺失；调整预算后测得的是另一种配置。

## 2. 用前缀和 checkpoint 构造 c、q

c 由可命中的缓存前缀提供，q 由 fresh input 提供。K3 的前缀命中按 1024-token hash block 对齐；这与 `LMCACHE_CHUNK_SIZE` 数值相同，但来源不同。

当前 state checkpoint 配置下，设服务端实际 prompt 长度为 P：

$$
\mathrm{anchor}=\left\lfloor\frac{P-1}{1024}\right\rfloor\times1024
$$

| 目标 q | 构造方式及边界 |
|---|---|
| `q < 1024` | 目标 `P = c + q`；无额外 anchor 截断，实际 q 仍有合成长度抖动 |
| `q ≥ 1024` 且对齐 1024 | 取 `P = c + q + δ`，使 anchor 落在 `c + q`；现有 harness 使用 `δ = 2` |
| `q ≥ 1024` 且未对齐 | 当前 checkpoint 规则可能拆成多个 forward，不能直接当单步目标 |

`δ` 会额外产生一个尾部 forward。只有纯模型探针按实际形状筛选时，才能直接排除这一步；TTFT 包含目标块、尾部及其它服务开销。

测量前先用短 primer 在 c 处留下 checkpoint。现有 harness 的 primer 目标 ISL 为 `c + 130`。这是 K3 特定规则，换模型需重新验证。

AIPerf 的 synthetic token 长度不一定等于服务端实际长度。K3 示例的模板补偿为 synthetic 路径 88、Mooncake trace 路径 89 token；以实际 tokenizer / chat template 的测量为准。重跑应改变 nonce / seed，防止 fresh 部分命中上轮缓存。

## 3. 构造实际 batch

`--concurrency b` 仅限制客户端并发，不能保证 Prefill 请求同一步进入。当前 K3 harness 分两种：

| 脚本 | 构造与输出 |
|---|---|
| [run_stage_b1.sh](../bench_run/run_stage_b1.sh) | b=1，primer 后测量，记录 TTFT 与实际 q |
| [run_stage_bn.sh](../bench_run/run_stage_bn.sh) | b≥2，fixed schedule 排队后核对目标 batch |
| [gen_b2_trace.py](../bench_run/gen_b2_trace.py) | 生成 primer、一个长占位请求和七个 blocker，再投递 b 条目标请求 |
| [analyze_bn_fs.py](../bench_run/analyze_bn_fs.py) | 对照引擎调度日志和 AIPerf 记录，报告目标 batch 数与 TTFT 差分 |

服务端和 AIPerf client 容器需事先启动，并以相同绝对路径挂载仓库、trace 和输出目录。测纯模型时间时，在服务端启用探针。

```bash
mkdir -p out/prefill/bench
export MODEL=/mnt/models/Kimi-K3
export URL=http://127.0.0.1:8010
export CLIENT=k3-aiperf-client
export SERVER=k3-prefill-bench

bash bench_run/run_stage_b1.sh 217 \
  "$PWD/out/prefill/profiling/final_points_unique.csv" \
  "$PWD/out/prefill/bench/aiperf_b1" "$PWD/out/prefill/bench/b1.csv"

SRC="$PWD/out/prefill/profiling/final_points_unique.csv" \
ART="$PWD/out/prefill/bench/aiperf_bn" \
CSV="$PWD/out/prefill/bench/bn.csv" \
TR="$PWD/out/prefill/bench/traces" \
bash bench_run/run_stage_bn.sh 217 8
```

`217` 是示例的优先级截止值，应按新点表的覆盖率选择。脚本会按已有结果跳过部分点；换点表或配置时使用新的输出目录，避免 priority 重号。

## 4. 验收与时间口径

以引擎记录为准，检查目标窗口中 `(done_tokens, chunk_tokens, batch_reqs)` 是否落到目标 `(c, q, b)`；排除 cold primer、混入的其它请求及尾部块。q<1024 的现有验收容差为 ±16 token，并保存 `q_actual`。

两类时间分别保留：

- `fwd_gpu_ms`：探针的 `run_model` GPU event 时间，供纯模型回代。
- TTFT 及扣排队后的 `ttft_net`：客户端 / 调度边界的诊断量。历史 `target_fwd_gpu_ms` 还扣了客户端常数和尾部模型值，属于差分估计。

历史 blocker 对照曾出现约 96 ms 的偏置，不能作为跨配置通用常数。共享前缀同样需要保留限制：当前 b≥2 Prefill harness 的目标请求共用 primer 前缀，而真实 batch 可能持有独立 KV；应通过异构 / 独立 KV 对照验证，不能默认影响为零。

## 5. 生成点表并回代

[K3 的参考表](../experiments/exp1_kimi_k3/Reproduce.md)为 `RESULTS_223_points.csv`，包含测量坐标、实际 q 和旧 TTFT 差分字段。用 [join_prefill_probe.py](../scripts/join_prefill_probe.py)将探针桶与该参考表连接，再由 [throughput_from_probe.py](../scripts/throughput_from_probe.py)回代 profiling 形状。

新实验应准备相同 schema 的参考表，并重新核对匹配覆盖率。按形状聚合得到的 Prefill `period_ms` 混入 primer / blocker 的间隔，只保留作诊断，不作为同形状稳定吞吐的分母。
