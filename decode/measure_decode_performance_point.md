# Decode 同构测点与异构校准

目标是测量 `(L, b)` 下的模型时间、稳定引擎步长与客户端 ITL。先让每条请求的 KV / 模型状态就绪，再选取 `b` 稳定、每条 query 长度为 1 的 Decode 窗口。

## 1. 测量条件

模型、硬件、并行度、KV dtype、kernel 和状态恢复路径与目标配置一致。K3 示例使用 TP8 + DCP8、`max-num-seqs = 64`、`max-num-batched-tokens = 8192`、hash block 1024，并开启 `ATOM_LOG_STEP_MEMBERSHIP=1` 和 [forward 探针](../docs/forward_probe.md)。

每条测量请求使用自己的 KV 前缀，避免同一 batch 多条请求读同一份物理页。现有生成器按 `(L, 槽位)` 分配 `hash_ids`，组内不重合；跨组复用必须以实际 cache hit 验证，不能只依据 ID 相同推断命中。

生成步数默认 `O = 300`，丢弃前 20 个可用样本。记录窗口中点的实际 L，并检查长度和步长漂移。`O / L` 较小有助于近似局部测量，但不保证 kernel 切换处的性能稳定。

## 2. 生成负载

先按 [Decode 流程](SKILL.md)得到主测点和校准请求 CSV，然后生成两批独立 trace：

```bash
python3 scripts/gen_decode_bench_traces.py \
  -i out/decode/decode_points_top150.csv --mode points \
  -o out/decode/bench/traces_points --nonce 101
python3 scripts/gen_decode_bench_traces.py \
  -i out/decode/decode_calib_requests.csv --mode calib \
  -o out/decode/bench/traces_calib --nonce 102
```

每组输出一个 JSONL trace，`index.csv` 记录执行顺序、b、各长度与输出步数。模板补偿 `TPL = 89` 和 block 1024 是 K3 当前合成路径的值，换 tokenizer / 模型时应修改并验证。

**Prime 和 meas 使用同一个 trace 文件、同一个 seed。** Mooncake 合成路径的 `output_length` 会影响 prompt 内容；prime 的短输出通过 `max_tokens = 1` 实现，不能另生成一个 `output_length = 1` 的 trace。

## 3. 运行测量

服务端和 AIPerf client 容器需已启动。主机与容器应以相同绝对路径访问 trace、artifact 和 probe；probe 文件位置与引擎的 `ATOM_FWDPROBE` 一致。

```bash
export MODEL=/mnt/models/Kimi-K3
export URL=http://127.0.0.1:8010
export CLIENT=k3-aiperf-client
export SERVER=k3-decode-instr
export PROBE=/absolute/path/to/fwdprobe_decode.jsonl

TR="$PWD/out/decode/bench/traces_points" \
ART="$PWD/out/decode/bench/art_points" \
CSV="$PWD/out/decode/bench/points_results.jsonl" \
bash scripts/run_decode_bench.sh

TR="$PWD/out/decode/bench/traces_calib" \
ART="$PWD/out/decode/bench/art_calib" \
CSV="$PWD/out/decode/bench/calib_results.jsonl" \
REFS="$PWD/out/decode/decode_calib_points.csv" \
bash scripts/run_decode_bench.sh
```

Runner 每组最多 prime 两次，然后运行 meas，并记录测量前后 probe 的字节边界。`ONLY` 可选择逗号分隔的 tag；同一结果文件已有 OK 的 tag 会跳过。换配置或点集时使用新结果文件。

历史文件 `traces_*_results.csv` 实际为 **JSONL**；新示例使用 `.jsonl` 后缀，分析器都按逐行 JSON 读取。

## 4. 验收

[analyze_decode_group.py](../scripts/analyze_decode_group.py)已实现：

| 自动检查 | 当前默认规则 |
|---|---|
| 样本过滤 | 纯 Decode、非 dummy、b 等于目标、所有 q=1 |
| 暖机及数量 | 丢前 20 个可用样本，余下至少 100 个 |
| 平均长度 | 窗口平均 L 与目标偏差 ≤2% |
| 校准特征 | 提供参考值时，CV_L 偏差 ≤0.02、r_max 偏差 ≤0.05 |
| 缓存 | 有 usage 指标时，平均 cache-read 不低于目标 L 的一半；这是排除冷跑的粗门槛 |
| 批次波动 | 错 b 的记录数超过保留样本数时判 BAD |

正式采样还需核对：请求全部成功、无抢占 / 重入、KV 共享方式符合设计、实际 cache hit 和 residual Prefill 合理。仅有 `accept=OK` 不证明这些条件全部成立。

稳定窗口宜保留至少 200 步，并检查记录是否连续。过滤后拼接的离散样本不能自动视为一个连续窗口；缺少 cache 指标、probe 尾部未落盘等情况也应在结果中说明。

## 5. 三种测量值

| 字段 | 解释 |
|---|---|
| `fwd_gpu_ms_p50` | `run_model` GPU event 时间的中位数 |
| `D_ms` | 相邻 `forward` 返回间隔的中位数，稳定时 `TPS = b / D × 1000` |
| `itl_ms_p50` | AIPerf 客户端 ITL，包含 streaming / 网络等影响 |

`D_ms` 与 ITL 接近可支持稳定性判断，但两者不要求严格相等。`D − fwd_gpu` 是同形状周期与模型时间的差额，不能据此逐项识别调度或采样成本。

## 6. 异构校准与回代

同一个 `(L_eq, b)` 测 A 同构、B 真实长度向量、C 对齐 r_max、D 对齐 CV_L 四种形状。保持长度之和不变，使用实际落地特征拟合：

$$
\log(D_{\mathrm{shape}}/D_A)=\beta_1CV_L^2+\beta_2\log r_{\max}
$$

生成方式与可识别性见[等价法](ragged_packed_decode_workload_equivalence.md#3-为什么需要异构校准)。当前长度全按 page 对齐，`CV_block` 与 `CV_L` 共线，默认不加入第三项。

```bash
python3 scripts/fit_decode_imbalance.py \
  -r out/decode/bench/calib_results.jsonl \
  -p out/decode/decode_calib_points.csv \
  -o out/decode/decode_imbalance_fit.md
```

只有实测显示修正没有稳定收益时，才在验证范围内使用未修正基线。若选择 GPU forward 作最终时间口径，也应检查 `--metric fwd_gpu_ms_p50` 的校准结果。主表汇总及吞吐命令见[Decode 流程](SKILL.md)。

K3 中建 KV 的时间远大于正式 Decode 采样。规划机时时先用少量点测冷 Prefill 和 cache hit，不把“跨组前缀理论可复用”直接当成实际节省。
