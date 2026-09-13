# 独立 AIPerf Decode cache-replay

用两遍相同 prompt 在单个 serving engine 上测量缓存命中后的 Decode serving 性能：第一遍准备 KV / 模型状态，第二遍重放并计量。这种方法保留缓存查找、残余 Prefill、调度和客户端通信，适合测 ITL 与输出吞吐。

需要纯模型时间及固定 batch 的实测表时，使用[Decode 测点流程](measure_decode_performance_point.md)和[forward 探针](../docs/forward_probe.md)。

## 1. 服务端条件

- 两遍请求到同一个缓存实例；多副本或 DP 场景需要可验证的固定路由。
- 开启 prefix cache，容量足以容纳当前批的 KV，并记录模型、量化、KV dtype、并行度和 kernel 版本。
- `max-num-seqs ≥ B`，`max-model-len ≥ ISL + OSL`。
- 对 KDA、GDN 等有额外状态的模型，确认 checkpoint 恢复正确。普通 KV 命中不足以证明整个模型状态可复用。

普通 vLLM 服务的命令示例：

```bash
vllm serve /path/to/model \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --max-num-seqs 64 \
  --max-model-len 32768 --enable-prefix-caching
```

ATOM 的具体参数随版本变化，应沿用目标实验的启动配置。K3 实验的本地 harness 使用了 `lmcache_offload` 恢复路径，见[复现说明](../experiments/exp1_kimi_k3/Reproduce.md)；不能把这种测量称为完全没有缓存恢复成本的 Decode-only。

## 2. 两遍请求

以下直接使用 AIPerf CLI，不依赖额外的启动或 sweep 脚本。先按环境修改模型与 tokenizer；AIPerf 版本需支持这些参数。

```bash
MODEL=/path/to/model
TOKENIZER=/path/to/model
BASE_URL=http://127.0.0.1:8000
ISL=8192
OSL=300
B=16
ART="$PWD/out/cache_replay"

for pass in cache_build measurement; do
  aiperf profile \
    --model "$MODEL" --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
    --url "$BASE_URL" --endpoint-type chat --streaming --use-server-token-count \
    --synthetic-input-tokens-mean "$ISL" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OSL" --output-tokens-stddev 0 \
    --num-dataset-entries "$B" --request-count "$B" --concurrency "$B" \
    --extra-inputs ignore_eos:true --extra-inputs "min_tokens:$OSL" \
    --extra-inputs "max_tokens:$OSL" --random-seed 42 \
    --output-artifact-dir "$ART/$pass"
done
```

两遍保持数据集参数、seed 和 prompt 一致，只使用 measurement 目录中的指标。相同 seed 是构造条件，还要比较实际 `inputs.json` 中的 prompt / token 内容；组内 B 条 prompt 也应分别对应目标的 KV 共享方式。

若使用仓库的 Mooncake trace harness，prime 与 meas 必须使用**同一个 trace 文件**。该合成路径的 `output_length` 会影响 prompt，缩短 prime 只能修改请求生成上限，详见[测点流程](measure_decode_performance_point.md)。

## 3. 验收

| 检查项 | 需要的证据 |
|---|---|
| 输入可重放 | 两遍实际 prompt / token 序列一致，模型及模板不变 |
| 请求完成 | B 条请求成功，实际 OSL 达到目标，无 error / cancel |
| 缓存命中 | measurement 的 usage、缓存 counter 增量或引擎 trace |
| 实际长度 | 服务端报告 ISL；chat template 可能增加 token |
| 批次稳定性 | 若用 `B / ITL`，确认所选稳定窗口内实际 batch 为 B |
| 状态恢复 | stateful 模型的 checkpoint / adapter 验证及残余 Prefill 长度 |

缓存指标不可用时，结果应标明“命中未经验证”。有 counter 时使用测量前后的增量，而不是累积 hit ratio；字段名以安装的引擎版本为准。

## 4. 结果含义

- `output_token_throughput = Σ OSL / 客户端窗口`，包含首输出边界与 ramp / drain。
- ITL 不包含 TTFT，但仍受实际 batch、抢占、SSE 合并和网络影响。
- 固定 `ISL, OSL` 覆盖的是增长中的上下文，代表位置约为 `ISL + OSL / 2`。
- 每 GPU 吞吐除以整个 engine 的物理 GPU 数；不能只依据 TP 参数推断总卡数。

Cache 命中后通常仍需计算 prompt 尾部以得到首 token logits，且 block / checkpoint 对齐可能增加重算范围。仅把 output token 分子乘 `(OSL − 1) / OSL` 不能从分母中移除这段开销。

本方法不直接测量真实 P→D 传输、生产到达分布或多副本路由。用于 P/D 容量规划前，应在目标部署下校准。

## 参考

- [Dynamo v0.9.1 profile_decode.py](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/profile_decode.py)
- [Dynamo v0.9.1 aiperf.py](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/aiperf.py)
- [AIPerf synthetic dataset](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/synthetic-dataset-generation)
- [ATOM state checkpoint PR #1771](https://github.com/ROCm/ATOM/pull/1771)；具体状态类型仍需验证支持范围。
