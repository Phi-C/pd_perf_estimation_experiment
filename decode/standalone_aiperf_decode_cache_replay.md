# 脱离 Dynamo 的 vLLM / Native ATOM + AIPerf Decode-Dominant Cache-Replay 性能测试

> 目标：不依赖 DynamoGraphDeployment、Dynamo Frontend 或 Dynamo Profiler，由开发人员直接启动普通 vLLM 或 Native ATOM 服务，通过两遍 AIPerf Prompt replay 测量一个或一组 Decode-dominant 性能点。
>
> 该方法复用了 Dynamo v0.9.1 的核心测量思想，并增加了相同请求体校验和可选的 Prefix Cache 命中率校验。它跳过绝大多数重复 Prompt Prefill，但不等于严格“零 Prefill”的 post-handoff Decode-only。

## 1. 方案结论

一个 Decode-dominant 性能点由以下参数定义：

```text
模型和并行配置
ISL：固定输入长度
OSL：固定输出长度
B：请求 cohort 大小/目标并发
GPU 数量
KV Cache 容量和数据类型
```

测试执行两遍完全相同的请求：

```text
Pass 1：cache build
    B 个固定 ISL/OSL 的 Prompt
    → vLLM 或 Native ATOM 正常执行真实 Prefill
    → Prompt KV 保留在本地 Prefix Cache

Pass 2：measurement
    使用相同 seed 重新生成相同的 B 个 Prompt
    → 期望命中 Pass 1 的 Prefix Cache
    → 测量 ITL 和 output throughput
```

服务端必须是：

```text
一个普通 aggregated vLLM engine
或一个 Native ATOM standalone OpenAI server
+ Prefix Cache 开启
+ 单服务副本或严格 sticky routing
```

服务端不能使用：

```text
DecodeBenchConnector
独立 Prefill/Decode worker
P→D KV-transfer connector
负载均衡到多个无共享 Cache 的 engine/DP rank
```

该方案测量的是本地 Prefix Cache 高命中后的 serving-level Decode-dominant 性能，不测量真实 P→D KV 网络传输。vLLM 和 Native ATOM 都会保留至少一个 Prompt Token或一个 Cache block用于重新计算采样 logits，因此 Pass 2仍包含小段 residual Prefill/边界 forward。

更准确的时间组成是：

```text
Pass 2 wall time
= Prefix Cache lookup
+ residual Prompt / first-output boundary forward
+ steady Decode
+ scheduler/ramp-up/drain
```

只有从 KV/state已经就绪、handoff token已经存在之后截取稳定 Decode窗口，才接近严格的 post-handoff Decode-only。

---

## 2. 与 Dynamo v0.9.1 源码的对应关系

### 2.1 profile_decode.py：定义采样点

Dynamo v0.9.1 的 [`profile_decode.py`](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/profile_decode.py) 负责：

1. 固定详细 Decode profiling 的 `OSL=500`；
2. 扫描不同 ISL；
3. 根据 KV 容量计算最大 concurrency；
4. 扫描 `num_request=B`；
5. 调用 AIPerf 单点测量；
6. 将结果保存成 Planner 使用的性能曲面。

核心关系为：

```python
osl = 500

for isl in ...:
    for num_request in sweep_num_request:
        itl, thpt_per_gpu = get_itl_and_thpt_per_gpu(
            isl, osl, num_request
        )
```

它没有逐次固定真实 Decode context，而是把整条生成轨迹映射到平均 context：

$$
L_{\text{ctx}}
=
ISL+\frac{OSL}{2}
$$

并估算 active KV usage：

$$
U_{\text{KV}}
=
\frac{
    B\left(ISL+\frac{OSL}{2}\right)
}{
    \text{max KV tokens}
}
$$

### 2.2  aiperf.py: 测量一个采样点


Dynamo v0.9.1 的 [`aiperf.py`](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/aiperf.py) 对每个点构造：

```text
--synthetic-input-tokens-mean ISL
--synthetic-input-tokens-stddev 0
--output-tokens-mean OSL
--output-tokens-stddev 0
--extra-inputs ignore_eos:true
--extra-inputs min_tokens:OSL
--extra-inputs max_tokens:OSL
--concurrency B
--num-dataset-entries B
--request-count B
```

并在 `benchmark_decode()` 中使用同一个随机 seed执行两遍 AIPerf：

```text
第一遍：构造 Prefix Cache
第二遍：相同 Prompt replay并正式计量
```

本文提供的独立脚本保留了这些核心参数，不依赖 Dynamo的 Python包、DGD转换器或 Planner。

---

## 3. 提供的脚本

仓库根目录新增：

```text
start_vllm_decode_cache_replay_server.sh
    启动普通 vLLM aggregated server并显式开启Prefix Cache
    仅适用于vLLM；Native ATOM启动命令见第5.3节

run_aiperf_decode_point.sh
    运行一个固定(ISL, OSL, B)两遍cache-replay性能点
    OpenAI请求可发送给vLLM或Native ATOM

run_aiperf_decode_sweep.sh
    扫描多个ISL和B，调用单点脚本并汇总decode_points.csv
```

这些脚本只依赖：

- Bash；
- Python 3；
- `curl`；
- `vllm` CLI或可导入的 `atom` Python包；
- `aiperf` CLI。

它们不导入任何 Dynamo模块。

---

## 4. 环境准备

### 4.1 安装并检查命令

在用于运行 serving engine/AIPerf 的 Linux环境中安装对应版本，然后确认：

```bash
# vLLM服务端
vllm --version

# 或Native ATOM服务端
python3 -c 'import atom; print(atom.__file__)'

aiperf --help
python3 --version
curl --version
```

AIPerf版本至少需要支持：

```text
aiperf profile
--synthetic-input-tokens-mean
--synthetic-input-tokens-stddev
--output-tokens-mean
--output-tokens-stddev
--concurrency
--request-count
--num-dataset-entries
--random-seed
--output-artifact-dir / --artifact-dir
```

当前脚本使用 `--output-artifact-dir`；AIPerf将其和 `--artifact-dir` 定义为别名。

### 4.2 给脚本添加执行权限

```bash
chmod +x \
  start_vllm_decode_cache_replay_server.sh \
  run_aiperf_decode_point.sh \
  run_aiperf_decode_sweep.sh
```

### 4.3 选择服务端容量

如果计划测试：

```text
最大 ISL = ISL_max
OSL = OSL_probe
最大并发 = B_max
```

至少应满足：

$$
\text{max model length}
\ge
ISL_{\max}+OSL_{\text{probe}}
$$

以及：

```text
serving engine --max-num-seqs >= B_max
```

KV容量至少需要容纳当前单点：

$$
\text{max KV tokens}
\gtrsim
B\left(ISL+OSL\right)
$$

还应为 block padding、AIPerf运行和其他 runtime buffer保留余量。接近 KV容量极限时，第二遍可能发生 eviction或 recompute。

---

## 5. 启动模型服务

### 5.1 启动vLLM

在第一个终端执行：

```bash
./start_vllm_decode_cache_replay_server.sh \
  --model Qwen/Qwen3-32B \
  --tp 8 \
  --max-model-len 32768 \
  --max-num-seqs 128
```

脚本实际构造的核心命令为：

```bash
vllm serve Qwen/Qwen3-32B \
  --served-model-name Qwen/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

启动脚本默认只监听 `127.0.0.1`，避免其他请求污染 Cache和性能数据。AIPerf位于另一台主机时，显式传入 `--host 0.0.0.0` 或指定服务网卡地址，并配置 API key和网络访问控制。

如果 API使用不同的 model name：

```bash
./start_vllm_decode_cache_replay_server.sh \
  --model /models/Qwen3-32B \
  --served-model-name qwen3-32b \
  --tp 8 \
  --max-model-len 32768
```

此时运行 AIPerf时必须使用：

```text
--model qwen3-32b
--tokenizer /models/Qwen3-32B
```

### 5.2 传递其他vLLM参数

`--` 后面的参数会原样传给 `vllm serve`：

```bash
./start_vllm_decode_cache_replay_server.sh \
  --model <model> \
  --tp 8 \
  --max-model-len 65536 \
  --max-num-seqs 64 \
  -- \
  --block-size 16
```

脚本会拒绝在 `--` 后覆盖它已经管理的 serving参数，也会拒绝 DP、config file、KV-transfer和关闭 Prefix Cache的参数；这些选项会破坏单engine cache-replay前提或使实际配置与启动脚本记录不一致。

固定版本测试时至少记录：

- vLLM或ATOM commit/version；
- model revision；
- TP/PP/DP/EP；
- model dtype和量化方式；
- KV Cache dtype；
- block size；
- GPU型号、数量和拓扑；
- `max-model-len`；
- `max-num-seqs`；
- `gpu-memory-utilization`；
- CUDA/ROCm和通信库版本。

### 5.3 启动Native ATOM

Native ATOM提供 OpenAI兼容的 `/v1/chat/completions`、`/v1/completions`、`/health` 和 `/metrics`。它的 Prefix Cache默认关闭，必须显式添加 `--enable_prefix_caching`。

在第一个终端执行：

```bash
python3 -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3-32B \
  --host 127.0.0.1 \
  --server-port 8000 \
  -tp 8 \
  -dp 1 \
  --kv_cache_dtype fp8 \
  --enable_prefix_caching \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90
```

注意：

- `--server-port` 是 OpenAI HTTP端口；ATOM的 `--port` 是 engine内部通信端口；
- 保持 `-dp 1` 最容易保证两遍请求落到同一份 Prefix Cache；
- 使用 DP时必须提供严格、可验证的 sticky rank routing，不能让同一 Prompt在两遍之间漂移；
- 使用 `atom.entrypoints.openai_server`；ATOMesh standalone的指标暴露能力与 Native OpenAI server不同；
- 不启用 Mooncake、MORI-IO、LMCache offload或其他 P/D/KV-transfer路径；
- 测试期间不要混入其他请求。

当前 Native ATOM OpenAI streaming响应会返回包含 `prompt_tokens`、`completion_tokens` 和 `total_tokens` 的 usage chunk，因此可与 AIPerf的 `--use-server-token-count` 配合使用。

### 5.4 Native ATOM的模型状态要求

普通 MHA/MLA模型主要依赖 paged KV blocks。对于以下 stateful模型，仅命中普通 KV block并不足以恢复合法 Decode状态：

- DeepSeek-V4 compressor ring/index state；
- GDN/Mamba/conv recurrent state；
- KDA等其他每请求状态；
- MTP/speculative decoding的 draft/target state。

对于 DeepSeek-V4/GDN等该变更覆盖的模型，Native ATOM至少应包含 [PR #1771](https://github.com/ROCm/ATOM/pull/1771)（merge commit `042e8f1c`）或等价后续实现。该版本增加 content-addressed per-request state checkpoint；旧版本的请求可能拿到未关联当前 Prefix的 state slot，不能把结果当作有效 Decode数据。KDA或其他未由该 PR覆盖的状态类型，仍需各自经过正确性验证的 content-addressed state adapter，不能仅凭版本号假定支持。

如果目标 Prompt边界没有可恢复的state checkpoint，ATOM会缩短actual prefix hit并重新执行较长 Prefill；此时HTTP流程虽然成功，但该点不再是有效的Decode-dominant点。

还必须记录并验证：

- `--state-checkpoint-interval-tokens`；
- actual prefix-cache hit与compressed hit；
- lost-to-checkpoint比例；
- state checkpoint eviction；
- Pass 2实际 residual Prefill token数。

### 5.5 服务端不应添加的配置

不要添加：

```text
--kv-transfer-config ... DecodeBenchConnector ...
真实P/D connector
Mooncake/MORI-IO/LMCache恢复路径
多个独立engine replica或无sticky routing的DP rank
```

原因是第一遍必须真实执行 Prompt Prefill并把 KV保存在第二遍能够访问的同一个 engine 中。

---

## 6. 运行单个 Decode-dominant 性能点

### 6.1 基本示例

在第二个终端运行：

```bash
./run_aiperf_decode_point.sh \
  --model Qwen/Qwen3-32B \
  --tokenizer Qwen/Qwen3-32B \
  --url http://127.0.0.1:8000 \
  --isl 1024 \
  --osl 500 \
  --concurrency 16 \
  --gpus 8
```

这个点表示：

```text
固定目标 ISL = 1024
固定目标 OSL = 500
数据集 Prompt 数 = 16
正式请求数 = 16
目标并发 = 16
物理 GPU 数 = 8
```

两遍均只包含这一批16个请求；完成后不会继续补入第17个请求。

### 6.2 对Native ATOM运行AIPerf

AIPerf命令本身不需要更换 backend；它仍通过 OpenAI Chat Completions调用 Native ATOM：

```bash
./run_aiperf_decode_point.sh \
  --model Qwen/Qwen3-32B \
  --tokenizer Qwen/Qwen3-32B \
  --url http://127.0.0.1:8000 \
  --isl 1024 \
  --osl 500 \
  --concurrency 16 \
  --gpus 8
```

当前 Native ATOM同时支持 `max_tokens` 和 `max_completion_tokens`，并通过 `ignore_eos=true` 保证生成到上限。ATOM没有 `min_tokens` 字段，但当前协议以 `extra="ignore"` 处理未知字段，所以脚本发送的 `min_tokens` 会被忽略，不影响 `max_tokens + ignore_eos` 固定 OSL。

旧版 ATOM如果只接受 `max_tokens`，增加：

```bash
--use-legacy-max-tokens
```

现有 `run_aiperf_decode_point.sh` 的请求、结果和 payload hash逻辑可以复用，但 Prefix Cache counter解析目前只识别 vLLM指标；Native ATOM的指标适配见第6.6节。

### 6.3 指定输出目录和seed

```bash
./run_aiperf_decode_point.sh \
  --model Qwen/Qwen3-32B \
  --isl 4096 \
  --osl 500 \
  --concurrency 32 \
  --gpus 8 \
  --seed 12345 \
  --output-dir ./results/decode_isl4096_osl500_b32
```

### 6.4 远端或带API Key的服务

```bash
./run_aiperf_decode_point.sh \
  --model qwen3-32b \
  --tokenizer Qwen/Qwen3-32B \
  --url https://inference.example.com \
  --api-key "$API_KEY" \
  --isl 4096 \
  --osl 500 \
  --concurrency 16 \
  --gpus 8
```

远端 endpoint如果不公开 `/health`：

```bash
--skip-health-check
```

远端 endpoint如果不公开 `/metrics`，脚本仍能运行，但不能自动验证 Prefix Cache命中率。此时不要使用 `--require-cache-metrics`，并通过服务端日志或其他 telemetry确认命中情况。

### 6.5 严格要求vLLM cache指标

```bash
./run_aiperf_decode_point.sh \
  --model <model> \
  --isl 8192 \
  --osl 500 \
  --concurrency 16 \
  --gpus 8 \
  --min-cache-hit-ratio 0.95 \
  --require-cache-metrics
```

当前 vLLM通常公开：

```text
vllm:prefix_cache_queries
vllm:prefix_cache_hits
```

Prometheus counter可能以 `_total` 后缀暴露。脚本同时识别这两种形式，并使用 Pass 2前后的 counter差值计算：

$$
\text{hit ratio}
=
\frac{
    \Delta\text{prefix cache hits}
}{
    \Delta\text{prefix cache queries}
}
$$

由于 Prefix Cache通常保留至少一个 Prompt边界 Token供模型计算 logits，加上 block对齐，命中率不一定严格等于 `1.0`。默认阈值采用 `0.90`；应结合 ISL和 block size选择更合理的门限。

### 6.6 Native ATOM Cache指标适配

当前 Native ATOM OpenAI server公开：

```text
atom:prefix_cache_cached_tokens
atom:prefix_cache_full_tokens
atom:prefix_cache_hit_ratio
atom:prefix_cache_compressed_hit_ratio
atom:prefix_cache_lost_to_checkpoint_ratio
atom:prefix_cache_lost_unrecoverable_ratio
```

Prometheus Counter实际通常带 `_total` 后缀。Pass 2的实际 token hit ratio应使用区间增量，而不是累计 gauge：

$$
\text{ATOM hit ratio}_{P2}
=
\frac{
    \Delta\texttt{atom:prefix\_cache\_cached\_tokens\_total}
}{
    \Delta\texttt{atom:prefix\_cache\_full\_tokens\_total}
}
$$

也可以在 Pass 2前后读取：

```text
GET /debug/cache_stats
```

当前脚本尚未解析这些 ATOM metric name。因此：

- 不适配脚本时，不要为 ATOM添加 `--require-cache-metrics`，否则该点会被判 invalid；
- 结果只能先标记为“请求一致、Cache命中待外部验证”；
- 正式采数前应扩展 `prefix_cache_counters()`，同时支持 vLLM和ATOM两个指标族；
- stateful模型还要检查 compressed hit与actual hit的差值，不能只看 Prompt文本相同。

### 6.7 兼容旧式max_tokens字段

如果目标服务要求旧式 `max_tokens` 字段：

```bash
--use-legacy-max-tokens
```

脚本不开放任意 AIPerf参数透传，防止调用方覆盖 model、URL、ISL、OSL、B、seed、warmup、request rate、运行次数或artifact目录，造成汇总配置与实际执行不一致。需要新增参数时，应在脚本中以专用选项显式接入并纳入校验。

---

## 7. 单点脚本内部流程

### 7.1 请求形状

两个 pass使用相同参数：

```text
ISL mean = ISL
ISL stddev = 0
OSL mean = OSL
OSL stddev = 0
ignore_eos = true
min_tokens = OSL
max_tokens = OSL
concurrency = B
num_dataset_entries = B
request_count = B
random_seed = seed
```

所以它复现的是 Dynamo v0.9.1 单点测量的核心请求形状。

### 7.2 Pass 1：构造Cache

Pass 1向普通 vLLM或Native ATOM engine发送 B个 Prompt：

```text
真实Prompt Prefill
→ 真实KV/state写入engine GPU Cache
→ 请求生成固定OSL
→ 请求结束后可复用的Prompt blocks保留在Prefix Cache
```

脚本没有额外启用 AIPerf内部 warmup。Pass 1本身同时承担：

- 模型/kernel warmup；
- Prompt KV构造；
- Prefix Cache warmup。

### 7.3 Pass 2：Replay并测量

Pass 2使用完全相同的：

- model；
- tokenizer；
- ISL/OSL；
- `num_dataset_entries`；
- `request_count`；
- random seed。

理论上 AIPerf会生成完全相同的 B个请求 payload。服务端处理路径为：

```text
Prefix Cache lookup
→ 命中绝大多数Prompt KV
→ 计算残余Prompt block和首输出边界
→ 正常逐Token Decode
```

这条路径不是严格零 Prefill：

- vLLM将最大cache hit长度限制为 `prompt_length - 1`，并可能因block对齐重新计算整个block；
- Native ATOM的 `can_allocate()` 同样明确不复用最后一个Prompt hash block，以保证至少执行一次forward获得sampler logits；
- 对普通attention模型，ATOM完全命中时通常仍有 `1～hash_block_size` 个新Prompt token；
- state checkpoint或SWA gate未命中时，Native ATOM可能重新计算更多Prompt token。

因此本流程应称为 cache-replay Decode-dominant serving benchmark。

### 7.4 请求一致性校验

每次 AIPerf运行都会输出 `inputs.json`。脚本：

1. 读取两个 pass的 `inputs.json`；
2. 取出所有实际 request payload；
3. 排除 session ID，只比较发送内容；
4. 对规范化 JSON计算 SHA-256；
5. 验证两个 hash和 payload数量完全一致。

如果请求体不一致，该性能点被标记为 invalid。

这比只相信相同 random seed更严格，但它仍验证的是 HTTP payload，不是服务端内部最终 Token IDs。若客户端和服务端 tokenizer/chat template不同，还应检查 server-reported ISL。

### 7.5 Cache命中校验

脚本在 Pass 2前后抓取 `/metrics`：

```text
metrics_before_measurement.prom
metrics_after_measurement.prom
```

现有脚本找到 vLLM Prefix Cache counters时，会计算仅属于 Pass 2的 hit ratio。加入第6.6节所述适配后，Native ATOM使用 `full_tokens/cached_tokens` 计算同一窗口。

当指标不可用时：

- 默认：给出 warning，保留结果；
- 使用 `--require-cache-metrics`：将该点标为 invalid；
- 也可以从 vLLM/ATOM日志、trace、`/debug/cache_stats` 或外部 Prometheus查询验证。

---

## 8. 结果目录

默认结果路径：

```text
aiperf_decode_points/
└── isl1024_osl500_b16_seed100_<timestamp>/
    ├── cache_build/
    │   ├── inputs.json
    │   └── profile_export_aiperf.json
    ├── measurement/
    │   ├── inputs.json
    │   └── profile_export_aiperf.json
    ├── cache_build.log
    ├── measurement.log
    ├── metrics_before_cache_build.prom
    ├── metrics_before_measurement.prom
    ├── metrics_after_measurement.prom
    └── summary.json
```

具体 AIPerf版本可能在 pass目录下增加一层 trial/sweep子目录；脚本会递归查找 `inputs.json` 和 `profile_export_aiperf.json`。

### 8.1 summary.json

汇总文件至少包含：

```text
valid
problems/warnings

configuration:
    ISL、OSL、B、GPU数、seed
    目标ISL + 目标OSL/2
    B × (ISL + OSL/2)

input_replay_validation:
    两遍payload SHA-256
    payload数量
    exact_payload_match

request_execution_validation:
    两遍successful request count
    error request count
    cancelled/error summary状态

prefix_cache_validation:
    Pass 2 engine-specific cache counter delta
    hit ratio
    校验阈值

measurement:
    实际平均ISL/OSL
    基于实测ISL/OSL计算的平均context
    average/p50/p90/p99 ITL
    aggregate output tok/s
    output tok/s/GPU
    估算的post-handoff throughput
```

只有同时满足以下硬条件时，脚本才把点标记为 valid：

1. 两遍 AIPerf都成功；
2. 两遍 request payload完全相同；
3. 两遍 payload数量都等于 `B`；
4. 两遍成功请求数都等于 `B`，并且没有error/cancel；
5. 实际平均OSL达到目标，ITL和throughput为有效正数；
6. 如果 cache counter可用，hit ratio不低于阈值；
7. 如果要求 cache metrics，则相关 counter必须可读取。

---

## 9. 如何解释ITL和throughput

### 9.1 ITL

AIPerf的 `inter_token_latency` 排除了请求发出到首 Token之间的 TTFT，主要反映首 Token之后的逐 Token间隔。

它仍可能包含：

- Scheduler等待；
- batch变化；
- preemption；
- Cache miss导致的局部 Prefill干扰；
- output streaming和runtime开销。

所以它是 serving-level ITL，不是单独 attention kernel时间。

Native ATOM还需要检查 SSE chunk粒度。ATOM frontend可能把同一请求的多个 Token合并到一个 SSE chunk，而 AIPerf按接收到的 chunk间隔形成 ITL样本。若一个 chunk含有 \(k>1\) 个Token，AIPerf观察到的是 chunk latency，不是严格逐Token ITL；吞吐和usage token数仍可正确，但mean/p99 ITL可能显著偏大或样本数偏少。

因此 Native ATOM应至少满足一项：

- 验证测量窗口内每个 SSE data chunk只包含一个Token；
- 从 Native ATOM engine trace直接统计逐Token/逐iteration时间；
- 对 MTP记录每步accepted tokens，并按accepted token口径重新构造ITL；
- 同时保存 `atom:mtp_average_tokens_per_forward` 和acceptance rate。

### 9.2 Output throughput

AIPerf的：

```text
output_token_throughput
```

是整个 engine的 aggregate output tok/s。每 GPU吞吐为：

$$
C_{D/GPU}
=
\frac{
    \text{aggregate output tok/s}
}{
    \text{physical GPUs in engine}
}
$$

这里必须除以整个 engine使用的物理 GPU数，不是只除以 TP degree。

### 9.3 首Token边界

Cache replay复用了真实 Prompt KV/state，但 Prefix Cache通常不缓存下一 Token logits。vLLM和Native ATOM都必须执行Prompt边界forward得到输出 token 1；受block对齐和state checkpoint gate影响，这同一次边界forward可能需要重新计算最后一个或多个 Prompt token。

如果每个请求返回 `OSL=M`：

```text
真实输出边界/Decode forward数约为M
严格post-handoff Decode forward数为M-1
```

脚本给出的：

```text
post_handoff_throughput_estimate
```

使用近似：

$$
C_{\text{post-handoff}}
\approx
C_{\text{raw output}}
\times
\frac{OSL-1}{OSL}
$$

该修正只调整 Token计数，不能从 wall time中严格移除 residual Prompt/首Token边界 forward、Cache lookup和ramp-up。需要精确值时应从 server trace中截取 KV/state已就绪后的稳定 post-handoff窗口。

---

## 10. Batch与Context口径

### 10.1 B是请求cohort大小，不是硬编码模型batch

脚本设置：

```text
concurrency = B
request_count = B
num_dataset_entries = B
```

所以总共只有 B个正式请求。请求结束后没有新请求补入。

但是第 `t` 次模型 forward的实际逻辑 batch为：

$$
B_t
=
\text{Scheduler在该步选择的ready sequences数量},
\qquad
B_t\le B
$$

以下阶段可能出现 `B_t<B`：

- 请求到达存在时间差；
- Scheduler在收齐 B个请求前已经启动第一步；
- KV admission被拆分；
- 首尾 ramp-up/drain；
- preemption；
- `max-num-seqs<B`；
- DP/Attention-DP分流。

如果 B个请求固定 OSL、忽略 EOS且全部被同时接纳，中间大部分迭代通常可以接近：

```text
logical batch = B
每条sequence一个query token
一次forward生成B个Token
```

要证明这一点，必须查看 vLLM/ATOM trace或每步 scheduler/model-runner指标。

### 10.2 Context会持续增长

固定：

```text
ISL=N
OSL=M
```

不表示每步 context严格等于 N。请求覆盖：

```text
N, N+1, ..., N+M-1
```

脚本按照 Dynamo v0.9.1的约定，将该点的代表 context记录为：

$$
L_{\text{ctx,avg}}
\approx
N+\frac{M}{2}
$$

如果需要严格的 `context=L` 性能，可以：

1. 从 trace中按当前 context bucket筛选迭代；
2. 缩短 probe OSL；
3. 对不同 token position分别统计；
4. 使用固定 KV长度的 model-runner/kernel microbenchmark进行补充。

---

## 11. 运行多个性能点

### 11.1 扫描ISL和并发

```bash
./run_aiperf_decode_sweep.sh \
  --model Qwen/Qwen3-32B \
  --tokenizer Qwen/Qwen3-32B \
  --url http://127.0.0.1:8000 \
  --isls 1024,4096,16384 \
  --concurrencies 1,2,4,8,16,32 \
  --osl 500 \
  --gpus 8
```

该命令执行笛卡尔积：

```text
3个ISL × 6个B = 18个性能点
```

每个点内部都执行两遍 AIPerf。Sweep为每个点使用不同 seed，降低不同点之间的 Prompt Cache交叉命中。

### 11.2 输出

每个点具有独立的 `summary.json`，Sweep根目录额外生成：

```text
decode_points.csv
```

主要字段：

```text
valid
isl
osl
concurrency
target_average_context
measured_average_context
active_context_tokens
input_payloads_match
prefix_cache_hit_ratio
measured_isl_avg_tokens
measured_osl_avg_tokens
itl_avg_ms
itl_p50_ms
itl_p90_ms
itl_p99_ms
output_throughput_total_tps
output_throughput_per_gpu_tps
post_handoff_throughput_estimate_tps
summary_json
```

该 CSV可以作为后续二维插值、绘图或 Planner适配的输入。

### 11.3 Sweep期间的Cache管理

多个点共享同一个 serving engine时，旧点的 Cache可能仍存在。每个新点都立即执行：

```text
自己的Pass 1
→ 自己的Pass 2
```

LRU通常会优先驱逐旧点，但仍必须通过每个点的 hit ratio验证当前 Prompt是否完整保留。

严格隔离测试可以在每个点前重启服务，但成本较高。无论采用哪种方式，测试期间都不应混入其他业务流量。

---

## 12. 建议的性能点有效性门槛

一个性能点至少应满足：

```text
两遍HTTP payload完全相同
Pass 2 Prefix Cache hit ratio达到预期
Pass 2 residual Prefill与engine预期一致
所有B个请求成功
实际OSL达到目标
实际server-side ISL接近目标
无OOM、timeout和preemption
max-num-seqs >= B
KV容量足以容纳B × (ISL + OSL)
测试期间没有其他流量
```

用于容量规划时还应记录：

- mean/p50/p90/p99 ITL；
- aggregate和per-GPU output throughput；
- 实际 running batch；
- 实际 context分布；
- GPU利用率和HBM带宽；
- KV Cache usage；
- preemption和queue；
- 重复运行3–5次后的方差。

只看到较低的 ITL或较高 throughput，不能自动证明该点有效；首先要确认 Cache确实命中且实际工作负载与目标一致。

---

## 13. 常见问题

### 13.1 第二遍Cache命中率低

检查：

1. vLLM是否带 `--enable-prefix-caching`，或Native ATOM是否带 `--enable_prefix_caching`；
2. 服务是否在两遍之间重启；
3. 是否经过多个 replica或无sticky routing的DP rank；
4. KV容量是否能够容纳当前点；
5. 是否存在其他请求驱逐 Cache；
6. 两遍 `inputs.json` hash是否一致；
7. tokenizer/model/chat template是否改变；
8. Prefix Cache block size导致的尾部未命中是否符合预期。

### 13.2 Payload一致但仍有Prefill

可能原因：

- Cache已被驱逐；
- 请求落到其他实例；
- vLLM只命中 block-aligned prefix，尾部仍需计算；
- Native ATOM有意保留最后一个hash block用于产生logits；
- Native ATOM的SWA/state checkpoint gate缩短了actual hit；
- Cache counter包含多个类型或多个实例；
- 服务端实际 Token IDs因模板或模型配置变化而不同。

### 13.3 实际ISL不是目标值

AIPerf按客户端 tokenizer生成目标长度，但 OpenAI chat服务端可能额外应用 chat template和special tokens。

应：

- 保证 AIPerf tokenizer与服务端一致；
- 使用 `--use-server-token-count`；
- 检查 `profile_export_aiperf.json` 中的实际 `input_sequence_length`；
- 建模时优先保存实际 ISL，而不是只保存命令行目标。

### 13.4 实际batch小于B

检查：

- `--max-num-seqs`；
- KV capacity；
- vLLM/ATOM scheduler trace；
- 请求是否同时到达；
- 是否发生preemption；
- 是否使用DP；
- 是否在首尾阶段取样。

客户端 `--concurrency B` 不能强制每个模型 forward都是 batch B。

### 13.5 `/metrics`不可用

可以：

- 不使用 `--require-cache-metrics`；
- 指定独立 `--metrics-url`；
- 查询外部 Prometheus；
- 从 vLLM/ATOM日志读取 Prefix Cache hit信息；
- Native ATOM可读取 `/debug/cache_stats`；
- 使用 trace确认第二遍没有大段 Prompt Prefill。

缺少 Cache验证时，结果只能标记为“假定命中”，不应直接作为高置信度 Decode-dominant数据。

### 13.6 Native ATOM stateful模型结果异常

检查：

- ATOM版本是否包含 PR #1771或后续等价修复；
- `state_checkpoint_interval_tokens` 是否覆盖目标context；
- `atom:prefix_cache_compressed_hit_ratio` 是否较高；
- actual hit是否因 `lost_to_checkpoint` 明显低于compressed hit；
- state checkpoint是否被evict；
- DeepSeek-V4 SWA/index或GDN/KDA state是否完整进入checkpoint；
- MTP的accepted tokens和SSE chunk粒度是否正确计入ITL。

HTTP请求成功、两遍payload相同，只能证明replay输入一致，不能证明Native ATOM恢复了全部模型状态。

---

## 14. 与完整 Dynamo Profiler 的区别

本方案保留：

- 固定 ISL/OSL；
- `concurrency=request_count=dataset_entries=B`；
- ignore EOS；
- 两遍相同 Prompt cache-replay；
- ITL和output throughput提取；
- `ISL+OSL/2`平均 context口径。

本方案额外增加：

- 两遍 `inputs.json` payload hash校验；
- Pass 2 Prefix Cache counter差值；
- 单点 `summary.json`；
- Sweep `decode_points.csv`；
- 无Dynamo依赖的直接脚本入口。

本方案没有实现 Dynamo Profiler的：

- 自动部署DGD；
- P/D并行配置搜索；
- 从engine日志自动推导 `max_kv_tokens`；
- 根据容量自动生成 concurrency range；
- 自动构建 `raw_data.npz`；
- Planner二维插值和SLA反查；
- 在线 correction factor。

因此它的定位是：

```text
独立、可审计的Decode-dominant性能采样器
```

而不是完整的自动容量规划系统。

---

## 15. 测量边界

该方案可以回答：

```text
在给定模型、硬件、并行配置、平均context和请求cohort大小下，
vLLM或Native ATOM本地Prefix Cache高命中后的
serving-level ITL和output throughput是多少？
```

该方案不能回答：

```text
真实Prefill节点性能
真实P→D KV传输性能
多个服务副本的路由均衡
生产请求到达分布
真实长短请求混合
完整PD系统的TTFT和goodput
严格排除residual Prompt/首Token边界的post-handoff Decode性能
```

真实系统容量应分别测量：

$$
C_{\text{system}}
=
\min(
    C_{\text{Prefill}},
    C_{\text{KV transfer/router}},
    C_{\text{Decode}}
)
$$

并使用真实 P/D端到端压测校准该独立 Decode模型。

---

## 16. 参考

- [Dynamo v0.9.1 `profile_decode.py`](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/profile_decode.py)
- [Dynamo v0.9.1 `aiperf.py`](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/aiperf.py)
- [Dynamo v0.9.1 vLLM ConfigModifier](https://github.com/ai-dynamo/dynamo/blob/v0.9.1/benchmarks/profiler/utils/config_modifiers/vllm.py)
- [AIPerf synthetic dataset说明](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/synthetic-dataset-generation)
- [AIPerf输出文件说明](https://docs.nvidia.com/aiperf/tutorials/metrics-analysis/working-with-profile-export-files)
- [vLLM Metrics文档](https://docs.vllm.ai/en/latest/design/metrics/)
- [vLLM full cache hit仍需重算最后Token](https://docs.vllm.ai/en/v0.9.2/api/vllm/v1/core/kv_cache_manager.html)
- [Native ATOM Serving & Benchmarking Guide](https://github.com/ROCm/ATOM/blob/main/docs/serving_benchmarking_guide.md)
- [Native ATOM Scheduling & KV Cache Guide](https://github.com/ROCm/ATOM/blob/main/docs/scheduling_kv_cache_guide.md)
- [Native ATOM PR #1771：stateful模型的content-addressed state checkpoint](https://github.com/ROCm/ATOM/pull/1771)
