# 指标与统计口径

比较两个吞吐值之前，先确认四件事：分子数哪些 token，分母包含哪些时间，统计哪段运行，以及结果覆盖多少样本。

## 1. Token 和时间的定义

| 符号 / 字段 | 含义 |
|---|---|
| `ISL` / `OSL` | 一条请求的完整输入 / 输出 token 数 |
| Prefill `c` / `q` | 本步已有的 KV 上下文 / 本步新计算的 token 数 |
| Decode `L` / `b` | 本步开始前的 KV 长度 / 实际 batch size |
| `fwd_gpu_ms` | GPU event 包围 `run_model` 的时间，当前探针包含模型 forward 和 logits 计算 |
| `step_wall_ms` | CPU 时钟包围 `ModelRunner.forward` 的时间 |
| `step_period_ms` / `D_ms` | 相邻 `forward` 返回时刻的间隔；在稳定、饱和运行时代表引擎步长 |
| TTFT / ITL | 客户端发出请求到首 token 的延迟 / 首 token 之后的 token 间隔 |

Prefill 中 `c` 不等于 ISL：同一请求的后续 chunk 会把前面新算的 token 也计入 `c`。统计 fresh token 用 `Σ chunk_tokens`；完整 ISL 对每个请求实例只计一次。

## 2. AIPerf 的客户端指标

设公共计量窗口为 $T$，通常从首条计量请求发出到最后一条响应结束。窗口包含排队、网络和尾部排空；warmup 是否排除，应以本次报告的阶段设置为准。

$$
\mathrm{inputTPS}=\frac{\sum ISL}{T},\qquad
\mathrm{outputTPS}=\frac{\sum OSL}{T},\qquad
\mathrm{totalTPS}=\frac{\sum ISL+\sum OSL}{T}
$$

对应字段为 `input_token_throughput`、`output_token_throughput` 和 `total_token_throughput`。三者共用窗口，不能直接把其中一个当作某个 P/D 节点的容量。启用 `--use-server-token-count` 时使用服务端 usage；否则需核对客户端 tokenizer 的计数。

对请求 $k$，记发送、首 token、最后响应时刻为 $s_k,f_k,e_k$：

$$
TTFT_k=f_k-s_k,\qquad
TPOT_k=\frac{e_k-f_k}{OSL_k-1}\quad(OSL_k>1)
$$

`ISL / TTFT` 和 `1 / TPOT` 是单请求速率。TPOT 是整条生成阶段的平均值；AIPerf 的 ITL 分布还取决于流式 chunk 和 token 计数方式，多 token chunk 时不能直接把一个收包间隔当作一步 GPU 时间。

首输出 token 通常来自最后一次 Prefill 的 logits 和采样。普通单 token Decode 从后续 token 开始，因此 post-handoff 计数常为 `OSL − 1`。具体边界应和引擎日志保持一致。

### effective 与 active

客户端可以把每条请求的 token 匀速摊到所属阶段，再叠加成系统速率曲线：

$$
R_P(t)=\sum_k\frac{ISL_k}{f_k-s_k}\mathbf{1}_{[s_k,f_k)}(t),\qquad
R_D(t)=\sum_k\frac{OSL_k-1}{e_k-f_k}\mathbf{1}_{[f_k,e_k)}(t)
$$

- `effective_*` 对完整计量窗口取时间平均。
- `active_*` 只对对应阶段至少有一条活跃请求的区间取时间平均；分母是这些区间的并集长度。

两类指标都以客户端时刻划分。TTFT 内的排队、KV 传输和网络时间不会自动消失，`active_prefill` 也不等于 GPU 忙时吞吐。字段细节需核对实际 AIPerf 版本。

## 3. 从 Forward 时间计算吞吐

对 forward $i$，令 $n_i$ 为它处理的 token 数，$t_i$ 为选定口径的耗时：

$$
T_{\mathrm{fwd}}=\sum_i t_i,\qquad
\mathrm{TPS}=\frac{\sum_i n_i}{\sum_i t_i}
=\sum_i\frac{t_i}{T_{\mathrm{fwd}}}\frac{n_i}{t_i}
$$

Prefill 的 fresh 分子为本步所有 `q_i` 之和；普通 Decode 的分子为实际 `b_i`。若报 Prefill ISL 口径，可把每条请求的 ISL 归到它最后一个 Prefill forward，避免 chunk 重复计数。

选择不同的 $t_i$，回答的问题也不同：

| 分母 | 回答的问题 |
|---|---|
| 客户端完整窗口 | 实际交付速率 |
| 同形状稳定运行的 `Σ step_period` | 当前引擎连续处理这些 batch 的速度 |
| `Σ fwd_gpu` | 假设其它开销可完全隐藏时，这些 batch 的纯模型吞吐 |

`Σ fwd_gpu` 给出的是有条件的估计上限。GPU event 还可能包含 stream 上的等待与通信，不能解释成算子理论峰值。跨 stream、跨 rank 的完整执行边界需要另行确认。

CPU 与 GPU 可以重叠执行，`step_wall − fwd_gpu` 不能直接解释为 CPU 开销。在相同形状的稳定窗口中，`D − fwd_gpu` 可作为步周期与模型时间的差额；它不是逐项测出的调度、采样时间之和。

### 子集与外推

查表只命中部分 forward 时，分子和分母都限制在这个子集，同时报告 forward、fresh / ISL 或 output token 的覆盖率。

把子集时间除以 token 覆盖率，再用全量 token 作分子，吞吐值不会改变。这隐含“未覆盖部分每 token 成本相同”，不能增加证据覆盖率。若用模型补齐缺失点，应单列补算时间、模型适用范围和外推比例。

## 4. 按请求形状加权时的区别

若 $w_j$ 是请求占比、$T_j$ 是每个形状独立实测的吞吐，则

$$
\overline T_{\mathrm{shape}}=\sum_j w_jT_j
$$

是形状加权性能指标。它通常不等于混合负载的总吞吐。

在各形状独立、按占比分段处理的近似下，设每条请求按当前吞吐口径计 $x_j$ 个 token，可估算：

$$
T_{\mathrm{mix}}\approx
\frac{\sum_j w_jx_j}{\sum_j w_jx_j/T_j}
$$

分母是平均每条请求摊到的处理时间；$x_j$ 全相等时退化为加权调和平均。真实混合 batch、缓存共享和调度相互影响时，应使用 forward 路线或混合负载实测。

## 5. 时间拆分与 P/D 配比

可以用 `窗口 = 模型时间 + 其余时间` 做记账，但由不同实验的点表推算模型时间后得到的余量，应称为**残差**。没有同次运行的队列和时间证据，就不能把它全部认定为空转或某类软件开销。

P/D 节点的 token 类型不同，配平的是请求速率：

$$
r_P=\frac{T_P}{\mathbb{E}[ISL]},\qquad
r_D=\frac{T_D}{\mathbb{E}[OSL]},\qquad
\frac{N_D}{N_P}=\frac{r_P}{r_D}
$$

若 $T_P$ 使用 fresh token，分母也改为平均 fresh token；若 $T_D$ 只数 post-handoff token，分母同步改为 `OSL − 1`。两侧都需要同一目标负载下的全量容量估计。改变节点数会改变每节点并发和 batch 分布，$T_D$ 需随之更新；KV 传输能力也要换算成 req/s 后共同约束系统容量。
