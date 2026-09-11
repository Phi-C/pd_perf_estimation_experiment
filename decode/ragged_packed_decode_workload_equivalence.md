# Ragged/Paged Batch 的 Decode 性能估计：工作量等价法

## 1. 问题定义

已有一组同构 batch 的 Decode 性能数据，每个测量点由以下二元组描述：

\[
(L,b)
\]

其中：

- \(L\)：每条请求在一次 Decode step 中对应的 ISL 或当前 KV cache length；
- \(b\)：batch size；
- 同一个测量点中的所有请求具有相同的 \(L\)；
- 性能指标包括 ITL、单步 Decode latency 和 output-token throughput。

现在需要估计一个异构请求 batch：

\[
\mathcal B=\{L_i\}_{i=0}^{n-1}
\]

这些请求都已经完成 Prefill，本轮只需要为每条请求执行一步 Decode，各生成 1 个新 token。各请求的 ISL 或当前 KV cache length 可以不同。

目标是利用已有同构测量数据，估计该异构 batch 的单步 Decode latency、每条请求的 ITL 和整个 batch 的 output-token throughput。

本文主要考虑支持变长 KV cache 的 **ragged/paged Decode batch**。如果执行引擎把所有请求 padding 到最大长度，则应改用最大长度模型，而不是本文的一阶工作量等价模型。

## 2. 长度定义与计数约定

需要首先区分初始输入长度和当前 KV cache length：

- \(ISL_i\)：请求最初的 input sequence length；
- \(O_i\)：在本轮之前已经完成的 Decode step 数；
- \(L_i\)：本轮 Decode 实际对应的当前上下文长度。

如果请求已经生成了若干输出 token，则通常有：

\[
L_i=ISL_i+O_i
\]

如果所有请求刚完成 Prefill，并且尚未执行其他 Decode step，则通常可以直接使用：

\[
L_i=ISL_i
\]

不同 benchmark 对“当前 token 是否已经计入 KV length”可能采用不同约定。如果本轮 Attention 实际参与的 KV token 数为 \(L_i+1\)，后续推导仍然成立，因为所有请求都增加了相同的常数 1。实际使用时，应保证异构请求和同构性能表采用相同的长度定义。

## 3. 单条请求的一步 Decode 工作量

### 3.1 非 Attention 工作量

普通单 token Decode 中，每条请求只处理 1 个新 query token。因此，QKV projection、output projection、MLP 和 normalization 等非 Attention 工作量对每条请求近似固定：

\[
W_{\mathrm{token},i}=1
\]

对于 \(n\) 条请求：

\[
W_{\mathrm{token,total}}=n
\]

因此，只要原始异构 batch 和等价同构 batch 的 batch size 相同，非 Attention token 工作量就已经自动匹配。

### 3.2 Attention 工作量

在一步 Decode 中，每条请求只有 1 个 query token。该 query 需要对请求的整个历史 KV cache 执行 Attention，因此有效 query-key 对数量近似为：

\[
A_{\mathrm{decode}}(L_i)=L_i
\]

如果计数约定包含当前 token，则为：

\[
A_{\mathrm{decode}}(L_i)=L_i+1
\]

两者对等价 ISL 的推导结果相同。

对于固定模型结构、head 数、head dimension 和 KV 数据类型：

- QK 和 probability-V 的计算量近似正比于 \(L_i\)；
- KV cache 读取字节数近似正比于 \(L_i\)；
- Decode 常常具有明显的内存带宽敏感性。

所以，可以使用当前 KV length 作为单步 Decode 的一阶长度相关工作量指标。

整个异构 batch 的 Attention 总工作量为：

\[
A_{\mathrm{total}}
=\sum_{i=0}^{n-1}L_i
\]

## 4. 构造等价同构 Batch

为了使用已有的同构 \((L,b)\) 性能表，将异构 batch 映射为：

\[
(L_{\mathrm{eq}},b=n)
\]

要求等价同构 batch 与异构 batch 具有相同的 KV Attention 总工作量。

### 4.1 两条请求

对于：

\[
L_0,L_1
\]

原始异构 batch 的总工作量为：

\[
A_{\mathrm{total}}=L_0+L_1
\]

等价同构 batch 包含两条长度均为 \(L_{\mathrm{eq}}\) 的请求，因此要求：

\[
2L_{\mathrm{eq}}=L_0+L_1
\]

解得：

\[
\boxed{
L_{\mathrm{eq}}
=\frac{L_0+L_1}{2}
}
\]

如果两条请求刚刚完成 Prefill，并且同构表使用初始 ISL 作为横坐标，则：

\[
\boxed{
ISL_{\mathrm{eq}}
=\frac{ISL_0+ISL_1}{2}
}
\]

原始异构 batch：

\[
\{L_0,L_1\}
\]

被映射为同构测量空间中的：

\[
\boxed{
(L_{\mathrm{eq}},b=2)
}
\]

### 4.2 推广到任意数量的请求

对于包含 \(n\) 条请求的 Decode batch：

\[
\mathcal B=\{L_i\}_{i=0}^{n-1}
\]

要求：

\[
nL_{\mathrm{eq}}
=\sum_{i=0}^{n-1}L_i
\]

因此：

\[
\boxed{
L_{\mathrm{eq}}
=\frac{1}{n}\sum_{i=0}^{n-1}L_i
}
\]

也就是说，普通单 token Decode 的一阶工作量等价长度就是各请求当前 KV length 的算术平均值。

这和 Prefill 不同：Prefill 的 query length 可能大于 1，Attention 工作量包含关于 query length 的二次项；Decode 中每条请求的 query length 固定为 1，因此长度相关 Attention 工作量对 \(L_i\) 近似线性。

## 5. 利用同构性能表估计单步 Decode 时间

对于包含 \(n\) 条请求的 batch，记已有的 batch-size-\(n\) 单步 Decode latency 曲线为：

\[
D_n(L)=D(L,b=n)
\]

在计算出 \(L_{\mathrm{eq}}\) 后，通过一维插值得到异构 batch 的单步 Decode 时间：

\[
\boxed{
\widehat D_{\mathrm{mixed}}
=\operatorname{Interp}
\left(D_n,L_{\mathrm{eq}}\right)
}
\]

推荐：

- 优先在原始长度 \(L\) 上使用局部线性插值，因为一阶 KV 读取工作量对 \(L\) 近似线性；
- 如果测量曲线在某些长度处发生 kernel、Split-K 或 partition 策略切换，应在切换边界处分段，避免跨边界做高阶插值；
- 如果 \(L_{\mathrm{eq}}\) 落在测量范围之外，应标记为外推结果；
- 如果性能表中没有对应的 batch size，则还需要在 batch-size 方向建立插值或并行效率模型，不能直接用其他 batch size 的 latency 替代。

## 6. ITL 和 Throughput 估计

### 6.1 ITL

如果 \(n\) 条请求被放入同一个 Decode forward，并且不包含排队、抢占和额外调度等待，则它们共享同一个 batch step latency：

\[
\boxed{
\widehat{ITL}_i
\approx
\widehat D_{\mathrm{mixed}}
\qquad(0\le i<n)
}
\]

严格来说，ITL 是相邻两个输出 token 的到达时间间隔。这里使用同构 ITL 曲线估计的前提是，该指标能够代表一次 Decode step 的服务时间。若测量 ITL 包含调度等待，则还需要单独建立调度模型。

### 6.2 Output-token Throughput

本轮每条请求生成 1 个 token，整个 batch 共生成 \(n\) 个 token，因此：

\[
\boxed{
\widehat{TPS}
=\frac{n}{\widehat D_{\mathrm{mixed}}}
}
\]

如果已有同构表只保存 throughput，可以先转换成单步时间：

\[
D_n(L)=\frac{n}{TPS_n(L)}
\]

然后对时间进行插值，最后重新计算异构 batch 的 throughput。不要直接平均不同 ISL 下的 throughput，因为吞吐率不是可加的工作量。

## 7. 数值示例

考虑两条刚刚完成 Prefill 的请求：

\[
ISL_0=1024,qquad ISL_1=4096
\]

两条请求尚未生成其他输出 token，因此：

\[
L_0=1024,qquad L_1=4096
\]

等价 ISL 为：

\[
ISL_{\mathrm{eq}}
=\frac{1024+4096}{2}
=2560
\]

所以原始异构 batch 可映射为：

\[
\boxed{
(ISL=2560,b=2)
}
\]

假设同构表中只有：

\[
D_2(2048),\qquad D_2(3072)
\]

则可以进行局部线性插值：

\[
\widehat D_{\mathrm{mixed}}
=D_2(2048)
+\frac{2560-2048}{3072-2048}
\left[D_2(3072)-D_2(2048)\right]
\]

在这个例子中，2560 正好位于 2048 和 3072 的中点，因此：

\[
\widehat D_{\mathrm{mixed}}
=\frac{D_2(2048)+D_2(3072)}{2}
\]

吞吐估计为：

\[
\widehat{TPS}
=\frac{2}{\widehat D_{\mathrm{mixed}}}
\]

## 8. 参考实现

```python
from collections.abc import Sequence


def current_kv_length(
    initial_sequence_length: int,
    decoded_tokens: int = 0,
) -> int:
    """返回本轮 Decode 开始前的当前 KV cache length。"""
    if initial_sequence_length < 0 or decoded_tokens < 0:
        raise ValueError("lengths must be non-negative")
    return initial_sequence_length + decoded_tokens


def equivalent_decode_length(
    kv_lengths: Sequence[float],
) -> tuple[float, int]:
    """将 ragged/paged Decode batch 映射到等价同构 batch。"""
    batch_size = len(kv_lengths)
    if batch_size == 0:
        raise ValueError("kv_lengths must not be empty")
    if any(length < 0 for length in kv_lengths):
        raise ValueError("KV lengths must be non-negative")

    equivalent_length = sum(kv_lengths) / batch_size
    return equivalent_length, batch_size


kv_lengths = [
    current_kv_length(1024),
    current_kv_length(4096),
]

length_eq, batch_size = equivalent_decode_length(kv_lengths)

# 用户根据自己的同构性能表实现该插值函数。
decode_step_time = interpolate_decode_time(
    sequence_length=length_eq,
    batch_size=batch_size,
)

output_token_throughput = batch_size / decode_step_time
```

## 9. 仅靠同构数据的不可识别性

工作量等价法选择了“KV 工作量可加、ragged/paged kernel 能够较好地进行跨请求负载均衡”这一假设。但仅靠同构数据，无法判断异构 batch 的性能究竟由总长度还是最大长度主导。

考虑两个性能模型。

总工作量模型：

\[
T_{\mathrm{sum}}(L_0,L_1)
=\alpha+\beta(L_0+L_1)
\]

最大长度模型：

\[
T_{\mathrm{max}}(L_0,L_1)
=\alpha+2\beta\max(L_0,L_1)
\]

当 batch 同构，即 \(L_0=L_1=L\) 时，两者完全相同：

\[
T(L,L)=\alpha+2\beta L
\]

但对于异构 batch，两者会给出不同预测。因此，无论测量多少个同构 \((L,b)\) 点，都无法从中唯一学习长度不均衡造成的性能变化。

这说明：

- \(L_{\mathrm{eq}}=\operatorname{mean}(L_i)\) 是合理的一阶基线；
- 要建模真实 kernel 的长度不均衡效应，仍然需要少量异构校准数据。

## 10. 长度不均衡校准

### 10.1 基础模型

首先使用同构性能曲线得到：

\[
D_{\mathrm{base}}
=D_n(L_{\mathrm{eq}})
\]

然后建立异构修正：

\[
\boxed{
D_{\mathrm{real}}
=D_{\mathrm{base}}
\exp\left(f_n(z_{\mathrm{imbalance}})\right)
}
\]

要求：

\[
f_n(0)=0
\]

从而在 batch 完全同构时：

\[
D_{\mathrm{real}}=D_{\mathrm{base}}
\]

### 10.2 不均衡特征

对于两条请求，可以使用：

\[
\delta_L
=\frac{|L_0-L_1|}{L_0+L_1}
\]

其中 \(0\le\delta_L\le1\)。完全同构时 \(\delta_L=0\)。

对于任意 batch size，可以使用：

\[
CV_L
=\frac{\operatorname{std}(L_i)}
{\operatorname{mean}(L_i)}
\]

\[
r_{\max}
=\frac{\max_iL_i}{L_{\mathrm{eq}}}
\]

如果 KV cache 使用 page/block size \(P\)，还可以定义：

\[
N_i=\left\lceil\frac{L_i}{P}\right\rceil
\]

\[
CV_{\mathrm{block}}
=\frac{\operatorname{std}(N_i)}
{\operatorname{mean}(N_i)}
\]

block 级特征往往比连续 token 长度更容易反映实际任务划分、分页元数据和尾部浪费。

### 10.3 简单修正模型

两请求场景可以从以下模型开始：

\[
\log
\frac{D_{\mathrm{real}}}{D_{\mathrm{base}}}
=\gamma_n(L_{\mathrm{eq}})\delta_L^2
\]

也就是：

\[
\boxed{
D_{\mathrm{real}}
=D_{\mathrm{base}}
\exp\left(
\gamma_n(L_{\mathrm{eq}})\delta_L^2
\right)
}
\]

对于任意 batch size，可以使用：

\[
\log
\frac{D_{\mathrm{real}}}{D_{\mathrm{base}}}
=\beta_1CV_L^2
+\beta_2\log r_{\max}
+\beta_3CV_{\mathrm{block}}^2
\]

采用平方项可以保证交换请求顺序不会改变预测结果，并且同构时修正自然退化为 0。

这些系数不一定为正。特定 kernel 的 Split-K、partition 或负载均衡策略可能使某些异构 shape 出现局部性能改善，因此应通过实际测量拟合，而不是预设“不均衡一定降低性能”。

### 10.4 最小异构校准集

不需要测量所有异构 batch。可以在若干高频 \(L_{\mathrm{eq}}\) 区域中分别选择：

- 低不均衡：\(L_0\approx L_1\)；
- 中不均衡；
- 高不均衡：一条短 context、一条长 context；
- 位于 KV block、Split-K 或 partition 切换边界附近的组合。

对每个校准 batch 计算残差：

\[
R
=\log
\frac{D_{\mathrm{measured}}}
{D_n(L_{\mathrm{eq}})}
\]

然后使用 \(\delta_L\)、\(CV_L\)、\(r_{\max}\) 和 block 级特征拟合 \(R\)。如果验证结果显示该残差很小且没有系统性趋势，可以暂时令修正项为 1。

## 11. 没有异构校准数据时

在没有异构校准数据时，建议至少同时计算两个场景。

工作量等价估计：

\[
D_{\mathrm{work}}
=D_n\left(\operatorname{mean}(L_i)\right)
\]

最大长度场景：

\[
D_{\mathrm{max\text{-}shape}}
=D_n\left(\max_iL_i\right)
\]

后者可以视为 padding 或最大 shape 主导的保守场景，但二者不是严格的数学上下界。

同时可以报告：

\[
\rho_L
=\frac{\max_iL_i}{L_{\mathrm{eq}}}
\]

当 \(\rho_L\) 接近 1 时，工作量等价估计通常更可信；当它很大时，应降低预测置信度或补测异构样本。

## 12. 单步测量与多步平均的区别

该方法要求已有性能数据能够表示指定长度位置上的单步 Decode 时间。

如果 throughput 是连续生成 \(O\) 个 token 时测得的，那么 KV length 会从 \(ISL\) 增长到 \(ISL+O-1\)。此时测得的平均 throughput 实际为：

\[
TPS_n(ISL,O)
=\frac{nO}
{\sum_{t=0}^{O-1}D_n(ISL+t)}
\]

它不能直接看作 \(ISL\) 位置的一步性能。

要估计当前单步 Decode，优先使用：

- 每个生成位置分别记录的 ITL；
- 单步 Decode microbenchmark；
- 或者明确记录测量窗口内每一步的当前 KV length。

如果只能获得多步平均 throughput，还必须知道固定的 output length \(O\)，并对整个增长过程进行建模，不能直接将平均 throughput 赋给初始 ISL。

## 13. 误差来源与适用边界

### 13.1 主要误差来源

- 不同 KV length 造成的 GPU 任务负载不均衡；
- KV page/block 数量及最后一个 block 的填充程度；
- Split-K、partition size 或 kernel 实现的切换；
- batch size 较小时长 context 导致的 SM 利用率变化；
- CUDA Graph、workspace 和调度元数据的固定开销；
- KV cache 数据类型、GQA/MQA 结构和 head dimension；
- tensor parallel、context parallel 和通信开销；
- ITL 中包含的排队、抢占和调度等待。

### 13.2 适用场景

该方法适用于：

- 每条请求本轮执行普通单 token Decode；
- Prefill 已完成，历史 token 的 KV cache 已存在；
- 使用支持变长 KV cache 的 ragged、paged 或 varlen Decode kernel；
- 已有同一模型、硬件、并行配置、KV dtype 和执行引擎下的同构性能数据；
- 需要在缺少全面异构测量数据时获得第一版工程估计。

### 13.3 不应直接使用的场景

- 执行引擎完整 padding 到 batch 最大 ISL；
- speculative decoding，每条请求一次处理多个 draft token；
- 不同请求本轮处理的 query token 数不同；
- sliding-window Attention，但仍直接使用未截断的完整 ISL；
- prefix sharing 或 KV cache 压缩导致实际读取量不再与 ISL 成正比；
- batch 中混合 Prefill 和 Decode 请求；
- 使用 Mamba、线性 Attention 或其他上下文复杂度不同的模型结构。

对于 sliding-window Attention，应使用有效 KV 长度：

\[
L_i^{\mathrm{effective}}
=\min(L_i,W)
\]

其中 \(W\) 为滑动窗口大小，然后再计算等价长度。

## 14. 总结

对于包含 \(n\) 条请求的普通单 token ragged/paged Decode batch，首先使用每条请求当前的 KV cache length：

\[
L_i=ISL_i+O_i
\]

构造等价同构长度：

\[
\boxed{
L_{\mathrm{eq}}
=\frac{1}{n}\sum_{i=0}^{n-1}L_i
}
\]

然后在相同 batch size 的同构性能曲线上插值：

\[
\boxed{
\widehat D_{\mathrm{mixed}}
=\operatorname{Interp}
\left(D(L,b=n),L_{\mathrm{eq}}\right)
}
\]

对应的 output-token throughput 为：

\[
\boxed{
\widehat{TPS}
=\frac{n}{\widehat D_{\mathrm{mixed}}}
}
\]

该映射保持了非 Attention token 工作量和 KV Attention 总工作量，是估计异构 Decode batch 性能的合理一阶基线。但同构数据无法识别长度不均衡造成的 kernel 效率变化，因此对于高精度模型，还需要使用少量异构 batch 拟合校准项。

## 参考资料

- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Zihao Ye et al., [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
- [vLLM PagedAttention kernel design](https://docs.vllm.ai/en/v0.8.0/design/kernel/paged_attention.html)
- [FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html)
