
# Ragged/Packed Batch 的 Prefill 性能估计：工作量等价法

## 1. 问题定义

已有一组同构 batch 的 Prefill 性能数据，每个测量点由以下三元组描述：

\[
(q,c,b)
\]

其中：

- \(q\)：每条请求的 query length，即本次需要计算的新 token 数；
- \(c\)：每条请求的 context length，即已经存在于 KV cache 中的上下文 token 数；
- \(b\)：batch size；
- 同一个测量点中的所有请求具有相同的 \(q\) 和 \(c\)。

现在需要估计一个包含两条异构请求的 batch：

\[
(q_0,c_0),\qquad(q_1,c_1)
\]

目标是利用已有的同构测量数据，估计该 batch 的 Prefill 执行时间、TTFT 和 throughput。

本文只考虑使用 **ragged/packed batch** 的执行引擎：不同请求的有效 token 被紧凑排列，Attention kernel 只处理真实 token，不把所有请求 padding 到统一的最大长度。

## 2. 核心假设

工作量等价法建立在以下假设上：

1. context token 已经存在于 KV cache 中，本次 Prefill 只对 \(q_i\) 个新 token 执行模型计算。
2. 非 Attention 部分，例如 QKV projection、output projection、MLP 和 normalization，其主要工作量近似正比于新 token 总数。
3. Attention 部分的主要工作量近似正比于实际计算的 query-key 对数量。
4. ragged/packed kernel 不会因为长度不同而产生大规模 padding 计算。
5. 两个具有相同“新 token 总量”和“Attention 总工作量”的 batch，其 Prefill 时间近似相同。

该方法是基于工作量匹配的近似模型。它不能完全表达 GPU 利用率、kernel 形状、调度开销和长度不均衡带来的效率变化。

## 3. 单条请求的工作量

### 3.1 非 Attention 工作量

对一条请求 \((q,c)\)，非 Attention 模块只处理本次新增的 \(q\) 个 token，因此可使用下式作为其工作量指标：

\[
W_{\mathrm{token}}(q,c)=q
\]

对于两条请求组成的 batch，新 token 总量为：

\[
Q=q_0+q_1
\]

### 3.2 Attention 工作量

对于 causal Attention：

- 每个新 token 都需要关注已有的 \(c\) 个 context token；
- 第一个新 token 关注自身，第二个新 token 关注前两个新 token，依此类推。

因此，一条请求的有效 query-key 对数量为：

\[
A(q,c)=qc+\frac{q(q+1)}{2}
\]

其中：

- \(qc\) 表示新 token 对已有 context KV 的 Attention；
- \(q(q+1)/2\) 表示新 token 之间的 causal self-attention。

两条异构请求的 Attention 总工作量为：

\[
A_{\mathrm{total}}
=A(q_0,c_0)+A(q_1,c_1)
\]

展开后为：

\[
A_{\mathrm{total}}
=q_0c_0+\frac{q_0(q_0+1)}{2}
+q_1c_1+\frac{q_1(q_1+1)}{2}
\]

## 4. 构造等价同构 Batch

为了使用已有的 \((q,c,b=2)\) 性能表，需要把异构 batch 映射成一个等价同构 batch：

\[
(q_{\mathrm{eq}},c_{\mathrm{eq}},2)
\]

要求该等价 batch 同时满足：

1. 新 token 总量相同；
2. Attention 总工作量相同。

### 4.1 匹配新 Token 总量

等价 batch 中有两条相同请求，因此：

\[
2q_{\mathrm{eq}}=q_0+q_1
\]

从而：

\[
\boxed{
q_{\mathrm{eq}}=\frac{q_0+q_1}{2}
}
\]

### 4.2 匹配 Attention 总工作量

要求：

\[
2A(q_{\mathrm{eq}},c_{\mathrm{eq}})
=A_{\mathrm{total}}
\]

代入 Attention 工作量公式：

\[
2\left(
q_{\mathrm{eq}}c_{\mathrm{eq}}
+\frac{q_{\mathrm{eq}}(q_{\mathrm{eq}}+1)}{2}
\right)
=A_{\mathrm{total}}
\]

解得：

\[
\boxed{
c_{\mathrm{eq}}
=\frac{A_{\mathrm{total}}}{q_0+q_1}
-\frac{q_{\mathrm{eq}}+1}{2}
}
\]

最终，异构 batch：

\[
\{(q_0,c_0),(q_1,c_1)\}
\]

被映射为同构测量空间中的：

\[
\boxed{
(q_{\mathrm{eq}},c_{\mathrm{eq}},b=2)
}
\]

这个映射不是简单地分别对 query length 和 context length 求算术平均。它保留了更重要的两个计算量指标：新 token 总数和有效 Attention 对总数。

### 4.3 推广到三条请求

如果 ragged/packed batch 中有三条请求：

\[
(q_0,c_0),\qquad(q_1,c_1),\qquad(q_2,c_2)
\]

新 token 总数为：

\[
Q=q_0+q_1+q_2
\]

Attention 总工作量为：

\[
A_{\mathrm{total}}
=\sum_{i=0}^{2}
\left(
q_ic_i+\frac{q_i(q_i+1)}{2}
\right)
\]

将其映射为等价同构 batch：

\[
(q_{\mathrm{eq}},c_{\mathrm{eq}},b=3)
\]

匹配新 token 总量可得：

\[
\boxed{
q_{\mathrm{eq}}=\frac{q_0+q_1+q_2}{3}
}
\]

匹配 Attention 总工作量：

\[
3A(q_{\mathrm{eq}},c_{\mathrm{eq}})
=A_{\mathrm{total}}
\]

解得：

\[
\boxed{
c_{\mathrm{eq}}
=\frac{A_{\mathrm{total}}}{q_0+q_1+q_2}
-\frac{q_{\mathrm{eq}}+1}{2}
}
\]

随后在已有的 \(b=3\) 同构性能表中，对 \((q_{\mathrm{eq}},c_{\mathrm{eq}})\) 进行二维插值。

### 4.4 推广到任意数量的请求

对于包含 \(n\) 条请求的 ragged/packed batch：

\[
\mathcal{B}=\{(q_i,c_i)\}_{i=0}^{n-1}
\]

定义：

\[
Q=\sum_{i=0}^{n-1}q_i
\]

\[
A_{\mathrm{total}}
=\sum_{i=0}^{n-1}
\left(
q_ic_i+\frac{q_i(q_i+1)}{2}
\right)
\]

等价同构点为：

\[
\boxed{
q_{\mathrm{eq}}=\frac{Q}{n}
}
\]

\[
\boxed{
c_{\mathrm{eq}}
=\frac{A_{\mathrm{total}}}{Q}
-\frac{q_{\mathrm{eq}}+1}{2}
}
\]

因此，原始异构 batch 被映射为：

\[
\boxed{
(q_{\mathrm{eq}},c_{\mathrm{eq}},b=n)
}
\]

等价 context length 还可以改写为：

\[
\boxed{
c_{\mathrm{eq}}
=\frac{\sum_i q_ic_i}{Q}
+\frac{\sum_i(q_i-q_{\mathrm{eq}})^2}{2Q}
}
\]

其中，第一项是按 query length 加权的 context length；第二项是 query length 不均衡产生的额外 causal self-attention 工作量。query length 差异越大，第二项通常越大。这也说明 \(c_{\mathrm{eq}}\) 一般不等于各请求 context length 的算术平均值。

## 5. 利用同构性能表估计 Prefill 时间

对于包含 \(n\) 条请求的 batch，记已有的 batch-size-\(n\) Prefill 时间曲面为：

\[
T_n(q,c)=T(q,c,b=n)
\]

在计算出 \((q_{\mathrm{eq}},c_{\mathrm{eq}})\) 后，通过二维插值得到异构 batch 的 Prefill 时间：

\[
\boxed{
\widehat T_{\mathrm{prefill}}
=\operatorname{Interp}
\left(T_n,q_{\mathrm{eq}},c_{\mathrm{eq}}\right)
}
\]

三条请求时使用 \(T_3(q,c)\)，两条请求时使用 \(T_2(q,c)\)。如果性能表中没有对应的 batch size，则还需要在 batch-size 方向建立插值或并行效率模型，不能直接用 \(b=2\) 的时间代替 \(b=3\)。

推荐使用以下插值方式：

- 如果测量点形成规则网格，使用双线性插值；
- 如果测量点是不规则分布，使用基于三角剖分的线性插值；
- 如果长度跨度很大，例如从几十到数万 token，可以在 \(\log q\)、\(\log c\) 坐标上插值；
- 应优先采用局部、单调的插值方法，避免高阶多项式在测量点之间产生不合理振荡；
- 如果等价点落在已有测量区域之外，应标记为外推结果，并降低预测置信度。

## 6. Throughput 估计

假设 throughput 的定义是本次 Prefill 处理的 query token 数除以 batch 执行时间，则异构 batch 的 throughput 为：

\[
\boxed{
\widehat{\mathrm{TPS}}
=\frac{\sum_{i=0}^{n-1}q_i}{\widehat T_{\mathrm{prefill}}}
}
\]

如果已有表只保存了同构 batch 的 throughput，可以先将其转换成时间：

\[
T(q,c,b)=\frac{bq}{\mathrm{TPS}(q,c,b)}
\]

然后对时间进行插值，最后再计算异构 batch 的 throughput。不要直接对 throughput 做算术平均，因为吞吐率不是可加的工作量。

如果系统使用其他 throughput 定义，例如 request/s 或 total-input-token/s，则只需保持分子定义一致：

\[
\mathrm{request/s}=\frac{n}{\widehat T_{\mathrm{prefill}}}
\]

## 7. TTFT 估计

如果测量数据中的 TTFT 近似等于 batch 的模型执行时间，并且 batch 中的 \(n\) 条请求同时进入一个 non-chunked Prefill forward，则可直接在对应 batch-size 的 TTFT 曲面上使用同样的等价映射：

\[
\widehat{\mathrm{TTFT}}_{\mathrm{batch}}
=\operatorname{Interp}
\left(
\mathrm{TTFT}_n,
q_{\mathrm{eq}},
c_{\mathrm{eq}}
\right)
\]

由于这些请求在同一次 batch forward 返回，通常可以近似认为：

\[
\widehat{\mathrm{TTFT}}_i
\approx
\widehat{\mathrm{TTFT}}_{\mathrm{batch}}
\qquad(0\le i<n)
\]

端到端 TTFT 还可能包含：

\[
\mathrm{TTFT}_i
=T_{\mathrm{queue},i}
+T_{\mathrm{schedule}}
+T_{\mathrm{prefill}}
+T_{\mathrm{first\ decode}}
+T_{\mathrm{communication}}
\]

工作量等价法主要估计其中与 batch 形状相关的 Prefill 执行部分。排队时间、调度等待和通信时间不能由 \((q,c,b)\) 单独确定。

如果系统启用了 chunked prefill，不同请求可能在不同的 chunk 完成。此时每条请求的 TTFT 还取决于 chunk size、执行次序以及 decode 插入策略，不能只通过一个等价同构点准确估计。

## 8. 数值示例

考虑以下两个请求：

\[
(q_0,c_0)=(128,1024)
\]

\[
(q_1,c_1)=(512,256)
\]

### 8.1 新 Token 总量

\[
Q=128+512=640
\]

因此：

\[
q_{\mathrm{eq}}=\frac{640}{2}=320
\]

### 8.2 Attention 总工作量

第一条请求：

\[
A_0
=128\times1024+\frac{128\times129}{2}
=139328
\]

第二条请求：

\[
A_1
=512\times256+\frac{512\times513}{2}
=262400
\]

所以：

\[
A_{\mathrm{total}}=139328+262400=401728
\]

### 8.3 等价 Context Length

\[
c_{\mathrm{eq}}
=\frac{401728}{640}-\frac{320+1}{2}
=467.2
\]

因此，原始异构 batch 可近似映射为：

\[
\boxed{(q,c,b)=(320,467.2,2)}
\]

随后在已有 \(b=2\) 性能表中对 \((320,467.2)\) 做二维插值，即可得到 Prefill 时间和 TTFT，再由总 query token 数 \(640\) 计算吞吐。

这里不能简单使用：

\[
c_{\mathrm{avg}}=\frac{1024+256}{2}=640
\]

因为两条请求的 query length 不同，较长 query 会产生更多 context Attention 访问以及更多 query-query Attention 工作量。

## 9. 参考实现

```python
def attention_work(q: float, c: float) -> float:
    """Causal prefill 中的有效 query-key 对数量。"""
    return q * c + q * (q + 1.0) / 2.0


def equivalent_homogeneous_point(
    requests: list[tuple[float, float]],
) -> tuple[float, float, int]:
    """将任意数量请求的 ragged batch 映射到等价同构 batch。"""
    batch_size = len(requests)
    if batch_size == 0:
        raise ValueError("requests must not be empty")

    total_query_tokens = sum(q for q, _ in requests)
    if total_query_tokens <= 0:
        raise ValueError("total query length must be positive")

    total_attention_work = sum(
        attention_work(q, c)
        for q, c in requests
    )

    q_eq = total_query_tokens / batch_size
    c_eq = (
        total_attention_work / total_query_tokens
        - (q_eq + 1.0) / 2.0
    )

    return q_eq, c_eq, batch_size


requests = [
    (128, 1024),
    (512, 256),
    (256, 512),
]

q_eq, c_eq, batch_size = equivalent_homogeneous_point(requests)

# 用户根据自己的性能表实现 interpolate_prefill_time。
prefill_time = interpolate_prefill_time(
    query_length=q_eq,
    context_length=c_eq,
    batch_size=batch_size,
)

query_token_throughput = (
    sum(q for q, _ in requests) / prefill_time
)
```

## 10. 误差来源与改进方式

工作量完全相同并不意味着实际执行时间严格相同，主要误差来源包括：

- 不同 query length 产生不同的 Attention kernel tile 数量和边界浪费；
- batch 内请求的长度差异可能影响负载均衡和 GPU occupancy；
- GEMM 性能不仅取决于 token 总量，也取决于矩阵形状和对齐方式；
- FlashAttention、PagedAttention 或其他 varlen kernel 的实现细节不同；
- kernel launch、调度和元数据处理包含固定开销；
- tensor parallel、pipeline parallel 和通信开销可能随形状非线性变化；
- prefix cache block 布局和 KV cache 分页可能影响内存访问效率；
- 测量中的 TTFT 可能混入排队、首个 decode step 和通信时间。

为了校准这些误差，可以补测少量异构 batch，并定义长度不均衡特征，例如：

\[
r_q=\frac{\max_i q_i}{q_{\mathrm{eq}}}
\]

以及：

\[
r_L=
\frac{\max_i(c_i+q_i)}
{\left[\sum_i(c_i+q_i)\right]/n}
\]

然后在基础估计上拟合一个校正因子：

\[
T_{\mathrm{corrected}}
=T_{\mathrm{equivalent}}
\times g(r_q,r_L)
\]

少量异构样本应优先覆盖：

- query length 比例为 \(1:2\)、\(1:4\)、\(1:8\) 的组合；
- context length 比例为 \(1:2\)、\(1:4\)、\(1:8\) 的组合；
- 长 query 配短 context、短 query 配长 context 的交叉组合；
- 总 token 工作量接近、但长度分布明显不同的组合。

## 11. 适用边界

该方法适用于：

- context 已经存在于 KV cache；
- 本次只 Prefill 新增 query token；
- 使用支持变长输入的 ragged、packed 或 varlen kernel；
- 已有同一模型、硬件、并行配置和执行引擎下的同构性能数据；
- 需要在缺少异构测量数据时获得第一版工程估计。

该方法不应直接用于以下情况：

- context token 也需要在本次 Prefill 中重新计算；
- 执行引擎把异构请求完整 padding 到最大长度；
- 开启 chunked prefill，但没有纳入 chunk 调度过程；
- batch 中混合了 Prefill 和 Decode 请求；
- 模型包含使计算量不再主要由 token 数和 Attention 对数决定的特殊结构；
- 等价点远离已有测量范围，需要大幅外推。

## 12. 总结

对于包含任意 \(n\) 条请求的 ragged/packed Prefill batch，工作量等价法的核心计算为：

\[
A(q,c)=qc+\frac{q(q+1)}{2}
\]

\[
Q=\sum_{i=0}^{n-1}q_i,qquad
q_{\mathrm{eq}}=\frac{Q}{n}
\]

\[
c_{\mathrm{eq}}
=\frac{\sum_{i=0}^{n-1}A(q_i,c_i)}{Q}
-\frac{q_{\mathrm{eq}}+1}{2}
\]

然后利用已有的同构性能曲面估计：

\[
\widehat T_{\mathrm{prefill}}
=\operatorname{Interp}
\left(T(q,c,n),q_{\mathrm{eq}},c_{\mathrm{eq}}\right)
\]

以及：

\[
\widehat{\mathrm{TPS}}
=\frac{Q}{\widehat T_{\mathrm{prefill}}}
\]

相较于分别平均 query length 和 context length，这种方法同时保持了线性 token 工作量和非线性 Attention 工作量，是从同构测量数据估计 ragged/packed 异构 batch 性能时更合理的基线。

## 参考资料

- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Gyeong-In Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
