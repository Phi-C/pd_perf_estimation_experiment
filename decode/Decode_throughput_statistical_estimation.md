# Decode：按请求形状分布估计性能

这条路线以请求为样本，固定并发测量代表形状的 decode serving 性能。它不需要完整的 running batch trace；有逐步 batch 信息时，可使用[Decode 工作量等价法](ragged_packed_decode_workload_equivalence.md)。

## 1. 用平均上下文描述请求

一条请求对应 $(I,L_{\mathrm{ctx}})$：

$$
I=ISL,\qquad L_{\mathrm{ctx}}\approx ISL+OSL/2
$$

`L_ctx` 是整条生成路径的代表位置，不是首个 Decode step 的实际 KV 长度。输出长度较长或性能沿上下文变化明显时，应分段测量。

## 2. 分箱并选择代表点

示例网格为 `ΔI = 8192`、`ΔL_ctx = 1024`。统计每格请求数 $N_{ij}$，以格内坐标的取整均值作代表点。按请求数降序选格，累计覆盖率达到 $\alpha$ 后停止；入选集合内的权重为：

$$
w_{ij}=\frac{N_{ij}}{N_S},\qquad N_S=\sum_{(i,j)\in S}N_{ij}
$$

保留原始 OSL 分布并检查代表点的输出长度：

$$
O_j=2(L_{\mathrm{ctx},j}-I_i)>0
$$

短输出请求可能因取整得到非法长度，应重新选代表值。`OSL = 1` 的请求没有可用于 ITL 的后续生成区间，应单独统计。

## 3. 在同一并发下测量

每点固定 `(ISL = I_i, OSL = O_j, concurrency = C)`，按[独立 cache-replay 方法](standalone_aiperf_decode_cache_replay.md)准备缓存并重放；只使用测量轮的指标。

| 输出 | 使用条件 |
|---|---|
| AIPerf `output_token_throughput` | 客户端完整窗口口径，包含首 token 边界和 ramp / drain |
| $C/ITL$ | 仅在实际 running batch 稳定为 C、ITL 能代表单步时间时近似成立，ITL 单位为秒 |
| $b/D$ | 引擎侧稳定 batch 吞吐，需要逐步验证 b 并测量 step period |

所有点使用相同的计量方式。记录实际 ISL / OSL、两遍 prompt 一致性、组内请求差异、缓存命中率和成功请求数。AIPerf 的并发参数本身不能证明实际 batch 大小。

## 4. 汇总及范围

$$
\overline T_{\mathrm{shape}}(C)=\sum_{(i,j)\in S}w_{ij}T_{ij}(C)
$$

这是固定并发下的形状加权性能。若估计完成这批请求的吞吐，使用[工作量 / 时间汇总](../docs/metrics.md#4-按请求形状加权时的区别)，其每请求工作量为与指标一致的 `OSL` 或 `OSL − 1`。

Cache-replay 反映本地缓存命中后的 serving 路径。实际 P/D 部署还包含 KV 传输、路由和负载变化，需要端到端验证。当前仓库的 [gen_decode_points.py](../scripts/gen_decode_points.py) 按 membership 区间选 forward 测点，和本文的请求级二维分箱是两条不同流程。
