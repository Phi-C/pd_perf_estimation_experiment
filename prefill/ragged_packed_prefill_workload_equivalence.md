# Prefill 工作量等价法

目标：把一次 ragged / packed Prefill batch 映射到同构性能表中的 `(q, c, b)`，估计该次 forward 的时间。上下文 KV 已准备好，每条请求只新算本步的 query token。

## 1. 等价点的推导

设 batch 有 $b$ 条请求，第 $i$ 条的已有上下文为 $c_i$，本步 query 长度为 $q_i>0$。对普通 causal attention，有效 query-key 对数为：

$$
A(q_i,c_i)=q_ic_i+\frac{q_i(q_i+1)}{2}
$$

保留 batch size，同时匹配新 token 总数与 attention 工作量：

$$
Q=\sum_i q_i,\qquad A_{\mathrm{total}}=\sum_i A(q_i,c_i)
$$

$$
bq_{\mathrm{eq}}=Q,\qquad
bA(q_{\mathrm{eq}},c_{\mathrm{eq}})=A_{\mathrm{total}}
$$

解得：

$$
\boxed{q_{\mathrm{eq}}=Q/b},\qquad
\boxed{c_{\mathrm{eq}}=A_{\mathrm{total}}/Q-(q_{\mathrm{eq}}+1)/2}
$$

也可以写成：

$$
c_{\mathrm{eq}}=\frac{\sum_i q_ic_i}{Q}
+\frac{\sum_i(q_i-q_{\mathrm{eq}})^2}{2Q}
$$

第一项是按 query 加权的 context，第二项补偿 query 长度不均衡带来的 causal attention 工作量。因此 `c_eq` 通常不等于 `mean(c_i)`。

这个映射精确匹配上述两个工作量指标；由它推断执行时间仍是近似。对于 KDA、线性 attention 等混合架构，需要用实测验证这些指标的代表性。

## 2. 查表与汇总

在相同模型、硬件、并行配置的同构时间表 $t(q,c,b)$ 中，对固定 $b$ 的曲面查询：

$$
\widehat t_{\mathrm{fwd}}=\operatorname{Interp}(t(\cdot,\cdot,b),q_{\mathrm{eq}},c_{\mathrm{eq}})
$$

规则网格可用双线性插值，稀疏点可用局部线性模型；测量范围之外标记外推。按 kernel 切换、对齐和 batch size 分段，避免用一条平滑曲线跨越性能突变。

若性能表存的是 fresh tok/s，先转成时间：

$$
t(q,c,b)=\frac{bq}{TPS(q,c,b)}
$$

整个形状序列的吞吐为：

$$
TPS_{\mathrm{fresh}}=\frac{\sum_f Q_f}{\sum_f\widehat t_f}
$$

分子用原始 batch 的 token 数。`final_context_length`、`final_query_length` 是选点后的网格坐标，不能用它们替换原始工作量。ISL 口径另按请求实例去重，见[指标约定](../docs/metrics.md)。

## 3. 数值例子

| 请求 | q | c | $A(q,c)$ |
|---|---:|---:|---:|
| 0 | 128 | 1,024 | 139,328 |
| 1 | 512 | 256 | 262,400 |

这里 $Q=640$，$A_{\mathrm{total}}=401728$，所以等价点是：

$$
(q_{\mathrm{eq}},c_{\mathrm{eq}},b)=(320,467.2,2)
$$

在 `b = 2` 的时间表查询该点，再用 `640 / t` 计算 fresh 吞吐。直接取 `c` 的算术平均 640 会改变 attention 工作量。

实现见 [parse_prefill_log.py](../scripts/parse_prefill_log.py) 中的 `equivalent_point()`；后续脚本优先读取 CSV 的 `eq_*` 列。

## 4. 需要校准的部分

相同工作量可能落到不同 kernel tile、访存布局或并行效率区间。可记录：

$$
r_q=\frac{\max_iq_i}{q_{\mathrm{eq}}},\qquad
r_L=\frac{\max_i(c_i+q_i)}{\operatorname{mean}_i(c_i+q_i)}
$$

对工作量接近、长度分布不同的 batch 做成对测量，再拟合 $t_{\mathrm{corrected}}=t_{\mathrm{equivalent}}g(r_q,r_L)$。优先检查大 batch、长短 query 混合及 prefix 共享方式不同的点。按不均衡程度划分的 high / mid / low 仅是筛选启发式，不是统计置信区间。

## 5. 使用边界

- Chunked prefill 按每个实际 forward 应用映射；一条请求可能跨多个 batch，不能把一个等价点直接当成它的 TTFT。
- TTFT 还包括排队、调度、通信和首输出边界。GPU forward 时间只解释其中一部分。
- 混合 Prefill / Decode、完整 padding、未恢复的 KV / 模型状态，需要另建模型。
- 同构测点的 prefix 共享模式也影响访存。用共享前缀的点表估计独立 KV 的 batch 时，应验证偏差。

完整操作见[Prefill 流程](SKILL.md)和[测点方法](measure_prefill_performance_point.md)。

## 参考

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
