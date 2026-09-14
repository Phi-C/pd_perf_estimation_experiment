# Decode 工作量等价法

目标：利用同构 `(L, b)` 的性能表，估计一次异构 ragged / paged Decode batch 的执行时间。这里每条请求每步处理一个 query token，历史 KV 和其它模型状态已准备好。

## 1. 等价长度

第 $i$ 条请求的当前 KV 长度为：

$$
L_i=ISL_i+O_i
$$

$O_i$ 是本轮之前已经完成的 Decode step 数。是否把当前 token 计入 KV 长度，应和测量表采用同一约定；对所有请求统一加 1 不改变以下平均关系。

同构与异构 batch 的 `b` 相同，非 attention 的 token 工作量已匹配。普通全上下文 attention 的 KV 工作量近似正比于 $\sum_iL_i$，因此：

$$
\boxed{L_{\mathrm{eq}}=\frac{1}{b}\sum_iL_i},\qquad
\text{等价点}=(L_{\mathrm{eq}},b)
$$

Prefill 的 query 长度可变，需要额外匹配二次项；Decode 的 query 长度为 1，因此这一阶映射是算术平均。

## 2. 时间表与吞吐

记同一 batch size 的单步时间曲线为 $D_b(L)$：

$$
\widehat D=\operatorname{Interp}(D_b,L_{\mathrm{eq}}),\qquad
TPS_{\mathrm{batch}}=\frac{b}{\widehat D}
$$

若表中存的是吞吐，先用 `D = b / TPS` 转成时间，再插值。多步汇总为 $\sum_fb_f/\sum_fD_f$。

例如两条请求的 KV 长度为 1,024 和 4,096，等价点为 `(2560, 2)`。若时间表仅有 `D₂(2048)` 与 `D₂(3072)`，局部线性插值得到两者的均值，再计算 `2 / D`。

时间表可以采用 GPU forward 或引擎 step period，但整张表和最终报告必须保持同一口径。客户端 ITL 只有在稳定 batch、没有明显排队和流式影响时才近似等于引擎步长。

长度轴优先做局部插值；遇到 kernel / partition 切换时分段。`b` 轴也可能出现台阶，不能默认平滑。测量范围之外应显式标记外推。

## 3. 为什么需要异构校准

同构数据无法唯一决定长度不均衡的影响。以两条请求为例：

$$
D_{\mathrm{sum}}=a+\beta(L_0+L_1),\qquad
D_{\mathrm{max}}=a+2\beta\max(L_0,L_1)
$$

当 $L_0=L_1$ 时两式完全一致，异构时却不同。增加同构测点无法解决这种不可识别性。

可记录不均衡特征：

$$
CV_L=\frac{\operatorname{std}(L_i)}{\operatorname{mean}(L_i)},\qquad
r_{\max}=\frac{\max_iL_i}{L_{\mathrm{eq}}}
$$

若 page 大小为 $P$，还可对 $N_i=\lceil L_i/P\rceil$ 定义 $CV_{\mathrm{block}}$。一个可选的修正模型为：

$$
D_{\mathrm{real}}=D_b(L_{\mathrm{eq}})
\exp\left(\beta_1CV_L^2+\beta_2\log r_{\max}+\beta_3CV_{\mathrm{block}}^2\right)
$$

系数由实测决定，不预设正负。同构时特征为零，修正自动退化为 1。

### 成对校准设计

同一个 `(L_eq, b)` 下保持长度之和不变，测以下形状：

| 形状 | 用途 |
|---|---|
| A：b 条等长请求 | 该点自己的同构基准 |
| B：真实 batch 的长度向量 | 检验实际异构形态 |
| C：一长、多短，对齐 B 的最大长度比 | 改变 CV，保留相近的 r_max |
| D：两长、多短，对齐 B 的 CV | 改变 r_max，保留相近的 CV |

拟合因变量为 `log(D_shape / D_A)`，特征取实际落地长度。仅用一长、多短的样本容易让特征高度相关，应检查矩阵秩、条件数及验证误差。对齐或长度上下限也可能使目标形状无法实现，需要记录实际偏差。

当所有长度恰好是 page 的整数倍时，`CV_block = CV_L`，不能分别识别两个平方项的系数。验证未发现稳定改进时，可在已验证范围内使用未修正基线；这不证明所有异构负载都无额外成本。

没有校准数据时，可同时计算 `D_b(mean L)` 和 `D_b(max L)` 作为敏感性场景。它们不是严格数学上下界，差距也不能直接用 `max L / mean L` 代替。

## 4. 多步测量对应一个区间

连续生成 $O$ 步时，KV 长度会增长：

$$
TPS_b(L_0,O)=\frac{bO}{\sum_{t=0}^{O-1}D_b(L_0+t)}
$$

因此多步平均值不能直接标成起点 `L₀` 的性能。可以逐位置测量，或缩短窗口、记录实际中点，并验证窗口内步长稳定。`O / L` 小仅表示长度漂移小，不能单独保证性能误差同样小。

## 5. 使用边界

该映射需要完整的 batch 成员与当前长度。只有每请求前几步日志时，观测到的 batch 大小是下界，无法恢复稳态等价点。

完整 padding、混合 Prefill / Decode、MTP / speculative decoding、KV 压缩或跨请求共享，会改变工作量关系。Sliding-window attention 应使用有效长度 `min(L, W)`；线性 attention 和混合架构需要实测验证长度模型。

实现与测量分别见 [Decode 流程](SKILL.md)和[Decode 测点方法](measure_decode_performance_point.md)。K3 案例中的波模型与异构校准结论见[实验报告](../experiments/exp1_kimi_k3/Summary.md)。

## 参考

- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [FlashInfer](https://arxiv.org/abs/2501.01005)
- [vLLM PagedAttention kernel design](https://docs.vllm.ai/en/v0.8.0/design/kernel/paged_attention.html)
