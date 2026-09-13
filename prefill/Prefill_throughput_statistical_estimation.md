# Prefill：按请求形状分布估计性能

这条路线用少量代表点描述给定并发下的 serving 性能。它适用于缺少完整 batch trace、但知道缓存前缀与新增输入分布的情况。先阅读[统计口径](../docs/metrics.md)，需要按实际 forward 定价时使用[工作量等价法](ragged_packed_prefill_workload_equivalence.md)。

## 1. 确定样本粒度

每个样本记为 $(c,q)$：`c` 是已缓存的上下文，`q` 是本次新增输入。

- 请求级数据：一条请求一个样本，权重表示请求占比。
- Chunk 日志：一条 `(forward, request)` 记录一个样本，权重表示 chunk 出现频率。同一请求可能出现多次。

当前 [prefill_grid_alpha.py](../scripts/prefill_grid_alpha.py) 使用第二种粒度。它不能直接产出请求级分布，也没有保留完整 batch 的 `b`。

## 2. 分箱、取代表点、选覆盖率

设两个网格跨度为 $\Delta c,\Delta q$：

$$
i=\lfloor c/\Delta c\rfloor,\qquad j=\lfloor q/\Delta q\rfloor
$$

K3 示例取 `Δc = 8192`、`Δq = 1024`。这些是采样网格参数，换模型和长度范围后可以调整。

每个非空格子统计样本数 $N_{ij}$，用格内 `c`、`q` 的取整均值作为代表点。按 $N_{ij}$ 降序选择格子，直到累计覆盖率首次达到 $\alpha$，再在入选集合 $S$ 内归一化：

$$
\sum_{(i,j)\in S}N_{ij}\ge\alpha N,\qquad
w_{ij}=\frac{N_{ij}}{\sum_{(u,v)\in S}N_{uv}}
$$

未选中的长尾要报告占比。若代表点不满足 cache block、checkpoint 或 token budget 约束，应投影到可测坐标，并记录目标值与实际值的偏差。

```bash
python3 scripts/prefill_grid_alpha.py \
  -i prefill/prefill_forwards_profiling.csv \
  -o out/prefill_grid --alpha 0.8 --cbin 8192 --qbin 1024
```

输出 `out/prefill_grid_allcells.csv` 和 `out/prefill_grid_selected.csv`；这里的覆盖率按 `(forward, request)` 记录数计算。

## 3. 固定并发实测

所有点使用相同的模型、硬件、并行配置和客户端并发 $C$。构造 cached prefix `c` 与 fresh input `q`，设置 `OSL = 1`，并回读引擎记录核对实际形状。具体方法见[Prefill 测点](measure_prefill_performance_point.md)。

完整输入吞吐使用 `c + q` 作 token 口径；fresh 吞吐只数 `q`。前缀命中率和 chat template 会影响实际长度，不能仅依据命令行参数认定测点有效。`concurrency = C` 也不能保证每次 forward 的 `b = C`。

## 4. 汇总

形状加权性能为：

$$
\overline T_{\mathrm{shape}}(C)=\sum_{(i,j)\in S}w_{ij}T_{ij}(C)
$$

报告时写明它是入选形状的加权均值。若要估计处理完整请求集合的吞吐，应按[工作量 / 时间公式](../docs/metrics.md#4-按请求形状加权时的区别)汇总，并验证混合调度的影响。Chunk 级分布不能直接代入每请求的 P/D 容量配平。
