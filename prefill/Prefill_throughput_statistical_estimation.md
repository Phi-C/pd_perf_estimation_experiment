# Agent 任务下 Prefill 节点的理想吞吐估计

目标：用 trace / 数据集里真实出现过的 request 形状，估计 Prefill 节点在**只跑这些形状、且每个形状都测到稳定吞吐**时的理想吞吐。做法是把形状映射到二维网格上，只测覆盖足够多 request 的格子，再按格子里的 request 数加权。

下文用 $c$ 表示 `context_length`（已在 KV cache 中的上下文），$q$ 表示 `query_length`（本步新算的 token 数）。一条 request 对应平面上一个点 $(c, q)$。

## 1. 分箱

对全部 $N$ 条 request 的 $(c, q)$ 做等距网格：

- $c$ 方向每 $8192$ 一格（与 `max_num_batched_tokens` 对齐）
- $q$ 方向每 $1024$ 一格（与 `block_size × dcp_world_size` 对齐）

格子 $(i, j)$ 覆盖

$$
c \in [8192\,i,\ 8192\,(i+1)),
\qquad
q \in [1024\,j,\ 1024\,(j+1))
$$

任意 request 恰好落入一格。记 $N_{ij}$ 为落入 $(i, j)$ 的点数，则 $\sum_{i,j} N_{ij} = N$。

## 2. 代表点

空格子丢掉。非空格子用格内点的**取整均值**当测量坐标 $(c_i, q_j)$，避免用格子边角去测一个实际很少出现的形状：

$$
c_i = \mathrm{round}\left(\frac{1}{N_{ij}}\sum_{n=1}^{N_{ij}} c_n\right),
\qquad
q_j = \mathrm{round}\left(\frac{1}{N_{ij}}\sum_{n=1}^{N_{ij}} q_n\right)
$$

$c_n$、$q_n$ 是该格内第 $n$ 个点的 `context_length`、`query_length`。

## 3. 按覆盖率选格并赋权

按 $N_{ij}$ **从高到低**选格子，累加点数，直到首次满足

$$
\sum_{(i,j)\in S} N_{ij} \ge \alpha N
$$

$S$ 是入选集合，$\alpha$ 是覆盖率阈值（例如 $0.8$）。排在后面的稀有形状不测。

权重只在入选格子上归一化（入选格子的权重和为 $1$）：

$$
N_S = \sum_{(i,j)\in S} N_{ij},
\qquad
w_{ij} = \frac{N_{ij}}{N_S}
\quad (i,j)\in S
$$

## 4. 定点实测，再加权

对每个 $(i,j)\in S$，固定形状 $(c_i, q_j)$ 测得吞吐 $T_{ij}$（tok/s）。理想吞吐取入选格子的加权平均：

$$
\hat{T} = \sum_{(i,j)\in S} w_{ij}\, T_{ij}
$$

$w_{ij}$ 是该形状在入选集合里出现的频繁程度，$T_{ij}$ 是这个形状单独能跑到的吞吐。

* 测 $T_{ij}$ 时沿用 `run_aiperf_workload_shape.sh` 的 prefix/fresh 拆分：`--isl = c_i + q_j`，cached prefix 长度 $= c_i$，fresh $= q_j$。把 `--osl` 设为 1（并 `min_tokens=max_tokens=1`），把 decode 压到最短，使 TTFT 和 input tok/s 反映的是 prefill，而不是脚本默认 OSL=1000 那种 decode SLA 测量
* 测 $T_{ij}$ 时要在aiperf侧指定`concurrency`
