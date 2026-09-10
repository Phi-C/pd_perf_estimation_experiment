# Agent 任务下 Decode 节点的理想吞吐估计

目标：用 trace / 数据集里真实出现过的 request 形状，估计 Decode 节点在**只跑这些形状、且每个形状都测到稳定吞吐**时的理想吞吐。做法与 prefill 侧相同：把形状映射到二维网格上，只测覆盖足够多 request 的格子，再按格子里的 request 数加权。

下文一条 request 对应平面上一个点 $(I, L_{\mathrm{ctx}})$：

- $I$：`ISL`（input sequence length，decode 开始时 KV 里已有的 prompt token 数）
- $L_{\mathrm{ctx}} = I + O/2$（decode 过程中上下文长度的中点；$O$ 是该请求的 `OSL`）

报的吞吐是 **output tok/s**。

## 1. 分箱

对全部 $N$ 条 request 的 $(I, L_{\mathrm{ctx}})$ 做等距网格：

- $I$ 方向每 $8192$ 一格
- $L_{\mathrm{ctx}}$ 方向每 $1024$ 一格

格子 $(i, j)$ 覆盖

$$
I \in [8192\,i,\ 8192\,(i+1)),
\qquad
L_{\mathrm{ctx}} \in [1024\,j,\ 1024\,(j+1))
$$

任意 request 恰好落入一格。记 $N_{ij}$ 为落入 $(i, j)$ 的点数，则 $\sum_{i,j} N_{ij} = N$。

## 2. 代表点

空格子丢掉。非空格子用格内点的**取整均值**当测量坐标 $(I_i, L_{\mathrm{ctx},j})$，避免用格子边角去测一个实际很少出现的形状：

$$
I_i = \mathrm{round}\left(\frac{1}{N_{ij}}\sum_{n=1}^{N_{ij}} I_n\right),
\qquad
L_{\mathrm{ctx},j} = \mathrm{round}\left(\frac{1}{N_{ij}}\sum_{n=1}^{N_{ij}} L_{\mathrm{ctx},n}\right)
$$

$I_n$、$L_{\mathrm{ctx},n}$ 是该格内第 $n$ 个点的 `ISL`、`L_ctx`。$L_{\mathrm{ctx},j}$ 不要吸到 0（没有上下文就没有 decode）。

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

先**指定一个并发** $C$（`--concurrency` $C$）。所有入选格子都在这个 $C$ 下测，得到的 $\hat{T}$ 才是「该并发下的理想吞吐」；不要把不同 $C$ 的 $T_{ij}$ 混进同一次加权。

对每个 $(i,j)\in S$，固定形状 $(I_i, L_{\mathrm{ctx},j})$，在并发 $C$ 下测得吞吐 $T_{ij}$（output tok/s）。理想吞吐取入选格子的加权平均：

$$
\hat{T}(C) = \sum_{(i,j)\in S} w_{ij}\, T_{ij}(C)
$$

$w_{ij}$ 是该形状在入选集合里出现的频繁程度，$T_{ij}(C)$ 是这个形状在并发 $C$ 下的 decode 吞吐。

由 $L_{\mathrm{ctx}} = I + O/2$ 反推该点要生成的长度 $O_j = 2(L_{\mathrm{ctx},j} - I_i)$。必须 $L_{\mathrm{ctx},j} > I_i$，否则 $O_j$ 无意义（至少为 1）。

测 $T_{ij}(C)$ 按 `standalone_aiperf_decode_cache_replay.md`（两遍 cache-replay），参数对应那份文档里的 `(ISL, OSL, B)`：

```text
--isl          = I_i
--osl          = O_j
--concurrency  = C      （即那份文档里的 B）
```

细节（payload 校验、Prefix Cache 命中、服务端怎么起）仍看那份文档，下面只钉和 $\hat{T}(C)$ 有关的口径。

**$T_{ij}(C)$ 用哪个数。** 优先用 Pass 2 的 ITL（不含 TTFT）：

$$
T_{ij}(C) \approx \frac{C}{\mathrm{ITL}_{ij}(C)}
$$

也可以用 Pass 2 的 `output_throughput_total_tps`（含首 token 边界，略偏 e2e）。不要用 Input tok/s，也不要把 Pass 1（建 cache 的冷 prefill）算进去。

**什么样的点才准入加权。** 未过门槛的点不进入 $\hat{T}(C)$：

- 两遍 payload 一致，成功请求数 $= C$
- Pass 2 Prefix Cache hit 过阈值
- `inputs.json` 里 $C$ 条 prompt **不能有两条完全相同**（整段复制会让两条请求共用同一份 KV，decode 读同一批页，ITL 会偏低）。允许偶尔共享一小段公共前缀——这是 agent 任务的常态（系统提示、工具说明、同一段对话历史）。脚本只校验两遍 payload 一致，不检查组内是否出现完全重复，采数前要打开 `inputs.json` 看一眼。

**测到的是什么。** 这是本机 Prefix Cache 命中后的 Decode-dominant serving 吞吐，不是零 prefill 的纯 kernel，也不是真实 1P1D decode 节点（那份方法禁止 P/D 和 KV-transfer）。客户端 `--concurrency C` 不能保证每步 running batch 都是 $C$，ramp/drain 时 $B_t < C$。
