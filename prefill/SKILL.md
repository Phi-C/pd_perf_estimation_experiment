---
name: pd-prefill-trace
description: 从一份 ATOM / vLLM PD-disaggregation 的 prefill node CI log 出发，一路做到待测性能点清单：解析 log 得 prefill_forwards.csv，做工作量等价映射得 (q_eq, c_eq, b)，分桶吸附去重得 final_points_unique.csv，并可选地用 aiperf 把这些点实测出来。同时产出 Perfetto trace 和耗时归因（真正的 GPU 空转 vs. step 本身变慢）。当用户提到 prefill forward trace、perfetto、token budget 利用率、forward 之间的 gap、prefill 性能归因、等价同构 batch、采样点/待测点、ATOM CI job log 分析时使用。
---

# CI log → prefill_forwards.csv → final_points_unique.csv

一条流水线，把一份 prefill node 的 CI log 变成一张**可以照着跑性能测试的待测点清单**。

```
CI job log ──①解析──> prefill_forwards.csv ──①a切warmup──> prefill_forwards_profiling.csv
                            │                                        │
                            ├──①b Perfetto trace + 耗时归因           ├──②分布──> 采样策略
                            │   (保留 warmup, 画分界竖线)              │
                            └────────────────────────────────────────┴──③分桶/吸附/去重
                                                                          │
                                    final_points_unique.csv
                                              │
                                              └──③a diff 上轮已测──> points_todo.csv ──④aiperf──> 实测 TTFT/吞吐
                                                                                            │
                                                                            ⑤ 给 4305 个 forward 逐个定价
                                                                                            │
                                                                              三个吞吐值(实际→忙时→硬件)
```

**②③④ 一律用切完 warmup 的 profiling 段**，只有 ①b 的 trace 保留全量。

每一步的产物都可独立检查，出问题往回退一步即可。

---

## ① 解析 log → `prefill_forwards.csv`

```bash
gh api repos/ROCm/ATOM/actions/jobs/<JOB_ID>/logs > /tmp/job<JOB_ID>.log   # 约 70MB
python3 scripts/parse_prefill_log.py /tmp/job<JOB_ID>.log -o <outdir> --label "<用例名>"
```

**一行 = 一个 (forward, request) 对**，不是一个 forward。本例 7775 行 / 5986 个 forward。
后续任何按 forward 做的统计都必须先按 `forward_idx` 去重（见 ③ 的警告）。

⚠️ 这份 CSV **同时含 warmup 和 profiling 两段**，直接拿去做吞吐统计会错一倍，先做 ①a。

| 列 | 含义 |
|---|---|
| `forward_idx,time` | forward 序号 / 调度时刻（秒级） |
| `req_id,done_tokens,chunk_tokens,ISL,progress_pct` | 该请求本步的 chunk：`done_tokens` 是已在 KV cache 的上下文，`chunk_tokens` 是本步新算的 token |
| `batch_reqs,batch_tokens` | 整个 forward 的规模（每行重复） |
| `eq_batch_size,eq_query_length,eq_context_length` | **等价同构 batch**（每行重复），见下, 具体内容参考`ragged_packed_prefill_workload_equivalence.md` |
| `attn_work_pairs,imbalance_rq,imbalance_rL` | Attention 总工作量与长度不均衡度，判断等价映射可信度 |
| `final_context_length,final_query_length` | ③ 写回的角点坐标（每行重复） |

同时产出 `gap_events.json`、`idle_gaps.json`、`prefill_stat.md`。

自检：脚本会打印 `dup_sched_lines_dropped / instances / arrival_matched / idle_steps`。
`arrival_matched` 掉下来说明 arrival 行格式变了。

### log 里的坑（脚本已处理，换用例时留意）

1. **runner 时间戳不能用**。行首 `2026-08-29T12:21:02.6193014Z` 是日志转发时刻，比引擎晚 12~30s
   且批量刷新，只能定顺序。用 `[atom HH:MM:SS]`（秒级）。
2. **秒级分辨率 + 同一秒最多 6 个 forward**。脚本在秒内均匀平铺，所以 **slice 宽度表示顺序和密度，
   不是真实耗时**。slice 上的 `step_latency_s` 同样**不是单次 forward 的真实耗时**：它是把那一秒的
   总时长按该秒内 forward 个数均分出来的，5986 个 forward 只有 21 个不同取值（1/2、1/3、15/46 秒这种）。
   聚合到秒以上它是守恒的（总和 = 真实墙钟 3485 s），但秒内被抹平——1926 个秒桶含多个 forward，其中
   656 个混着 q 相差 4 倍以上的 forward，摊到一起完全失真。要单次 forward 的真实耗时只能实测
   （见 bench_run/）。log 里还有约 10 处 1 秒回退，脚本 clamp。
3. **log 会重放行**。同一秒里整段（arrival + 若干 scheduled 行）重复出现。判据：同一秒内该 batch 里
   **每一条**请求的 done 都比已处理的进度小 → 丢弃。本例丢了 60 行；不去重会凭空多出 60 个 forward。
4. **req_id 会复用**（warmup → profiling 阶段边界处从 0 重新开始）。按"done 相对当前实例回退"切分实例，
   **不能**简单用 `done == 0` —— LMCache 前缀命中时首块 done 就等于命中长度（本例 2605/2840 命中）。
5. **ISL 以 arrival 行为准**。少数请求不会跑完全部 chunk，只用 `max(done+new)` 会低估。
6. **首次 forward 极慢是 JIT**，不是调度问题：Triton autotune（shape-keyed）+ aiter 首次 dispatch。

### 等价同构 batch（工作量等价法）

依据 `ragged_packed_prefill_workload_equivalence.md`。目的：把一个 ragged/packed 的异构 forward
映射成同构性能表里的**一个点**。用本步的 **q_i = `chunk_tokens`** 和 **c_i = `done_tokens`**，
注意 c_i **不是** ISL：

```
A(q, c) = q·c + q(q+1)/2                  一条请求的有效 query-key 对数
Q       = Σ q_i
q_eq    = Q / n                            匹配"新 token 总量"
c_eq    = Σ A(q_i,c_i) / Q − (q_eq+1)/2    匹配"Attention 总工作量"
        → 等价点 (q_eq, c_eq, b=n)
```

**c_eq 不是各请求 context 的算术平均。** b≥5 的 forward 上两者常差 15~25%
（F843: c_eq=116,239 而算术平均=153,523，低 24%）。等价形式
`c_eq = Σq_i·c_i/Q + Σ(q_i−q_eq)²/(2Q)`：第一项是按 query length 加权的 context，
第二项是 query 长度不均衡带来的额外 causal self-attention。

可信度用 `r_q = max q_i / q_eq`、`r_L = max(q_i+c_i) / mean(q_i+c_i)` 判断（原文 §10）：
`≤1.2` 高、`≤2.5 且 ≤1.8` 中、否则低，写进 slice 的 `eq_confidence`。本例 b=1 占 83%（映射平凡且精确）；
**b≥4 时 r_q 中位数 2.5+，落在"低/中"** —— 拿这些点做估计要按原文补少量异构样本拟合校正因子。

**适用边界**（原文 §11）：本数据开了 chunked prefill，等价点只能估**单个 forward 的执行时间**，
不能直接当成单条请求的 TTFT —— TTFT 还取决于它被切成几个 chunk 以及中间的空转。

---

## ①a ⚠️ 先切掉 warmup —— 不做这步所有吞吐结论都会错一倍

aiperf 跑两个阶段：**Warmup** 和 **Profiling**。CI 指标只统计 Profiling 段，
但 `parse_prefill_log.py` 解析的是整份 log，**产出的 `prefill_forwards.csv` 两段都在里面**。

warmup 跑在**冷 prefix cache** 上，每个 token 都得真算；profiling 段命中率 95.7%。
后果是 warmup 只占 28% 的墙钟，却贡献了 **50%** 的 GPU 实算 token。
拿全 trace 算吞吐 = 把一份 benchmark 的分母摊到两倍的计算量上。

### 分界线从 log 里取，不要反推

```bash
grep -aE "Credit phase start|Initialized [0-9]+ phase" /tmp/job<JOB_ID>.log
```

```
12:20:48.816  Initialized 2 phase(s): ['Warmup', 'Profiling']
12:20:50.405  Credit phase start: warmup      | target: 354 requests
12:43:44.970  Credit phase start: profiling   <- 分界线
```

`t_split = (profiling 起点) − (第一个 forward 的 wall_clock)`。本例 12:43:44.970 − 12:20:50 = **1374.97 s**。

> 试过用 aiperf 汇总表的 `Total Usage Prompt Tokens` 反推累积 ISL 找切点，
> 得到 1400 s，偏 25 s。**能从 log 直接读就别拟合。**

### 切分与校验

```bash
python3 - <<'EOF'
import csv
sec = lambda t: (lambda h, m, s: h*3600 + m*60 + s)(*map(int, t.split(":")))
rows = list(csv.DictReader(open("prefill_forwards.csv")))
CUT  = min(sec(r["time"]) for r in rows) + 1375     # t_split, 向上取整到秒
keep = [r for r in rows if sec(r["time"]) >= CUT]
with open("prefill_forwards_profiling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(keep)
print(f"{len(keep)}/{len(rows)} 行, {len({r['forward_idx'] for r in keep})} 个 forward")
EOF
```

**必须拿 aiperf 汇总表的三个数对账**（`Usage` 表 + `[aiperf] prefix cache hit` 行）：

| 校验项 | 本例 trace | aiperf 报告 | 差 |
|---|---:|---:|---:|
| 请求数（warmup 段） | 355 | `target: 354 requests` | +1（跨边界那条） |
| ISL / Total Usage Prompt Tokens | 320.76 M | 317,480,133 | +1.0% |
| 新算 token | 13.89 M | 13,782,213 | +0.8% |
| 缓存命中率 | 95.67% | 95.66% | +0.01pp |

三项都在 1% 内才算切对。对不上就是 `t_split` 取错了。

### 两段的实际差异（本例）

| 段 | forward | 请求 | ISL | 新算 tok | 命中率 | GPU 时间 |
|---|---:|---:|---:|---:|---:|---:|
| warmup | 1,681 | 355 | 66.95 M | **13.52 M** | 79.8% | 1,342 s |
| profiling | 4,305 | 2,472 | 320.76 M | **13.89 M** | 95.7% | 2,143 s |

### 顺带：CI 报的吞吐是什么口径

CI summary 行的头条数字是 **`Input Token Throughput`**（本例 **80,783.71 tok/s**）。
分子是**含 prefix cache 命中的完整 prompt token**（`Total Usage Prompt Tokens`），
分母是 **3930 s**——不是 benchmark 窗口。三个全局吞吐同用这个分母，可交叉验证：

```
Total Prompt      317,480,133 / Input tok/s  80,783.71 = 3930 s
Total Completion    2,216,245 / Output tok/s    563.93 = 3930 s
Request Count           2,460 / Request tput      0.63 = 3905 s
```

⚠️ **两个容易对错的地方：**

1. 日志里的 `Benchmark Duration: 3621.10 sec` 是**信用窗口，不是吞吐分母**。
   两者差 309 s 是尾部请求排空——`Request Latency` 最大值 823 s，
   最后几条在窗口关闭后很久才结束。
2. `Active Prefill Throughput`（本例 88,119）是**逐请求平均值**
   （min 238 / max 535,224），**不是全局比值**，不要用它对口径。
   它和 `320.76M ÷ 3621 = 87,674` 只差 0.5%，是巧合。

用本 trace 按 CI 口径复算，可以完全闭合：

```
320.76 M ÷ 3930 s = 81,619 tok/s   vs CI 实报 80,784   差 +1.0%
                                      (恰好等于两边 ISL 差 320.76 vs 317.48 M)
```

如果你要的是"GPU 每秒真算多少新 token"，分子必须换成 `Σ chunk_tokens`：
本例 `13.89 M ÷ 3930 = 3,535 tok/s`。**两者差 23 倍，比较吞吐前先对齐口径。**

---

## ①b（可选）Perfetto trace + 耗时归因

```bash
python3 scripts/gen_prefill_trace.py -d <outdir> --label "<用例名>" \
        --phase-split 1374.97          # ①a 算出的 t_split, 强烈建议带上
```

产出 `prefill_forwards_trace_full.json.gz`，拖进 https://ui.perfetto.dev。
slice 名带 `≙ (q=…, c=…, b=…)`，另有三条 counter 轨 `q_eq` / `c_eq` / `attn_work`。

**trace 保留 warmup**（看冷缓存行为、JIT autotune 要用），但 `--phase-split` 会画一条
**全局竖线**加一条 `aiperf Phase` 轨（最上面，warmup 灰 / profiling 绿），
一眼能看出哪些 forward 该计入 CI 指标。不传这个参数就没有分界线，
很容易把 warmup 的 forward 误当成 benchmark 数据——这是本流水线最贵的一个坑。

### ⚠️ 时间模型：最容易犯的错

log 里的 `[atom HH:MM:SS]` 是**调度时刻**，所以

```
t[k+1] − t[k]  ==  第 k 个 engine step 的完整延迟（forward 计算 + KV offload/RDMA 收尾 + 下次调度）
```

**两次 forward 之间没有天然空洞**，引擎在这段时间里是连续忙的。

错误画法：把 forward 画成固定宽度，剩下的标成 gap。这样一个跑了 3s 的 long-context forward 会被画成
"1s forward + 2s 空档"，"长 context 计算"就被错误归因成**空档**原因。长 context 只让 slice 变宽。

正确画法（脚本已实现）：
- **busy step**：forward slice 铺满 `[t_k, t_{k+1})`，不画 gap
- **starvation step**：slice 只占 `D̂`（同 ctx 桶、同样"跑完了 request"的 busy step 均值），
  剩下的才是真正的 GPU 空转，画在独立 idle 轨上

D̂ 取的是偏慢的一类 step，所以推算出的空转时间是**下界**。换成 p25/p75 重跑可看敏感度。

**唯一能在两次 forward 之间造出洞的原因**：`running` 和 `waiting` 都空了。

### 归因时必须做混淆控制

不要只看"有 X 事件的 step 平均更慢"就下结论。至少固定 **ctx 桶** 和 **这一步是否跑完了某条 request**
两个变量再比。本项目踩过两次：

- LMCache `Failed to allocate memory block` 一度被归因成 328s。控制两个变量后，有告警的 step
  **一次都不比同类 step 慢，多数还更快** —— 告警只是伴随高吞吐窗口出现。已撤回。
- "请求收尾 KV offload+RDMA" 一度吃掉 47% 的空档，因为**跑完最后一块的那个 forward 同时会触发 RDMA
  并清空队列**。把 starvation 提为独立且更高优先级的类别后才分清。

---

## ② 分布统计 → 决定往哪里布点

```bash
python3 scripts/plot_eq_distribution.py -d <outdir> --label "<用例名>"   # eq_point_distribution.md + .svg
python3 scripts/plot_eq_grid512.py      -d <outdir>                      # 512×512 等距网格版
```

- **统计粒度是 forward batch**，先按 `forward_idx` 去重，否则多请求 batch 被重复计 N 次，分布往大 b 偏。
- **两个维度都得用对数分桶**。`q_eq` 跨约 4 个数量级、`c_eq` 跨约 3 个，等距桶会把 95% 的点堆进一个桶。
- **`q_eq` 的上界恒等于 `max_num_batched_tokens`**，并在该值形成尖刺（本例 37.6% 的 forward 精确等于
  8192）。画图时给尖刺固定宽度，否则 `[8192,8193)` 这个宽度为 1 的桶在对数轴上细到看不见。
- **`c_eq` 可能为 0**（首块且无前缀命中），对数轴放不下，用底部单独灰带承载。理论上还可能为负。
- 不依赖 matplotlib/numpy（CI 机器上通常没有），标准库直接拼 SVG。

看图是为了决定采样点往哪布：找尖刺（重点采）、找稀疏角落（少采或标成外推区）、
看两维是否独立（独立就能用规则网格，不必沿对角线采）。

---

## ③ 分桶 → 角点 → 去重 → `final_points_unique.csv`

```bash
python3 scripts/gen_prefill_trace.py -d <outdir> --label "<用例名>"   # --rank-by time 需要它
python3 scripts/gen_final_points.py  -d <outdir> --c-bin 8192 --q-bin 1024
```

三步：

1. **分桶** `c_bin = floor(c_eq/8192)`，`q_bin = floor(q_eq/1024)`。
   跨度取这两个值是因为 8192 = `max_num_batched_tokens`，1024 = `block_size × dcp_world_size`。
2. **代表点 → 最近角点**：每格对 `c_eq`、`q_eq` **各自独立**取中位数，某一维中位数距低边 ≥ 半格就取
   高边否则取低边（两维独立判断等价于二维欧氏最近角）。这一步把格子数压成更少的角点数，
   因为**相邻格子会共用角点**。
   例外：`q_eq` 中位数 < `--no-snap-below`（默认 = `--q-bin`）的格子不吸附，直接取中位数 —— 见下。
3. **对 (final_c, final_q, eq_batch_size) 去重** —— b 是第三个维度，同一个角点可以带多个 b 值。

产物：

| 文件 | 内容 |
|---|---|
| `prefill_forwards.csv` | 原地追加 `final_context_length,final_query_length` 两列（重跑会覆盖旧的） |
| `final_grid_cells.csv` | 每个非空格子一行：两维中位数、角点、`n_forwards`、该格出现过的 b 值 |
| `final_points_unique.csv` | 每个 unique `(c,q,b)` 一行，**已按优先级排好**，取前 N 行即待测清单；带 `exec_seconds` / `n_forwards` / `attn_work_pairs` 和三条累计覆盖曲线 |

本例的收敛链条：

```
7775 行 → 5986 个 forward → 434 个非空格子 → 379 个角点 → 673 个待测点 (c,q,b)
```

379 → 673 是 b 撑起来的（同一角点平均带 1.8 个 b，最多 8 个）；
434 → 379 是角点共用压下来的。**这两个数不一致不是 bug**，问"为什么格子 434 个但清单 673 行"时看这里。

### ⚠️ 这一步最容易踩的两个坑

**（1）忘了按 `forward_idx` 去重。** CSV 一行是一个 (forward, request) 对，多请求 batch 的每一行都
重复同样的 `eq_*`。不去重就是按行统计，b=9 的 forward 被计 9 次，格子中位数和 `n_forwards` 全部往
大 b 偏。本例按行算会得到 588 个点、80% 覆盖需要 208 个点；正确去重后是 606 个点、151 个点。
**去重后待测点反而变多**，因为 b=1 的权重回来了，稀有格子不再被大 b 淹没。

**（2）q 维吸附会毁掉尾巴 forward —— 所以 `--no-snap-below` 默认等于 `--q-bin`。**
chunked prefill 的最后一块真实 q 常在 500~700，全落在 `q_bin=0`。硬吸附有两个后果：

- 中位数 ≥ 半格（512）→ 拉到 q=1024，**翻倍高估**。本例 30 个格子 / 1266 个 forward（21.1%）
  被放大超过 1.5 倍，而且正是排名最靠前的那批点。
- 中位数 < 半格 → 拉到 **q = 0**，这是个**测不了的坐标**（没有新 token 就没有 forward）。
  本例会产生 24 个这样的格子 / 52 个待测点，最高排到 priority 33 —— 照着清单往下跑，
  第 33 个就撞墙。

所以默认豁免整个 `q_bin=0` 行：这些格子的 `final_q` 直接取中位数。代价是待测点从 606 涨到 673，
覆盖 80% 时间从 190 点涨到 214 点。`--no-snap-below 0` 可以关掉，脚本会对上面两种情况各打一条警告。
c 维没有这个问题 —— 8192 的跨度相对 c 的量级（1e4~2e5）很窄，也不存在 c=0 测不了的问题
（c=0 就是无前缀命中的首块，完全可测）。

### 按覆盖率裁剪：`--rank-by`

`final_points_unique.csv` 已按优先级排序，取前 N 行即可。三条累计覆盖曲线直接读：
`cum_time_pct` / `cum_forward_pct` / `cum_work_pct`。

**默认 `--rank-by time`** —— 用 trace 里的 `step_latency_s`，**并且已扣掉 GPU 空转**
（⚠️ 注意 `step_latency_s` 是秒内均分的估计值，不是单次 forward 真值，见上面"log 里的坑"第 2 条。
排序在聚合层面仍然合理——总和守恒——但单个点的时间权重会被同秒的其它 forward 拉平，
不要拿它当单点耗时用。）
（profiling 段 forward 执行 2143 s；全 trace 是 3485 s，另有 1489 s 是引擎 running/waiting
都空、纯等新请求，那部分不该算进任何点头上）。需要先跑 `gen_prefill_trace.py`，找不到 trace 会
直接报错让你改用 `work|count`。

本例（**profiling 段**，603 个点）覆盖 80% 时间各需多少点：

| 排序依据 | 需要点数 | 此时次数覆盖 | 此时工作量覆盖 |
|---|---|---|---|
| **`time`** | **217** | 84.9% | 59.4% |
| `count` | 约 230 | — | — |
| `work` | 约 300 | — | — |

按时间排的曲线：50% → 64 点，80% → 217，90% → 337，95% → 438。

对照全 trace（含 warmup，673 个点）：80% → 214 点。**点数几乎没变，但点集变了** ——
见下面 ③a。

**`work` 是三者里最差的**，这点反直觉、值得记住：Attention 工作量当时间的代理并不好，
它把大 context 的点抬得过高（排到 80% 时间时工作量已覆盖 90.9%，过度集中在少数大点上），
因为真实耗时里有很大一块 —— KV offload、RDMA 收尾、kernel launch —— 不随 `q·c` 走。
`count` 则会把 q≈1024 那排的尾巴 forward 顶到最前面：次数多但每次都便宜。
按时间排顺带把另外两个指标都带到 82%+，是最均衡的。

**限制**：log 只有秒级时间戳，同一秒最多 6 个 forward，脚本在秒内均匀平铺，所以
**单个 forward 的耗时不可信**（见 ①b）。但这里是按点聚合的 —— 每个点平均 10 个 forward，
头部的点上百个 —— 平铺误差基本抵消，点级的时间占比可用。要更准只能加 per-step 埋点。

长尾绝大多数是 b≥3 的稀有组合（b=9 全部才 4 个点，673 个点里有 200+ 个只对应 1 次 forward），
也正是最难测的那批 —— 见 ④。

---

## ③a 换口径重跑之后：哪些点真的要重测 → `points_todo.csv`

```bash
python3 scripts/diff_points.py -d <outdir> --measured bench_run/RESULTS_214_points.csv --cover 80
```

换了口径（比如从全 trace 切到 profiling 段）重跑 ③，点集会变。**但"新点"≠"要测的点"** ——
拟合好的 7 参数模型在已测点张成的包络内是插值，精度可信；只有掉到包络外的才是外推，
必须真去机器上测。脚本按 b 分别算已测点的 (c,q) 矩形包络，把点分三类写进 `points_todo.csv`
（`extrap` 排最前，直接拿去排测试计划）：

| status | 含义 | 本例 217 点中 | 占 profiling 时间 |
|---|---|---:|---:|
| `measured` | 上轮已测，查表 | 117 | 55.6% |
| `interp` | 包络内，模型插值 | 87 | 22.5% |
| `extrap` | **包络外，要补测** | **13** | **2.0%** |

即：口径一换，100 个点没测过，但其中 87 个落在已测包络里，**真正要上机的只有 13 个**
（b=1×3, b=2×5, b=4×1, b=5×2, b=7×2，约 2 h 机时）。`--cover 100` 看全部 603 点：
extrap 118 个 / 7.0% 时间。

**⚠️ 别用最近邻距离做这个判断。** 第一版我按相对距离筛，标出 63 个"较远点"——
其中大多数是 c 完全命中、q 恰好差一个网格步长（1024 vs 2048 读成 Δq=50%）的点，
纯属误报。**矩形包络才是对的判据**：模型对 c、q 各自单调，包络内就是插值。

`reason` 列会写清楚外推方向（`q 310 < [336,8192]`），排测试计划时优先补包络边界上的点 ——
补一个边界点往往能把一批 extrap 转成 interp。

**⚠️ 但先别急着去测。** 本例把这 13 个 extrap 点真测了一遍（13/13 accept OK），代回
profiling 段总时间只差 **-0.13%**（13 点只覆盖 1.04% 的时间）。包络判据挑出来的是
"形式上没覆盖"，不等于"预测不准"——模型对 c、q 单调且平滑，外推一两个网格步长损失有限。

但**必须同 b 分组比**，否则会得出反的结论：

| | 外推 (13 新点) | 内插 (210 旧点) |
|---|---:|---:|
| b=1  | 6.7%  | 3.9%  |
| b≥2  | 13.9% | 18.3% |

b≥2 外推确实不比内插差（那片区域本来就拟合得差）；但 **b=1 外推 6.7% 明显劣于内插 3.9%**
——拟合得好的区域，外推是有代价的。所以"包络外可以不测"只在模型本来就拟合得差的区域成立。

所以正确用法是：**先用旧模型预测这批 extrap 点，看它们占多少时间**。像本例只占 2.0%，
即使全错也只是 2% 的误差，直接跳过。只有当 extrap 占比大（比如 >15%）或落在明显不同的
物理区（如 c=0 这种边界）时才值得上机。

---

## ④ 用 aiperf 实测这些点

详见 `measure_prefill_performance_point.md`（手写，非生成）和
`bench_run/rows423_428_b2_result.md`（一次 b=2 的完整实测记录）。这里只放结论性的注意事项：

- **服务端参数必须和 CI 一致且全程不改**（`--max-num-batched-tokens 8192`、`--max-num-seqs 64`、
  `enable_prefix_caching`、`chunked_prefill`）。改了就不是在复现 CI 的工况。
- **要 primer**。q=8192 的请求自己的 checkpoint 落在 `c+8192`，永远不会在 c 处留下可续跑的
  checkpoint，前缀复用无从谈起。得先用一个 q=128 的短请求（ISL=c+130 → anchor=c）在 c 处打点。
- **TTFT 必须排除建缓存的那次 forward**（`context_length=0`、无前缀命中），那和要测的
  "带 prefix cache 的 chunk prefill" 不是一回事。
- **b≥2 要靠 blocker 硬凑**，而 blocker 会引入系统性偏置：vLLM 的调度异步跑在执行前面，目标 batch 的
  日志行可能在最后一个 blocker step 还在执行时就打出来了。本例实测 **+96.2 ms**（对照：同 harness
  但每组只发一条请求）。**每个 b 值都要单独跑一次对照量这个偏置**，否则跨 b 比较得到的是 harness 差异。
  好消息是**斜率对常数偏置免疫**，只比"对 context 的敏感度"时不受影响。
- **已验证。** `run_stage_bn.sh` 的 fixed-schedule + 32 blocker
  方案对 b=2..9 全部成立，上一轮 `stage_bn_timing.csv` **44/44 全 OK**
  （b=3×6, b=4×7, b=5×2, b=6×2, b=7×1, b=8×1, b=9×1）。
  关键是 blocker 数取 8 的倍数：ISL=1024 的冷请求 8 个正好零余量填满 8192，排干后队列里
  只剩目标那 b 条，没有第三方能被拉进来 —— **余量与 q 无关，所以 q<1024 的点也成立**。
  唯一的硬约束是 `b*q ≤ 8192`（预算不变时物理上凑不出更大的 batch）。
- 单点耗时几乎与 c、q 无关（被 fixed-schedule 的时间表撑着）：**b=1 约 51 s/点，b=2 约 152 s/点**。
  673 个点全跑约 25 h（不含重跑），覆盖 80% 时间的 214 个点约 5 h，覆盖 50% 的 75 个点约 1.2 h。

---

## ⑤ 三个吞吐值 —— 实测点表的最终用途

④ 测出来的点表本身不是目的。目的是**给 profiling 段的 forward 逐个定价**,
把"这批 batch 形状如果背靠背跑完要多久"算出来, 得到一条三级阶梯。

全部限定 profiling 段 (①a 切完 warmup)。基础量: 4,305 forward / 2,472 请求 /
ISL 320.76 M / fresh 13.89 M (命中率 95.67%, 两个口径差 **23.09x**)。

| # | 量 | ISL 口径 | fresh 口径 | 分母 | 估算成分 | 备注 |
|---|---|---:|---:|---|---|---|
| **1** | 实际交付吞吐 | **81,619** | 3,535 | aiperf 窗口 3,930 s | **零** —— 分子分母全直读 | |
| **2** | 忙时吞吐 (剔空转) | **>= 149,680** | >= 6,482 | GPU 忙 <= 2,143 s | 分母是估算的**上界** ⟹ 结果是**下界**| 第`k`个forward的时间为`t[k+1] - t[k]`, `t[k]`是第`k`个prefill batch调度完成、马上要launch forward的时刻 |
| **3** | 理想上限 | **204,801** | 10,066 | Σ 实测纯 forward = 1,034.8 s | 单价全是引擎内埋点实测, 但**只覆盖 66.1% 的 ISL** | 估计值, $\Sigma_{i} (t_i / T) tput_i$: 其中$t_i$和$tput_i$都是通过测量纯模型forward的时间得到的 |

```
81,619  ──剔除无请求可调度的空转──>  >=149,680  ──剔除 step 内一切非模型开销──>  204,801
 1.00x                                >=1.83x                                  2.51x
```

⚠️ **#3 是子集口径。** 3,400/4,305 个 forward 能在点表里查到单价, 它们占 fresh 的 75.0%、
ISL 的 66.1%。**剩下 905 个 forward 没有全段外推**(下面 ⑤a 末尾说明为什么)。

### #1 实际交付吞吐 — 直读

见 ①a "CI 报的吞吐是什么口径"。要点: 分母是 **3,930 s**(三个全局吞吐交叉验证得到),
**不是** `Benchmark Duration: 3621.10 sec`(那是信用窗口)。分子用完整 ISL 得 81,619,
换成 `Σ chunk_tokens` 得 3,535。

### #2 忙时吞吐 — trace 窗口减空转

分母 = trace 窗口 3,602 s − 空转 >= 1,457 s。空转来自 ①b 的 `idle_gaps.json`,
只把"running 与 waiting 队列**均为空**"记作 idle, 且时长保守折算 —— 两个偏向同向,
所以空转是下界、忙时间是上界、吞吐是下界。**是不等式, 但方向确定。**


### #3 固定模型优化下的天花板 — 查点表 + 引擎内埋点

```bash
python3 scripts/throughput_from_probe.py     # 看 "口径 B" 那一段
```

把每个 forward 按 `(final_context_length, final_query_length, eq_batch_size)` 去查
⑤a 产出的探针表, 取 `fwd_gpu_ms` 当单价:

```
4,305 个 forward → 命中 3,400 (79.0%), Σ 实测纯 forward = 1,034.8 s
覆盖 fresh 75.0% / ISL 66.1%
→ ISL 204,801 / fresh 10,066 tok/s
```

### ⑤a 装 forward 探针 (#3 的数据来源)

`fwd_gpu_ms` 拿不到就没有 #3。它必须从引擎内部量, 外部黑盒测不出来。

**埋点**: `scripts/_fwdprobe.py`(可复用副本, 无第三方依赖, 复制即用;
生产副本在被插桩的引擎包里 `atom/_fwdprobe.py`)。三个挂点, 不设环境变量时全是 no-op、零开销:

| 装饰器 | 挂在 | 产出 |
|---|---|---|
| `@probe_gpu` | `ModelRunner.run_model` | `fwd_gpu_ms` — HIP event, **延迟读取** |
| `@probe_wall` | `ModelRunner.forward` (prefill) | `step_wall_ms` / `step_period_ms` — perf_counter |
| `@probe_wall` | `<Decode>ModelRunner.forward` | 同上 (decode 侧, 本流水线不用) |

```bash
docker run ... -e ATOM_FWDPROBE=/path/fwdprobe.jsonl -e ATOM_FWDPROBE_LAG=8 ...
```

⚠️ **`probe_wall` 必须挂在 `@torch.inference_mode()` 之上(最外层)**, 否则量不到完整 forward:

```python
@_fwdprobe.probe_wall
@torch.inference_mode()
@with_eplb_forward_monitor
def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
```

**换引擎只需改 `_batch_meta()`** —— 它按 ATOM 的 `ScheduledBatch` 取字段
(`num_scheduled_tokens` / `context_lens` / `total_tokens_num_prefill` ...)。
取不到时写 `meta_err` 而不抛异常, **探针永远不会弄崩一个 forward**。
⚠️ ctx 的约定是 `ctx = context_lens - num_scheduled_tokens`, 与 bench harness 的 (c,q) 一致 ——
**c 是已在 KV cache 里的上下文, 不是 ISL**(同 ① 里 `done_tokens` vs `ISL` 那个区分)。

**为什么 GPU 时间必须用 event 而不是 perf_counter**: kernel launch 是异步的,
`perf_counter` 包住 `self.model(...)` 量到的是**launch 时间不是 GPU 时间**。
`torch.cuda.Event` 映射到 hipEvent, 在 stream 上打时间戳。
**关键是 `elapsed_time()` 的读取延迟 `DRAIN_LAG=8` 次 forward** —— 立即读会隐式
同步、把流水线串行化, 那恰好就是我们不想扰动的东西。

**跑法**: 用与 ④ 完全相同的 harness 重跑一遍点表(`run_points_instr.sh` 跑 b=1,
`run_bn_instr.sh` 跑 b>=2), 探针在引擎内侧被动记录, 不影响 aiperf 侧的测量。
每点约 116 s(b>=2), 210 个点约 3.5 h。**context length 这个变量是 harness 造的,
不是探针造的** —— primer 请求先把 c 个 token 的前缀种进 KV cache, 测量请求靠显式
`hash_ids` 复用它(③ 的 `gen_b2_trace.py`), 探针只是把命中之后的 `ctx` 如实记下来。
本例探针侧 ctx 有 151 个不同取值 / 0..753,664, 前缀命中率 95.7%, 与 trace 侧一致。

**合流**:

```bash
python3 join_probe.py    # fwdprobe.jsonl + 参考点表 -> RESULTS_probe_vs_bench.csv
```

按 `(ctx, q, b)` 分桶, 只保留 `dummy=0 / decode=0 / ctx 与 q 组内齐一`的纯 prefill forward,
取每桶中位数。**q<1024 的桶用 ±16 容差匹配** —— 那一档没有 ISL anchor 兜底,
aiperf 合成长度有几个 token 抖动(同 ③ 的 `--no-snap-below` 那个坑), 不给容差会全部 join 不上。
本例落地 210 个点 / 252 行。12,967 次 forward 里 9,510 次是 GPU-bound(host 先返回、
GPU 还在算), 符合预期。


### 两个注意事项

1. **权重取时间占比时, 加权平均在代数上恒等于聚合吞吐**: `Σ(t_i/T)(tok_i/t_i) = Σtok_i/T`。
   所以算出来的**就是** total/total, 不是"平均每个 forward 的吞吐"。后者要按**次数**加权,
   会被大量廉价小 batch 拉高, 是另一个量。曾经记的 9,541 tok/s 是次数加权那一种, **已作废**。
2. **ISL 不能按 `prefill_forwards_profiling.csv` 的行直接累加。** 那是 (forward, request) 对,
   被 chunk 切开的请求会在它跨的每个 forward 里各记一遍完整 ISL, 全段加出 484 M 而真实只有
   320.76 M(虚高 1.5x)。正确做法: 每个请求的 ISL 只归到它**最后一个** forward。

---

## 校验清单（换用例后跑一遍）

0. **warmup 切干净了没**：`prefill_forwards_profiling.csv` 的 ISL / 新算 token / 命中率
   与 aiperf 汇总表三项对账，全部在 1% 内（见 ①a）。这条不过，后面全部作废。

```
① 每个 request 实例最终 progress 恰好 100%，且没有 >100%
① 首块之后 done 链条不断裂（首块 done>0 是 LMCache 前缀命中，正常）
① arrival 匹配率接近 100%
①b trace 里同一 track 的 slice 无重叠
③ gen_final_points.py 打印的 forward 数 == CSV 的 unique forward_idx 数（不是行数）
③ 格子数 ≥ 角点数，待测点数 ≥ 角点数
③ --rank-by time 时不该出现"找不到执行时间"的告警（trace 和 CSV 应同批生成）
③ final_points_unique.csv 里不该有 final_query_length == 0 的行（测不了的坐标）
⑤ 探针路线的匹配 forward 数与覆盖率要一起报（本例 3400/4305 = 79.0%，ISL 覆盖 66.1%）
⑤ throughput_from_*.py 里 assert 加权和 == Σtok/T 必须通过（口径自检）
```

等价点可以手算抽查：任取一个 `batch_reqs≥5` 的 forward，按公式算 `q_eq/c_eq` 和 CSV 对齐
（`equivalent_point()` 是唯一实现，trace 端优先读 CSV 列，两边不会漂）。

## 换 engine / 换用例

所有正则集中在 `parse_prefill_log.py` 顶部，匹配不到就产出 0，不会崩。适配新 engine 通常只需改
`RE_SCHED` 和 `RE_ARR`。`gen_prefill_trace.py` 和 `gen_final_points.py` 只依赖 CSV 的列名。
换 engine 后 ③ 的两个分桶跨度要跟着改：`--c-bin` 跟 `max_num_batched_tokens`，
`--q-bin` 跟 `block_size × dcp_world_size`。

## 已知边界

**"engine busy" ≠ "GPU busy"**。收尾那段（KV offload + RDMA）GPU 大概率是闲的，但 budget 利用率
counter 仍读 100%。要分开必须加 per-step 埋点，光靠 log 做不到。
