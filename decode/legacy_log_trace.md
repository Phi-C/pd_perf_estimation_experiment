# 只有 PD-DECODE 日志时能分析什么

旧 ATOM 日志中的 `[PD-DECODE]` 通常只打印每条请求的前 3 个 decode step。可以用它诊断请求进入 Decode 的过程，无法恢复完整的稳态 batch 分布。

## 运行

```bash
python3 scripts/parse_decode_log.py /path/to/job.log \
  -o out/decode_legacy --label '旧日志诊断'
python3 scripts/gen_decode_trace.py -d out/decode_legacy
```

输出 `decode_forwards.csv`、`decode_requests.csv`、`decode_backpressure.json`、`decode_gap_events.json`、统计 Markdown 和 Perfetto 文件。使用独立目录，因为这里的请求表与 [StepMembers 流程](SKILL.md)结构不同。

## 关键字段

| 字段 / 日志 | 含义 |
|---|---|
| `forward_idx` | 启发式分组后的 observed forward 序号 |
| `batch_reqs_observed` | 可见成员数，仅反映日志中观测到的部分 |
| `[PD-DECODE] seq` | batch 槽位，会随 step 重排 |
| `[PD-TRANSITION] seq` | Decode 节点 request ID |
| `PD backpressure` | 稀疏采样的 waiting / running / resident 状态 |

解析器通过 ISL、已记录 step 链和槽位线索匹配请求，处理日志重放。重复 ISL 会造成歧义，应检查匹配率及 `chain_bad`，不要仅凭长度做唯一身份连接。

## 时间和统计限制

引擎日志只有秒级时间，runner 行首时间还包含转发延迟。Perfetto 中的窄 slice 只表示观察顺序和密度；相邻 slice 的空白可能有大量未打印的 decode step，不能认定为 GPU 空转。

只有完整成员集合才能计算 `L_eq` 与真实 b。该旧流程不产出等价 batch 列，也不能用于 ITL、纯模型吞吐或稳态容量估计。要获得这些量，需要 StepMembers 或其它逐步成员记录以及可靠测时。
