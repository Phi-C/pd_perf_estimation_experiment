# Forward 探针与测量边界

[_fwdprobe.py](../scripts/_fwdprobe.py) 在 ATOM 引擎内记录 GPU forward 与 CPU forward 的时间。它依赖引擎已有的 PyTorch，离线分析脚本不需要加载模型。

## 1. 安装挂点

把脚本复制到目标引擎包的 `atom/_fwdprobe.py`，在对应 model runner 中导入，并添加装饰器：

```python
from atom import _fwdprobe

@_fwdprobe.probe_gpu
def run_model(self, input_ids, batch=None, ...):
    ...

@_fwdprobe.probe_wall
@torch.inference_mode()
def forward(self, batch):
    ...
```

这里展示挂点位置，函数签名应保留目标引擎原实现。`probe_wall` 放在最外层；Prefill / Decode 若使用不同 runner，两处都需安装。

启动引擎前配置：

```bash
export ATOM_FWDPROBE=/path/to/fwdprobe.jsonl
export ATOM_FWDPROBE_LAG=8
```

未设置输出路径时装饰器直接返回原函数。当前实现只写 rank 0；跨 rank 的瓶颈和不同 stream 的事件覆盖需单独确认。换引擎时检查 `_batch_meta()` 的字段和 `ctx = context_lens - num_scheduled_tokens` 约定。

## 2. 输出解释

每条 JSONL 记录包含形状 `q[] / ctx[] / b / prefill / decode / dummy` 及以下时间：

| 字段 | 时钟与边界 | 用途 |
|---|---|---|
| `fwd_gpu_ms` | GPU event 包围 `run_model`，当前挂点含 logits | 对同形状模型时间定价 |
| `step_wall_ms` | `forward` 进入到返回的 CPU 时钟 | 观察 host 调用耗时 |
| `step_period_ms` | 本次与上次 `forward` 返回时刻之差 | 同形状、饱和运行时估计稳定步长 |
| `t` | host 墙钟记录时刻 | 定位记录，不是 GPU event 的绝对时间 |

GPU kernel 异步执行，CPU 调用返回不表示 GPU 已结束。`step_wall_ms` 不能当作 GPU 时间，也不能和 `fwd_gpu_ms` 直接相减来拆分调度开销。

探针延迟读取 event，并先用 `query()` 检查完成状态，避免在每个 forward 内强制同步。最后几条记录可能仍在缓冲区；需要在测量结束且 event 已完成后触发 `flush()`。仅等待几秒不保证缓冲区自动落盘。分析时保留实际样本数。

## 3. Prefill 表的生成

```bash
python3 scripts/join_prefill_probe.py \
  --probe /path/to/fwdprobe.jsonl \
  --reference bench_run/RESULTS_223_points.csv \
  --output out/prefill/bench/RESULTS_probe_vs_bench.csv
```

脚本保留非 dummy、纯 Prefill、组内 `ctx` 与 `q` 均一致的 batch，按 `(c, q, b)` 分桶取中位数。`q < 1024` 允许 ±16 token 的参考点匹配误差，这是 K3 合成输入抖动的约定，换配置需重新验证。

同一个参考点若匹配多个小 q 桶，吞吐脚本按桶样本数合并其时间中位数。`host_ms` 和 `period_ms` 保留为诊断量：Prefill harness 交错执行 primer、blocker 和目标请求，按形状聚合的返回间隔不能视为该形状的稳定 step period。

`target_fwd_gpu_ms` 是旧表由 TTFT 扣除客户端和尾部估计得到的字段名；它不等于直接测得的 `fwd_gpu_ms`。

## 4. Decode 的采样窗口

[analyze_decode_group.py](../scripts/analyze_decode_group.py) 从测量前后记录的 probe 字节区间读取样本，保留 `prefill == 0`、非 dummy、`b == 目标值` 且所有 `q == 1` 的记录，丢弃开头的 warmup 步，再对照实际 `L_eq / CV_L / r_max`。

```bash
python3 scripts/analyze_decode_group.py \
  --probe /path/to/fwdprobe.jsonl --start 0 --end 123456 \
  --artifact-dir /path/to/aiperf/measurement \
  -b 32 --L-eq 163840 --drop 20 --min-steps 100
```

字节边界替换为该轮实际值。自动过滤不证明剩余样本连续、无抢占或 KV 独立，仍需结合 `StepMembers`、trace 生成方式和缓存命中信息验收。详见[Decode 测点方法](../decode/measure_decode_performance_point.md)。
