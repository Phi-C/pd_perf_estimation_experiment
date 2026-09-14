#!/usr/bin/env python3
"""b>=2 fixed-schedule 的对齐分析 —— analyze_b2_fs.py 的任意 b 版本。

从引擎日志挑出 b、done、new 均符合目标的 batch，再按时间邻近关系匹配 AIPerf 请求。
请求先按 prompt 长度过滤，再按 request_start_ns 排序；每个目标 batch 贪心匹配
调度前 MAXQ 毫秒内最近的 b 条未用请求。这里没有跨端统一 request ID，匹配仍需验收。

TTFT 口径: aiperf 报的 TTFT 含"排在 blocker 后面等待"的时间, 不能和 b=1 比。
ttft_net = ttft_raw - queue，queue = 日志调度时刻 - 请求发出时刻。
日志转发、时钟和匹配误差会影响差分；它不是 GPU forward 直接测时。

q<1024 用 ±16 容差: 那一档没有 anchor 兜底, aiperf 合成长度有 ±10 token 抖动
(和 b=1 的 run_stage_b1.sh 同一个原因), 引擎侧实测 q 记进 q_actual, 建模用 q_actual。
"""
import json, re, argparse, datetime as dt, statistics as st

MAXQ = 6000.0   # 目标 batch 与其请求发出时刻的最大间隔(ms), 必须 < trace 的 gap

LINE = re.compile(r"Scheduled prefill batch: (\d+) reqs, \d+ new tokens "
                  r"\(done: \[([0-9, ]+)\], new: \[([0-9, ]+)\]\), req_ids: \(([0-9, ]*)\)")

def pct(v, p):
    if not v: return "NA"
    v = sorted(v); k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--jsonl", required=True)
    ap.add_argument("-c", type=int, required=True); ap.add_argument("-q", type=int, required=True)
    ap.add_argument("-b", type=int, required=True)
    a = ap.parse_args(); c, q, B = a.c, a.q, a.b
    tol = 0 if q >= 1024 else 16
    lo_meas = q * 0.8          # primer 的 new=130 要被排除, 所有 q>=354 时这个阈值安全

    batches = []           # [(T, b)] 目标 batch 的调度时刻
    n_meas_engine = 0; qact = []; n_split = 0
    for line in open(a.log, errors="ignore"):
        m = LINE.search(line)
        if not m: continue
        nreq = int(m.group(1))
        d = [int(x) for x in m.group(2).split(",")]
        n = [int(x) for x in m.group(3).split(",")]
        ids = [int(x) for x in m.group(4).split(",") if x.strip()]
        if len(d) != nreq or len(n) != nreq or len(ids) != nreq: continue
        n_meas_engine += sum(1 for i in range(nreq) if d[i] == c and n[i] >= lo_meas)
        if nreq == B and all(x == c for x in d) and all(abs(x - q) <= tol for x in n):
            qact += n
            ts = line.split(" ", 1)[0][:26]
            batches.append(dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=dt.timezone.utc).timestamp() * 1000)
        elif any(x == c for x in d) and any(x >= lo_meas for x in n):
            n_split += 1
    batches.sort()
    n_tgt_batch = len(batches)

    recs = [json.loads(l) for l in open(a.jsonl)]
    recs = [r for r in recs
            if c + q - 64 <= r["metrics"]["usage_prompt_tokens"]["value"] <= c + q + 1024]
    recs.sort(key=lambda r: r["metadata"]["request_start_ns"])

    # 贪心配对: 按 T 升序, 每个 batch 吃掉最接近它、尚未被用掉的 b 条记录
    used = [False] * len(recs)
    raw, queue, net = [], [], []
    for T in batches:
        cand = [i for i, r in enumerate(recs) if not used[i]
                and 0 < T - r["metadata"]["request_start_ns"] / 1e6 < MAXQ]
        cand = cand[-B:]
        if len(cand) != B: continue          # 这一组配不齐, 整组丢弃, 不做半拉子统计
        for i in cand:
            used[i] = True
            t = recs[i]["metrics"]["time_to_first_token"]["value"]
            qw = T - recs[i]["metadata"]["request_start_ns"] / 1e6
            raw.append(t); queue.append(qw); net.append(t - qw)

    isl = st.mean(r["metrics"]["usage_prompt_tokens"]["value"] for r in recs) if recs else 0
    crt = st.mean(r["metrics"].get("usage_prompt_cache_read_tokens", {}).get("value", 0)
                  for r in recs) if recs else 0
    qa = round(st.median(qact), 1) if qact else "NA"
    print(json.dumps(dict(
        n_meas_engine=n_meas_engine, n_meas_aiperf=len(recs), n_target_batch=n_tgt_batch,
        n_split_batch=n_split, n_req=len(net), q_actual=qa,
        isl_mean=round(isl, 1), cache_read_mean=round(crt, 1),
        ttft_raw_avg=round(st.mean(raw), 3) if raw else "NA",
        queue_avg=round(st.mean(queue), 3) if queue else "NA",
        ttft_net_avg=round(st.mean(net), 3) if net else "NA",
        ttft_net_p50=pct(net, 50), ttft_net_p90=pct(net, 90), ttft_net_p99=pct(net, 99),
        fresh_tok_s=round(B * q / (st.mean(net) / 1000), 1) if net else "NA",
        per_user_tok_s=round(q / (st.mean(net) / 1000), 1) if net else "NA")))

main()
