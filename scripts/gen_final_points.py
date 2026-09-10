#!/usr/bin/env python3
"""把 prefill_forwards.csv 的等价点分桶 -> 角点 -> 去重, 产出按优先级排好的待测点清单。

三步:
  1) 分桶: c_bin = floor(c_eq/CB), q_bin = floor(q_eq/QB)
  2) 每格取代表点(两维各自独立取中位数), 吸附到最近的角点 -> (final_c, final_q)
  3) 对 (final_c, final_q, eq_batch_size) 去重, 按 --rank-by 排优先级

产物:
  prefill_forwards.csv    原地追加/覆盖 final_context_length, final_query_length 两列
  final_grid_cells.csv    每个非空格子一行(中位数、角点、点数、出现过的 b)
  final_points_unique.csv 每个 unique (c,q,b) 一行, 已按优先级排序, 取前 N 行即待测清单

⚠️ 统计粒度是 forward batch: CSV 里同一个 forward 的每一行都重复同样的 eq_* 值,
   必须先按 forward_idx 去重, 否则多请求 batch 被重复计 N 次, 中位数和覆盖数都会往大 b 偏。
"""
import argparse, csv, gzip, json, os, sys, statistics as st
from collections import defaultdict, Counter

DROP_PREFIX = ("grid_", "final_")   # 重跑时先清掉自己上次写的列


def load_exec_seconds(d):
    """从 Perfetto trace 读每个 forward 的执行时间(秒), starvation step 的 GPU 空转已扣除。

    ⚠️ log 只有秒级时间戳, 同一秒里的多个 forward 是均匀平铺的, 所以**单个 forward 的耗时
       不可信**。但这里是按点聚合(每个点平均 10 个 forward), 平铺误差基本抵消, 点级占比可用。
    """
    for name in ("prefill_forwards_trace_full.json.gz", "prefill_forwards_trace_full.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        t = json.load(op(p))
        ev = t["traceEvents"] if isinstance(t, dict) else t
        # forward slice 的 dur 单位是 us; IDLE slice 是独立轨, 没有 forward_idx, 自动排除
        return {a["forward_idx"]: e["dur"] / 1e6 for e in ev
                if e.get("ph") == "X" and "step_latency_s" in (a := e.get("args") or {})
                and "forward_idx" in a}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default=".", help="含 prefill_forwards.csv 的目录")
    ap.add_argument("--c-bin", type=int, default=8192, help="eq_context_length 的格子跨度")
    ap.add_argument("--q-bin", type=int, default=1024, help="eq_query_length 的格子跨度")
    ap.add_argument("--rank-by", choices=("time", "work", "count"), default="time",
                    help="待测点优先级依据。time=实测执行时间(默认, 需要先跑 gen_prefill_trace.py); "
                         "work=Attention 工作量; count=forward 次数")
    ap.add_argument("--no-snap-below", type=int, default=None,
                    help="q_eq 中位数小于该值的格子不吸附, final_q 直接取中位数四舍五入。"
                         "默认 = --q-bin, 即整个 q_bin=0 行都豁免。两个原因: (1) chunked prefill "
                         "的尾巴 forward 真实 q 常在 500~700, 吸附到 q_bin 上界等于翻倍高估; "
                         "(2) 中位数小于半格时会被吸附到 q=0 —— 没有新 token 就没有 forward, "
                         "那是个测不了的坐标。设 0 可关闭。")
    a = ap.parse_args()
    CB, QB = a.c_bin, a.q_bin
    if a.no_snap_below is None:
        a.no_snap_below = QB
    P = os.path.join(a.dir, "prefill_forwards.csv")
    rows = list(csv.DictReader(open(P)))
    fields = [f for f in rows[0] if not f.startswith(DROP_PREFIX)]

    exec_s = load_exec_seconds(a.dir)
    if exec_s is None:
        if a.rank_by == "time":
            sys.exit(f"--rank-by time 需要 {a.dir}/prefill_forwards_trace_full.json[.gz], "
                     f"先跑 scripts/gen_prefill_trace.py, 或改用 --rank-by work|count")
        exec_s = {}

    # --- 1) 分桶。cell 统计按 forward 去重, 每行的 cell key 仍逐行算(同 forward 必同 cell) ---
    cells, seen = defaultdict(list), set()
    for r in rows:
        c, q, b = (float(r["eq_context_length"]), float(r["eq_query_length"]),
                   int(float(r["eq_batch_size"])))
        r["_key"] = key = (int(c // CB), int(q // QB))
        fi = int(r["forward_idx"])
        if fi not in seen:                        # <- 去重, 别删
            seen.add(fi)
            cells[key].append((c, q, b, float(r["attn_work_pairs"]), exec_s.get(fi, 0.0)))
    if exec_s:
        missing = len(seen - set(exec_s))
        if missing:
            print(f"⚠️ {missing}/{len(seen)} 个 forward 在 trace 里找不到执行时间, 按 0 计")

    # --- 2) 代表点 -> 最近角点。两维独立判断, 等价于二维欧氏最近角 ---
    rep = {}
    for (cb, qb), pts in cells.items():
        mc = st.median(p[0] for p in pts); mq = st.median(p[1] for p in pts)
        fc = cb * CB + (CB if mc - cb * CB >= CB / 2 else 0)
        fq = (round(mq) if mq < a.no_snap_below
              else qb * QB + (QB if mq - qb * QB >= QB / 2 else 0))
        rep[(cb, qb)] = (mc, mq, fc, fq, pts)

    # --- 写回 prefill_forwards.csv ---
    out = fields + ["final_context_length", "final_query_length"]
    with open(P, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out); w.writeheader()
        for r in rows:
            _, _, fc, fq, _ = rep[r["_key"]]
            w.writerow({**{k: r[k] for k in fields},
                        "final_context_length": fc, "final_query_length": fq})

    # --- 每格明细 ---
    with open(os.path.join(a.dir, "final_grid_cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c_bin","q_bin","c_lo","c_hi","q_lo","q_hi","n_forwards",
                    "median_eq_context_length","median_eq_query_length",
                    "final_context_length","final_query_length","b_values"])
        for (cb, qb), (mc, mq, fc, fq, pts) in sorted(rep.items()):
            w.writerow([cb, qb, cb*CB, (cb+1)*CB, qb*QB, (qb+1)*QB, len(pts),
                        round(mc,1), round(mq,1), fc, fq,
                        "|".join(str(b) for b in sorted({p[2] for p in pts}))])

    # --- 3) 去重 + 排优先级 ---
    cnt, work, secs = Counter(), defaultdict(float), defaultdict(float)
    for (mc, mq, fc, fq, pts) in rep.values():
        for _, _, b, aw, es in pts:
            k = (fc, fq, b)
            cnt[k] += 1; work[k] += aw; secs[k] += es
    # 默认按实测时间。attn 工作量当时间的代理并不好 —— 它把大 context 的点抬得过高, 真实耗时里
    # 有很大一块(KV offload/RDMA 收尾/kernel launch)不随 q·c 走。按次数排则会把 q≈1024 那排的
    # 尾巴 forward 顶到最前面: 次数多但每次都便宜。本例覆盖 80% 时间: time 190 点, count 206,
    # work 268 —— 代理排序反而最差。
    keyf = {"time": lambda k: -secs[k], "work": lambda k: -work[k],
            "count": lambda k: -cnt[k]}[a.rank_by]
    order = sorted(cnt, key=lambda k: (keyf(k), k))
    tn, tw, ts_ = sum(cnt.values()), sum(work.values()), sum(secs.values())
    cn = cw = cs = 0.0
    with open(os.path.join(a.dir, "final_points_unique.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["priority","final_context_length","final_query_length","eq_batch_size",
                    "n_forwards","exec_seconds","attn_work_pairs",
                    "cum_time_pct","cum_forward_pct","cum_work_pct"])
        for i, k in enumerate(order, 1):
            cn += cnt[k]; cw += work[k]; cs += secs[k]
            w.writerow([i, *k, cnt[k], round(secs[k], 2), int(work[k]),
                        round(100*cs/ts_, 2) if ts_ else "",
                        round(100*cn/tn, 2), round(100*cw/tw, 2)])

    # --- 自检输出 ---
    nf = len(seen)
    corners = {(v[2], v[3]) for v in rep.values()}
    print(f"{len(rows)} 行 / {nf} 个 forward -> {len(cells)} 个非空格子 "
          f"-> {len(corners)} 个角点 -> {len(cnt)} 个待测点 (c,q,b)")
    print("b 分布:", sorted(Counter(b for _, _, b in cnt).items()))
    tail = [v for v in rep.values() if v[1] < QB]
    inflated = [v for v in tail if v[3] > v[1] * 1.5]
    if inflated:
        print(f"⚠️ q 中位数 < {QB} 且被吸附放大 >1.5x 的格子: {len(inflated)}/{len(tail)}, "
              f"覆盖 {sum(len(v[4]) for v in inflated)} 个 forward "
              f"({sum(len(v[4]) for v in inflated)/nf:.1%}) —— 见 --no-snap-below")
    zero = [k for k in cnt if k[1] == 0]
    if zero:
        print(f"⚠️ final_query_length == 0 的待测点: {len(zero)} 个 —— 无新 token 即无 forward, "
              f"测不了。--no-snap-below 设成 --q-bin(默认)可消除")
    print(f"优先级按 {a.rank_by} 排 (--rank-by)"
          + (f", forward 执行时间合计 {ts_:.0f}s" if ts_ else ""))
    cn = cw = cs = 0.0; hit = {}
    for i, k in enumerate(order, 1):
        cn += cnt[k]; cw += work[k]; cs += secs[k]
        for th in (50, 80, 90, 95):
            if th not in hit and ts_ and cs/ts_ >= th/100:
                hit[th] = (i, cn/tn, cw/tw)
    for th in sorted(hit):
        i, fn, fw = hit[th]
        print(f"  覆盖 {th}% 时间 -> {i:3d} 点 (次数 {fn:5.1%}, 工作量 {fw:5.1%})")


main()
