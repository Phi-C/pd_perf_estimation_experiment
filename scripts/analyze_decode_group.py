#!/usr/bin/env python3
"""把一个 decode 组的两路口径对齐成一行结果。

  引擎侧 (_fwdprobe)  : fwd_gpu_ms（纯模型 forward）、step_wall_ms、step_period_ms
  客户端侧 (aiperf)   : ITL（含调度/网络/流式开销）

两者之差就是 §3 里 D_n(L) 到用户可见 ITL 之间的那层壳。

验收（§2.3 / §5）：只认 prefill==0 && !dummy && b==目标 && 所有 q==1 的 forward，
再把 ctx[] 还原出 L_eq / CV_L / r_max 和目标形状对账 —— 混进 prefill chunk 或
少一条请求的 forward 会让 D 偏小，必须显式判掉而不是平均掉。
"""
import argparse, json, math, os, statistics as st, sys
import numpy as np


def load_probe(path, start, end, b, drop):
    recs, bad = [], dict(prefill=0, dummy=0, wrong_b=0, wrong_q=0)
    with open(path, "rb") as f:
        f.seek(start)
        blob = f.read((end - start) if end > start else -1)
    for ln in blob.decode("utf-8", "replace").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue                      # 尾行可能被截断
        if r.get("prefill"):   bad["prefill"] += 1;  continue
        if r.get("dummy"):     bad["dummy"] += 1;    continue
        if r.get("b") != b:    bad["wrong_b"] += 1;  continue
        if any(q != 1 for q in r.get("q", [])): bad["wrong_q"] += 1; continue
        recs.append(r)
    return recs[drop:], bad, len(recs)


def feat(ctx):
    L = np.asarray(ctx, dtype=np.float64)
    m = L.mean()
    return dict(L_eq=float(m), L_max=float(L.max()), L_min=float(L.min()),
                CV_L=float(L.std() / m) if m else 0.0,
                r_max=float(L.max() / m) if m else 1.0)


def med(v):
    return float(st.median(v)) if v else float("nan")


def aiperf_itl(art_dir):
    """从 aiperf 产物里取客户端侧 ITL / 输出吞吐。优先逐请求 jsonl，退化到汇总 json。"""
    out = dict(itl_ms_p50=None, itl_ms_mean=None, itl_n=0, src=None,
               ttft_ms_p50=None, cache_read_tokens=None)
    for root, _, files in os.walk(art_dir):
        for fn in files:
            if fn.endswith(".json") and "aiperf" in fn:
                try:
                    d = json.load(open(os.path.join(root, fn)))
                except Exception:
                    continue
                r = d.get("records") or d
                g = lambda k, f="p50": (r.get(k) or {}).get(f)
                if g("inter_token_latency") is not None:
                    out.update(itl_ms_p50=g("inter_token_latency"),
                               itl_ms_mean=g("inter_token_latency", "avg"),
                               itl_n=g("inter_token_latency", "count") or 0,
                               ttft_ms_p50=g("time_to_first_token"),
                               cache_read_tokens=g("usage_prompt_cache_read_tokens", "avg"),
                               src=fn)
                    return out
    jl = None
    for root, _, files in os.walk(art_dir):
        for fn in files:
            if fn == "profile_export.jsonl":
                jl = os.path.join(root, fn)
    if jl:
        gaps = []
        for ln in open(jl):
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ts = d.get("response_timestamps") or d.get("responses") or []
            ts = [t.get("timestamp", t) if isinstance(t, dict) else t for t in ts]
            ts = [t for t in ts if isinstance(t, (int, float))]
            if len(ts) >= 3:
                # 第一个 gap 里混着 TTFT 的尾巴，丢掉
                gaps += [(ts[i + 1] - ts[i]) / 1e6 for i in range(1, len(ts) - 1)]
        if gaps:
            out.update(itl_ms_p50=round(med(gaps), 3),
                       itl_ms_mean=round(sum(gaps) / len(gaps), 3),
                       itl_n=len(gaps), src="profile_export.jsonl")
            return out
    for root, _, files in os.walk(art_dir):
        for fn in files:
            if fn.endswith(".json") and "aiperf" in fn:
                try:
                    d = json.load(open(os.path.join(root, fn)))
                except Exception:
                    continue
                for k in ("inter_token_latency", "inter_token_latencies"):
                    m = (d.get("records", d) or {}).get(k) if isinstance(d, dict) else None
                    if isinstance(m, dict):
                        sc = 1e-6 if m.get("unit", "ms") == "ns" else 1.0
                        out.update(itl_ms_p50=m.get("p50"), itl_ms_mean=m.get("avg"),
                                   itl_n=m.get("count", 0), src=fn)
                        return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--artifact-dir", default="")
    ap.add_argument("-b", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--L-eq", type=float, default=0.0)
    ap.add_argument("--cv-ref", type=float, default=None)
    ap.add_argument("--rmax-ref", type=float, default=None)
    ap.add_argument("--drop", type=int, default=20, help="§2.4 丢弃前 K 步暖机")
    ap.add_argument("--min-steps", type=int, default=100)
    a = ap.parse_args()

    if not os.path.exists(a.probe):
        print(json.dumps(dict(tag=a.tag, accept="BAD", reason="probe 文件不存在")))
        return
    recs, bad, n_raw = load_probe(a.probe, a.start, a.end, a.b, a.drop)

    r = dict(tag=a.tag, b=a.b, L_eq_target=a.L_eq,
             n_probe_all=n_raw, n_probe_used=len(recs),
             skip_prefill=bad["prefill"], skip_dummy=bad["dummy"],
             skip_wrong_b=bad["wrong_b"], skip_wrong_q=bad["wrong_q"])
    if len(recs) < a.min_steps:
        r.update(accept="BAD", reason=f"可用 decode forward 只有 {len(recs)} < {a.min_steps}")
        print(json.dumps(r, ensure_ascii=False)); return

    f0, f1 = feat(recs[0]["ctx"]), feat(recs[-1]["ctx"])
    # 窗口内每步 L_i 各 +1，D 对应的是窗口中点的 L，不是首步的 L
    L_mid = float(np.mean([np.mean(x["ctx"]) for x in recs]))
    fwd = [x["fwd_gpu_ms"] for x in recs if x.get("fwd_gpu_ms") is not None]
    wall = [x["step_wall_ms"] for x in recs if x.get("step_wall_ms") is not None]
    per = [x["step_period_ms"] for x in recs if x.get("step_period_ms") is not None]
    D = med(per) or float("nan")

    r.update(
        L_eq_meas=round(f0["L_eq"], 1), L_eq_mid=round(L_mid, 1), L_eq_end=round(f1["L_eq"], 1),
        L_max_meas=round(f0["L_max"], 1), L_min_meas=round(f0["L_min"], 1),
        CV_L_meas=round(f0["CV_L"], 4), r_max_meas=round(f0["r_max"], 4),
        fwd_gpu_ms_p50=round(med(fwd), 3), fwd_gpu_ms_mean=round(sum(fwd)/len(fwd), 3) if fwd else None,
        step_wall_ms_p50=round(med(wall), 3),
        step_period_ms_p50=round(D, 3),
        # §3: D_n(L) 就是 ITL；TPS 是 batch 的 decode token 吞吐，绝不是各请求吞吐之和的平均
        D_ms=round(D, 3), tps_tok_s=round(a.b / D * 1000.0, 2) if D == D and D > 0 else None,
        overhead_ms=round(D - med(fwd), 3) if fwd else None,
        gpu_busy_frac=round(med(fwd) / D, 4) if fwd and D else None,
    )
    if a.artifact_dir:
        r.update(aiperf_itl(a.artifact_dir))
        if r.get("itl_ms_p50") and D:
            r["itl_minus_D_ms"] = round(r["itl_ms_p50"] - D, 3)

    why = []
    if a.L_eq and abs(L_mid - a.L_eq) / a.L_eq > 0.02:
        why.append(f"L_eq 实测 {L_mid:.0f} 偏离目标 {a.L_eq:.0f} >2%")
    if a.cv_ref is not None and abs(f0["CV_L"] - a.cv_ref) > 0.02:
        why.append(f"CV_L 实测 {f0['CV_L']:.4f} != 目标 {a.cv_ref:.4f}")
    if a.rmax_ref is not None and abs(f0["r_max"] - a.rmax_ref) > 0.05:
        why.append(f"r_max 实测 {f0['r_max']:.4f} != 目标 {a.rmax_ref:.4f}")
    # prime 没把 KV 灌进前缀缓存的话，meas 会整条冷 prefill —— 引擎侧 D 仍然干净
    # （过滤器只认纯 decode forward），但客户端 ITL 会被排队时间稀释好几倍，
    # 两路口径就没法对账了。所以显式判掉，而不是让它混进结果表。
    cr = r.get("cache_read_tokens")
    if cr is not None and a.L_eq and cr < 0.5 * a.L_eq:
        why.append(f"meas 未命中 prime 的前缀缓存 (cache_read={cr:.0f} << L={a.L_eq:.0f})，ITL 不可用")
    if bad["wrong_b"] > len(recs):
        why.append(f"batch 不稳：{bad['wrong_b']} 个 forward 的 b 不等于 {a.b}")
    r["accept"] = "OK" if not why else "BAD"
    r["reason"] = "; ".join(why)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
