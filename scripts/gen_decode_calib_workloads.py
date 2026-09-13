#!/usr/bin/env python3
"""异构校准 workload 生成器 —— 给 §10 的回归造成对样本。

背景（`ragged_packed_decode_workload_equivalence.md` §9/§10，
      `decode/measure_decode_performance_point.md` §6）：

    同构点只能测出 D_n(L_eq)。要拟合

        log(D_real / D_base) = β1·CV_L² + β2·log r_max + β3·CV_block²

    必须有**同 (L_test, b)、同 mean(L_i)、只有形状不同**的成对测量。

每个校准点造 4 种形状（mean(L_i) 全部锁定在 L_test，b 相同）：

    A_homog     b 条等长 L_test                     -> D_base，CV=0, r=1
    B_replay    回放 CI 里真实出现过的 L_i 向量      -> CV_L / r_max / CV_block 同时对齐
    C_rmax      1 长 + (b-1) 短，对齐 B 的 r_max     -> CV_L 偏低
    D_cv        2 长 + (b-2) 短，对齐 B 的 CV_L      -> r_max 偏低
    E_pageedge  (可选) 在 C 上把长请求推过 page 边界 -> §10.4 的边界样本

C 和 D 存在的唯一理由是**打散共线性**：两级形状（m 条长 + b-m 条短）里

    r_max = 1 + CV_L · sqrt((b-m)/m)

CV_L 和 r_max 被 b、m 锁死。如果所有样本都用 m=1，回归的 CV_L² 列和
log r_max 列几乎完全共线，β1 / β2 分不开。B(真实分布) + C(m=1) + D(m=2)
三者的 (CV_L, r_max) 组合互不相同，才能把两个系数解耦。

输出：
    <prefix>_points.csv    一行 = 一个 (校准点, 形状)，带落地后的真实特征
    <prefix>_requests.csv  一行 = 一条测量请求，prefix_tokens 就是要造的 KV 长度
"""
import argparse, csv, math, os, sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------- features
def feats(L, page):
    """§10.2：CV_L / r_max / CV_block。L 是落地后的整数向量。"""
    L = np.asarray(L, dtype=np.float64)
    mean = L.mean()
    nb = np.ceil(L / page)
    return dict(
        b=int(L.size),
        L_eq=round(float(mean), 2),
        L_min=int(L.min()), L_max=int(L.max()), L_sum=int(L.sum()),
        CV_L=round(float(L.std() / mean), 4),
        r_max=round(float(L.max() / mean), 4),
        CV_block=round(float(nb.std() / nb.mean()), 4) if nb.mean() else 0.0,
    )


def land(L, mu, align, lo, hi):
    """把浮点长度向量落到 align 的整数倍，并把 sum 拉回 b*mu。

    §2.1：L 只能取 hash_block_size(=1024) 的整数倍。四舍五入会让均值漂，
    所以按 align 为单位把差额补回去 —— 补在离边界最远的元素上，避免把某条
    请求推出 [lo, hi]。返回 None 表示这个形状在约束下造不出来。
    """
    L = np.asarray(L, dtype=np.float64)
    # 先按比例把均值拉到 mu 再取整：缩放不改变 CV_L / r_max，而“先取整再补差额”
    # 会把整个均值偏移堆到最长的那条上（实测能把 r_max 打歪 7%）。
    if L.mean() > 0:
        L = L * (mu / L.mean())
    if L.min() < lo - align or L.max() > hi + align:
        return None
    u = np.clip(np.rint(L / align), max(1, lo // align), hi // align).astype(np.int64)
    want = int(round(len(u) * mu / align))
    for _ in range(4 * len(u) + 8):
        d = want - int(u.sum())
        if d == 0:
            break
        step = 1 if d > 0 else -1
        # 改动最“有余地”的那条：加时挑最短的，减时挑最长的（且不越界）
        cand = [i for i in range(len(u))
                if lo // align <= u[i] + step <= hi // align]
        if not cand:
            return None
        i = min(cand, key=lambda i: u[i]) if d > 0 else max(cand, key=lambda i: u[i])
        u[i] += step
    if int(u.sum()) != want:
        return None
    return (u * align).astype(np.int64)


def two_level(b, mu, m, cv=None, rmax=None):
    """m 条长 + (b-m) 条短，mean 恒为 mu。给 cv 或 rmax 之一。

    令长的偏离 A = ell - mu，则 short = mu - m·A/(b-m)，
        var = A²·m/(b-m)   =>   CV = (A/mu)·sqrt(m/(b-m))
        r_max = 1 + A/mu
    """
    if m >= b:
        return None
    if cv is not None:
        a = cv * math.sqrt((b - m) / m)          # A/mu
    else:
        a = rmax - 1.0
    if a <= 0:
        return None
    ell = mu * (1.0 + a)
    short = mu * (1.0 - a * m / (b - m))
    if short <= 0:
        return None
    return [ell] * m + [short] * (b - m)


# ---------------------------------------------------------------- selection
def pick_points(rows, n, strata):
    """按 r_max_p50 分层，层内按 token 占比降序，跨层 round-robin 取。

    §10.4 要的是“覆盖不均衡跨度”，不是“取最热的 N 个”—— 最热的点全挤在
    r_max≈3.7 那一堆，拟合不出斜率。同时用 b 的分桶做去重，避免同一个 b 刷屏。
    """
    rows = sorted(rows, key=lambda r: float(r["r_max_p50"]))
    bins = [[] for _ in range(strata)]
    for i, r in enumerate(rows):
        bins[min(strata - 1, i * strata // len(rows))].append(r)
    for bkt in bins:
        bkt.sort(key=lambda r: -float(r["tok_pct"]))
    out, seen_b = [], defaultdict(int)
    while len(out) < n and any(bins):
        for bkt in bins:
            if not bkt or len(out) >= n:
                continue
            # 层内优先挑 b 分桶出现次数最少的，保证 b 也有跨度
            j = min(range(len(bkt)),
                    key=lambda j: (seen_b[int(bkt[j]["b"]) // 8], -float(bkt[j]["tok_pct"])))
            r = bkt.pop(j)
            seen_b[int(r["b"]) // 8] += 1
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default="decode")
    ap.add_argument("--points-csv", default=None,
                    help="默认 <dir>/decode_points_top150.csv")
    ap.add_argument("-o", "--out-prefix", default=None,
                    help="默认 <dir>/decode_calib")
    ap.add_argument("-n", "--points", type=int, default=16,
                    help="校准点个数（§6 建议 15~20）")
    ap.add_argument("--strata", type=int, default=5, help="r_max 分层数")
    ap.add_argument("--shapes", default="A,B,C,D",
                    help="A_homog,B_replay,C_rmax,D_cv,E_pageedge 的首字母")
    ap.add_argument("--phase", default="profiling")
    ap.add_argument("--page", type=int, default=1024,
                    help="KV page = block_size * dcp_world_size (128*8)")
    ap.add_argument("--align", type=int, default=1024,
                    help="hash_block_size；L 只能取它的整数倍 (§2.1)")
    ap.add_argument("--min-len", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=1048576,
                    help="服务端 max_model_len；超了直接判 infeasible")
    ap.add_argument("--output-tokens", type=int, default=300, help="§2.4 的 O")
    a = ap.parse_args()

    pts_csv = a.points_csv or os.path.join(a.dir, "decode_points_top150.csv")
    pre = a.out_prefix or os.path.join(a.dir, "decode_calib")
    want = {c.upper() for c in a.shapes.split(",") if c.strip()}

    points = pick_points(list(csv.DictReader(open(pts_csv))), a.points, a.strata)

    # --- membership 区间起点的 L_i 向量（用于 B 形状回放） -------------------
    segs = defaultdict(list)
    seg_step = {}
    for r in csv.DictReader(open(os.path.join(a.dir, "decode_members.csv"))):
        rid = int(r["row_idx"])
        segs[rid].append(int(r["context_length"]))
        seg_step[rid] = int(r["step"])

    # --- 每个 (L_test,b) 下所有 forward 的特征，用来挑代表 -------------------
    cand = defaultdict(list)
    keys = {(int(p["L_test"]), int(p["b"])) for p in points}
    for r in csv.DictReader(open(os.path.join(a.dir, "decode_eq_batches.csv"))):
        if a.phase and r["phase"] != a.phase:
            continue
        k = (int(r["L_test"]), int(r["b"]))
        if k in keys:
            cand[k].append((int(r["seg_row_idx"]), int(r["step"]),
                            float(r["CV_L"]), float(r["r_max"]), float(r["CV_block"])))

    prow, rrow, warn = [], [], []
    for p in points:
        L_test, b = int(p["L_test"]), int(p["b"])
        tgt = (float(p["CV_L_p50"]), float(p["r_max_p50"]), float(p["CV_block_p50"]))
        # 代表 forward：三个特征归一化后离该点中位数最近的那一个
        c = cand[(L_test, b)]
        if not c:
            warn.append(f"point rank={p['rank']} ({L_test},{b}) 在 phase={a.phase} 下没有 forward")
            continue
        rid, step, *_ = min(
            c, key=lambda x: sum(((x[2 + i] - tgt[i]) / (tgt[i] or 1.0)) ** 2 for i in range(3)))
        real = np.asarray(segs[rid], dtype=np.int64) + (step - seg_step[rid])
        fr = feats(real, a.page)

        shapes = []
        if "A" in want:
            shapes.append(("A_homog", [float(L_test)] * b, "基准：b 条等长"))
        if "B" in want:
            shapes.append(("B_replay", real.astype(float).tolist(),
                           f"回放 seg_row_idx={rid} step={step} 的真实 L_i"))
        if "C" in want:
            shapes.append(("C_rmax", two_level(b, L_test, 1, rmax=fr["r_max"]),
                           "1 长 + (b-1) 短，对齐 B 的 r_max"))
        if "D" in want:
            shapes.append(("D_cv", two_level(b, L_test, 2, cv=fr["CV_L"]),
                           "2 长 + (b-2) 短，对齐 B 的 CV_L"))
        if "E" in want:
            base = two_level(b, L_test, 1, rmax=fr["r_max"])
            if base:
                # §10.4：把最长那条推到刚跨过一个 page 边界（+1 token 落在新 page）
                e = list(base)
                e[0] = (math.floor(e[0] / a.page) + 1) * a.page + 1
                e[1:] = [(b * L_test - e[0]) / (b - 1)] * (b - 1)
                shapes.append(("E_pageedge", e, "C 的长请求推过 page 边界 (§10.4)"))
                if a.align % a.page == 0 and not any("E_pageedge" in w for w in warn):
                    warn.append("E_pageedge 与 C_rmax 落地后完全相同：align 是 page 的整数倍，"
                                "那个 +1 token 会被重新取整抹掉。前缀缓存法根本造不出 page 边界样本，"
                                "要探边界只能绕开 §2.1（自己塞 KV 或改 block_size）。")

        for name, vec, note in shapes:
            pid = f"r{p['rank']}_{L_test}_{b}"
            if vec is None:
                prow.append(dict(point_id=pid, rank=p["rank"], L_test=L_test, b=b,
                                 shape=name, feasible=0, note=f"{note}；约束下无解"))
                continue
            u = land(vec, L_test, a.align, a.min_len, a.max_len)
            if u is None:
                prow.append(dict(point_id=pid, rank=p["rank"], L_test=L_test, b=b,
                                 shape=name, feasible=0,
                                 note=f"{note}；落到 {a.align} 倍数后越界"))
                continue
            f = feats(u, a.page)
            f.pop("b")                      # 与外层 b 同值，避免重复关键字
            prow.append(dict(
                point_id=pid, rank=p["rank"], L_test=L_test, b=b, shape=name,
                feasible=1,
                **f,
                L_eq_err_pct=round(100 * (f["L_eq"] - L_test) / L_test, 4),
                CV_L_ref=fr["CV_L"], r_max_ref=fr["r_max"], CV_block_ref=fr["CV_block"],
                x_cv2=round(f["CV_L"] ** 2, 6),
                x_logr=round(math.log(f["r_max"]), 6),
                x_cvb2=round(f["CV_block"] ** 2, 6),
                src_seg_row_idx=rid, src_step=step,
                output_tokens=a.output_tokens, note=note,
            ))
            for i, L in enumerate(sorted(u.tolist(), reverse=True)):
                rrow.append(dict(point_id=pid, shape=name, req_idx=i,
                                 prefix_tokens=int(L), output_tokens=a.output_tokens,
                                 unique_lead_tokens=a.align))

    if not prow:
        sys.exit("没有可用的校准点")

    cols = ["point_id", "rank", "L_test", "b", "shape", "feasible", "L_eq", "L_eq_err_pct",
            "L_min", "L_max", "L_sum", "CV_L", "r_max", "CV_block",
            "CV_L_ref", "r_max_ref", "CV_block_ref", "x_cv2", "x_logr", "x_cvb2",
            "src_seg_row_idx", "src_step", "output_tokens", "note"]
    with open(pre + "_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(prow)
    with open(pre + "_requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rrow[0]))
        w.writeheader()
        w.writerows(rrow)

    ok = [r for r in prow if r["feasible"]]
    print(f"{pre}_points.csv   : {len(prow)} 行 ({len(points)} 个点 × 形状), 可行 {len(ok)}")
    print(f"{pre}_requests.csv : {len(rrow)} 条请求")
    print(f"  Σ prefix tokens (串行冷跑上界): {sum(r['prefix_tokens'] for r in rrow):,}")
    print(f"  单点 KV 峰值 max Σ L_i        : {max(r['L_sum'] for r in ok):,}")
    print(f"  |L_eq - L_test| 最大偏差       : {max(abs(r['L_eq_err_pct']) for r in ok):.4f}%")

    # 共线性诊断：这三列就是回归的自变量，相关系数越低越好
    X = np.array([[r["x_cv2"], r["x_logr"], r["x_cvb2"]] for r in ok], dtype=float)
    names = ["CV_L^2", "log r_max", "CV_block^2"]
    print("\n设计矩阵相关系数（|ρ|>0.95 说明系数分不开）：")
    C = np.corrcoef(X.T)
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  ρ({names[i]:>10s}, {names[j]:>10s}) = {C[i, j]:+.4f}")
    def cond(M):
        Mc = M - M.mean(0)
        sv = np.linalg.svd(Mc, compute_uv=False)
        return sv[0] / sv[-1] if sv[-1] > 1e-12 else float("inf")
    print(f"  条件数 cond(X, 3 列) = {cond(X):.4g}")
    print(f"  条件数 cond(X, 去掉 CV_block^2) = {cond(X[:, :2]):.2f}")
    if abs(C[0, 2]) > 0.999:
        print("  ⚠️ CV_block^2 与 CV_L^2 完全共线：L 必须取 hash_block_size 的整数倍，")
        print("     于是 N_i = L_i/P 精确成立、CV_block ≡ CV_L。β3 在本 setup 下不可识别，")
        print("     回归退化为 log(D_real/D_base) = β1·CV_L² + β2·log r_max。")
    for name in sorted({r['shape'] for r in ok}):
        sub = [r for r in ok if r["shape"] == name]
        print(f"  {name:<11s} n={len(sub):<3d} CV_L {min(r['CV_L'] for r in sub):.3f}..{max(r['CV_L'] for r in sub):.3f}"
              f"  r_max {min(r['r_max'] for r in sub):.3f}..{max(r['r_max'] for r in sub):.3f}")
    for w_ in warn:
        print("⚠️ " + w_)


if __name__ == "__main__":
    main()
