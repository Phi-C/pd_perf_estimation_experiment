#!/usr/bin/env python3
"""把待测点 / 校准 workload 变成 aiperf mooncake_trace 文件（decode 侧）。

每个组（一个 (点, 形状) 或一个 (L_test, b) 点）出**一个** trace，跑两遍：

    prime : 同一个文件 + --extra-inputs max_tokens:1  —— 只为把 KV 灌进前缀缓存
    meas  : 同一个文件 + --extra-inputs max_tokens:O  —— 命中缓存后几乎立刻进 decode

⚠️ 必须是同一个文件。记录里的 `output_length` **参与合成 prompt 的内容** ——
hash_ids 和 input_length 一模一样、只把 output_length 从 300 改成 1，前缀缓存
就一条都不命中（实测 67s 冷 prefill，cache_read=0）。所以"灌 KV"这件事只能靠
`--extra-inputs max_tokens` 在客户端截断生成，不能靠改 trace 里的 output_length。

分两遍跑（而不是在一个 fixed-schedule 里排时间戳）是自计时的：prime 跑完 aiperf
才返回，不用猜 t0。挤在一跑里的话，5.5M token 的冷 prefill 要 ~70s，300 步 decode
只有 ~18s，测量窗口早跑没了。

§2.2 每条测量请求必须有**独立** KV：hash_ids 逐请求唯一，绝不跨请求共享。
「request i 命中自己 primer 留下的页」不算共享 —— primer 已经结束，测量时刻
b 条请求各自持有一组互不相交的物理页。

ISL 精确落到 L：mooncake 路径下实际 ISL = input_length + TPL(89)，
所以 input_length = L - TPL，L 是 1024 的整数倍 => 正好占满 L/1024 个 block。
"""
import argparse, csv, json, os
from collections import defaultdict

TPL = 89        # chat template 开销（bench_run/gen_b2_trace.py 实测值）
BLK = 1024      # --synthetic-input-tokens-block-size


class IdPool:
    """按 (L, 槽位) 分配 block id —— 组内唯一、跨组复用。

    组内：同一组里两条请求的 (L, ridx) 必不相同（ridx 不同），所以 hash_ids 不相交，
          §2.2 的「独立 KV」成立。长度不同的两条也不会共享前缀 —— key 里带了 L，
          短的那条不是长的那条的 id 前缀。
    跨组：同 L 同槽位的请求拿到同一串 id。把组按 L 排序、同 L 内 b 大的先跑，
          后面的组整条命中前缀缓存，prime 从冷 prefill 变成秒级。实测 top150
          能把 614M token 的冷 prefill 压到 83M（-86%）。
    """

    def __init__(self, nonce=0):
        self.base = 1_000_000 + nonce * 500_000_000
        self.cur = self.base
        self.m = {}

    def get(self, L, ridx):
        k = (L, ridx)
        if k not in self.m:
            self.m[k] = self.cur
            self.cur += L // BLK
        b = self.m[k]
        return [b + i for i in range(L // BLK)]


def emit(path, Ls, pool, out_len, t0=0, gap=0):
    rows = []
    for ridx, L in enumerate(Ls):
        rows.append(dict(timestamp=t0 + ridx * gap,
                         input_length=L - TPL,
                         output_length=out_len,
                         hash_ids=pool.get(L, ridx)))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--csv", required=True,
                    help="decode_calib_requests.csv 或 decode_points_top150.csv")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--mode", choices=["calib", "points"], required=True,
                    help="calib: 按 (point_id, shape) 分组；points: 每行一个 (L_test,b) 同构点")
    ap.add_argument("-O", "--output-tokens", type=int, default=None,
                    help="覆盖 output_length；calib 模式默认读 CSV 的 output_tokens")
    ap.add_argument("--prime-gap", type=int, default=0,
                    help="primer 之间的时间戳间隔(ms)；0 = 全部同时发，让引擎自己排")
    ap.add_argument("--nonce", type=int, default=0,
                    help="id 空间偏移。两批 trace 必须用不同的 nonce，否则 top150 会"
                         "悄悄命中校准批留下的页，prime/meas 的因果关系就断了")
    ap.add_argument("--top", type=int, default=0, help="points 模式只取前 N 行")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    groups = []          # (tag, [L_i...], O)
    if a.mode == "calib":
        g = defaultdict(list)
        O = {}
        for r in csv.DictReader(open(a.csv)):
            g[(r["point_id"], r["shape"])].append(int(r["prefix_tokens"]))
            O[(r["point_id"], r["shape"])] = int(r["output_tokens"])
        for k in sorted(g, key=lambda k: (k[0], k[1])):
            groups.append((f"{k[0]}__{k[1]}", sorted(g[k], reverse=True),
                           a.output_tokens or O[k]))
    else:
        for i, r in enumerate(csv.DictReader(open(a.csv))):
            if a.top and i >= a.top:
                break
            L, b = int(r["L_test"]), int(r["b"])
            groups.append((f"p{r['rank']}_L{L}_b{b}", [L] * b, a.output_tokens or 300))

    # 运行顺序 = 缓存复用顺序：先按平均长度排，同长度里 b 大的先跑（它把所有槽位
    # 都灌满，后面 b 小的组是纯命中）。index.csv 就是 runner 的执行顺序。
    groups.sort(key=lambda g: (sum(g[1]) / len(g[1]), -len(g[1]), g[0]))
    pool = IdPool(a.nonce)
    idx = []
    for gidx, (tag, Ls, O) in enumerate(groups):
        emit(os.path.join(a.outdir, tag + ".jsonl"), Ls, pool, O, gap=a.prime_gap)
        idx.append(dict(gidx=gidx, tag=tag, b=len(Ls), L_sum=sum(Ls),
                        L_eq=round(sum(Ls) / len(Ls), 2), L_max=max(Ls),
                        L_min=min(Ls), output_tokens=O))
    with open(os.path.join(a.outdir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(idx[0])); w.writeheader(); w.writerows(idx)

    tot = sum(r["L_sum"] for r in idx)
    uniq = sum(L // BLK * BLK for (L, _) in pool.m)
    print(f"{len(idx)} 组 -> {a.outdir}")
    print(f"  Σ prime tokens  : {tot:,}")
    print(f"  去重后真冷 prefill: {uniq:,}  (-{100-100*uniq/tot:.0f}%)  @19.6K tok/s ≈ {uniq/19600/3600:.2f} h")
    print(f"  单组 KV 峰值    : {max(r['L_sum'] for r in idx):,}")


if __name__ == "__main__":
    main()
