#!/usr/bin/env python3
"""Generate a K3 Prefill Mooncake trace with primer and fixed-schedule groups.

A primer leaves the checkpoint at c. Each measurement group uses one long hog
and seven 1024-token blockers to let the target requests queue together. The
actual batch must still be checked in the engine log. --members supports b>2.

The target requests share a primer prefix in this historical harness. Keep
this limitation when comparing against independent-KV batches. Template and
block constants are specific to the K3 setup; see the Prefill measurement doc.
"""
import json, sys, argparse

TPL = 89          # chat template 开销 (mooncake trace 路径下实测比 synthetic 路径多 1)
BLK = 1024        # --synthetic-input-tokens-block-size
Q   = 4096        # 每个请求在目标 batch 里贡献的新 token 数
DELTA = 2         # ISL = c + Q + DELTA, 保证 anchor 落在 c+Q

def hash_ids(total_len, shared_blocks, uniq_base):
    n = -(-total_len // BLK)
    ids = list(range(1, shared_blocks + 1))
    ids += [uniq_base + i for i in range(n - shared_blocks)]
    assert len(ids) == n, (len(ids), n)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--t0", type=int, default=45000, help="第一对的时间戳(ms), 要留够 primer 冷 prefill 的时间")
    ap.add_argument("--gap", type=int, default=4000)
    ap.add_argument("--lead", type=int, default=40, help="blocker 提前量(ms)")
    ap.add_argument("--blockers", type=int, default=1, help="兼容旧命令；当前构造固定使用一个 hog 和七个 blocker")
    ap.add_argument("--members", type=int, default=2, help="每组同时发几条测量请求(对照用 1)")
    ap.add_argument("--qq", type=int, default=Q, help="每条测量请求的目标块大小")
    # nonce: 测量请求/blocker 的 uniq hash_id 基址偏移。同一个点重跑时若 hash_ids 不变,
    # 生成的 prompt 内容逐字节相同 -> 整条 99328 token 被上一轮留下的 GPU prefix cache
    # 全量命中, 目标 chunk 那次 forward 根本不发生(日志里只剩 done:[c+q], new:[2] 的尾巴),
    # 验收必然 0/N。共享前缀 id 1..NB 不加 nonce —— 那是 primer 要复用的 checkpoint。
    ap.add_argument("--hog", type=int, default=8192 * 8 + 1024, help="占位请求的 ISL，使用 8192*k+1024 以配合七个 blocker")
    ap.add_argument("--nonce", type=int, default=0)
    ap.add_argument("-o", required=True)
    a = ap.parse_args()
    c = a.c
    assert c % BLK == 0, "c 必须 1024 对齐"
    NB = c // BLK
    out = []
    # primer
    out.append({"timestamp": 0, "input_length": c + 130 - TPL, "output_length": 1,
                "hash_ids": hash_ids(c + 130 - TPL, NB, 900_000)})
    for k in range(a.pairs):
        T = a.t0 + k * a.gap
        # blocker: 冷请求, ISL=1024 整块 -> anchor=floor_1024(1023)=0<=done, 不切块,
        # 一次 forward。8 个正好填满 max_num_batched_tokens=8192, 零余量 —— 测量请求
        # 挤不进 blocker 的那一步。发 blockers 个(必须是 8 的倍数), 让队列在 T 时刻还没排空:
        # FIFO 下两条测量请求被压在 blocker 后面, blocker 排干后它们必然独占一步 -> [q, q]。
        # prompt 很小(1024 token), 投递快, 不会像共享前缀的大 blocker 那样拖成好几步。
        # hog + 7 blocker: 让 hog 排空前的每一步余量都 < 1024, 测量请求一条都挤不进去。
        #   hog ISL = 8192*j + 1024 -> anchor=8192*j, 前 j 步每步正好吃满 8192(余量 0),
        #     尾块固定 1024。(尾块必然 <=1024, 所以单靠一条 hog 总有一步余量 >=1024 会漏。
        #      实测 ISL=65536 时 anchor=64512, 倒数第二步 new=7168、余量 1024, 漏进 1 条。)
        #   7 条 ISL 正好 1024 的 blocker 和那个 1024 的尾块拼成 8192, 余量 0。
        #     注意 input_length 要 +1: BLK-TPL 送进去实际 ISL 是 1023, 7 条只占 7161,
        #     余量 1031 >= 1024, 照样漏。
        # 只有 8 条请求要投递(hog 有 ~2.8s 掩护), 不存在之前 64 个 blocker 投不过来的问题。
        out.append({"timestamp": T - a.lead, "input_length": a.hog - TPL, "output_length": 1,
                    "hash_ids": [700_000 + a.nonce * 1_000_000 + 1000 * k + i
                                 for i in range(-(-a.hog // BLK))]})
        for bi in range(7):
            out.append({"timestamp": T - a.lead + 200, "input_length": BLK - TPL + 1,
                        "output_length": 1,
                        "hash_ids": [800_000 + a.nonce * 1_000_000 + 100 * k + bi]})
        # q>=1024 需要 DELTA 把 anchor 推到 c+qq(代价是一次 DELTA token 的尾巴 forward);
        # q<1024 时 anchor=floor_1024(c+qq-1)=c<=done, 本来就不切块, 加 DELTA 反而会切错。
        dl = DELTA if a.qq >= BLK else 0
        for j in range(a.members):
            L = c + a.qq + dl - TPL
            out.append({"timestamp": T, "input_length": L, "output_length": 1,
                        "hash_ids": hash_ids(L, NB, 100_000 + a.nonce * 1_000_000 + 1000 * k + 100 * j)})
    with open(a.o, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    dl = DELTA if a.qq >= BLK else 0
    print(f"{a.o}: {len(out)} entries | c={c} NB={NB} | members={a.members} "
          f"测量请求 ISL={c+a.qq+dl} anchor={c+a.qq if dl else c} delta={dl} | "
          f"primer ISL={c+130} | hog ISL={a.hog}")

main()
