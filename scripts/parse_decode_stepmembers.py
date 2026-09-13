#!/usr/bin/env python3
"""Rebuild every decode forward batch from the `[StepMembers]` delta log.

Input : a decode-node log containing
          [atom HH:MM:SS] [StepMembers] step=S decode n=N tokens=T +[..] -[..]
          [atom HH:MM:SS] [PD-TRANSITION] seq R: num_tokens=ISL, ...
Output: decode_steps.csv / decode_requests.csv / decode_stat.md

The [StepMembers] line is a *delta*: it is emitted only on steps where the
membership changed, so the set it reports holds until the next line.  Replaying
the deltas gives the full running batch of every decode forward, which is what
the workload-equivalence mapping needs (it wants every member's L_i, not a
sample).  `n=` is the redundancy that makes a dropped line fail loudly instead
of silently corrupting the replay.
"""
import argparse, csv, math, os, re, statistics, sys
from collections import Counter

RE_MEM = re.compile(
    r"\[atom (\d+):(\d+):(\d+)\].*?\[StepMembers\] step=(\d+) decode n=(\d+) "
    r"tokens=(\d+) \+\[(.*?)\] -\[(.*?)\]"
)
RE_TRANS = re.compile(r"\[PD-TRANSITION\] seq (\d+): num_tokens=(\d+)")
RE_BLOCK = re.compile(r"\[PD-TRANSITION\] seq \d+: num_tokens=\d+, num_prompt=\d+, blocks=(\d+)")


def _ids(s):
    s = s.strip()
    return [int(x) for x in s.split(",")] if s else []


def parse(path):
    isl, blocks, lines = {}, {}, []
    seen_steps = set()
    dup = 0
    for ln in open(path, errors="replace"):
        m = RE_TRANS.search(ln)
        if m:
            isl[int(m.group(1))] = int(m.group(2))
            b = RE_BLOCK.search(ln)
            if b:
                blocks[int(m.group(1))] = int(b.group(1))
            continue
        m = RE_MEM.search(ln)
        if m:
            step = int(m.group(4))
            if step in seen_steps:          # CI logs replay whole blocks verbatim
                dup += 1
                continue
            seen_steps.add(step)
            lines.append(
                dict(
                    t=int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)),
                    hhmmss=f"{m.group(1)}:{m.group(2)}:{m.group(3)}",
                    step=step,
                    n=int(m.group(5)),
                    tokens=int(m.group(6)),
                    joins=_ids(m.group(7)),
                    leaves=_ids(m.group(8)),
                )
            )
    lines.sort(key=lambda r: r["step"])
    return isl, blocks, lines, dup


def features(Ls, page):
    """L_eq plus the imbalance features of §10.2 of the equivalence note."""
    n = len(Ls)
    mean = sum(Ls) / n
    if n > 1 and mean > 0:
        cv = statistics.pstdev(Ls) / mean
        nb = [math.ceil(L / page) for L in Ls]
        mb = sum(nb) / n
        cvb = statistics.pstdev(nb) / mb if mb else 0.0
    else:
        cv = cvb = 0.0
    rmax = max(Ls) / mean if mean else 1.0
    return mean, cv, rmax, cvb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--outdir", default="decode")
    ap.add_argument("--label", default="")
    ap.add_argument("--page", type=int, default=1024,
                    help="KV page for CV_block: block_size * dcp_world_size (128*8)")
    ap.add_argument("--max-span", type=int, default=256,
                    help="split a stable stretch longer than this so L_eq stays local")
    ap.add_argument("--no-members", action="store_true",
                    help="skip decode_members.csv (the per-(forward, request) expansion)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    isl, blocks, lines, dup = parse(a.log)
    if not lines:
        sys.exit("no [StepMembers] lines: was ATOM_LOG_STEP_MEMBERSHIP=1 set?")

    cur, done = set(), Counter()
    member_rows = []
    first_step, last_step, joins_n = {}, {}, Counter()
    rows, mismatch, missing_isl = [], 0, set()

    for k, ln in enumerate(lines):
        cur -= set(ln["leaves"])
        for j in ln["joins"]:
            joins_n[j] += 1
            first_step.setdefault(j, ln["step"])
        cur |= set(ln["joins"])
        if len(cur) != ln["n"]:
            mismatch += 1
        nxt = lines[k + 1]["step"] if k + 1 < len(lines) else ln["step"] + 1
        span = nxt - ln["step"]
        members = sorted(cur)
        for i in members:
            last_step[i] = nxt - 1
            if i not in isl:
                missing_isl.add(i)

        off = 0
        while off < span:                       # keep each emitted row local in L
            d = min(a.max_span, span - off)
            Ls = [isl.get(i, 0) + done[i] + off + (d - 1) / 2.0 for i in members]
            L_eq, cv, rmax, cvb = features(Ls, a.page)
            rid = len(rows)
            if not a.no_members:
                for i, L in zip(members, Ls):
                    member_rows.append(
                        dict(row_idx=rid, step=ln["step"] + off, time=ln["hhmmss"],
                             span_steps=d, b=len(members), req_id=i, ISL=isl.get(i, ""),
                             gen_tokens=done[i] + off,
                             gen_tokens_end=done[i] + off + d,
                             total_decode_steps=0,   # back-filled below
                             query_tokens=1,
                             context_length=isl.get(i, 0) + done[i] + off,
                             context_at_span_mid=round(L, 1),
                             kv_blocks=blocks.get(i, ""))
                    )
            rows.append(
                dict(
                    row_idx=rid,
                    step=ln["step"] + off,
                    time=ln["hhmmss"],
                    span_steps=d,
                    b=len(members),
                    tokens_reported=ln["tokens"] if off == 0 else "",
                    L_eq=round(L_eq, 1),
                    L_min=int(min(Ls)),
                    L_max=int(max(Ls)),
                    L_sum=int(sum(Ls)),
                    CV_L=round(cv, 4),
                    r_max=round(rmax, 4),
                    CV_block=round(cvb, 4),
                    eq_confidence="high" if rmax <= 1.2 else ("mid" if rmax <= 2.5 else "low"),
                )
            )
            off += d
        for i in members:
            done[i] += span

    with open(os.path.join(a.outdir, "decode_steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if member_rows:
        # one row = one *stable stretch*, not one forward: weight by span_steps.
        # total_decode_steps is the request's full output length (== decode_requests.csv).
        for r in member_rows:
            r["total_decode_steps"] = done[r["req_id"]]
        with open(os.path.join(a.outdir, "decode_members.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(member_rows[0]))
            w.writeheader()
            w.writerows(member_rows)

    reqs = []
    for i in sorted(set(first_step)):
        reqs.append(dict(req_id=i, ISL=isl.get(i, ""), kv_blocks=blocks.get(i, ""),
                         first_step=first_step[i], last_step=last_step.get(i, ""),
                         decode_steps=done[i], join_events=joins_n[i]))
    with open(os.path.join(a.outdir, "decode_requests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(reqs[0]))
        w.writeheader()
        w.writerows(reqs)

    span_total = lines[-1]["step"] - lines[0]["step"] + 1
    tok = sum(done.values())
    hist = Counter()
    for r in rows:
        hist[r["b"]] += r["span_steps"]
    stat = [
        f"# Decode forward 统计 (StepMembers 重放){' ' + a.label if a.label else ''}",
        "",
        f"log: `{a.log}`",
        "",
        f"- `[StepMembers]` 行: **{len(lines)}** (重放去重 {dup} 行)",
        f"- step 区间: {lines[0]['step']} .. {lines[-1]['step']} (跨度 {span_total})",
        f"- 重放出的 forward batch 行: **{len(rows)}** (按 --max-span {a.max_span} 切段)",
        f"- 请求: **{len(reqs)}**, 缺 ISL {len(missing_isl)}, 重复 join(抢占) {sum(1 for r in reqs if r['join_events'] > 1)}",
        f"- ⚠️ n= 自检失配: **{mismatch}** (必须为 0)",
        f"- 重放出的 decode token 总数: **{tok:,}**  ← 与 aiperf 的 completion tokens 对账",
        f"- step 加权平均 batch size: **{sum(k*v for k,v in hist.items())/sum(hist.values()):.2f}**",
        "",
        "## batch size 的 step 占比",
        "",
        "| b | steps | 占比 |",
        "|---:|---:|---:|",
    ]
    tot = sum(hist.values())
    for b in sorted(hist):
        stat.append(f"| {b} | {hist[b]:,} | {100*hist[b]/tot:.2f}% |")
    v = sorted(x for x in isl.values())
    stat += ["", "## 进入 decode 的请求 ISL 分布", "", "| 分位 | ISL |", "|---|---:|"]
    for name, q in (("min", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9), ("p99", .99), ("max", 1)):
        stat.append(f"| {name} | {v[min(len(v)-1, int(q*(len(v)-1)))]:,} |")
    open(os.path.join(a.outdir, "decode_stat.md"), "w").write("\n".join(stat) + "\n")

    print(f"lines={len(lines)} dup={dup} rows={len(rows)} members={len(member_rows)} reqs={len(reqs)} "
          f"n_mismatch={mismatch} missing_isl={len(missing_isl)} tokens={tok}")


if __name__ == "__main__":
    main()
