#!/usr/bin/env bash
# Stage B: 测 final_points_unique.csv 前 N 名里 b>=2 的等价点。
# 派生自 run_rows423_428_b2_fs.sh, 三处推广:
#   1) b 不再写死 2 —— gen_b2_trace.py --members b / analyze_bn_fs.py -b b 对任意 b 都成立;
#   2) q 不再写死 4096 —— q<1024 时 gen 里 DELTA=0(anchor 本来就 <=done, 不切块),
#      验收用 ±16 容差并把引擎侧实测 q 记进 q_actual 列, 建模用 q_actual;
#   3) t0 随 c 放大 —— primer 是 c token 的冷 prefill, c=750k 实测要 ~107s,
#      t0 不够长的话 blocker 和测量请求会堆在 primer 后面, 前几组全废。
#
# 为什么 blocker 数必须是 8 的倍数: ISL=1024 的冷请求一次 forward, 8 个正好填满
# max_num_batched_tokens=8192、零余量。每个 blocker step 都精确填满, 排干之后队列里
# 只剩这 b 条测量请求, 没有第三方能被拉进来 —— 余量大小与 q 无关, 所以 q<1024 也成立。
#
# 跳过 b*q > 8192 的点: 预算不变的前提下物理上凑不出这个 batch(见 SKILL.md ③)。
# 用法: ./run_stage_bn.sh [TOPN] [PAIRS]
set -uo pipefail
RUN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOPN=${1:-214}; PAIRS=${2:-8}; ONLY=${3:-}   # ONLY=逗号分隔的 priority, 试跑用
LEAD=${LEAD:-1200}; TAGSUF=${TAGSUF:-}
# SRC/ART/CSV/TR 可用环境变量覆盖 —— 换口径重选点后 priority 会和上一轮撞车,
# 必须写到独立的 CSV, 否则 "已有结果" 的 skip 会把新点误跳过。
SRC=${SRC:-$RUN/../profiling/final_points_unique.csv}
ART=${ART:-$RUN/aiperf_stage_bn}; CSV=${CSV:-$RUN/stage_bn_timing.csv}
TR=${TR:-$RUN/bn_traces}
MODEL=${MODEL:-/mnt/models/Kimi-K3}; URL=${URL:-http://127.0.0.1:8010}
CLIENT=${CLIENT:-k3-aiperf-client}; SERVER=${SERVER:-k3-prefill-bench}
mkdir -p "$ART" "$TR"
docker exec "$SERVER" true 2>/dev/null || { echo "server 容器不在"; exit 1; }

HDR="priority,c,q,q_actual,b,n_forwards,pairs,seconds,rc,n_meas_engine,n_target_batch,n_split_batch,n_req,accept,ttft_raw_avg_ms,queue_avg_ms,ttft_net_avg_ms,ttft_net_p50_ms,ttft_net_p90_ms,ttft_net_p99_ms,fresh_tok_s,per_user_tok_s,isl_mean,cache_read_mean"
[[ -s "$CSV" ]] || echo "$HDR" > "$CSV"

# 取前 TOPN 名里 b>=2 且 b*q<=8192 的点, 按 (c, priority) 排 —— 同 c 的点连着跑, 前缀不容易被挤掉
mapfile -t PTS < <(awk -F, -v n="$TOPN" 'NR>1 && NR<=n+1 && $4>=2 && $4*$3<=8192 {print $2","$3","$4","$1","$5}' "$SRC" \
                   | sort -t, -k1,1n -k4,4n)
if [[ -n "$ONLY" ]]; then
  mapfile -t PTS < <(printf "%s\n" "${PTS[@]}" | awk -F, -v o=",$ONLY," '{ if (index(o, ","$4",")) print }')
fi
SKIP=$(awk -F, -v n="$TOPN" 'NR>1 && NR<=n+1 && $4>=2 && $4*$3>8192' "$SRC" | wc -l)
echo "共 ${#PTS[@]} 个 b>=2 点待测 (另有 $SKIP 个 b*q>8192 不可行, 已跳过)"

NONCE=$(( ($(date +%s) / 60) % 900 + 1 ))   # 见下面 --nonce 的说明
echo "NONCE=$NONCE (hash_ids 基址偏移, 防命中上一轮自己的前缀缓存)"
i=0; T0=$(date +%s)
for p in "${PTS[@]}"; do
  IFS=, read -r c q b prio nf <<<"$p"; i=$((i+1))
  grep -q "^$prio," "$CSV" && { echo "  skip p$prio (已有结果)"; continue; }
  tag="p${prio}_c${c}_q${q}_b${b}${TAGSUF}"; SEED=$(( 7000 + prio ))
  # ⚠️ 必须传 --nonce: hash_ids 决定请求内容, 同一个点重跑时若 hash_ids 不变, 整条 prompt
  # 会命中上一轮自己留下的前缀缓存 —— 目标 forward 变成 done=c+q/new=1 的尾巴, .sched 里
  # 一条 new:[q,...] 都抓不到, 结果全 NA/BAD。NONCE 每次运行变一次, 组内所有点共用。
  # (run_stage_b1.sh 早就有这个机制, 这里是漏了。)
  # primer 冷 prefill 约 c/8192 步 x ~0.6s, 给 2 倍余量再加 30s 底
  t0=$(( 30000 + c * 150 / 1000 ))
  python3 "$RUN/gen_b2_trace.py" -c "$c" --qq "$q" --members "$b" --pairs "$PAIRS" \
     --t0 "$t0" --gap 9000 --lead "$LEAD" --blockers 32 --nonce "$NONCE" -o "$TR/$tag.jsonl" > "$ART/$tag.trace.txt" 2>&1 \
     || { echo "[$(date -u +%H:%M:%S)] ($i/${#PTS[@]}) $tag GEN_FAIL: $(cat "$ART/$tag.trace.txt")"; continue; }
  echo "[$(date -u +%H:%M:%S)] ($i/${#PTS[@]}) $tag start | $(cat "$ART/$tag.trace.txt")"

  since=$(date -u +%Y-%m-%dT%H:%M:%S); t_0=$(date +%s.%N)
  docker exec "$CLIENT" aiperf profile \
    --model "$MODEL" --tokenizer "$MODEL" --tokenizer-trust-remote-code \
    --url "$URL" --endpoint-type chat --streaming --use-server-token-count \
    --custom-dataset-type mooncake_trace --input-file "$TR/$tag.jsonl" \
    --fixed-schedule --synthetic-input-tokens-block-size 1024 --workers-max 16 \
    --extra-inputs ignore_eos:true --extra-inputs min_tokens:1 --extra-inputs max_tokens:1 \
    --random-seed "$SEED" --ui simple --output-artifact-dir "$ART/$tag" > "$ART/$tag.log" 2>&1
  rc=$?; secs=$(echo "$(date +%s.%N) - $t_0" | bc)

  docker logs -t --since "$since" "$SERVER" 2>&1 | grep -E "Scheduled prefill batch" > "$ART/$tag.sched"
  jl=$(find "$ART/$tag" -name profile_export.jsonl 2>/dev/null | head -1)
  j=""
  [[ -n "$jl" ]] && j=$(python3 "$RUN/analyze_bn_fs.py" --log "$ART/$tag.sched" --jsonl "$jl" \
                          -c "$c" -q "$q" -b "$b" 2>"$ART/$tag.err")
  [[ -z "$j" ]] && j='{}'
  read -r nme ntgt nsplit nreq qact raw qav net p50 p90 p99 fr pu isl crt <<<"$(python3 - "$j" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); g=lambda k: d.get(k,"NA")
print(g("n_meas_engine"),g("n_target_batch"),g("n_split_batch"),g("n_req"),g("q_actual"),
      g("ttft_raw_avg"),g("queue_avg"),g("ttft_net_avg"),g("ttft_net_p50"),g("ttft_net_p90"),
      g("ttft_net_p99"),g("fresh_tok_s"),g("per_user_tok_s"),g("isl_mean"),g("cache_read_mean"))
PY
)"
  # 验收: 至少一半的组干净落成目标 batch, 且 cache_read 命中 c
  acc=BAD
  [[ "$ntgt" =~ ^[0-9]+$ ]] && [[ "$ntgt" -ge $((PAIRS/2)) ]] \
    && python3 -c "import sys;sys.exit(0 if abs(float('$crt')-$c)<=64 else 1)" 2>/dev/null && acc=OK

  echo "$prio,$c,$q,$qact,$b,$nf,$PAIRS,$secs,$rc,$nme,$ntgt,$nsplit,$nreq,$acc,$raw,$qav,$net,$p50,$p90,$p99,$fr,$pu,$isl,$crt" >> "$CSV"
  el=$(( $(date +%s) - T0 ))
  printf "[%s]   -> %s ttft_net=%sms (raw=%s queue=%s) q_act=%s target_batch=%s/%s split=%s | %.0fs | %dm\n" \
    "$(date -u +%H:%M:%S)" "$acc" "$net" "$raw" "$qav" "$qact" "$ntgt" "$PAIRS" "$nsplit" "$secs" $((el/60))
done
echo "TOTAL_SECONDS=$(( $(date +%s) - T0 ))"
