#!/usr/bin/env bash
# Stage A: final_points_unique.csv 里 b=1 的点。
# 构造: primer(ISL=c+130 -> anchor=floor_1024(c+129)=c) 在 c 处留 checkpoint, 然后
#   q>=1024 (必然 1024 对齐): ISL=c+q+2 -> anchor=c+q, 首个 forward 恰好 q (ISL 抖动被 anchor 吸收)
#   q< 1024:                  ISL=c+q   -> anchor=c<=done, 不切块, 单次 forward=q
#     ⚠️ 这一档没有 anchor 兜底, aiperf 合成长度有 ±10 token 抖动, 实测 q 会偏几个 token。
#        所以验收用容差, 并把引擎侧实测的 q 记进 q_actual 列 —— 建模时用 q_actual, 不要用 q。
# 按 c 分组: 同一个 c 的所有点共用一个 primer(种子 = f(c), 前缀内容相同), 连续跑保证前缀不被挤掉。
# 用法: ./run_stage_b1.sh [MAXPRIO] [点表CSV] [ART目录] [输出CSV]
set -uo pipefail
RUN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC=${2:-$RUN/../profiling/final_points_unique.csv}
MAXP=${1:-214}
ART=${3:-$RUN/aiperf_stage_b1}; CSV=${4:-$RUN/stage_b1_timing.csv}
MODEL=${MODEL:-/mnt/models/Kimi-K3}; URL=${URL:-http://127.0.0.1:8010}
CLIENT=${CLIENT:-k3-aiperf-client}; SERVER=${SERVER:-k3-prefill-bench}; TPL=88
mkdir -p "$ART"
# 种子 = 本次调用唯一的 base + c。同一次运行内同 c 共享种子(primer 才能复用);
# 跨运行 base 变化, 保证请求内容是新的 —— 否则重跑一个点会命中上一轮自己留下的前缀缓存,
# 整条 prompt 变成 cache hit (done=c+q, new=2), TTFT 直接掉一半, 测的就不是 prefill 了。
NONCE=$(( ($(date +%s) % 90000) * 1000 ))
HDR="priority,c,q,q_actual,b,n_forwards,reqs,seconds,rc,ttft_avg_ms,ttft_p50_ms,ttft_p90_ms,n_target,n_prime,n_other,accept,isl_mean,cache_read_mean,fresh_tok_s"
[[ -s "$CSV" ]] || echo "$HDR" > "$CSV"

# 按 c 升序分组, 组内按 priority
mapfile -t PTS < <(python3 - "$SRC" "$MAXP" <<'PY'
import csv,sys
rs=[r for r in csv.DictReader(open(sys.argv[1]))
    if int(r["eq_batch_size"])==1 and int(r["priority"])<=int(sys.argv[2])]
rs.sort(key=lambda r:(int(r["final_context_length"]), int(r["priority"])))
for r in rs: print(r["priority"], r["final_context_length"], r["final_query_length"], r["n_forwards"])
PY
)
echo "共 ${#PTS[@]} 个 b=1 点, 按 c 分组"
T0=$(date +%s); i=0; LASTC=-1
for line in "${PTS[@]}"; do
  read -r prio c q nf <<<"$line"; i=$((i+1))
  grep -q "^$prio," "$CSV" && { echo "[skip] p$prio"; LASTC=$c; continue; }
  if   [[ $c -gt 400000 ]]; then REQS=6;  WARM=2
  elif [[ $c -gt 200000 ]]; then REQS=10; WARM=2
  else                           REQS=20; WARM=4; fi
  if [[ $q -ge 1024 ]]; then mean=$((q - TPL + 2)); else mean=$((q - TPL)); fi
  tag="p${prio}_c${c}_q${q}"; SEED=$(( NONCE + c % 1000 ))   # 见顶部 NONCE 说明
  PFX=(); [[ $c -gt 0 ]] && PFX=(--num-prefix-prompts 1 --prompt-prefix-length "$c")

  if [[ $c -ne $LASTC && $c -gt 0 ]]; then
    echo "[$(date -u +%H:%M:%S)] --- primer c=$c ---"
    docker exec "$CLIENT" aiperf profile --model "$MODEL" --tokenizer "$MODEL" \
      --tokenizer-trust-remote-code --url "$URL" --endpoint-type chat --streaming \
      --use-server-token-count "${PFX[@]}" \
      --synthetic-input-tokens-mean 42 --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean 1 --output-tokens-stddev 0 \
      --extra-inputs ignore_eos:true --extra-inputs min_tokens:1 --extra-inputs max_tokens:1 \
      --concurrency 1 --request-count 1 --random-seed "$SEED" --ui simple \
      --output-artifact-dir "$ART/primer_c$c" > "$ART/primer_c$c.log" 2>&1
  fi
  LASTC=$c
  echo "[$(date -u +%H:%M:%S)] ($i/${#PTS[@]}) p$prio c=$c q=$q reqs=$REQS"

  since=$(date -u +%Y-%m-%dT%H:%M:%S); t0=$(date +%s.%N)
  docker exec "$CLIENT" aiperf profile --model "$MODEL" --tokenizer "$MODEL" \
    --tokenizer-trust-remote-code --url "$URL" --endpoint-type chat --streaming \
    --use-server-token-count "${PFX[@]}" \
    --synthetic-input-tokens-mean "$mean" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 1 --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true --extra-inputs min_tokens:1 --extra-inputs max_tokens:1 \
    --warmup-request-count "$WARM" --concurrency 1 --request-count "$REQS" \
    --random-seed "$SEED" --ui simple --output-artifact-dir "$ART/$tag" > "$ART/$tag.log" 2>&1
  rc=$?; secs=$(printf %.1f "$(echo "$(date +%s.%N) - $t0" | bc)")

  read -r avg p50 p90 isl crt pstart <<<"$(python3 - "$ART/$tag" <<'PY'
import json,sys,glob,os
f=sorted(glob.glob(os.path.join(sys.argv[1],"**","profile_export_aiperf.json"),recursive=True))
if not f: print(" ".join(["NA"]*6)); raise SystemExit
r=json.load(open(f[-1])); d=r.get("records",r)
g=lambda k: (d.get(k,{}) or {}).get("avg","NA") if isinstance(d.get(k),dict) else "NA"
t=d.get("time_to_first_token",{}) or {}
print(t.get("avg","NA"),t.get("p50","NA"),t.get("p90","NA"),
      g("input_sequence_length"),g("usage_prompt_cache_read_tokens"),r.get("start_time",""))
PY
)"
  [[ -z "$pstart" || "$pstart" == "NA" ]] && pstart="$since"
  docker logs --since "$pstart" "$SERVER" 2>&1 \
    | grep -oE "done: \[[0-9]+\], new: \[[0-9]+\]" > "$ART/$tag.shapes"
  # 容差: q>=1024 要求精确(anchor 保证), q<1024 允许 ±16 token
  read -r nt qa npr not <<<"$(python3 - "$ART/$tag.shapes" "$c" "$q" <<'PY'
import re,sys,collections,statistics
p=[tuple(map(int,re.findall(r"\d+",l))) for l in open(sys.argv[1])]
c,q=int(sys.argv[2]),int(sys.argv[3])
tol = 0 if q>=1024 else 16
hit=[n for d,n in p if d==c and abs(n-q)<=tol]
prime=0 if c==0 else sum(1 for d,n in p if d==0)
print(len(hit), round(statistics.median(hit),1) if hit else "NA", prime, len(p)-len(hit)-prime)
PY
)"
  acc=BAD; [[ "$nt" =~ ^[0-9]+$ ]] && [[ "$nt" -ge $((REQS*8/10)) && "$npr" -eq 0 ]] && acc=OK
  fresh=$(python3 -c "
try: print(round(float('$qa')/(float('$avg')/1000),1))
except Exception: print('NA')")
  echo "$prio,$c,$q,$qa,1,$nf,$REQS,$secs,$rc,$avg,$p50,$p90,$nt,$npr,$not,$acc,$isl,$crt,$fresh" >> "$CSV"
  printf "[%s]   -> %s ttft=%sms q_act=%s target=%s/%s %ss | %dm\n" \
    "$(date -u +%H:%M:%S)" "$acc" "$avg" "$qa" "$nt" "$REQS" "$secs" $(( ($(date +%s)-T0)/60 ))
done
echo "TOTAL_SECONDS=$(( $(date +%s) - T0 ))"
