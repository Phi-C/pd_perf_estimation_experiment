#!/usr/bin/env bash
# 逐组测 decode 性能点：吞吐 / ITL / 纯模型 forward 时间。
#
# 每组两跑（不是一跑里排时间戳）：
#   ① prime : 同一个 trace + max_tokens=1 —— 把各自的 KV 灌进前缀缓存。冷 prefill，慢。
#   ② meas  : 同一个 trace + max_tokens=O —— 整条命中自己的前缀，秒进 decode。
# 两遍必须用**同一个 trace 文件**：mooncake 记录里的 output_length 参与合成 prompt，
# 改它就等于换了 prompt，前缀缓存一条都不命中（§7.2.2）。截断生成只能靠 max_tokens。
# 分两次跑是自计时的：prime 跑完 aiperf 才返回，不用猜 t0。挤在一跑里的话，
# 5.5M token 的冷 prefill 要 ~70s，300 步 decode 才 ~18s，测量窗口早跑没了。
#
# 两路口径同时留证据：aiperf 侧 ITL 从 artifact 里读，引擎侧 fwd_gpu_ms /
# step_period_ms 从 _fwdprobe 的字节区间里切 —— meas 前后各记一次文件长度。
set -uo pipefail
if [[ ${1:-} == --help ]]; then
  cat <<'USAGE'
TR=/absolute/path/to/traces bash scripts/run_decode_bench.sh

The server and AIPerf client containers must already be running. Host and
containers must share the trace, artifact and probe paths at identical paths.
Set ART, CSV (JSONL content), REFS, PROBE, MODEL, URL, CLIENT and SERVER as needed.
ONLY selects comma-separated tags; DROP defaults to 20 warmup steps.
See decode/measure_decode_performance_point.md for setup and acceptance checks.
USAGE
  exit 0
fi
ROOT=${PD_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
DEC=$ROOT/decode; RUN=${BENCH_RUN:-$DEC/bench_run}
TR=${TR:?trace 目录，如 $RUN/traces_calib}
ART=${ART:-$RUN/art_$(basename "$TR")}
CSV=${CSV:-$RUN/$(basename "$TR")_results.csv}
REFS=${REFS:-}                  # 可选：decode_calib_points.csv，用来对账 CV_L / r_max
PROBE=${PROBE:-$RUN/fwdprobe_decode.jsonl}
MODEL=${MODEL:-/mnt/models/Kimi-K3}; URL=${URL:-http://127.0.0.1:8010}
CLIENT=${CLIENT:-k3-aiperf-client}; SERVER=${SERVER:-k3-decode-instr}
ONLY=${ONLY:-}; DROP=${DROP:-20}
mkdir -p "$ART"
docker exec "$SERVER" true 2>/dev/null || { echo "server 容器不在"; exit 1; }
: > "$PROBE.keep" 2>/dev/null; rm -f "$PROBE.keep"
[[ -f "$PROBE" ]] || : > "$PROBE"

mapfile -t ROWS < <(python3 - "$TR/index.csv" "$REFS" "$ONLY" <<'PY'
import csv,sys
idx,refs,only=sys.argv[1],sys.argv[2],sys.argv[3]
R={}
if refs:
    for r in csv.DictReader(open(refs)):
        R[f"{r['point_id']}__{r['shape']}"]=(r["CV_L"],r["r_max"])
sel=set(only.split(","))-{""}
for r in csv.DictReader(open(idx)):
    if sel and r["tag"] not in sel: continue
    cv,rm=R.get(r["tag"],("",""))
    print("|".join([r["tag"],r["b"],r["L_eq"],r["output_tokens"],cv,rm]))
PY
)
echo "共 ${#ROWS[@]} 组待测 -> $CSV"
i=0; T0=$(date +%s)
for row in "${ROWS[@]}"; do
  IFS='|' read -r tag b Leq O cv rm <<<"$row"; i=$((i+1))
  [[ -s "$CSV" ]] && grep -q "\"tag\": \"$tag\".*\"accept\": \"OK\"" "$CSV" && { echo "  skip $tag (已有 OK 结果)"; continue; }

  # prime 可能要跑两遍。实测：一串 hash_ids 的**首次**冷 prefill 之后，紧接着的
  # 那一跑仍然 cache_read=0；再跑一遍才稳定命中。原因大概率是 lmcache_offload
  # 这条 kv_connector 把 KV 异步搬走后才对下一次 lookup 可见。不确认命中就发 meas
  # 的话，测量请求会整条冷 prefill、错峰进 decode，客户端 ITL 直接虚高 4 倍（§7.2.2）。
  t_a=$(date +%s); rc_p=0; np=0
  for att in 1 2; do
    np=$att
    docker exec "$CLIENT" aiperf profile \
      --model "$MODEL" --tokenizer "$MODEL" --tokenizer-trust-remote-code \
      --url "$URL" --endpoint-type chat --streaming --use-server-token-count \
      --custom-dataset-type mooncake_trace --input-file "$TR/$tag.jsonl" \
      --fixed-schedule --synthetic-input-tokens-block-size 1024 --workers-max 64 \
      --extra-inputs ignore_eos:true --extra-inputs min_tokens:1 --extra-inputs max_tokens:1 \
      --random-seed 42 --ui simple --output-artifact-dir "$ART/$tag.prime$att" \
      > "$ART/$tag.prime$att.log" 2>&1
    rc_p=$?
    hit=$(python3 "$ROOT/scripts/cache_read.py" "$ART/$tag.prime$att" 2>/dev/null || echo 0)
    awk -v h="$hit" -v l="$Leq" 'BEGIN{exit !(h > 0.5*l)}' && break
  done
  t_b=$(date +%s)

  off0=$(stat -c %s "$PROBE")
  docker exec "$CLIENT" aiperf profile \
    --model "$MODEL" --tokenizer "$MODEL" --tokenizer-trust-remote-code \
    --url "$URL" --endpoint-type chat --streaming --use-server-token-count \
    --custom-dataset-type mooncake_trace --input-file "$TR/$tag.jsonl" \
    --fixed-schedule --synthetic-input-tokens-block-size 1024 --workers-max 64 \
    --extra-inputs ignore_eos:true --extra-inputs "min_tokens:$O" --extra-inputs "max_tokens:$O" \
    --random-seed 42 --ui simple --output-artifact-dir "$ART/$tag.meas" \
    > "$ART/$tag.meas.log" 2>&1
  rc_m=$?; t_c=$(date +%s)
  # LAG=8：等待已产生的写入完成；没有后续 forward 时，尾部记录仍可能未 flush。
  sleep 2; off1=$(stat -c %s "$PROBE")

  args=(--probe "$PROBE" --start "$off0" --end "$off1" --artifact-dir "$ART/$tag.meas"
        -b "$b" --tag "$tag" --L-eq "$Leq" --drop "$DROP")
  [[ -n "$cv" ]] && args+=(--cv-ref "$cv"); [[ -n "$rm" ]] && args+=(--rmax-ref "$rm")
  j=$(python3 "$ROOT/scripts/analyze_decode_group.py" "${args[@]}" 2>"$ART/$tag.err")
  [[ -z "$j" ]] && j="{\"tag\": \"$tag\", \"accept\": \"BAD\", \"reason\": \"analyzer 崩了\"}"
  j=$(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);d.update(prime_s=int(sys.argv[2]),meas_s=int(sys.argv[3]),rc_prime=int(sys.argv[4]),rc_meas=int(sys.argv[5]),n_prime_runs=int(sys.argv[6]));print(json.dumps(d,ensure_ascii=False))' \
       "$j" "$((t_b-t_a))" "$((t_c-t_b))" "$rc_p" "$rc_m" "$np")
  echo "$j" >> "$CSV"
  python3 -c 'import json,sys;d=json.loads(sys.stdin.read());print("  %-28s %s b=%s D=%sms fwd=%sms tps=%s itl=%s %s"%(d["tag"],d.get("accept"),d.get("b"),d.get("D_ms"),d.get("fwd_gpu_ms_p50"),d.get("tps_tok_s"),d.get("itl_ms_p50"),d.get("reason","")))' <<<"$j"
  el=$(( $(date +%s) - T0 ))
  echo "[$(date -u +%H:%M:%S)] ($i/${#ROWS[@]}) prime=$((t_b-t_a))s meas=$((t_c-t_b))s 累计 ${el}s"
done
