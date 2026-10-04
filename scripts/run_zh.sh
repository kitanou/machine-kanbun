#!/bin/bash
# Issue #19 A: Chinese (ZH-L0 / L1 / IRC / IRL / IRID / IRIDL). Waits for grid E to finish (one model at a time).
cd "$(dirname "$0")/.."
while ! grep -q DONE results/ir_e_timing.log 2>/dev/null; do sleep 20; done
V6="ZH-L0 ZH-L1 ZH-IRC ZH-IRL ZH-IRID ZH-IRIDL"
V5="ZH-L1 ZH-IRC ZH-IRL ZH-IRID ZH-IRIDL"      # 32k: full-text L0 exceeds the context window
LOG=results/zh_timing.log
run_stage() {  # name model lengths variants...
  local name=$1 model=$2 lengths=$3; shift 3; local t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    python3 -m machine_kanbun longqa --model "$model" --lengths $lengths --variants "$@" --n-questions 48 --out results/zh
    rc=$?; [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $name" >> $LOG
  done
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b; Q=qwen/qwen3-8b
run_stage "gemma 2k"  $G "2000" $V6
run_stage "qwen 2k"   $Q "2000" $V6
run_stage "gemma 8k"  $G "8000" $V6
run_stage "gemma 16k" $G "16000" $V6
run_stage "gemma 32k" $G "32000" $V5
echo DONE >> $LOG
