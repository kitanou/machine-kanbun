#!/bin/bash
# Issue #17 label-localisation ablation (resumable). New variants only: IR-C == the existing MKW runs
# (identical text, same 48 questions); JA IR-L == IR-C. 8 variants x gemma 2k/8k/16k/32k + qwen 2k.
cd "$(dirname "$0")/.."
V8="EN-IRL KO-IRL JA-IRID EN-IRID KO-IRID JA-IRIDL EN-IRIDL KO-IRIDL"
LOG=results/ir_run_timing.log
run_stage() {  # name model lengths
  local name=$1 model=$2 lengths=$3 t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    python3 -m machine_kanbun longqa --model "$model" --lengths $lengths --variants $V8 --n-questions 48 --out results/ir
    rc=$?
    [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $name" >> $LOG
  done
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b; Q=qwen/qwen3-8b
run_stage "gemma 2k"  $G "2000"
run_stage "qwen 2k"   $Q "2000"
run_stage "gemma 8k"  $G "8000"
run_stage "gemma 16k" $G "16000"
run_stage "gemma 32k" $G "32000"
echo DONE >> $LOG
