#!/bin/bash
# Issue #11 plan A (resumable): gemma 2k/8k/16k/32k x 11 variants, qwen 2k x 11 variants,
# direct EN/KO/JA -> MKW conversion (gemma 2k/8k, qwen 2k). 48 questions, seed 1.
# 32k runs compressed forms only (full-text L0 exceeds the 19456-token context). LM Studio stalls
# (client exit 75) trigger a model reload and resume.
cd "$(dirname "$0")/.."
V11="JA-L0 JA-L1 JA-MKW EN-L0 EN-L1 EN-MKW KO-L0 KO-L1 KO-MKW EN-MKW@JA KO-MKW@JA"
V8="JA-L1 JA-MKW EN-L1 EN-MKW KO-L1 KO-MKW EN-MKW@JA KO-MKW@JA"
LOG=results/ml_run_timing.log

run_stage() {  # name model command...
  local name=$1 model=$2; shift 2
  local t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    "$@"; rc=$?
    [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $name" >> $LOG
  done
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
lq() { python3 -m machine_kanbun longqa --model "$1" --lengths $2 --variants $3 --n-questions 48 --out results/ml; }
mc() { python3 -m machine_kanbun mlconv --model "$1" --length "$2" --n-questions 48; }

G=google/gemma-4-12b; Q=qwen/qwen3-8b
run_stage "gemma 2k"  $G lq $G "2000" "$V11"
run_stage "gemma 8k"  $G lq $G "8000" "$V11"
run_stage "gemma 16k" $G lq $G "16000" "$V11"
run_stage "gemma 32k" $G lq $G "32000" "$V8"
run_stage "gemma convert 2k" $G mc $G 2000
run_stage "gemma convert 8k" $G mc $G 8000
run_stage "qwen 2k"   $Q lq $Q "2000" "$V11"
run_stage "qwen convert 2k" $Q mc $Q 2000
echo DONE >> $LOG
