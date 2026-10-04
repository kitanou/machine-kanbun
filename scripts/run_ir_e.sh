#!/bin/bash
# Issue #19 E: label families (IR-CS / IR-EN / IR-SYM x JA/EN/KO), resumable, stall-safe.
cd "$(dirname "$0")/.."
V9="JA-IRCS EN-IRCS KO-IRCS JA-IREN EN-IREN KO-IREN JA-IRSYM EN-IRSYM KO-IRSYM"
LOG=results/ir_e_timing.log
run_stage() {
  local name=$1 model=$2 lengths=$3 t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    python3 -m machine_kanbun longqa --model "$model" --lengths $lengths --variants $V9 --n-questions 48 --out results/ir
    rc=$?; [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $name" >> $LOG
  done
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b; Q=qwen/qwen3-8b
run_stage "gemma 2k" $G "2000"
run_stage "qwen 2k"  $Q "2000"
run_stage "gemma 8k" $G "8000"
echo DONE >> $LOG
