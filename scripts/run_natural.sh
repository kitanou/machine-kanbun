#!/bin/bash
# Issue #19 C/D: natural sentences -> IR -> QA / round trip. Waits for the Chinese grid to finish.
cd "$(dirname "$0")/.."
while ! grep -q DONE results/zh_timing.log 2>/dev/null; do sleep 30; done
LOG=results/natural_timing.log
run_model() {
  local model=$1 t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$model load failed" >> $LOG; return 1; }
    python3 -m machine_kanbun natural --model "$model"
    rc=$?; [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $model" >> $LOG
  done
  echo "$model rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
run_model google/gemma-4-12b
run_model qwen/qwen3-8b
echo DONE >> $LOG
