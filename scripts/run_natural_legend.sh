#!/bin/bash
# Issue #19 C: QA over the saved IR with the operator legend (separates IR information loss from operator reading).
cd "$(dirname "$0")/.."
while ! grep -q DONE results/natural_timing.log 2>/dev/null; do sleep 20; done
LOG=results/natural_legend_timing.log
for model in google/gemma-4-12b qwen/qwen3-8b; do
  t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$model load failed" >> $LOG; break; }
    python3 -m machine_kanbun natural --model "$model" --legend
    rc=$?; [ $rc -ne 75 ] && break
  done
  echo "$model rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
done
echo DONE >> $LOG
