#!/bin/bash
# Issue #11 timing pilot: small runs only, to estimate the cost of the full grid. Logs stage wall times.
cd "$(dirname "$0")/.."
LOG=results/ml_pilot_timing.log
V9="JA-L0 JA-L1 JA-MKW EN-L0 EN-L1 EN-MKW KO-L0 KO-L1 KO-MKW EN-MKW@JA"
stage() { local name=$1; shift; local t0=$(date +%s); "$@" >> results/ml_pilot_run.log 2>&1; echo "$name $(( $(date +%s) - t0 ))s rc=$?" >> $LOG; }
: > $LOG
for model in google/gemma-4-12b qwen/qwen3-8b; do
  lms unload --all >/dev/null 2>&1
  stage "load $model" lms load "$model" -c 20000 --parallel 1 -y
  stage "$model variants 2k n=8" python3 -m machine_kanbun longqa --model "$model" --lengths 2000 --variants $V9 --n-questions 8 --out results/ml_pilot
  stage "$model convert 2k (JA EN KO)" python3 -m machine_kanbun mlconv --model "$model" --langs JA EN KO --length 2000 --n-questions 8
  if [[ $model == google* ]]; then
    stage "$model variants 8k n=8 (EN-L0 KO-L0 KO-MKW)" python3 -m machine_kanbun longqa --model "$model" --lengths 8000 --variants EN-L0 KO-L0 KO-MKW --n-questions 8 --out results/ml_pilot
  fi
done
echo DONE >> $LOG
