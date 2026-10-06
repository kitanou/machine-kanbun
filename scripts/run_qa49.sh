#!/bin/bash
# Issue #49: multiple-choice status recall QA over compressed chat histories (gemma, deterministic scoring).
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
stage() {
  local name=$1 model=$2 ctx=$3; shift 3; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c $ctx --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn
    rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break
    echo "retry $attempt (rc=$rc): $name" >> $LOG
  done
  echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b
echo "長い履歴のSeCF変換を実行中" > results/jp3/progress_qa.txt
stage "49 secf convert long" $G 20000 $PY -m machine_kanbun.jpchatqa prep $G long
echo "短い履歴のQA(35×3×6)を実行中" > results/jp3/progress_qa.txt
stage "49 qa short" $G 20000 $PY -m machine_kanbun.jpchatqa run $G short
echo "長い履歴のQA(21×3×6)を実行中" > results/jp3/progress_qa.txt
stage "49 qa long" $G 20000 $PY -m machine_kanbun.jpchatqa run $G long
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > results/jp3/progress_qa.txt
echo "DONE 49" >> $LOG
