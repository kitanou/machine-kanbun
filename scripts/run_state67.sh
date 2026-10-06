#!/bin/bash
# Issue #67: question format x representation for the "迷い -> いいえ" misreading.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b
mkdir -p results/jp4
P=results/jp4/progress_state67.txt
echo "状態QA(4形式×3表現×105件)" > $P
t0=$(date +%s)
for attempt in 1 2 3 4; do
  lms unload --all >/dev/null 2>&1; lms load $G -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "FAILED load" > $P; exit 1; }
  PYTHONPATH=. $PY -m machine_kanbun.jpstate run $G 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: 67 state" >> $LOG
done
echo "$(date +%H:%M) 67 state rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
[ $rc -eq 0 ] || { echo "FAILED run" > $P; exit 1; }
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
