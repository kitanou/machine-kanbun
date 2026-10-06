#!/bin/bash
# Issue #54: three-tier chat history (oldest SeCF-L1 / middle SCF-L1 / recent NF); reuses the #49 QA harness.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b
stage() { local name=$1; shift; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1; lms load $G -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: $name" >> $LOG
  done; echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; }
P=results/jp3/progress_qa54.txt
echo "短い履歴(35×3×3条件)" > $P; stage "54 short" $PY -m machine_kanbun.jpchatqa run $G --tier short
echo "長い履歴(21×3×3条件)" > $P; stage "54 long" $PY -m machine_kanbun.jpchatqa run $G --tier long
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
