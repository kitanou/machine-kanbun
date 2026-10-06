#!/bin/bash
# Issue #57 (Meta #56 P1): same-budget comparison of NF-short vs SCF-L1 / SeCF-L1 (oldest exchanges cut first).
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b
stage() { local name=$1; shift; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1; lms load $G -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: $name" >> $LOG
  done; echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; }
P=results/jp3/progress_budget57.txt
echo "履歴生成(35シナリオ×24往復)" > $P; stage "57 histories" $PY -m machine_kanbun.jpchatbudget hist $G
echo "SeCF-L1変換(新履歴)" > $P;        stage "57 secf1" $PY -m machine_kanbun.jpchatbudget prep $G
echo "QA(35×2並び×13条件)" > $P;        stage "57 qa" $PY -m machine_kanbun.jpchatbudget run $G
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
