#!/bin/bash
# Issue #62: memories whose state changes (raw list vs SeCF-L1 consolidation): prep (conversion + consolidation) -> retrieval + QA.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b
mkdir -p results/jp4
P=results/jp4/progress_upd62.txt
run() { local name=$1; shift; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1; lms load $G -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: $name" >> $LOG
  done; echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; return $rc; }
echo "SeCF-L1変換(718件)と統合(60組)" > $P; run "62 prep" $PY -m machine_kanbun.jpmemupd prep $G || { echo "FAILED prep" > $P; exit 1; }
echo "検索とQA(5条件×60問)" > $P;            run "62 run" $PY -m machine_kanbun.jpmemupd run $G || { echo "FAILED run" > $P; exit 1; }
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
