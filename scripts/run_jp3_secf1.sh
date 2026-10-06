#!/bin/bash
# Issue #33 redo: SeCF-L1 (concise natural Japanese, #19 style) instead of the structured SeCF-L5-like form.
# SeCF-L1 conversion (gemma) -> answers (gemma) -> blind judging (qwen3-8b) -> status-recall QA (#49 harness).
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b; Q=qwen/qwen3-8b
stage() { local name=$1 model=$2 ctx=$3; shift 3; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1; lms load "$model" -c $ctx --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: $name" >> $LOG
  done; echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; }
P=results/jp3/progress_secf1.txt
echo "SeCF-L1変換(短い履歴)" > $P;      stage "33r secf1 convert short" $G 20000 $PY -m machine_kanbun.jpchat prep_secf1 $G secf1
echo "SeCF-L1変換(長い履歴)" > $P;      stage "33r secf1 convert long" $G 20000 $PY -m machine_kanbun.jpchat prep_secf1 $G secf1long
echo "回答生成(短い履歴 10条件)" > $P;    stage "33r gen short" $G 20000 $PY -m machine_kanbun.jpchat gen $G secf1
echo "回答生成(長い履歴 4条件)" > $P;     stage "33r gen long" $G 20000 $PY -m machine_kanbun.jpchat gen $G secf1long
echo "匿名採点(qwen3-8b)" > $P;          stage "33r judge short" $Q 8192 $PY -m machine_kanbun.jpchat judge $Q $G secf1
                                          stage "33r judge long" $Q 8192 $PY -m machine_kanbun.jpchat judge $Q $G secf1long
echo "想起QA(短、SeCF-L1)" > $P;         stage "33r qa short" $G 20000 $PY -m machine_kanbun.jpchatqa run $G --secf1 short
echo "想起QA(長、SeCF-L1)" > $P;         stage "33r qa long" $G 20000 $PY -m machine_kanbun.jpchatqa run $G --secf1 long
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
