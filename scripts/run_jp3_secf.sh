#!/bin/bash
# Issue #33, SeCF-L1 part: SeCF conversion of the short histories (gemma) -> answers under 13 SeCF conditions (gemma) -> blind judging (qwen3-8b).
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
G=google/gemma-4-12b; Q=qwen/qwen3-8b
echo "SeCF変換(会話履歴35件の全発話)を実行中" > results/jp3/progress_secf.txt
stage "33 secf convert gemma" $G 20000 $PY -m machine_kanbun.jpchat prep_secf
echo "SeCF条件の回答生成(13条件×35)を実行中" > results/jp3/progress_secf.txt
stage "33 secf generation gemma" $G 20000 $PY -m machine_kanbun.jpchat gen $G secf
echo "qwen3-8bで匿名採点を実行中" > results/jp3/progress_secf.txt
stage "33 secf judge qwen" $Q 8192 $PY -m machine_kanbun.jpchat judge $Q $G secf
lms unload --all >/dev/null 2>&1
echo "ALL DONE(SeCF部分)" > results/jp3/progress_secf.txt
echo "DONE 33c" >> $LOG
