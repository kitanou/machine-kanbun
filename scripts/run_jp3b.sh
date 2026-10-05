#!/bin/bash
# Issue #33, long-history variant (24 exchanges): token savings become visible. Runs after scripts/run_jp3.sh finished.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
while ! grep -q "^DONE 33" $LOG 2>/dev/null; do sleep 30; done
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
stage "33 long histories gemma" $G 20000 $PY -m machine_kanbun.jpchat hist $G long
stage "33 long generation gemma" $G 20000 $PY -m machine_kanbun.jpchat gen $G long
stage "33 long judge qwen" $Q 8192 $PY -m machine_kanbun.jpchat judge $Q $G long
lms unload --all >/dev/null 2>&1
echo "DONE 33b" >> $LOG
