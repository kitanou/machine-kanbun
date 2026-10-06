#!/bin/bash
# Issue #69: SeCF-L1 long-term memory x prefix/KV cache. usage: run_cache69.sh [--pilot]
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
mkdir -p results/jp6
SUF=$([ "$1" = "--pilot" ] && echo _pilot)
P=results/jp6/progress$SUF.txt
step() { local name=$1 mode=$2 flag=$3; local t0=$(date +%s) rc
  echo "$name" > $P
  for attempt in 1 2 3; do
    PYTHONPATH=. $PY -m machine_kanbun.jpcache $mode $flag 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: 69 $mode" >> $LOG
  done
  echo "$(date +%H:%M) 69 $mode $SUF rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; return $rc; }
step "実験B(ヒット・部分ヒット・ミス × 記憶量)" hit "$1" || { echo "FAILED hit" > $P; exit 1; }
step "実験A(並列スロット数 × 4 利用者)" slots "$1" || { echo "FAILED slots" > $P; exit 1; }
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
