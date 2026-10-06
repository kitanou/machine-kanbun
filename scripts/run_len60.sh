#!/bin/bash
# Issue #60: SeCF-L1 length dependence (compression, TTFT, break-even). usage: run_len60.sh [--pilot]
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b
mkdir -p results/jp5
P=results/jp5/progress$([ "$1" = "--pilot" ] && echo _pilot).txt
run() { local name=$1; shift; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1; lms load $G -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: $name" >> $LOG
  done; echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; return $rc; }
echo "SeCF-L1変換(長さ別サンプル)" > $P; run "60 convert $1" $PY -m machine_kanbun.jplen convert $G $1 || { echo "FAILED convert" > $P; exit 1; }
echo "TTFT測定(NFとSeCF-L1)" > $P;       run "60 ttft $1" $PY -m machine_kanbun.jplen ttft $G $1 || { echo "FAILED ttft" > $P; exit 1; }
echo "意味の類似度(埋め込み)" > $P;    PYTHONPATH=. $PY -m machine_kanbun.jplen sim $1 2>&1 | grep --line-buffered -v -i warn
lms unload --all >/dev/null 2>&1
PYTHONPATH=. $PY -m machine_kanbun.jplen report $1 > /dev/null 2>&1
echo "ALL DONE" > $P
