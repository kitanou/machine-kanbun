#!/bin/bash
# Issue #69 follow-up: where does the cache give out? A2 = total cached tokens beyond the loaded context; A3 = number of cached prefixes.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
P=results/jp6/progress_stress.txt
echo "追加測定(A2 総トークン数、A3 キャッシュの本数)" > $P
t0=$(date +%s)
for attempt in 1 2 3; do
  PYTHONPATH=. $PY -m machine_kanbun.jpcache ${1:-stress} 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: 69 stress" >> $LOG
done
echo "$(date +%H:%M) 69 stress rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
[ $rc -eq 0 ] || { echo "FAILED stress" > $P; exit 1; }
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
