#!/bin/bash
# extra question samples (pooled with the first sample in the report) to raise the power of the #25 token-matched comparison
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp2/timing.log
G=google/gemma-4-12b
for g in tm600b tm300b; do
  t0=$(date +%s)
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1
    lms load "$G" -c 20000 --parallel 1 -y >/dev/null 2>&1
    PYTHONPATH=. $PY -m machine_kanbun.jpctxrun $g --model $G 2>&1 | grep --line-buffered -v -i warn
    rc=${PIPESTATUS[0]}; [ $rc -ne 75 ] && break
  done
  echo "$(date +%H:%M) gemma $g rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
done
lms unload --all >/dev/null 2>&1
echo "DONE D" >> $LOG
[ -x scripts/run_jp2e.sh ] && scripts/run_jp2e.sh
