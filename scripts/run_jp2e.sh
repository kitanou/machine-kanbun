#!/bin/bash
# Issue #27 refined L1 (r1, r2) on gemma after the failure analysis; resumable (completed reps are skipped).
cd "$(dirname "$0")/.."
G=google/gemma-4-12b
for attempt in 1 2 3; do
  lms unload --all >/dev/null 2>&1
  lms load "$G" -c 20000 --parallel 1 -y >/dev/null 2>&1
  PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpnatrun qa $G 2>&1 | grep --line-buffered -v -i warn
  rc=${PIPESTATUS[0]}; [ $rc -ne 75 ] && break
done
for attempt in 1 2 3; do
  lms unload --all >/dev/null 2>&1
  lms load "$G" -c 20000 --parallel 1 -y >/dev/null 2>&1
  PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpmlrun qa $G 2>&1 | grep --line-buffered -v -i -E "warn|jieba|Building|Loading|Dumping|Prefix"
  rc=${PIPESTATUS[0]}; [ $rc -ne 75 ] && break
done
echo "$(date +%H:%M) 31 native-marker L1n gemma rc=$rc" >> results/jp2/timing.log
lms unload --all >/dev/null 2>&1
echo "$(date +%H:%M) 27 refined L1 gemma rc=$rc" >> results/jp2/timing.log
echo "DONE E" >> results/jp2/timing.log
[ -x scripts/run_jp2g.sh ] && scripts/run_jp2g.sh
