#!/bin/bash
# Issues #25/#30: context experiments (gemma-4-12b, then qwen3-8b core). Stall-safe stages; appends to results/jp2/timing.log.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp2/timing.log
stage() {  # name model group [extra args]
  local name=$1 model=$2 group=$3; shift 3; local t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. $PY -m machine_kanbun.jpctxrun $group --model "$model" "$@" 2>&1 | grep --line-buffered -v -i warn
    rc=${PIPESTATUS[0]}; [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $name" >> $LOG
  done
  echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b
for g in tm600 tm300 ablation300 tm100; do stage "gemma $g" $G $g; done
echo "DONE gemma" >> $LOG
