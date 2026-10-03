#!/bin/bash
# Issue #4 long-context grid. Resumable: finished variants are skipped.
# Each model is loaded with a 20k context (gemma-4-12b clamps to 19456 anyway) and parallel=1. LM Studio occasionally hangs at
# "prompt processing 98.8%"; the client then exits with 75 and we reload the model and resume.
# Full-text 32k (L0/JSON) is infeasible on this 16GB machine, so 32k runs compressed forms only;
# 16k runs a reduced variant set.
cd "$(dirname "$0")/.."
V2=(json L0 L1 L3:none L3:full L5:none L5:full L5:minimal L5:category adaptive:none adaptive:minimal)
V16=(json L0 L1 L3:none L5:none L5:full L5:minimal adaptive:minimal)
V32=(L1 L5:none L5:full L5:minimal adaptive:minimal)

run_stage() {  # model lengths variants...
  local model=$1 lengths=$2; shift 2
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "load failed: $model"; return 1; }
    python3 -m machine_kanbun longqa --model "$model" --lengths $lengths --variants "$@" --n-questions 48
    rc=$?
    [ $rc -ne 75 ] && return $rc
    echo "stalled (attempt $attempt): reloading $model"
  done
}

for model in ${@:-google/gemma-4-12b qwen/qwen3-8b}; do
  run_stage "$model" "2000 8000" "${V2[@]}"
  run_stage "$model" "16000" "${V16[@]}"
  run_stage "$model" "32000" "${V32[@]}"
done
