#!/bin/bash
# Issue #10 L1 ablation. Usage: scripts/run_ablation.sh <model> <lengths> <seed> [variants...]
# Default variants = anchors (L0 L1 L5) + 13 independent ablations + 11 cumulative ladder steps.
# Reloads the model and resumes if LM Studio stalls (client exits 75).
cd "$(dirname "$0")/.."
model=$1; lengths=$2; seed=$3; shift 3
if [ $# -eq 0 ]; then
  set -- $(python3 -c "
from machine_kanbun.ablate import single_variants, ladder_variants
print(' '.join(['L0','L1','L5:none']+single_variants()+ladder_variants()))")
fi
for attempt in 1 2 3 4 5; do
  lms unload --all >/dev/null 2>&1
  lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "load failed: $model"; exit 1; }
  python3 -m machine_kanbun longqa --model "$model" --lengths $lengths --variants "$@" --seed "$seed" \
      --n-questions 48 --out results/ablation
  rc=$?
  [ $rc -ne 75 ] && exit $rc
  echo "stalled (attempt $attempt): reloading $model"
done
