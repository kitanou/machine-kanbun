#!/bin/bash
# Issue #72: personal-AI long-term memory x Memory Density. Stages resume (done items are skipped); any failure prints FAILED and stops.
cd "$(dirname "$0")/.." || exit 1
export MD_USERS=${MD_USERS:-10}
for st in ${STAGES:-gen extract convert qa budget}; do
  .venv-ja/bin/python -m machine_kanbun.jpmemdens $st >> results/jp7/run.log 2>&1 || { echo "FAILED at $st" | tee -a results/jp7/run.log; exit 1; }
  echo "stage $st done $(date +%H:%M)" >> results/jp7/run.log
done
echo "ALL DONE" >> results/jp7/run.log
