#!/bin/bash
cd "$(dirname "$0")/.."
SUF=${1:-}
while true; do
  h=$(wc -l < results/jp6/hit$SUF.jsonl 2>/dev/null || echo 0); c=$(wc -l < results/jp6/slots$SUF.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #69$SUF $(tr '\n' ' ' < results/jp6/progress$SUF.txt)| B ${h}件、A ${c}件"
  grep -q "^ALL DONE\|^FAILED" results/jp6/progress$SUF.txt && break
  sleep 1800
done
