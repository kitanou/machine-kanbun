#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  q=$(wc -l < results/jp4/state_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #67 $(tr '\n' ' ' < results/jp4/progress_state67.txt)| QA ${q}/1260"
  grep -q "^ALL DONE\|^FAILED" results/jp4/progress_state67.txt && break
  sleep 1800
done
