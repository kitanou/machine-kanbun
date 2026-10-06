#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  n=$(cat results/jp3/qa_short_google_gemma-4-12b.jsonl results/jp3/qa_long_google_gemma-4-12b.jsonl 2>/dev/null | grep -c -E '"cond": "T(3|2|3m)_')
  echo "[$(date +%H:%M)] #54 $(tr '\n' ' ' < results/jp3/progress_qa54.txt)| 三層条件の回答 ${n}/504"
  grep -q "^ALL DONE" results/jp3/progress_qa54.txt && break
  sleep 1800
done
