#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  n=$(cat results/jp3/qa_*_google_gemma-4-12b.jsonl 2>/dev/null | grep -c -E '"cond": "B_(legend|expl)')
  echo "[$(date +%H:%M)] #51 $(tr '\n' ' ' < results/jp3/progress_qa51.txt)| 追加条件の回答 ${n}/504"
  grep -q "^ALL DONE" results/jp3/progress_qa51.txt && break
  sleep 1800
done
