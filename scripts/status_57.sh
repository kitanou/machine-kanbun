#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  h=$(wc -l < results/jp3/histories_longpos.jsonl 2>/dev/null || echo 0)
  q=$(wc -l < results/jp3/budget_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #57 $(tr '\n' ' ' < results/jp3/progress_budget57.txt)| 履歴 ${h}/35、QA ${q}/910"
  grep -q "^ALL DONE" results/jp3/progress_budget57.txt && break
  sleep 1800
done
