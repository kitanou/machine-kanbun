#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  c=$(wc -l < results/jp3/secf1_cache.jsonl 2>/dev/null || echo 0)
  q=$(wc -l < results/jp4/qa_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #59 $(tr '\n' ' ' < results/jp4/progress.txt)| SeCF-L1変換キャッシュ ${c}件(目標は既存860+約1150)、QA ${q}/360"
  grep -q "^ALL DONE" results/jp4/progress.txt && break
  sleep 1800
done
