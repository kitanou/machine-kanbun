#!/bin/bash
# one progress line every 30 minutes for Issue #49
cd "$(dirname "$0")/.."
while true; do
  c=$(wc -l < results/jp3/secf_cache.jsonl 2>/dev/null || echo 0)
  s=$(wc -l < results/jp3/qa_short_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  l=$(wc -l < results/jp3/qa_long_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #49 $(tr '\n' ' ' < results/jp3/progress_qa.txt)| SeCF変換キャッシュ ${c}、QA 短 ${s}/630、長 ${l}/378"
  grep -q "^ALL DONE" results/jp3/progress_qa.txt && break
  sleep 1800
done
