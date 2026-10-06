#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  q=$(wc -l < results/jp4/upd_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  c=$(wc -l < results/jp4/consol_cache.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #62 $(tr '\n' ' ' < results/jp4/progress_upd62.txt)| 統合 ${c}/60、QA ${q}/300"
  grep -q "^ALL DONE\|^FAILED" results/jp4/progress_upd62.txt && break
  sleep 1800
done
