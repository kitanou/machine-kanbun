#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  c=$(wc -l < results/jp3/secf1_cache.jsonl 2>/dev/null || echo 0)
  g=$(cat results/jp3/gensecf1_*.jsonl results/jp3/gensecf1long_*.jsonl 2>/dev/null | wc -l)
  j=$(cat results/jp3/judgesecf1_*.jsonl results/jp3/judgesecf1long_*.jsonl 2>/dev/null | wc -l)
  q=$(cat results/jp3/qa_*_google_gemma-4-12b.jsonl 2>/dev/null | grep -c -E '"cond": "(E1|G1)_')
  echo "[$(date +%H:%M)] #33再実行 $(tr '\n' ' ' < results/jp3/progress_secf1.txt)| SeCF-L1変換 ${c}/約1280発話、回答 ${g}/434、採点 ${j}/434、QA ${q}/336"
  grep -q "^ALL DONE" results/jp3/progress_secf1.txt && break
  sleep 1800
done
