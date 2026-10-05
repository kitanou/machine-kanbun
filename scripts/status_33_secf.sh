#!/bin/bash
# one progress line every 30 minutes for the SeCF part of Issue #33
cd "$(dirname "$0")/.."
while true; do
  stage=$(tr '\n' ' ' < results/jp3/progress_secf.txt)
  c=$(wc -l < results/jp3/secf_cache.jsonl 2>/dev/null || echo 0)
  g=$(wc -l < results/jp3/gensecf_google_gemma-4-12b.jsonl 2>/dev/null || echo 0)
  j=$(cat results/jp3/judgesecf_*.jsonl 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] #33 SeCF ${stage}| SeCF変換 ${c}/560発話、回答生成 ${g}/455、採点 ${j}/455"
  grep -q "^ALL DONE" results/jp3/progress_secf.txt && break
  sleep 1800
done
