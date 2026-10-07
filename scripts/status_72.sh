#!/bin/bash
# one progress line every 30 minutes until ALL DONE / FAILED
cd "$(dirname "$0")/.." || exit 1
while true; do
  L=$(tail -n 1 results/jp7/run.log 2>/dev/null)
  echo "$(date +%H:%M) #72: stages=[$(grep -o 'stage [a-z]* done' results/jp7/run.log | tr '\n' ' ')] conv=$(wc -l < results/jp7/conversations.jsonl 2>/dev/null) mem=$(wc -l < results/jp7/memory_nf.jsonl 2>/dev/null) qa=$(wc -l < results/jp7/qa.jsonl 2>/dev/null) budget=$(wc -l < results/jp7/budget.jsonl 2>/dev/null) | $L"
  case "$L" in *"ALL DONE"*|*FAILED*) exit 0;; esac
  sleep 1800
done
