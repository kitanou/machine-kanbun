#!/bin/bash
cd "$(dirname "$0")/.."
SUF=${1:-}; N=$([ "$SUF" = "_pilot" ] && echo 54 || echo 240)
while true; do
  c=$(wc -l < results/jp5/convert$SUF.jsonl 2>/dev/null || echo 0)
  t=$(wc -l < results/jp5/ttft$SUF.jsonl 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] #60$SUF $(tr '\n' ' ' < results/jp5/progress$SUF.txt)| 変換 ${c}/${N}、TTFT ${t}/${N}"
  grep -q "^ALL DONE\|^FAILED" results/jp5/progress$SUF.txt && break
  sleep 1800
done
