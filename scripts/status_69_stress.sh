#!/bin/bash
cd "$(dirname "$0")/.."
while true; do
  n=$(cat results/jp6/slots_R*_S*.jsonl 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] #69追加 $(tr '\n' ' ' < results/jp6/progress_stress.txt)| 追加測定 ${n}件"
  grep -q "^ALL DONE\|^FAILED" results/jp6/progress_stress.txt && break
  sleep 1800
done
