#!/bin/bash
# prints one progress line every 10 minutes for Issue #19 (stage text is kept in results/synthesis/progress.txt)
cd "$(dirname "$0")/.."
while true; do
  stage=$(tr '\n' ' ' < results/synthesis/progress.txt)
  run=$(pgrep -f "machine_kanbun longqa|scripts/run_" >/dev/null && echo "実験実行中" || echo "実験なし")
  echo "[$(date +%H:%M)] #19 ${stage}| ${run}"
  grep -q "^ALL DONE" results/synthesis/progress.txt && break
  sleep 600
done
