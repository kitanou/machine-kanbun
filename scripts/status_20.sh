#!/bin/bash
# one progress line every 10 minutes for Issue #20 (stage text lives in results/jp/progress.txt)
cd "$(dirname "$0")/.."
while true; do
  stage=$(tr '\n' ' ' < results/jp/progress.txt)
  run=$(pgrep -f "machine_kanbun|scripts/run_jp|jp_" >/dev/null && echo "処理実行中" || echo "処理なし")
  echo "[$(date +%H:%M)] #20 ${stage}| ${run}"
  grep -q "^ALL DONE" results/jp/progress.txt && break
  sleep 600
done
