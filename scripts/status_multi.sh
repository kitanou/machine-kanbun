#!/bin/bash
# one progress line every 30 minutes for Issues #25-#31 (stage text lives in results/jp2/progress.txt)
cd "$(dirname "$0")/.."
while true; do
  stage=$(tr '\n' ' ' < results/jp2/progress.txt)
  auto=$(.venv-ja/bin/python scripts/jp2_status.py 2>/dev/null)
  run=$(pgrep -f "machine_kanbun|scripts/run_jp2" >/dev/null && echo "処理実行中" || echo "処理なし")
  echo "[$(date +%H:%M)] ${stage}| ${auto} | ${run}"
  grep -q "^ALL DONE" results/jp2/progress.txt && break
  sleep 1800
done
