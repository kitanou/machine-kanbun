#!/bin/bash
# one progress line every 30 minutes for Issue #33 (stage text lives in results/jp3/progress.txt)
cd "$(dirname "$0")/.."
while true; do
  stage=$(tr '\n' ' ' < results/jp3/progress.txt)
  h=$(wc -l < results/jp3/histories.jsonl 2>/dev/null || echo 0)
  g=$(cat results/jp3/gen_*.jsonl 2>/dev/null | wc -l)
  j=$(cat results/jp3/judge_*.jsonl 2>/dev/null | wc -l)
  run=$(pgrep -f "machine_kanbun|scripts/run_jp3" >/dev/null && echo "処理実行中" || echo "処理なし")
  echo "[$(date +%H:%M)] #33 ${stage}| 会話履歴 ${h}/35、回答生成 ${g}/525、採点 ${j}/525 | ${run}"
  grep -q "^ALL DONE" results/jp3/progress.txt && break
  sleep 1800
done
