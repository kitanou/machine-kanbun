#!/bin/bash
# Issue #29: attention / hidden-state analysis on Qwen3-1.7B (MLX). Runs after the LM Studio stages (GPU must be free).
cd "$(dirname "$0")/.."
lms unload --all >/dev/null 2>&1
LOG=results/jp2/timing.log
t0=$(date +%s)
PYTHONPATH=. .venv-mlx/bin/python -m machine_kanbun.jpattn 100 8 attn 2>&1 | grep --line-buffered -v -i warn
PYTHONPATH=. .venv-mlx/bin/python -m machine_kanbun.jpattn 250 4 attn_long 2>&1 | grep --line-buffered -v -i warn
echo "$(date +%H:%M) 29 attention $(( $(date +%s) - t0 ))s" >> $LOG
echo "DONE C" >> $LOG
[ -x scripts/run_jp2d.sh ] && scripts/run_jp2d.sh
