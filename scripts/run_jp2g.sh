#!/bin/bash
# Issue #29 second pass: same probe with per-head attention metrics and more prompts (runs after the LM Studio stages).
cd "$(dirname "$0")/.."
lms unload --all >/dev/null 2>&1
t0=$(date +%s)
PYTHONPATH=. .venv-mlx/bin/python -m machine_kanbun.jpattn 100 16 attn2 2>&1 | grep --line-buffered -v -i warn
echo "$(date +%H:%M) 29 attention pass 2 $(( $(date +%s) - t0 ))s" >> results/jp2/timing.log
echo "DONE G" >> results/jp2/timing.log
