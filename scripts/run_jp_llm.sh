#!/bin/bash
# Issue #20 experiments 2 & 5: LLM end-to-end (full context) and RAG mode. Waits for the retrieval benchmark.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
while [ ! -f results/jp/rag.json ]; do sleep 20; done
PYTHONPATH=. $PY -m machine_kanbun.jpretime > results/jp/retime.log 2>&1  # clean embedding timings, nothing else loaded
LOG=results/jp/llm_timing.log
stage() {  # name model mode ns...
  local name=$1 model=$2 mode=$3; shift 3; local t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. $PY -m machine_kanbun.jprun $mode --model "$model" --ns "$@"
    rc=$?; [ $rc -ne 75 ] && break
    echo "stalled (attempt $attempt): $name" >> $LOG
  done
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b; Q=qwen/qwen3-8b
stage "gemma full 10-600" $G llm 10 100 300 600
stage "gemma rag 1000/10000" $G rag 1000 10000
stage "qwen full 10-100" $Q llm 10 100
stage "qwen rag 1000/10000" $Q rag 1000 10000
echo DONE >> $LOG
