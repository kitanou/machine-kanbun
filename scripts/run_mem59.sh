#!/bin/bash
# Issue #59 (Meta #56 P2): SeCF-L1 as long-term memory representation (NF / SCF-L1 / SeCF-L1): conversion -> retrieval -> QA.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp3/timing.log
G=google/gemma-4-12b
mkdir -p results/jp4
P=results/jp4/progress.txt
run() { local name=$1 model=$2 ctx=$3; shift 3; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1
    if [ "$ctx" = "0" ]; then lms load "$model" -y >/dev/null 2>&1; else lms load "$model" -c $ctx --parallel 1 -y >/dev/null 2>&1; fi || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i warn; rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break; echo "retry $attempt: $name" >> $LOG
  done; echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG; }
echo "SeCF-L1変換(記憶1000件+クエリ150件)" > $P; run "59 prep" $G 20000 $PY -m machine_kanbun.jpmem prep
echo "検索(BM25・ベクトル・RRF)" > $P;           run "59 retrieval" text-embedding-qwen3-embedding-0.6b 0 $PY -m machine_kanbun.jpmem retr
echo "回答QA(6条件×60問)" > $P;                  run "59 qa" $G 20000 $PY -m machine_kanbun.jpmem qa $G
lms unload --all >/dev/null 2>&1
echo "ALL DONE" > $P
