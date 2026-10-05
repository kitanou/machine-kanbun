#!/bin/bash
# Remaining stages after the qwen3-8b failure (max_tokens=5 returned empty content): qwen replication, then #29 (C), extra samples (D), refined L1 (E).
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp2/timing.log
stage() {
  local name=$1 model=$2; shift 2; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    lms unload --all >/dev/null 2>&1
    lms load "$model" -c ${CTXLEN:-20000} --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i -E "warn|jieba|Building|Loading|Dumping|Prefix"
    rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break
    echo "retry $attempt (rc=$rc): $name" >> $LOG
  done
  echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
Q=qwen/qwen3-8b
CTXLEN=8192   # smaller KV cache: qwen3-8b hit Metal "Insufficient Memory" / hangs on 4-6k-token prompts with a 20000 context
stage "26 tokenizer-aware qwen (N=100)" $Q $PY -m machine_kanbun.jpctxrun taw100 --model $Q
stage "25 token-matched qwen (N=100, ratio 1.21)" $Q $PY -m machine_kanbun.jpctxrun tm100 --model $Q --ratio 1.21
# (27 natural chats on qwen3-8b dropped: the 4.5k-token prompts made LM Studio return degenerate text such as "!descr")
lms unload --all >/dev/null 2>&1
echo "DONE B" >> $LOG
[ -x scripts/run_jp2c.sh ] && scripts/run_jp2c.sh
