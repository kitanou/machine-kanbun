#!/bin/bash
# Issues #28 #26 #27 #31 (+ qwen replication of #25/#26): runs after scripts/run_jp2.sh finished. Stages are resumable; a stage is retried up to 4 times.
cd "$(dirname "$0")/.."
PY=.venv-ja/bin/python
LOG=results/jp2/timing.log
while ! grep -q "^DONE gemma" $LOG 2>/dev/null; do sleep 30; done
stage() {  # name model(or -) cmd...
  local name=$1 model=$2; shift 2; local t0=$(date +%s) rc
  for attempt in 1 2 3 4; do
    if [ "$model" != "-" ]; then
      lms unload --all >/dev/null 2>&1
      lms load "$model" -c 20000 --parallel 1 -y >/dev/null 2>&1 || { echo "$name load failed" >> $LOG; return 1; }
    fi
    PYTHONPATH=. "$@" 2>&1 | grep --line-buffered -v -i -E "warn|jieba|Building|Loading|Dumping|Prefix"
    rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] && break
    echo "retry $attempt (rc=$rc): $name" >> $LOG
  done
  echo "$(date +%H:%M) $name rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
}
G=google/gemma-4-12b; Q=qwen/qwen3-8b
stage "28 dual-index retrieval" - $PY -m machine_kanbun.jpdual
stage "28 dual-index QA gemma" - $PY -c "from machine_kanbun import jpdual; jpdual.run_qa('$G')"
stage "26 tokenizer-aware gemma" $G $PY -m machine_kanbun.jpctxrun taw300 --model $G
stage "27 natural chats generate" $G $PY -m machine_kanbun.jpnatrun gen $G 90
stage "27 natural chats QA gemma" $G $PY -m machine_kanbun.jpnatrun qa $G
stage "31 multilingual QA gemma" $G $PY -m machine_kanbun.jpmlrun qa $G
stage "26 tokenizer-aware qwen" $Q $PY -m machine_kanbun.jpctxrun taw300 --model $Q
stage "25 token-matched qwen (ratio 1.21)" $Q $PY -m machine_kanbun.jpctxrun tm300 --model $Q --ratio 1.21
stage "27 natural chats QA qwen" $Q $PY -m machine_kanbun.jpnatrun qa $Q
lms unload --all >/dev/null 2>&1
echo "DONE B" >> $LOG
[ -x scripts/run_jp2c.sh ] && scripts/run_jp2c.sh
