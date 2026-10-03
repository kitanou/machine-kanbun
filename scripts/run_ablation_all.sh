#!/bin/bash
# Issue #10 full plan (resumable): qwen 2k x 2 seeds (all variants), gemma 8k (reduced set), conversion cost.
cd "$(dirname "$0")/.."
scripts/run_ablation.sh qwen/qwen3-8b 2000 1
scripts/run_ablation.sh qwen/qwen3-8b 2000 2
scripts/run_ablation.sh google/gemma-4-12b 8000 1 L0 L1 L5:none ab=particle ab=kanji ab=subj ab=attr ab=tense ab=simple ab=neg ab=unc ab=cond ab=cause ab=cmp ab=struct ab=simple+neg+order abL=particle+kanji abL=particle+kanji+attr+tense abL=particle+kanji+attr+tense+simple+neg+unc abL=particle+kanji+attr+tense+simple+neg+unc+cond+cause+cmp abL=particle+kanji+attr+tense+simple+neg+unc+cond+cause+cmp+struct abL=particle+kanji+attr+tense+simple+neg+unc+cond+cause+cmp+struct+order
lms unload --all >/dev/null 2>&1
lms load google/gemma-4-12b -c 20000 --parallel 1 -y >/dev/null 2>&1
python3 -m machine_kanbun convcost --model google/gemma-4-12b --llm-lengths 2000 8000
