"""Issue #11: estimate the wall time of the full multilingual grid from measured pilot data.

Inputs (all measured on this machine): results/ml_pilot (2k, 8k), results/long + results/ablation
(per-length warm latency and prefill rate), results/ml/conversion.json (direct EN/KO/JA -> MKW).
Run: python3 scripts/estimate_ml.py  -> results/ml_pilot_report.md
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
CTX_LIMIT = 19456 - 300          # LM Studio effective context minus prompt/answer overhead
FACTS = {2000: 134, 8000: 480, 16000: 959, 32000: 1863}      # facts per length (seed 1)
LENGTHS = [2000, 8000, 16000, 32000]
N_Q = 48


def rows(glob):
    for f in sorted(R.glob(glob)):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    yield r


warm, rate = defaultdict(list), defaultdict(list)
for r in list(rows("ml_pilot/*.jsonl")) + list(rows("long/*.jsonl")) + list(rows("ablation/*.jsonl")):
    k = (r["model"], r["length"])
    if r["cold"]:
        rate[k].append(r["prompt_tokens"] / r["ttft"])
    else:
        warm[k].append(r["total"])

# measured context tokens per variant at 2k (model tokenizer), pilot
tok2k = {}
for r in rows("ml_pilot/*.jsonl"):
    if r["cold"] and r["length"] == 2000:  # 8k pilot rows must not overwrite the 2k reference
        tok2k[(r["model"], r["fmt"])] = r["prompt_tokens"] - r["base_tokens"]

conv = json.loads((R / "ml_pilot" / "conversion.json").read_text(encoding="utf-8"))
VARIANTS = [f"{l}-{rp}" for l in ("JA", "EN", "KO") for rp in ("L0", "L1", "MKW")] + ["EN-MKW@JA", "KO-MKW@JA"]


def ctx_tokens(model, v, length):
    base = tok2k.get((model, v.replace("KO-MKW@JA", "KO-MKW")))
    if base is None:
        base = tok2k[(model, "KO-MKW" if v.startswith("KO") else "EN-MKW")]
    return base * FACTS[length] / FACTS[2000]


def warm_s(model, length):
    ks = [k for k in warm if k[0] == model]
    if (model, length) in warm:
        return st.mean(warm[(model, length)])
    return st.mean(warm[max(ks, key=lambda k: k[1])])  # fall back to the longest measured length


def rate_s(model, length):
    ks = [k for k in rate if k[0] == model]
    if (model, length) in rate:
        return st.mean(rate[(model, length)])
    return st.mean(rate[max(ks, key=lambda k: k[1])])


def plan(model, lengths, n_q=N_Q):
    total, rows_, skipped = 0.0, [], []
    for L in lengths:
        t_len, n = 0.0, 0
        for v in VARIANTS:
            tk = ctx_tokens(model, v, L)
            if tk > CTX_LIMIT:
                skipped.append((L, v, int(tk)))
                continue
            t = 2 + tk / rate_s(model, L) + (n_q - 1) * warm_s(model, L)
            t_len += t
            n += 1
        rows_.append((L, n, t_len))
        total += t_len
    return total, rows_, skipped


def conversion_time(model, length, langs=("JA", "EN", "KO"), n_q=N_Q):
    """Direct conversion. Chunked into ~2k pieces beyond 8k (single pass would exceed the context window)."""
    tot = 0.0
    for lg in langs:
        c = conv.get(f"{model}|{lg}|2000")
        if not c:
            continue
        per_chunk = c["prompt_tokens"] / rate_s(model, 2000) + c["completion_tokens"] / c["decode_tps"]
        chunks = FACTS[length] / FACTS[2000]
        tot += per_chunk * chunks + 2 + (n_q - 1) * warm_s(model, length)  # + QA on the converted context
    return tot


H = lambda s: f"{s / 3600:.1f}h" if s >= 3600 else f"{s / 60:.0f}min"
out = ["# Issue #11 パイロット計測と本格実行の所要時間見積もり\n",
       "実測: gemma-4-12b / qwen3-8b (LM Studio, 20000 ctx, parallel 1)。見積もりは `scripts/estimate_ml.py` で再計算できる。\n",
       "## 実測値\n", "| モデル | 項目 | 値 |", "|---|---|---|"]
for m in ("google/gemma-4-12b", "qwen/qwen3-8b"):
    for L in LENGTHS:
        if (m, L) in rate:
            out.append(f"| {m} | {L} 相当: prefill 速度 / ウォーム1問 | {rate_s(m, L):.0f} tok/s / {warm_s(m, L):.1f} s |")
out += ["", "**2k 相当の文脈トークン(モデル実測, 文脈のみ)**\n", "| 変種 | gemma | qwen |", "|---|---|---|"]
for v in ["JA-L0", "JA-L1", "JA-MKW", "EN-L0", "EN-L1", "EN-MKW", "KO-L0", "KO-L1", "KO-MKW"]:
    out.append(f"| {v} | {tok2k.get(('google/gemma-4-12b', v), '-')} | {tok2k.get(('qwen/qwen3-8b', v), '-')} |")
out += ["", "**直接変換(各言語の L0 → MKW, 2k, 8 問で QA)**\n",
        "| モデル | 言語 | 変換時間 | prompt+出力 tok | デコード速度 | 正解IRとの項目F1 | 変換結果でのQA精度 |", "|---|---|---|---|---|---|---|"]
for k, c in conv.items():
    m, lg, _ = k.split("|")
    out.append(f"| {m} | {lg} | {c['total']:.0f}s | {c['prompt_tokens']}+{c['completion_tokens']} | {c['decode_tps']:.1f} | {c['f1']:.2f} | {c['acc']:.0%} |")
out.append("")
g, q = "google/gemma-4-12b", "qwen/qwen3-8b"
out += ["## 本格実行の見積もり(1 seed, 48 問/変種, 11 変種 = 3言語 × L0/L1/MKW + 言語固定質問2種)\n",
        "| モデル | 文脈長 | 実行可能な変種数 | 所要 |", "|---|---|---|---|"]
tg, rg, sg = plan(g, LENGTHS)
for L, n, t in rg:
    out.append(f"| {g} | {L} | {n}/11 | {H(t)} |")
out.append(f"| **gemma 小計** | | | **{H(tg)}** |")
tq, rq, sq = plan(q, [2000])
out.append(f"| {q} | 2000 | {rq[0][1]}/11 | {H(rq[0][2])} |")
out.append(f"| **qwen 小計 (2k のみ)** | | | **{H(tq)}** |")
conv_g = sum(conversion_time(g, L) for L in (2000, 8000))
conv_g2 = sum(conversion_time(g, L) for L in (16000, 32000))
conv_q = conversion_time(q, 2000)
out += ["", "**直接変換の QA 評価付きコスト(3言語分)**\n", "| モデル | 文脈長 | 所要 |", "|---|---|---|"]
for L in (2000, 8000, 16000, 32000):
    out.append(f"| {g} | {L}{' (2k チャンク分割)' if L >= 16000 else ''} | {H(conversion_time(g, L))} |")
out.append(f"| {q} | 2000 | {H(conv_q)} |")
base = tg + tq + conv_g + conv_q
out += ["", "## 合計\n", "| 案 | 内容 | 所要 |", "|---|---|---|",
        f"| A 最小 | gemma 2k/8k/16k/32k の 11 変種 + qwen 2k + 変換(gemma 2k/8k, qwen 2k) | **{H(base)}** |",
        f"| B A + 変換 16k/32k(チャンク分割) | 直接変換を全長で測る | {H(base + conv_g2)} |",
        f"| C A + 2k の seed 追加 ×2 | 検出力向上(2k: gemma+qwen) | {H(base + 2 * (rg[0][2] + tq))} |"]
out += ["", "## 実行不能として除外した変種(gemma, コンテキスト 19456 超過の見積もり)\n", "| 文脈長 | 変種 | 推定トークン |", "|---|---|---|"]
for L, v, t in sg:
    out.append(f"| {L} | {v} | {t} |")
(R / "ml_pilot_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
