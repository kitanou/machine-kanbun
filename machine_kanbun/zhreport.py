"""Issue #19 A report: Chinese (ZH) as the fourth language and the typological control."""
from __future__ import annotations

import statistics as st
from pathlib import Path
from typing import List

from . import irlabel, mlenc
from .gen import generate
from .irreport import cell, load, summarize
from .tokcross import counters

RES = Path(__file__).parent.parent / "results"
L4 = ("JA", "EN", "KO", "ZH")


def cv(xs):
    return st.pstdev(xs) / st.mean(xs)


def main():
    rows, _ = load()
    S = summarize(rows)
    L: List[str] = []
    P = L.append
    P("# Issue #19 A: 中国語(ZH)の追加\n")
    P("中国語は漢字を母語として使い、形態変化が少ない。IR-C(共通漢字ラベル)は中国語にとって**部分的に母語**(職/职, 医療/医疗 など日中で字形が違う語もある)で、"
      "IR-L は簡体字の中国語語彙。ZH の IR-C は `ZH-IRC`、IR-L は `ZH-IRL`。\n")
    P("## 4言語のトークンと言語間 CV\n")
    cnt = counters()
    reps = ("L0", "L1", "IRL", "IRC", "IRID")
    P("| トークナイザ | 文脈長 | 表現 | JA / EN / KO / ZH | CV(3言語 JA/EN/KO) | CV(4言語) |")
    P("|---|---|---|---|---|---|")
    for length in (2000, 8000):
        ents, _ = generate(length, cnt["o200k_base"], seed=1, n_questions=8)
        for name in ("o200k_base", "qwen3", "gemma2", "llama3.1", "mistral-nemo", "deepseek-v3"):
            if name not in cnt:
                continue
            for rep in reps:
                v = [cnt[name](mlenc.encode_ml(ents, lg, rep) if rep in ("L0", "L1") else irlabel.render(ents, lg, rep)) for lg in L4]
                P(f"| {name} | {length} | {rep} | " + " / ".join(map(str, v)) + f" | {cv(v[:3]):.1%} | {cv(v):.1%} |")
    models = sorted({m for m, _, _ in S})
    for model in models:
        for length in sorted({l for m, l, _ in S if m == model}):
            if (model, length, "ZH-L1") not in S:
                continue
            P(f"\n## {model} / {length}\n")
            P("| 言語 | 表現 | 文脈tok | 精度 | TTFTコールド(s) | vs L1 | vs L0 |")
            P("|---|---|---|---|---|---|---|")
            for lg in L4:
                for rep in ("L0", "L1", "IRL", "IRC", "IRID", "IRIDL"):
                    key = f"{lg}-{rep}"
                    s = S.get((model, length, key))
                    if rep == "IRC" and not s:
                        s = S.get((model, length, f"{lg}-IRC"))
                    if lg == "JA" and rep == "IRL":
                        s = S.get((model, length, "JA-IRL"))
                    if not s:
                        continue
                    l1, l0 = S.get((model, length, f"{lg}-L1")), S.get((model, length, f"{lg}-L0"))
                    P(f"| {lg} | {rep} | {s['ctx']} | {s['acc']:.1%} | {s['ttft']:.1f} | "
                      f"{(s['ctx'] / l1['ctx'] - 1) * 100:+.1f}% | {(s['ctx'] / l0['ctx'] - 1) * 100:+.1f}% |" if (l1 and l0) else
                      f"| {lg} | {rep} | {s['ctx']} | {s['acc']:.1%} | {s['ttft']:.1f} | - | - |")
            P("\n**ZH の対応あり比較**: " + "; ".join(
                f"{a}−{b} {cell(S, (model, length, f'ZH-{a}'), (model, length, f'ZH-{b}'))}" for a, b in (("IRC", "L1"), ("IRL", "L1"), ("IRC", "IRL"), ("IRID", "IRC"), ("L1", "L0"))
                if (model, length, f"ZH-{a}") in S and (model, length, f"ZH-{b}") in S))
    # headline: the two hypotheses of the issue
    P("\n## 課題の仮説の判定\n")
    for model in models:
        for length in sorted({l for m, l, _ in S if m == model}):
            v = {lg: S.get((model, length, f"{lg}-IRC")) for lg in L4}
            l1 = {lg: S.get((model, length, f"{lg}-L1")) for lg in L4}
            if not all(v.values()) or not all(l1.values()):
                continue
            ch = {lg: v[lg]["ctx"] / l1[lg]["ctx"] - 1 for lg in L4}
            P(f"- {model} {length}: L1→IR-C のトークン変化 JA {ch['JA']:+.1%} / EN {ch['EN']:+.1%} / KO {ch['KO']:+.1%} / ZH {ch['ZH']:+.1%}。"
              f"IR-C の言語間 CV(実測) 3言語 {cv([v[lg]['ctx'] for lg in L4[:3]]):.1%} → 4言語 {cv([v[lg]['ctx'] for lg in L4]):.1%}。")
    text = "\n".join(L)
    (RES / "synthesis" / "zh_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
