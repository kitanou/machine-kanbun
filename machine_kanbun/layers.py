"""Issue #19 synthesis: decompose the compression into three layers and verify the claims.

  Layer 1  L0 -> L1     natural-language compression (grammatical redundancy removal)
  Layer 2  L1 -> IR-L   semantic structuring, in the language's own vocabulary
  Layer 3  IR-L -> IR-C label normalisation to shared kanji labels (Japanese: IR-L == IR-C, so 0 by definition)

L1 here is the deterministic, hand-templated summary of the benchmark (not an LLM summary), so layer 1
is the reduction a good template achieves; a real summariser may be less aggressive.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path
from typing import Dict, List

import tiktoken

from . import irlabel, mlenc
from .gen import generate
from .irreport import cell, load, summarize

RES = Path(__file__).parent.parent / "results"
LANGS = ("JA", "EN", "KO")


def main():
    rows, _ = load()
    S = summarize(rows)
    enc = tiktoken.get_encoding("o200k_base")
    cnt = lambda s: len(enc.encode(s))
    L: List[str] = []
    P = L.append
    P("# Issue #19 研究整理: 三層分解と課題文の数値検証\n")
    P("三層: **Layer 1** L0→L1(自然言語の簡潔化)、**Layer 2** L1→IR-L(各言語語彙のまま構造化)、**Layer 3** IR-L→IR-C(共通漢字ラベルへの正規化。日本語は IR-L = IR-C なので 0)。"
      "L1 はベンチマークの決定的テンプレート要約で、LLM による要約ではない(実際の要約はここまで圧縮しない可能性がある)。\n")

    def tokens(model, length, lg, rep):
        if model == "o200k":
            ents, _ = generate(length, cnt, seed=1, n_questions=8)
            if rep == "L0":
                return cnt(mlenc.encode_ml(ents, lg, "L0"))
            if rep == "L1":
                return cnt(mlenc.encode_ml(ents, lg, "L1"))
            return cnt(irlabel.render(ents, lg, rep))
        s = S.get((model, length, f"{lg}-{rep}"))
        return s["ctx"] if s else None

    P("## 層ごとのトークン削減(同一 facts)\n")
    P("| 計測 | 文脈長 | 言語 | L0 | L1 | IR-L | IR-C | L0→L1 (層1) | L1→IR-L (層2) | IR-L→IR-C (層3) | L0→IR-C 合計 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in ("o200k", "google/gemma-4-12b", "qwen/qwen3-8b"):
        for length in (2000, 8000, 16000, 32000):
            for lg in LANGS:
                v = {r: tokens(model, length, lg, r) for r in ("L0", "L1", "IRL", "IRC")}
                if model != "o200k" and (v["L0"] is None or v["L1"] is None or v["IRC"] is None):
                    continue
                if v["IRL"] is None:
                    continue
                pc = lambda a, b: f"{v[b] / v[a] - 1:+.1%}"
                P(f"| {model} | {length} | {lg} | {v['L0']} | {v['L1']} | {v['IRL']} | {v['IRC']} | {pc('L0', 'L1')} | {pc('L1', 'IRL')} | {pc('IRL', 'IRC')} | {pc('L0', 'IRC')} |")
    # claims
    P("\n## 課題文の数値の検証\n")
    P("| 節 | 課題文の主張 | 実測 | 判定 |")
    P("|---|---|---|---|")
    g = "google/gemma-4-12b"
    chk = []
    for lg, ex in zip(LANGS, (-0.55, -0.51, -0.57)):
        a, b = tokens(g, 8000, lg, "L0"), tokens(g, 8000, lg, "L1")
        chk.append(("§1", f"gemma 8k {lg}: L0→L1 {ex:+.0%}", f"{a}→{b} ({b / a - 1:+.1%})", abs((b / a - 1) - ex) < 0.01))
    for lg, ex in zip(LANGS, (0.02, 0.06, -0.11)):
        a, b = tokens(g, 8000, lg, "L1"), tokens(g, 8000, lg, "IRC")
        chk.append(("§2", f"gemma 8k {lg}: MKW vs L1 約{ex:+.0%}", f"{a}→{b} ({b / a - 1:+.1%})", abs((b / a - 1) - ex) < 0.01))
    for lg, (il, ic) in zip(LANGS, ((3362, 3276), (3171, 3294), (3849, 3346))):
        a, b = tokens("o200k", 8000, lg, "IRL"), tokens("o200k", 8000, lg, "IRC")
        chk.append(("§3", f"o200k 8k {lg}: IR-L {il} / IR-C {ic}", f"{a} / {b} (IR-C vs IR-L {b / a - 1:+.1%})", (a, b) == (il, ic)))
    for lg, (c, i) in zip(LANGS, ((100.0, 60.4), (95.8, 64.6), (97.9, 56.2))):
        sc, si = S[(g, 2000, f"{lg}-IRC")]["acc"] * 100, S[(g, 2000, f"{lg}-IRID")]["acc"] * 100
        chk.append(("§5", f"gemma 2k {lg}: IR-C {c}% / IR-ID {i}%", f"{sc:.1f}% / {si:.1f}%", abs(sc - c) < 0.15 and abs(si - i) < 0.15))
    for sec, claim, meas, ok in chk:
        P(f"| {sec} | {claim} | {meas} | {'一致' if ok else '**不一致**'} |")
    P("\n(§5 は MKW 実行の採点を後で改訂したため、再採点後の値で比較している。不一致は採点改訂による。)\n")
    # accuracy per layer step
    P("\n## 層ごとの精度変化(対応あり: 同一問題。IR-L は EN/KO のみ)\n")
    P("| モデル | 文脈長 | 言語 | L0 | L1 | IR-L | IR-C | L1 − L0 | IR-L − L1 | IR-C − IR-L |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for model in (g, "qwen/qwen3-8b"):
        for length in sorted({l for m, l, _ in S if m == model}):
            for lg in LANGS:
                ks = {r: (model, length, f"{lg}-{r}") for r in ("L0", "L1", "IRL", "IRC")}
                if not all(k in S for k in ks.values()):
                    continue
                acc = {r: f"{S[k]['acc']:.0%}" for r, k in ks.items()}
                d = lambda a, b: cell(S, ks[a], ks[b]).split(" [")[0] if (lg != "JA" or (a, b) != ("IRC", "IRL")) else "0 (同一)"
                P(f"| {model} | {length} | {lg} | {acc['L0']} | {acc['L1']} | {acc['IRL']} | {acc['IRC']} | {d('L1', 'L0')} | {d('IRL', 'L1')} | {d('IRC', 'IRL')} |")
    P("\n## 層ごとの精度変化(文脈長プール, 対応あり)\n")
    P("層1=L1−L0、層2=IR-L−L1、層3=IR-C−IR-L(EN/KO のみ。日本語は IR-L = IR-C)。\n")
    P("| モデル | 層 | 対象言語 | 比較問題数 | Δ [95%CI] | b/c | p |")
    P("|---|---|---|---|---|---|---|")
    from .ablreport import diff_ci
    from .irreport import paired
    from .longreport import sign_test
    steps = [("層1 L1−L0", "L1", "L0", ("JA", "EN", "KO")), ("層2 IR-L−L1", "IRL", "L1", ("EN", "KO")),
             ("層3 IR-C−IR-L", "IRC", "IRL", ("EN", "KO")), ("層2+3 IR-C−L1", "IRC", "L1", ("JA", "EN", "KO")),
             ("全体 IR-C−L0", "IRC", "L0", ("JA", "EN", "KO"))]
    for model in (g, "qwen/qwen3-8b"):
        for name, a_, b_, grp in steps:
            for gg in (grp, ) if len(grp) == 1 else (grp, *[(x,) for x in grp]):
                bb = cc = n = 0
                for length in sorted({l for m, l, _ in S if m == model}):
                    for lg in gg:
                        ka, kb = (model, length, f"{lg}-{a_}"), (model, length, f"{lg}-{b_}")
                        if ka in S and kb in S:
                            x, y, m_ = paired(S, ka, kb)
                            bb, cc, n = bb + x, cc + y, n + m_
                if n:
                    lo, hi = diff_ci(bb, cc, n)
                    P(f"| {model} | {name} | {'+'.join(gg)} | {n} | {(bb - cc) / n * 100:+.1f}pt [{lo:+.1f},{hi:+.1f}] | {bb}/{cc} | {sign_test(bb, cc):.2f} |")
    P("\n## 言語間 CV(層ごと。同一 facts・3言語)\n")
    P("| 計測 | 文脈長 | L0 | L1 | IR-L | IR-C |")
    P("|---|---|---|---|---|---|")
    cv = lambda xs: st.pstdev(xs) / st.mean(xs)
    for model in ("o200k", g):
        for length in (2000, 8000, 16000, 32000):
            vals = {r: [tokens(model, length, lg, r) for lg in LANGS] for r in ("L0", "L1", "IRL", "IRC")}
            if any(None in v for v in vals.values()):
                continue
            P(f"| {model} | {length} | " + " | ".join(f"{cv(vals[r]):.1%}" for r in ("L0", "L1", "IRL", "IRC")) + " |")
    text = "\n".join(L)
    (RES / "synthesis" / "layers_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
