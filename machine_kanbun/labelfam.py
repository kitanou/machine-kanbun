"""Issue #19 E report: label families that keep content words fixed and vary only the structural words.

IR-CS  kanji labels      transparent to a model that knows kanji, shared across languages
IR-L   local labels      transparent, native words
IR-EN  English labels    transparent, one language for everything
IR-ID  opaque codes      short, familiar tokens, no meaning
IR-SYM rare symbols      no meaning, rare and token-expensive
(IR-C = IR-CS with kanji content words as well: the original MKW.)
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import List

from . import irlabel, mlenc
from .ablreport import diff_ci, svg_scatter
from .gen import generate
from .irreport import cell, load, paired, summarize
from .longreport import sign_test

RES = Path(__file__).parent.parent / "results"
LANGS = ("JA", "EN", "KO")
FAM = [("IRC", "IR-C 漢字(構造+内容)"), ("IRCS", "IR-CS 漢字構造語"), ("IRL", "IR-L 各言語"), ("IREN", "IR-EN 英語"), ("IRID", "IR-ID 抽象ID"), ("IRSYM", "IR-SYM レア記号")]
TRAITS = {"IRC": "意味あり・漢字・言語共通", "IRCS": "意味あり・漢字", "IRL": "意味あり・母語", "IREN": "意味あり・英語", "IRID": "意味なし・短い既知トークン", "IRSYM": "意味なし・レア記号"}
COLORS = {"IRC": "#c0392b", "IRCS": "#e67e22", "IRL": "#2471a3", "IREN": "#1e8449", "IRID": "#7f8c8d", "IRSYM": "#8e44ad"}


def main():
    rows, _ = load()
    S = summarize(rows)
    models = sorted({m for m, _, _ in S})
    L: List[str] = []
    P = L.append
    P("# Issue #19 E: ラベル種別の比較\n")
    P("内容語(名前・名詞・数値)は IR-L と同じに固定し、構造語(クラス・キー・グループ・時制・演算子・関係語)だけを変える。IR-C は構造語も内容語も漢字(元の MKW)。"
      "日本語の IR-L は IR-C と同一語彙(再実行なし)。\n")
    P("| 族 | 特徴 |")
    P("|---|---|")
    for k, name in FAM:
        P(f"| {name} | {TRAITS[k]} |")
    # offline tokenizer sizes
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    cnt = lambda s: len(enc.encode(s))
    P("\n## トークン数と言語間 CV(o200k)\n")
    P("| 文脈長 | " + " | ".join(f"{n} (JA / EN / KO) CV" for _, n in FAM) + " |")
    P("|---|" + "---|" * len(FAM))
    for length in (2000, 8000, 32000):
        ents, _ = generate(length, cnt, seed=1, n_questions=8)
        cells = []
        for k, _ in FAM:
            v = [cnt(irlabel.render(ents, lg, k)) for lg in LANGS]
            cells.append(" / ".join(map(str, v)) + f" **{st.pstdev(v) / st.mean(v):.1%}**")
        P(f"| {length} | " + " | ".join(cells) + " |")
    for model in models:
        P(f"\n## {model}\n")
        lens = sorted({l for m, l, _ in S if m == model})
        P("| 文脈長 | 族 | JA tok / 精度 | EN tok / 精度 | KO tok / 精度 | 言語間 CV(実測tok) | 精度(EN+KO) | 対 IR-L(EN+KO) | 対 IR-ID(EN+KO) |")
        P("|---|---|---|---|---|---|---|---|---|")
        for length in lens:
            for k, name in FAM:
                cs = {lg: S.get((model, length, f"{lg}-{k}")) for lg in LANGS}
                if not all(cs.values()):
                    continue
                toks = [cs[lg]["ctx"] for lg in LANGS]
                acc_ek = sum(sum(cs[lg]["ok"].values()) for lg in ("EN", "KO")) / sum(len(cs[lg]["ok"]) for lg in ("EN", "KO"))
                def pooled(ref):
                    bb = cc = n = 0
                    for lg in ("EN", "KO"):
                        a, b = (model, length, f"{lg}-{k}"), (model, length, f"{lg}-{ref}")
                        if a in S and b in S:
                            x, y, m_ = paired(S, a, b)
                            bb, cc, n = bb + x, cc + y, n + m_
                    return "-" if not n or k == ref else f"{(bb - cc) / n * 100:+.1f}pt (p={sign_test(bb, cc):.2f})"
                P(f"| {length} | {name} | " + " | ".join(f"{cs[lg]['ctx']} / {cs[lg]['acc']:.0%}" for lg in LANGS)
                  + f" | {st.pstdev(toks) / st.mean(toks):.1%} | {acc_ek:.1%} | {pooled('IRL')} | {pooled('IRID')} |")
        # tokens vs accuracy relation (does token length or meaning drive accuracy?)
        pts = []
        for length in lens:
            for k, name in FAM:
                for lg in ("EN", "KO"):
                    s = S.get((model, length, f"{lg}-{k}"))
                    ref = S.get((model, length, f"{lg}-IRL"))
                    if s and ref:
                        pts.append((k, s["ctx"] / ref["ctx"] - 1, s["acc"] - ref["acc"]))
        if pts:
            P("\n**IR-L に対する、トークン増減 と 精度差(EN/KO, 文脈長込み平均)**\n")
            P("| 族 | トークン増減 | 精度差 |")
            P("|---|---|---|")
            by = defaultdict(list)
            for k, t, a in pts:
                by[k].append((t, a))
            for k, name in FAM:
                if k in by:
                    P(f"| {name} | {st.mean(t for t, _ in by[k]):+.1%} | {st.mean(a for _, a in by[k]) * 100:+.1f}pt |")
    text = "\n".join(L)
    (RES / "synthesis" / "label_families.md").write_text(text + "\n", encoding="utf-8")
    print(text)
