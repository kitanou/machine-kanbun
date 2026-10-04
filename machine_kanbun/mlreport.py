"""Issue #11 report: {JA,EN,KO} x {L0,L1,MKW} x model x context length -> markdown + CSV + SVG."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import mlenc
from .ablreport import diff_ci, svg_scatter
from .longreport import pareto, sign_test

RES = Path(__file__).parent.parent / "results"
ML = RES / "ml"
LANGS, REPS = ("JA", "EN", "KO"), ("L0", "L1", "MKW")
BASE = [f"{l}-{r}" for l in LANGS for r in REPS]
COLORS = {"anchor": "#c0392b", "single": "#2471a3", "ladder": "#1e8449"}  # JA / EN / KO


def rescore(rows):
    """Re-score stored answers with the current scorer (answers are saved, the scorer evolves)."""
    import tiktoken

    from . import qa as qamod
    from .gen import generate
    enc = tiktoken.get_encoding("o200k_base")
    cnt = lambda s: len(enc.encode(s))
    qmaps = {}
    changed = 0
    for r in rows:
        if r["length"] not in qmaps:
            _, qs = generate(r["length"], cnt, seed=r["seed"], n_questions=48)
            qmaps[r["length"]] = {mlenc.question_text(q, lg): q for q in qs for lg in LANGS}
        q = qmaps[r["length"]].get(r["q"])
        if q is None:
            continue
        ok = qamod.score(q, r["answer"], ml=(r["qlang"] != "JA" or r["lang"] != "JA"))
        changed += ok != r["ok"]
        r["ok"] = ok
    return changed


def load():
    rows, errors = [], []
    for f in sorted(ML.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                (errors if "error" in r else rows).append(r)
    rescore(rows)
    return rows, errors


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["length"], r["fmt"])].append(r)
    out = {}
    for k, rs in g.items():
        cold = next(r for r in rs if r["cold"])
        warm = [r["total"] for r in rs if not r["cold"]]
        ctx = cold["prompt_tokens"] - cold["base_tokens"]
        out[k] = dict(rows=rs, acc=sum(r["ok"] for r in rs) / len(rs), n=len(rs), ctx=ctx, nf=cold["n_facts"], tpf=ctx / cold["n_facts"],
                      ttft=cold["ttft"], tps=cold["prompt_tokens"] / cold["ttft"], warm=st.mean(warm) if warm else 0,
                      ok={r["i"]: r["ok"] for r in rs})
    return out


def paired(S, a, b):
    x, y = S[a]["ok"], S[b]["ok"]
    idx = [i for i in x if i in y]
    bb = sum(x[i] and not y[i] for i in idx)
    cc = sum(y[i] and not x[i] for i in idx)
    return bb, cc, len(idx)


def cell(S, a, b) -> str:
    """'Δacc [CI] b/c p' for variant a vs reference b (paired by question index)."""
    if a not in S or b not in S:
        return "-"
    bb, cc, n = paired(S, a, b)
    lo, hi = diff_ci(bb, cc, n)
    return f"{(bb - cc) / n * 100:+.1f}pt [{lo:+.0f},{hi:+.0f}] {bb}/{cc} p={sign_test(bb, cc):.2f}"


def examples() -> str:
    import tiktoken

    from .gen import generate
    enc = tiktoken.get_encoding("o200k_base")
    c = lambda s: len(enc.encode(s))
    ents, _ = generate(2000, c, seed=1, n_questions=8)
    ent = next(e for e in ents if e.label.startswith("人物"))
    out = ["## 同一内容の3言語 × 3表現(実例)\n",
           "同じ facts(共通の Canonical Facts)を EN/KO/JA の L0・L1・MKW にしたもの。MKW は関係・演算子・概念語が漢字で共通、"
           "固有名詞(人名・愛称・都市)だけ原言語の表記。トークン数は o200k。\n"]
    for lang in LANGS:
        out.append(f"### {lang}\n")
        for rep in REPS:
            t = mlenc.encode_ml([ent], lang, rep)
            out.append(f"**{lang}-{rep}** — {c(t)} tok\n\n```text\n{t}\n```\n")
    return "\n".join(out)


def main():
    rows, errors = load()
    S = summarize(rows)
    L: List[str] = []
    P = L.append
    P("# Issue #11 多言語(EN/KO/JA)→機械漢文 評価レポート\n")
    P("同一 Canonical Facts から EN/KO/JA の L0・L1 と MKW を生成。質問は文脈と同じ言語(`X-L0/L1/MKW`)、"
      "または日本語固定(`X-MKW@JA`)。モデル行は LM Studio 実測の文脈トークン(空文脈の基準リクエストとの差)。"
      "各セル 48 問・1 seed。Δ は対応あり(同一問題)の差で `[95%CI] b/c p` を併記、p は符号検定(多重比較補正なし)。\n")
    P(examples())
    models = sorted({m for m, _, _ in S})
    csv_rows = []
    for model in models:
        for length in sorted({l for m, l, _ in S if m == model}):
            V = {v: S[(model, length, v)] for (m, l, v) in S if (m, l) == (model, length)}
            nf = next(iter(V.values()))["nf"]
            P(f"\n## {model} / {length} ({nf} facts)\n")
            P("| 変種 | 精度 | 文脈tok | tok/fact | vs 同言語L0 | vs 同言語L1 | TTFTコールド(s) | prefill tok/s | ウォーム1問(s) | vs 同言語L1(Δacc, p) |")
            P("|---|---|---|---|---|---|---|---|---|---|")
            for v in [x for x in BASE + ["EN-MKW@JA", "KO-MKW@JA"] if x in V]:
                s = V[v]
                lang = v[:2]
                l0, l1 = V.get(f"{lang}-L0"), V.get(f"{lang}-L1")
                r0 = f"{s['ctx'] / l0['ctx'] - 1:+.0%}" if l0 else "-"
                r1 = f"{s['ctx'] / l1['ctx'] - 1:+.0%}" if l1 else "-"
                vs = cell({k: x for k, x in V.items()}, v, f"{lang}-L1") if v != f"{lang}-L1" else "-"
                P(f"| {v} | {s['acc']:.1%} | {s['ctx']} | {s['tpf']:.2f} | {r0} | {r1} | {s['ttft']:.1f} | {s['tps']:.0f} | {s['warm']:.1f} | {vs} |")
                csv_rows.append(dict(model=model, length=length, variant=v, acc=round(s["acc"], 4), n=s["n"], ctx_tokens=s["ctx"],
                                     tok_per_fact=round(s["tpf"], 3), cold_ttft=round(s["ttft"], 2), prefill_tps=round(s["tps"], 1),
                                     warm_s=round(s["warm"], 2)))
            pts = [(v, s["ctx"], s["acc"] * 100, {"JA": "anchor", "EN": "single", "KO": "ladder"}[v[:2]]) for v, s in V.items()]
            P("\n**Pareto frontier (文脈tok, 精度)**: " + ", ".join(pareto([(v, s["ctx"], s["acc"]) for v, s in V.items()])))
            svgp = RES / f"ml_pareto_{model.replace('/', '_')}_{length}.svg"
            svg_scatter(pts, f"{model} {length}: tokens vs accuracy (red=JA blue=EN green=KO)", svgp, COLORS)
            P(f"グラフ: [{svgp.name}]({svgp.name})")
    # tokens across languages: the language-invariance question
    P("\n## 言語間のトークン数(MKW は言語非依存か)\n")
    P("同じ facts を各言語にしたときの文脈トークン。**変動係数(CV) = 3言語の標準偏差/平均**が小さいほど言語に依らない。\n")
    P("| 計測 | 文脈長 | " + " | ".join(f"{r} (JA / EN / KO)" for r in REPS) + " | CV: L0 / L1 / MKW |")
    P("|---|---|---|---|---|---|")
    import tiktoken

    from .gen import generate
    enc = tiktoken.get_encoding("o200k_base")
    cnt = lambda s: len(enc.encode(s))
    lens = sorted({l for _, l, _ in S})
    for length in lens:
        ents, _ = generate(length, cnt, seed=1, n_questions=8)
        row = {r: [cnt(mlenc.encode_ml(ents, lg, r)) for lg in LANGS] for r in REPS}
        P(f"| o200k | {length} | " + " | ".join(" / ".join(map(str, row[r])) for r in REPS) + " | "
          + " / ".join(f"{st.pstdev(row[r]) / st.mean(row[r]):.1%}" for r in REPS) + " |")
    for model in models:
        for length in sorted({l for m, l, _ in S if m == model}):
            row = {}
            for r in REPS:
                vals = [S.get((model, length, f"{lg}-{r}"), {}).get("ctx") for lg in LANGS]
                row[r] = vals if all(vals) else None
            P(f"| {model} | {length} | " + " | ".join((" / ".join(map(str, row[r])) if row[r] else "-") for r in REPS) + " | "
              + " / ".join((f"{st.pstdev(row[r]) / st.mean(row[r]):.1%}" if row[r] else "-") for r in REPS) + " |")
    # accuracy: MKW vs L1, cross-language, question language
    P("\n## 精度の比較(対応あり: 同一問題)\n")
    for model in models:
        P(f"### {model}\n")
        P("**(a) 各言語で MKW − L1 / MKW − L0**\n")
        P("| 文脈長 | 言語 | MKW − L1 | MKW − L0 |")
        P("|---|---|---|---|")
        for length in sorted({l for m, l, _ in S if m == model}):
            for lg in LANGS:
                P(f"| {length} | {lg} | {cell(S, (model, length, f'{lg}-MKW'), (model, length, f'{lg}-L1'))} | "
                  f"{cell(S, (model, length, f'{lg}-MKW'), (model, length, f'{lg}-L0'))} |")
        P("\n**(b) 言語間: EN − JA / KO − JA(同じ表現・同じ問題)**\n")
        P("| 文脈長 | 表現 | EN − JA | KO − JA |")
        P("|---|---|---|---|")
        for length in sorted({l for m, l, _ in S if m == model}):
            for r in REPS:
                P(f"| {length} | {r} | {cell(S, (model, length, f'EN-{r}'), (model, length, f'JA-{r}'))} | "
                  f"{cell(S, (model, length, f'KO-{r}'), (model, length, f'JA-{r}'))} |")
        P("\n**(c) 質問言語: 日本語固定 − 同言語(MKW 文脈は同一)**\n")
        P("| 文脈長 | EN-MKW@JA − EN-MKW | KO-MKW@JA − KO-MKW |")
        P("|---|---|---|")
        for length in sorted({l for m, l, _ in S if m == model}):
            P(f"| {length} | {cell(S, (model, length, 'EN-MKW@JA'), (model, length, 'EN-MKW'))} | "
              f"{cell(S, (model, length, 'KO-MKW@JA'), (model, length, 'KO-MKW'))} |")
        P("")
    # pooled MKW - L1 per language across lengths
    P("\n## MKW − L1 を文脈長でプール(言語別 / 全言語)\n")
    P("| モデル | 言語 | 比較問題数 | MKW 精度 | L1 精度 | Δ [95%CI] | b/c | p |")
    P("|---|---|---|---|---|---|---|---|")
    for model in models:
        for grp in [(lg,) for lg in LANGS] + [LANGS]:
            bb = cc = n = am = al = 0
            for length in sorted({l for m, l, _ in S if m == model}):
                for lg in grp:
                    a, b = (model, length, f"{lg}-MKW"), (model, length, f"{lg}-L1")
                    if a in S and b in S:
                        x, y, m_ = paired(S, a, b)
                        bb, cc, n = bb + x, cc + y, n + m_
                        am += sum(S[a]["ok"][i] for i in S[a]["ok"] if i in S[b]["ok"])
                        al += sum(S[b]["ok"][i] for i in S[b]["ok"] if i in S[a]["ok"])
            if n:
                lo, hi = diff_ci(bb, cc, n)
                P(f"| {model} | {'+'.join(grp)} | {n} | {am / n:.1%} | {al / n:.1%} | {(bb - cc) / n * 100:+.1f}pt [{lo:+.1f},{hi:+.1f}] | {bb}/{cc} | {sign_test(bb, cc):.2f} |")
    # category accuracy
    P("\n## 意味カテゴリ別精度(文脈長プール, 括弧内は問題数)\n")
    for model in models:
        by = defaultdict(lambda: defaultdict(list))
        for (m, l, v), d in S.items():
            if m == model and v in BASE:
                for r in d["rows"]:
                    by[v][r["cat"]].append(r["ok"])
        cats = sorted({c for v in by for c in by[v]})
        P(f"**{model}**\n")
        P("| カテゴリ | " + " | ".join(BASE) + " |")
        P("|---|" + "---|" * len(BASE))
        for c in cats:
            P(f"| {c} | " + " | ".join(f"{sum(by[v][c]) / len(by[v][c]):.0%}({len(by[v][c])})" if by[v].get(c) else "-" for v in BASE) + " |")
        P("")
    # conversion
    cp = ML / "conversion.json"
    if cp.exists():
        conv = json.loads(cp.read_text(encoding="utf-8"))
        P("\n## 直接変換(各言語の L0 → MKW, 日本語を経由しない)と損益分岐\n")
        P("F1 は正解 IR(固有名詞表記も一致を要求)との項目一致率、QA は変換結果だけを文脈にした 48 問の精度。"
          "T_saved = コールド TTFT(同言語 L0) − コールド TTFT(MKW)、R_be = T_conv / T_saved(再利用回数)。"
          "token 版 R_be は (変換の入力 + w×出力 tok) / (L0 − MKW の文脈 tok)。\n")
        P("| モデル | 言語 | 文脈長 | 変換時間 | 入力+出力 tok | 出力が上限? | デコード tok/s | F1 | 変換結果でのQA | 正解IRでのQA | T_saved | R_be(時間) | R_be(tok, w=1) | R_be(tok, w=5) |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, c in conv.items():
            m, lg, ln = k.split("|")
            ln = int(ln)
            l0, mk = S.get((m, ln, f"{lg}-L0")), S.get((m, ln, f"{lg}-MKW"))
            ts = (l0["ttft"] - mk["ttft"]) if l0 and mk else None
            sv = (l0["ctx"] - mk["ctx"]) if l0 and mk else None
            fm = lambda x: f"{x:.1f}" if x and x > 0 else "-"
            P(f"| {m} | {lg} | {ln} | {c['total']:.0f}s | {c['prompt_tokens']}+{c['completion_tokens']} | "
              f"{'**はい**' if c['completion_tokens'] >= 7990 else 'いいえ'} | {c['decode_tps']:.1f} | {c['f1']:.2f} | {c['acc']:.1%} | "
              f"{mk['acc']:.1%} | {ts:.1f}s | {fm(c['total'] / ts) if ts else '-'} | "
              f"{fm((c['prompt_tokens'] + c['completion_tokens']) / sv) if sv else '-'} | "
              f"{fm((c['prompt_tokens'] + 5 * c['completion_tokens']) / sv) if sv else '-'} |" if (l0 and mk) else
              f"| {m} | {lg} | {ln} | {c['total']:.0f}s | {c['prompt_tokens']}+{c['completion_tokens']} | - | {c['decode_tps']:.1f} | {c['f1']:.2f} | {c['acc']:.1%} | - | - | - | - | - |")
        P("\n(ルールベース変換は構造化 facts からの描画で数ミリ秒。生の文章からの抽出は含まない。)")
    if errors:
        P("\n## 失敗した変種\n")
        for e in errors:
            P(f"- {e['model']} len={e['length']} {e['fmt']}: {e['error'][:140]}")
    text = "\n".join(L)
    (RES / "ml_report.md").write_text(text + "\n", encoding="utf-8")
    with (RES / "ml_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(text)
