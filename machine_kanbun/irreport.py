"""Issue #17 report: IR-C / IR-L / IR-ID / IR-IDL x {JA,EN,KO} x model x length (+ L0 / L1 baselines)."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import irlabel, mlenc
from .ablreport import diff_ci, svg_scatter
from .longreport import pareto, sign_test
from .mlreport import rescore

RES = Path(__file__).parent.parent / "results"
LANGS = ("JA", "EN", "KO")
CONDS = ("IRC", "IRL", "IRID", "IRIDL")
NAMES = {"IRC": "IR-C 共通漢字", "IRL": "IR-L 各言語", "IRID": "IR-ID 抽象ID", "IRIDL": "IR-IDL 抽象ID+凡例"}
COLORS = {"anchor": "#c0392b", "single": "#2471a3", "ladder": "#1e8449"}  # JA / EN / KO


def load():
    rows, errors = [], []
    for d in ("ml", "ir", "zh"):
        if not (RES / d).exists():
            continue
        for f in sorted((RES / d).glob("*.jsonl")):
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    (errors if "error" in r else rows).append(r)
    rescore(rows)
    for r in rows:
        r["cond"] = r["fmt"].replace("-MKW", "-IRC")  # IR-C is exactly the existing MKW runs
    return rows, errors


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["length"], r["cond"])].append(r)
    out = {}
    for k, rs in g.items():
        cold = next(r for r in rs if r["cold"])
        ctx = cold["prompt_tokens"] - cold["base_tokens"]
        out[k] = dict(rows=rs, acc=sum(r["ok"] for r in rs) / len(rs), n=len(rs), ctx=ctx, nf=cold["n_facts"], base=cold["base_tokens"],
                      ttft=cold["ttft"], ok={r["i"]: r["ok"] for r in rs})
    for (m, l, c), v in list(out.items()):  # Japanese IR-L has the same vocabulary as IR-C (kanji are Japanese's own words)
        if c == "JA-IRC":
            out[(m, l, "JA-IRL")] = v
    # legend overhead = base prompt tokens with legend - without (same question, same system prompt otherwise)
    for (m, l, c), v in out.items():
        v["legend"] = 0
        if c.endswith("-IRIDL") and (m, l, c.replace("IRIDL", "IRID")) in out:
            v["legend"] = v["base"] - out[(m, l, c.replace("IRIDL", "IRID"))]["base"]
    return out


def cv(vals):
    return st.pstdev(vals) / st.mean(vals) if len(vals) > 1 else 0.0


def paired(S, a, b):
    x, y = S[a]["ok"], S[b]["ok"]
    idx = [i for i in x if i in y]
    return sum(x[i] and not y[i] for i in idx), sum(y[i] and not x[i] for i in idx), len(idx)


def cell(S, a, b) -> str:
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
    p = next(e for e in ents if e.label.startswith("人物"))
    out = ["## 3条件の実例(同じ人物, o200k トークン)\n",
           "IR-C は関係・演算子・概念語が漢字、IR-L はそれらを各言語の単語に、IR-ID は構造語を言語共通の意味のないコードに置換する"
           "(名前・名詞・数値は IR-L と同じ)。記号 `{ } ; : > ?` は全条件で共通。\n"]
    for lang in LANGS:
        out.append(f"### {lang}\n")
        for cond in CONDS[:3]:
            t = irlabel.render([p], lang, cond)
            out.append(f"**{NAMES[cond]}** — {c(t)} tok\n\n```text\n{t}\n```\n")
        leg = irlabel.legend(lang)
        out.append(f"**IR-IDL の凡例**(システムプロンプトに1回) — {c(leg)} tok\n\n```text\n{leg}\n```\n")
    return "\n".join(out)


def main():
    rows, errors = load()
    S = summarize(rows)
    models = sorted({m for m, _, _ in S})
    L: List[str] = []
    P = L.append
    P("# Issue #17 多言語 Semantic IR のラベル局所化実験\n")
    P("同じ facts・同じ 48 問・同じ文書で、IR のラベル語彙だけを変える。IR-C は既存の MKW 実行(同一テキスト)、"
      "日本語の IR-L は IR-C と同一語彙(漢字が日本語自身の語)。モデル列は LM Studio 実測の文脈トークン(凡例は別掲)。"
      "CV = 3 言語のトークン数の標準偏差/平均。Δ は対応あり(同一問題)で `[95%CI] b/c p`(符号検定, 補正なし)。\n")
    P(examples())
    # ---- token CV
    import tiktoken

    from .gen import generate
    enc = tiktoken.get_encoding("o200k_base")
    cnt = lambda s: len(enc.encode(s))
    lens = sorted({l for _, l, _ in S})
    P("\n## 言語間トークン CV(主要指標)\n")
    P("| 計測 | 文脈長 | " + " | ".join(f"{NAMES[c]} (JA / EN / KO)" for c in CONDS[:3]) + " | CV: C / L / ID |")
    P("|---|---|---|---|---|---|")
    csv_rows = []
    for length in lens:
        ents, _ = generate(length, cnt, seed=1, n_questions=8)
        v = {c: [cnt(irlabel.render(ents, lg, c)) for lg in LANGS] for c in CONDS[:3]}
        P(f"| o200k | {length} | " + " | ".join(" / ".join(map(str, v[c])) for c in CONDS[:3]) + " | " + " / ".join(f"{cv(v[c]):.1%}" for c in CONDS[:3]) + " |")
    for model in models:
        for length in sorted({l for m, l, _ in S if m == model}):
            v = {c: [S.get((model, length, f"{lg}-{c}"), {}).get("ctx") for lg in LANGS] for c in CONDS[:3]}
            if all(all(x) for x in v.values()):
                P(f"| {model} | {length} | " + " | ".join(" / ".join(map(str, v[c])) for c in CONDS[:3]) + " | " + " / ".join(f"{cv(v[c]):.1%}" for c in CONDS[:3]) + " |")
                for c in CONDS[:3]:
                    csv_rows.append(dict(kind="token_cv", model=model, length=length, cond=c, cv=round(cv(v[c]), 4)))
    # ---- main table
    for model in models:
        for length in sorted({l for m, l, _ in S if m == model}):
            V = {c: s for (m, l, c), s in S.items() if (m, l) == (model, length)}
            P(f"\n## {model} / {length}\n")
            P("| 条件 | 言語 | 文脈tok | 凡例tok | 合計tok | vs 同言語L0 | vs 同言語L1 | 精度 | TTFTコールド(s) | vs IR-C(Δacc) |")
            P("|---|---|---|---|---|---|---|---|---|---|")
            pts = []
            for cond in CONDS:
                for lg in LANGS:
                    k = f"{lg}-{cond}"
                    s = V.get(k)
                    if not s:
                        continue
                    tot = s["ctx"] + s["legend"]
                    l0, l1 = V.get(f"{lg}-L0"), V.get(f"{lg}-L1")
                    r0 = f"{tot / l0['ctx'] - 1:+.0%}" if l0 else "-"
                    r1 = f"{tot / l1['ctx'] - 1:+.0%}" if l1 else "-"
                    d = cell(S, (model, length, k), (model, length, f"{lg}-IRC")) if cond != "IRC" else "-"
                    P(f"| {NAMES[cond]} | {lg} | {s['ctx']} | {s['legend'] or '-'} | {tot} | {r0} | {r1} | {s['acc']:.1%} | {s['ttft']:.1f} | {d} |")
                    pts.append((k, tot, s["acc"] * 100, {"JA": "anchor", "EN": "single", "KO": "ladder"}[lg]))
                    csv_rows.append(dict(kind="cell", model=model, length=length, cond=cond, lang=lg, ctx_tokens=s["ctx"], legend_tokens=s["legend"],
                                         acc=round(s["acc"], 4), n=s["n"], cold_ttft=round(s["ttft"], 2)))
            for lg in LANGS:  # baselines for context
                for b in ("L0", "L1"):
                    s = V.get(f"{lg}-{b}")
                    if s:
                        pts.append((f"{lg}-{b}", s["ctx"], s["acc"] * 100, {"JA": "anchor", "EN": "single", "KO": "ladder"}[lg]))
            P("\n**ベースライン**: " + ", ".join(f"{lg}-{b} {V[f'{lg}-{b}']['acc']:.0%}/{V[f'{lg}-{b}']['ctx']}tok" for lg in LANGS for b in ("L0", "L1") if f"{lg}-{b}" in V))
            P("**Pareto frontier (合計tok, 精度)**: " + ", ".join(pareto([(n, t, a / 100) for n, t, a, _ in pts])))
            svgp = RES / f"ir_pareto_{model.replace('/', '_')}_{length}.svg"
            svg_scatter(pts, f"{model} {length} (red=JA blue=EN green=KO)", svgp, COLORS)
            P(f"グラフ: [{svgp.name}]({svgp.name})")
    # ---- pooled paired comparisons
    P("\n## 条件間の精度差(文脈長・言語でプール, 対応あり)\n")
    P("| モデル | 比較 | 対象言語 | 比較問題数 | Δ [95%CI] | b/c | p |")
    P("|---|---|---|---|---|---|---|")
    pairs = [("IRL", "IRC"), ("IRID", "IRC"), ("IRIDL", "IRC"), ("IRIDL", "IRID"), ("IRID", "IRL")]
    for model in models:
        for a, b in pairs:
            for grp in [("EN", "KO"), ("JA", "EN", "KO")] + [(lg,) for lg in LANGS]:
                if a == "IRL" and grp == ("JA",):
                    continue  # identical vocabulary
                bb = cc = n = 0
                for length in sorted({l for m, l, _ in S if m == model}):
                    for lg in grp:
                        ka, kb = (model, length, f"{lg}-{a}"), (model, length, f"{lg}-{b}")
                        if ka in S and kb in S:
                            x, y, m_ = paired(S, ka, kb)
                            bb, cc, n = bb + x, cc + y, n + m_
                if n:
                    lo, hi = diff_ci(bb, cc, n)
                    P(f"| {model} | {NAMES[a]} − {NAMES[b]} | {'+'.join(grp)} | {n} | {(bb - cc) / n * 100:+.1f}pt [{lo:+.1f},{hi:+.1f}] | {bb}/{cc} | {sign_test(bb, cc):.2f} |")
    # ---- categories
    P("\n## 意味カテゴリ別精度(EN+KO・文脈長プール, 括弧内は問題数)\n")
    for model in models:
        by = defaultdict(lambda: defaultdict(list))
        for (m, l, c), d in S.items():
            if m == model and c.split("-")[1] in CONDS and not c.startswith("JA-"):  # EN/KO only: JA IR-L == IR-C was not re-run
                for r in d["rows"]:
                    by[c.split("-")[1]][r["cat"]].append(r["ok"])
        cats = sorted({x for v in by.values() for x in v})
        P(f"**{model}**\n")
        P("| カテゴリ | " + " | ".join(NAMES[c] for c in CONDS) + " |")
        P("|---|" + "---|" * len(CONDS))
        for cat in cats:
            P(f"| {cat} | " + " | ".join(f"{sum(by[c][cat]) / len(by[c][cat]):.0%}({len(by[c][cat])})" if by[c].get(cat) else "-" for c in CONDS) + " |")
        P("")
    if errors:
        P("\n## 失敗した変種\n")
        for e in errors:
            P(f"- {e['model']} len={e['length']} {e['fmt']}: {e['error'][:140]}")
    text = "\n".join(L)
    (RES / "ir_report.md").write_text(text + "\n", encoding="utf-8")
    keys = sorted({k for r in csv_rows for k in r})
    with (RES / "ir_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    print(text)
