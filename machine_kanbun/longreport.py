"""Aggregate results/long/*.jsonl into the Issue #4 report + model profile JSON."""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import legend as lg
from .tokens import get_counters

RES = Path(__file__).parent.parent / "results"
FIXED_MKW = ("L2", "L3", "L4", "L5")


def vname(fmt: str, leg: str) -> str:
    return fmt if fmt in ("json", "L0", "L1") else f"{fmt}:{leg}"


def pareto(points):
    """points: [(name, cost, acc)] -> names not dominated (lower cost, higher acc)."""
    return [n for n, c, a in points
            if not any((c2 <= c and a2 >= a) and (c2 < c or a2 > a) for _, c2, a2 in points)]


def sign_test(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant pairs (b = variant right/L1 wrong, c = reverse)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load():
    rows, errors = [], []
    for f in sorted((RES / "long").glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                (errors if "error" in r else rows).append(r)
    return rows, errors


def legend_tokens(cond: str, rows, count) -> float:
    if cond in ("full", "minimal"):
        return count(lg.legend_text(cond))
    if cond == "category":
        ts = [count(t) for t in {lg.legend_text("category", r["cat"]) for r in rows} if t]
        return st.mean(ts) if ts else 0  # average per-question legend
    return 0


def summarize(rows, count):
    """-> {(model,length): {variant: stats}}"""
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["length"], r["fmt"], r["legend"])].append(r)
    out: Dict = defaultdict(dict)
    for (model, length, fmt, leg), rs in g.items():
        cold = next(r for r in rs if r["cold"])
        warm = [r["ttft"] for r in rs if not r["cold"]]
        ctx = cold["prompt_tokens"] - cold["base_tokens"]
        out[(model, length)][vname(fmt, leg)] = dict(
            fmt=fmt, legend=leg, n=len(rs), acc=sum(r["ok"] for r in rs) / len(rs), ctx=ctx, n_facts=cold["n_facts"],
            tpf=ctx / cold["n_facts"], leg_tok=legend_tokens(leg, rs, count), ttft=cold["ttft"],
            tps=cold["prompt_tokens"] / cold["ttft"] if cold["ttft"] else 0, warm=st.mean(warm) if warm else 0,
            total=cold["total"], mem=cold["used_peak_mb"] - cold["used_base_mb"], rss=cold["rss_peak_mb"] - cold["rss_base_mb"])
    return out


def main(reuse: List[int]):
    count = get_counters()["o200k_base"]
    rows, errors = load()
    L: List[str] = []
    P = L.append
    if not rows:
        print("no results in results/long/")
        return
    S = summarize(rows, count)
    P("# Issue #4 長文コンテキスト評価レポート\n")
    P("凡例トークンは o200k による推定、文脈トークンはモデル実測(prompt_tokens − 空文脈の基準リクエスト)。"
      "TTFT/prefill は各変種の1問目(コールド, prefix cache 無効化)。実効凡例コスト = 凡例Token / 再利用回数 R。\n")
    for (model, length) in sorted(S):
        V = S[(model, length)]
        l0, l1, js = V.get("L0"), V.get("L1"), V.get("json")
        P(f"\n## {model} / 文脈長 {length} (L0 相当, {next(iter(V.values()))['n_facts']} facts)\n")
        hdr = ("| 変種 | 精度 | 文脈tok | tok/fact | vs L0 | vs L1 | vs JSON | 凡例tok | " +
               " | ".join(f"実効tok(R={r})" for r in reuse) + " | TTFTコールド(s) | prefill tok/s | TTFTウォーム(s) | メモリΔ(MB) |")
        P(hdr)
        P("|" + "---|" * (hdr.count("|") - 1))
        order = ["json", "L0", "L1"] + sorted(v for v in V if v not in ("json", "L0", "L1"))
        for v in order:
            s = V.get(v)
            if not s:
                continue
            rel = lambda b: f"{s['ctx'] / b['ctx'] - 1:+.0%}" if b else "-"
            effs = " | ".join(f"{s['ctx'] + s['leg_tok'] / r:.0f}" for r in reuse)
            P(f"| {v} | {s['acc']:.1%} | {s['ctx']} | {s['tpf']:.2f} | {rel(l0)} | {rel(l1)} | {rel(js)} | {s['leg_tok']:.0f} | "
              f"{effs} | {s['ttft']:.1f} | {s['tps']:.0f} | {s['warm']:.2f} | {s['mem']:.0f} |")
        # break-even
        P("\n**凡例の損益分岐 (N_break_even = 凡例tok / 1factあたり削減量, 基準=L1 と L0)**\n")
        P("| 変種 | 凡例tok | 精度向上(凡例なし比) | S(vs L1) tok/fact | N_be(vs L1) | S(vs L0) | N_be(vs L0) |")
        P("|---|---|---|---|---|---|---|")
        for v, s in sorted(V.items()):
            if s["leg_tok"] <= 0:
                continue
            nol = V.get(f"{s['fmt']}:none")
            gain = f"{(s['acc'] - nol['acc']) * 100:+.1f}pt" if nol else "-"
            def be(b):
                if not b:
                    return "-", "-"
                sv = (b["ctx"] - s["ctx"]) / s["n_facts"]
                return f"{sv:.2f}", (f"{s['leg_tok'] / sv:.0f} facts" if sv > 0 else "∞")
            a1, b1 = be(l1)
            a0, b0 = be(l0)
            P(f"| {v} | {s['leg_tok']:.0f} | {gain} | {a1} | {b1} | {a0} | {b0} |")
        # L1 vs best fixed vs adaptive
        P("\n**L1 vs 最良固定MKW vs Adaptive vs JSON** (実効tok は R=1 / R=10)\n")
        P("| 系統 | 変種 | 精度 | 実効tok(R=1) | 実効tok(R=10) | TTFTコールド(s) |")
        P("|---|---|---|---|---|---|")
        fixed = [(v, s) for v, s in V.items() if s["fmt"] in FIXED_MKW]
        ada = [(v, s) for v, s in V.items() if s["fmt"] == "adaptive"]
        pick = lambda xs: max(xs, key=lambda t: (t[1]["acc"], -(t[1]["ctx"] + t[1]["leg_tok"]))) if xs else None
        for label, item in (("L1", ("L1", l1) if l1 else None), ("最良固定MKW", pick(fixed)), ("Adaptive", pick(ada)),
                            ("JSON", ("json", js) if js else None), ("L0", ("L0", l0) if l0 else None)):
            if item:
                v, s = item
                P(f"| {label} | {v} | {s['acc']:.1%} | {s['ctx'] + s['leg_tok']:.0f} | {s['ctx'] + s['leg_tok'] / 10:.0f} | {s['ttft']:.1f} |")
        # pareto
        P("\n**Pareto frontier**")
        for label, key in (("(実効tok R=1, 精度)", lambda s: s["ctx"] + s["leg_tok"]),
                           ("(実効tok R=10, 精度)", lambda s: s["ctx"] + s["leg_tok"] / 10),
                           ("(TTFTコールド, 精度)", lambda s: s["ttft"])):
            P(f"- {label}: " + ", ".join(pareto([(v, key(s), s["acc"]) for v, s in V.items()])))
        # promising checks
        P("\n**有望判定チェック(この長さ)**")
        if l1:
            beat = [v for v, s in V.items() if s["fmt"] in FIXED_MKW + ("adaptive",) and s["acc"] >= l1["acc"]
                    and s["ctx"] + s["leg_tok"] < l1["ctx"]]
            P(f"- L1と同等以上の精度でToken削減: {', '.join(beat) or 'なし'}")
            hi = [v for v, s in V.items() if s["fmt"] in FIXED_MKW + ("adaptive",) and s["acc"] > l1["acc"]
                  and s["ctx"] + s["leg_tok"] <= l1["ctx"] * 1.05]
            P(f"- 同Token量(±5%)でL1より高精度: {', '.join(hi) or 'なし'}")
        if js:
            jb = [v for v, s in V.items() if s["fmt"] in FIXED_MKW + ("adaptive",) and s["acc"] >= js["acc"]
                  and s["ctx"] + s["leg_tok"] < js["ctx"] * 0.6]
            P(f"- JSONより大幅(-40%以上)に少Tokenで同等以上: {', '.join(jb) or 'なし'}")
        if l0 and length >= 8000:
            fast = [f"{v}({s['ttft'] / l0['ttft'] - 1:+.0%})" for v, s in V.items()
                    if s["fmt"] in FIXED_MKW + ("adaptive",) and s["ttft"] < l0["ttft"] * 0.8]
            P(f"- 8k以上でTTFT(vs L0)が20%超改善: {', '.join(fast) or 'なし'}")
    # accuracy vs length
    P("\n## 文脈長別精度\n")
    for model in sorted({m for m, _ in S}):
        lens = sorted(l for m, l in S if m == model)
        vs = sorted({v for (m, l), V in S.items() if m == model for v in V})
        P(f"**{model}**\n")
        P("| 変種 | " + " | ".join(str(l) for l in lens) + " |")
        P("|---|" + "---|" * len(lens))
        for v in vs:
            P(f"| {v} | " + " | ".join(f"{S[(model, l)][v]['acc']:.0%}" if v in S[(model, l)] else "-" for l in lens) + " |")
        P("")
    # paired comparison against L1 (same documents, same questions), pooled over lengths
    P("\n## L1 との対応あり比較(同一問題での正誤、符号検定)\n")
    P("b = その変種のみ正解の問題数, c = L1 のみ正解の問題数。p は両側正確検定。"
      "同一文書の問題を全変種が共有するため変種間の誤りは相関しており、多重比較の補正はしていない。\n")
    for model in sorted({r["model"] for r in rows}):
        ok = {}
        for r in rows:
            if r["model"] == model:
                ok[(vname(r["fmt"], r["legend"]), r["length"], r["i"])] = r["ok"]
        base = {k[1:]: v for k, v in ok.items() if k[0] == "L1"}
        P(f"**{model}**\n")
        P("| 変種 | 比較対数 | 精度(変種) | 精度(L1) | b | c | p |")
        P("|---|---|---|---|---|---|---|")
        for v in sorted({k[0] for k in ok} - {"L1"}):
            ks = [k[1:] for k in ok if k[0] == v and k[1:] in base]
            if not ks:
                continue
            b = sum(ok[(v, *k)] and not base[k] for k in ks)
            c = sum(base[k] and not ok[(v, *k)] for k in ks)
            P(f"| {v} | {len(ks)} | {sum(ok[(v, *k)] for k in ks) / len(ks):.1%} | {sum(base[k] for k in ks) / len(ks):.1%} "
              f"| {b} | {c} | {sign_test(b, c):.2f} |")
        P("")
    # category / operator accuracy (pooled over lengths)
    prof: Dict = {}
    for model in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == model]
        P(f"\n## {model}: 意味カテゴリ別・演算子別精度 (全長プール)\n")
        prof[model] = {"operator_accuracy": {}, "category_accuracy": {}}
        for dim, key, tgt in (("カテゴリ", "cat", "category_accuracy"), ("演算子", "op", "operator_accuracy")):
            cells = defaultdict(list)
            for r in mr:
                if r[key] or dim == "カテゴリ":
                    cells[(r[key], vname(r["fmt"], r["legend"]))].append(r["ok"])
            vs = sorted({v for _, v in cells}, key=lambda v: (v not in ("json", "L0", "L1"), v))
            ks = sorted({k for k, _ in cells})
            P(f"**{dim}別**\n")
            P(f"| {dim} | " + " | ".join(vs) + " |")
            P("|---|" + "---|" * len(vs))
            for k in ks:
                P(f"| {k} | " + " | ".join(
                    f"{sum(cells[(k, v)]) / len(cells[(k, v)]):.0%}({len(cells[(k, v)])})" if cells.get((k, v)) else "-" for v in vs) + " |")
                prof[model][tgt][k] = {v: {"acc": sum(cells[(k, v)]) / len(cells[(k, v)]), "n": len(cells[(k, v)])}
                                       for v in vs if cells.get((k, v))}
            P("")
        prof[model]["variant_summary"] = {
            f"{length}": {v: {k: s[k] for k in ("acc", "ctx", "tpf", "leg_tok", "ttft", "tps")} for v, s in V.items()}
            for (m, length), V in S.items() if m == model}
    P("\n(括弧内は問題数。1問あたりの寄与が大きいので n が小さい演算子は参考値。)")
    if errors:
        P("\n## 失敗した変種\n")
        for e in errors:
            P(f"- {e['model']} len={e['length']} {vname(e['fmt'], e['legend'])}: {e['error'][:160]}")
    text = "\n".join(L)
    (RES / "long_report.md").write_text(text + "\n", encoding="utf-8")
    (RES / "model_profiles.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)
