"""Issue #10 report: L1 ablation results -> markdown, CSV and dependency-free SVG Pareto plots."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from . import ablate
from .longreport import pareto, sign_test

RES = Path(__file__).parent.parent / "results"
ABL = RES / "ablation"
NONINFERIOR_PT = 0.025  # ~1 question of 48: "indistinguishable from L1" tolerance


def vname(fmt: str, leg: str) -> str:
    return fmt if (fmt in ("json", "L0", "L1") or ablate.is_ablation(fmt)) else f"{fmt}:{leg}".replace(":none", "")


def short(v: str) -> str:
    if v.startswith("abL="):
        return f"ladder{len(ablate.parse(v))}"
    if v.startswith("ab="):
        return v[3:]
    return v


def load() -> Tuple[List[dict], List[dict]]:
    rows, errors = [], []
    for f in sorted(ABL.glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                r["variant"] = vname(r["fmt"], r["legend"])
                (errors if "error" in r else rows).append(r)
    return rows, errors


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["length"], r["seed"], r["variant"])].append(r)
    out = {}
    for k, rs in g.items():
        cold = next(r for r in rs if r["cold"])
        ctx = cold["prompt_tokens"] - cold["base_tokens"]
        out[k] = dict(rows=rs, acc=sum(r["ok"] for r in rs) / len(rs), n=len(rs), ctx=ctx, n_facts=cold["n_facts"],
                      tpf=ctx / cold["n_facts"], ttft=cold["ttft"], tps=cold["prompt_tokens"] / cold["ttft"] if cold["ttft"] else 0,
                      ok={r["i"]: r["ok"] for r in rs})
    return out


def paired(S, key, base_key):
    a, b = S[key]["ok"], S[base_key]["ok"]
    idx = [i for i in a if i in b]
    bb = sum(a[i] and not b[i] for i in idx)
    cc = sum(b[i] and not a[i] for i in idx)
    return bb, cc, sign_test(bb, cc), len(idx)


def diff_ci(b: int, c: int, n: int) -> Tuple[float, float]:
    """95% CI of the paired accuracy difference (variant - base) in points, Agresti-Min adjusted
    (+0.5 per cell) so that zero discordant pairs still leaves honest uncertainty."""
    if n == 0:
        return 0.0, 0.0
    n2, b2, c2 = n + 2, b + 0.5, c + 0.5
    d = (b2 - c2) / n2
    se = ((b2 + c2) - (b2 - c2) ** 2 / n2) ** 0.5 / n2
    return (d - 1.96 * se) * 100, (d + 1.96 * se) * 100


def tok_per_pt(base, v) -> str:
    saved = base["ctx"] - v["ctx"]
    lost = (base["acc"] - v["acc"]) * 100
    if saved <= 0:
        return "削減なし"
    return "損失なし" if lost <= 0 else f"{saved / lost:.0f} tok/pt"


def svg_scatter(points, title, path: Path):
    """points: [(label, x, y, kind)] kind in anchor|single|ladder. Pure SVG, no dependencies."""
    W, H, M = 760, 460, 60
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x0, x1 = min(xs) * 0.97, max(xs) * 1.03
    y0, y1 = max(0, min(ys) - 5), min(100, max(ys) + 3)
    sx = lambda x: M + (x - x0) / (x1 - x0 or 1) * (W - 2 * M)
    sy = lambda y: H - M - (y - y0) / (y1 - y0 or 1) * (H - 2 * M)
    col = {"anchor": "#c0392b", "single": "#2471a3", "ladder": "#1e8449"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>', f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
         f'<line x1="{M}" y1="{H - M}" x2="{W - M}" y2="{H - M}" stroke="#444"/><line x1="{M}" y1="{M}" x2="{M}" y2="{H - M}" stroke="#444"/>',
         f'<text x="{W / 2}" y="{H - 14}" text-anchor="middle">context tokens</text>',
         f'<text x="16" y="{H / 2}" transform="rotate(-90 16 {H / 2})" text-anchor="middle">QA accuracy (%)</text>']
    for t in range(5):
        xv, yv = x0 + (x1 - x0) * t / 4, y0 + (y1 - y0) * t / 4
        o.append(f'<text x="{sx(xv):.0f}" y="{H - M + 14}" text-anchor="middle">{xv:.0f}</text>')
        o.append(f'<text x="{M - 6}" y="{sy(yv) + 4:.0f}" text-anchor="end">{yv:.0f}</text>')
        o.append(f'<line x1="{M}" y1="{sy(yv):.0f}" x2="{W - M}" y2="{sy(yv):.0f}" stroke="#eee"/>')
    front = sorted([p for p in points if p[0] in pareto([(q[0], q[1], q[2]) for q in points])], key=lambda p: p[1])
    if len(front) > 1:
        o.append('<polyline fill="none" stroke="#999" stroke-dasharray="4 3" points="' +
                 " ".join(f"{sx(p[1]):.0f},{sy(p[2]):.0f}" for p in front) + '"/>')
    for lab, x, y, kind in points:
        o.append(f'<circle cx="{sx(x):.0f}" cy="{sy(y):.0f}" r="4.5" fill="{col[kind]}" fill-opacity="0.8"/>')
        o.append(f'<text x="{sx(x) + 6:.0f}" y="{sy(y) - 5:.0f}" fill="#333">{lab}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def examples_section() -> str:
    """Same entity rendered in every representation (seed 1), with o200k token counts."""
    from .encoder import POLICIES, encode_doc
    from .gen import generate
    from .tokens import get_counters

    count = get_counters()["o200k_base"]
    ents, _ = generate(2000, count, seed=1, n_questions=8)
    person = next(e for e in ents if e.label.startswith("人物"))
    project = next(e for e in ents if e.label.startswith("案件"))
    out = ["## 圧縮表現の例\n",
           "同じ事実(seed 1 の合成データ)を各表現にしたもの。トークン数は o200k。"
           "L0→L5 が機械漢文系の段階、`ab=` は L1 から1要素だけ変えた表現。\n"]
    for ent, title in ((person, "人物の例"), (project, "案件の例(条件・因果・比較・否定を多く含む)")):
        out.append(f"### {title}: {ent.label}\n")
        forms = [("L0 自然日本語", "L0"), ("L1 日本語要約(基準)", "L1"), ("L2 漢字圧縮", "L2"), ("L3 擬似漢文", "L3"),
                 ("L4 構造漢文", "L4"), ("L5 機械漢文", "L5"),
                 ("Adaptive(gemma 方針)", "adaptive"), ("JSON", "json")]
        forms += [(f"{name}  [{ablate.describe(name)}]", name) for name in
                  ["ab=particle", "ab=neg", "ab=unc", "ab=tense", "ab=struct", "ab=subj",
                   ablate.ladder_variants()[7], ablate.ladder_variants()[-2]]]
        base = count(encode_doc([ent], "L0", POLICIES["gemma"]))
        l1 = count(encode_doc([ent], "L1", POLICIES["gemma"]))
        for label, fmt in forms:
            text = encode_doc([ent], fmt, POLICIES["gemma"])
            n = count(text)
            out.append(f"**{label}** — {n} tok (L0比 {n / base - 1:+.0%}, L1比 {n / l1 - 1:+.0%})\n")
            out.append("```text\n" + text.strip() + "\n```\n")
    out.append(tokenizer_table())
    out.append("**L5 の記号の読み方**: `不`=否定 `無`=不在 `未`=まだ〜でない `非`=〜ではない `禁`=禁止 `疑`=不確実 "
               "`過/今/将`=過去/現在/将来 `若A→B`・`A?B`=AならB `A故B`=AゆえB `好:A>B`=AをBより好む `x{...;...}`=グループ\n")
    return "\n".join(out)


def tokenizer_table() -> str:
    """L1 vs machine-kanbun token counts for the same document under different tokenizers."""
    from .encoder import POLICIES, encode_doc
    from .gen import generate
    from .tokens import get_counters

    c = get_counters()["o200k_base"]
    rows_, _ = load()
    S_ = summarize(rows_)
    out = ["### トークナイザ依存(同一文書の L1 と機械漢文 L5)\n",
           "同じ文書でもトークナイザで L1 と L5 の大小関係が変わる。モデル行は LM Studio が返した実測値(文脈のみ)。\n",
           "| 文脈長 / seed | トークナイザ | L0 | L1 | L5 | L5 vs L1 | L5 vs L0 |", "|---|---|---|---|---|---|---|"]
    for (length, seed) in sorted({(l, s) for (_, l, s, _) in S_}):
        have = [m for m in sorted({m for (m, l, s, _) in S_ if (l, s) == (length, seed)})
                if all((m, length, seed, v) in S_ for v in ("L0", "L1", "L5"))]
        if not have:
            continue
        ents, _ = generate(length, c, seed=seed, n_questions=8)
        o = {f: c(encode_doc(ents, f, POLICIES["gemma"])) for f in ("L0", "L1", "L5")}
        out.append(f"| {length} / s{seed} | o200k (GPT-4o 系) | {o['L0']} | {o['L1']} | {o['L5']} | {o['L5'] / o['L1'] - 1:+.1%} | {o['L5'] / o['L0'] - 1:+.1%} |")
        for m in have:
            g = lambda v: S_[(m, length, seed, v)]["ctx"]
            out.append(f"| {length} / s{seed} | {m} (実測) | {g('L0')} | {g('L1')} | {g('L5')} | {g('L5') / g('L1') - 1:+.1%} | {g('L5') / g('L0') - 1:+.1%} |")
    return "\n".join(out) + "\n"


def main():
    rows, errors = load()
    if not rows:
        print("no results in results/ablation/")
        return
    S = summarize(rows)
    L: List[str] = []
    P = L.append
    P("# Issue #10 L1 アブレーション結果\n")
    P(examples_section())
    P("基準は L1(日本語要約)。各行は L1 の1要素だけを変換した表現(単独 `ab=`)または変換を累積した階段(`abL=`)。"
      "Δtok/Δacc は L1 比。p は同一問題での正誤の符号検定(両側, 多重比較補正なし)。1問 ≒ 2.1pt。\n")
    csv_rows = []
    cells = sorted({(m, l, s) for (m, l, s, _) in S})
    for (model, length, seed) in cells:
        V = {v: S[(model, length, seed, v)] for (m, l, s, v) in S if (m, l, s) == (model, length, seed)}
        base = V.get("L1")
        if not base:
            continue
        P(f"\n## {model} / {length} / seed {seed} ({base['n_facts']} facts)\n")
        P("| 変種 | 内容 | 精度 | 文脈tok | tok/fact | Δtok vs L1 | Δacc vs L1 | b/c | p | 削減tok/失精度pt | TTFTコールド(s) | prefill tok/s |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        groups = [("anchor", [v for v in ("L0", "L1", "L5") if v in V]),
                  ("single", [v for v in V if v.startswith("ab=")]),
                  ("ladder", sorted([v for v in V if v.startswith("abL=")], key=lambda v: len(ablate.parse(v))))]
        for kind, vs in groups:
            for v in vs:
                s = V[v]
                if v == "L1":
                    bc, p = "-", "-"
                else:
                    bb, cc, p_, _n = paired(S, (model, length, seed, v), (model, length, seed, "L1"))
                    bc, p = f"{bb}/{cc}", f"{p_:.2f}"
                desc = (f"累積{len(ablate.parse(v))}段: +{ablate.OP_JA[ablate.LADDER[len(ablate.parse(v)) - 1]]}" if v.startswith("abL=")
                        else ablate.describe(v)) if ablate.is_ablation(v) else {"L0": "自然日本語", "L1": "日本語要約(基準)", "L5": "機械漢文"}[v]
                P(f"| {short(v)} | {desc} | {s['acc']:.1%} | {s['ctx']} | {s['tpf']:.2f} | {s['ctx'] / base['ctx'] - 1:+.1%} | "
                  f"{(s['acc'] - base['acc']) * 100:+.1f}pt | {bc} | {p} | {tok_per_pt(base, s) if v != 'L1' else '-'} | "
                  f"{s['ttft']:.1f} | {s['tps']:.0f} |")
                csv_rows.append(dict(model=model, length=length, seed=seed, kind=kind, variant=v, ops=desc, acc=round(s["acc"], 4),
                                     n=s["n"], ctx_tokens=s["ctx"], tok_per_fact=round(s["tpf"], 3),
                                     d_tok_vs_L1=round(s["ctx"] / base["ctx"] - 1, 4), d_acc_vs_L1=round(s["acc"] - base["acc"], 4),
                                     p_vs_L1=("" if v == "L1" else round(paired(S, (model, length, seed, v), (model, length, seed, "L1"))[2], 4)),
                                     cold_ttft=round(s["ttft"], 2), prefill_tps=round(s["tps"], 1)))
        # pareto + minimal representation candidates
        pts = [(short(v), s["ctx"], s["acc"] * 100, "anchor" if v in ("L0", "L1", "L5") else "single" if v.startswith("ab=") else "ladder")
               for v, s in V.items()]
        P("\n**Pareto frontier (文脈tok, 精度)**: " + ", ".join(pareto([(short(v), s["ctx"], s["acc"]) for v, s in V.items()])))
        cands = []
        for v, s in V.items():
            if v in ("L0",):
                continue
            if v == "L1":
                p_ = 1.0
            else:
                p_ = paired(S, (model, length, seed, v), (model, length, seed, "L1"))[2]
            if s["acc"] >= base["acc"] - NONINFERIOR_PT - 1e-9 and p_ >= 0.05:
                cands.append((s["ctx"], v))
        cands.sort()
        P(f"**最小表現の候補** (精度が L1 より {NONINFERIOR_PT * 100:.1f}pt 以上は下がらず L1 と有意差なし, 文脈tok 昇順): "
          + ", ".join(f"{short(v)}({c})" for c, v in cands[:5]) + "\n")
        svgp = RES / f"ablation_pareto_{model.replace('/', '_')}_{length}_s{seed}.svg"
        svg_scatter(pts, f"{model} {length} seed{seed}: tokens vs accuracy", svgp)
        P(f"グラフ: [{svgp.name}]({svgp.name})\n")
    # per-op effect pooled across lengths/seeds (single ablations)
    P("\n## アブレーションの効果(長さ・seed をプール, L1 比。単独 → 累積の順)\n")
    P("95%CI は Agresti–Min 補正(不一致 0 件でも幅が残る)。この規模(n=48〜96)で検出できるのは概ね ±5pt 以上の差で、"
      "それ未満の差は「差がない」ではなく「検出できない」。\n")
    for model in sorted({m for m, *_ in S}):
        P(f"**{model}**\n")
        P("| 操作 | 内容 | 比較セル数 | 比較問題数 | 平均Δtok | 平均Δacc | 95%CI(Δacc) | b/c | p |")
        P("|---|---|---|---|---|---|---|---|---|")
        vs_ = {v for (m, _, _, v) in S if m == model and ablate.is_ablation(v)}
        for v in sorted(vs_, key=lambda x: (x.startswith("abL="), len(ablate.parse(x)) if x.startswith("abL=") else 0, x)):
            dt, da, bb, cc, nn = [], [], 0, 0, 0
            for (m, l, s, vv) in S:
                if m == model and vv == v and (m, l, s, "L1") in S:
                    b, c, _, n_ = paired(S, (m, l, s, v), (m, l, s, "L1"))
                    bb, cc, nn = bb + b, cc + c, nn + n_
                    dt.append(S[(m, l, s, v)]["ctx"] / S[(m, l, s, "L1")]["ctx"] - 1)
                    da.append(S[(m, l, s, v)]["acc"] - S[(m, l, s, "L1")]["acc"])
            if dt:
                lo, hi = diff_ci(bb, cc, nn)
                dsc = f"累積{len(ablate.parse(v))}段: +{ablate.OP_JA[ablate.LADDER[len(ablate.parse(v)) - 1]]}" if v.startswith("abL=") else ablate.describe(v)
                P(f"| {short(v)} | {dsc} | {len(dt)} | {nn} | {st.mean(dt):+.1%} | {(bb - cc) / nn * 100:+.1f}pt | "
                  f"[{lo:+.1f}, {hi:+.1f}] | {bb}/{cc} | {sign_test(bb, cc):.2f} |")
        P("")
    # category impact
    P("\n## 意味カテゴリ別の影響(単独アブレーションの精度 − L1, n≥6 のカテゴリのみ, プール)\n")
    for model in sorted({m for m, *_ in S}):
        by = defaultdict(lambda: defaultdict(list))
        for (m, l, s, v), d in S.items():
            if m == model:
                for r in d["rows"]:
                    by[v][r["cat"]].append(r["ok"])
        if "L1" not in by:
            continue
        cats = sorted(c for c, x in by["L1"].items() if len(x) >= 6)
        P(f"**{model}**\n")
        P("| 操作 | " + " | ".join(cats) + " |")
        P("|---|" + "---|" * len(cats))
        P("| L1(絶対値) | " + " | ".join(f"{sum(by['L1'][c]) / len(by['L1'][c]):.0%}" for c in cats) + " |")
        for v in sorted(v for v in by if v.startswith("ab=")):
            P(f"| {short(v)} | " + " | ".join(
                f"{(sum(by[v][c]) / len(by[v][c]) - sum(by['L1'][c]) / len(by['L1'][c])) * 100:+.0f}" if by[v].get(c) else "-" for c in cats) + " |")
        P("")
    # conversion cost
    cc_path = RES / "conversion_cost.json"
    if cc_path.exists():
        P(conversion_section(json.loads(cc_path.read_text(encoding="utf-8")), S))
    if errors:
        P("\n## 失敗した変種\n")
        for e in errors:
            P(f"- {e['model']} len={e['length']} {vname(e['fmt'], e['legend'])}: {e['error'][:160]}")
    text = "\n".join(L)
    (RES / "ablation_report.md").write_text(text + "\n", encoding="utf-8")
    with (RES / "ablation_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(text)


def conversion_section(cc: dict, S) -> str:
    out = ["\n## 変換コストと損益分岐 (TotalCost = Conversion + Prefill + Decode)\n"]
    out.append("**ルールベース変換**(構造化 facts からの描画。生の日本語文からの変換ではない)\n")
    out.append("| 文脈長 | L1 | L5 | adaptive | 全アブレーション中の最大 |")
    out.append("|---|---|---|---|---|")
    for length, d in sorted(cc.get("rule", {}).items(), key=lambda x: int(x[0])):
        mx = max(d.items(), key=lambda kv: kv[1])
        out.append(f"| {length} | {d['L1']:.2f}ms | {d['L5']:.2f}ms | {d['adaptive']:.2f}ms | {mx[1]:.2f}ms ({short(mx[0])}) |")
    out.append("\n(いずれも 1/1000 秒台。L0→圧縮の prefill 短縮(秒〜分)に比べ無視できるが、生の日本語文から facts を抽出する工程は含まない。)")
    out.append("\n**LLM 変換**(L0 → L1 風の要約を LLM に生成させる)と損益分岐\n")
    out.append("T_saved = コールドTTFT(L0) − コールドTTFT(L1)(実測), T_conv = LLM 変換の総時間, "
               "R_be = T_conv / T_saved(変換結果を何回再利用すれば元が取れるか)。"
               "token 版は 変換トークン(入力 + w×出力) / (L0文脈tok − L1文脈tok)。\n")
    out.append("| モデル | 文脈長 | 変換 prompt/出力 tok | T_conv(s) | 変換後の文脈tok | 変換結果での QA 精度 | T_saved(s) | R_be(時間) | R_be(token, w=1) | R_be(token, w=5) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for model, d in cc.get("llm", {}).items():
        for length, r in sorted(d.items(), key=lambda x: int(x[0])):
            l0 = next((S[k] for k in S if k[0] == model and k[1] == int(length) and k[3] == "L0"), None)
            l1 = next((S[k] for k in S if k[0] == model and k[1] == int(length) and k[3] == "L1"), None)
            if not (l0 and l1):
                out.append(f"| {model} | {length} | {r['prompt_tokens']}/{r['completion_tokens']} | {r['total']:.0f} | {r['ctx_tokens']} | {r['acc']:.1%} | - | - | - | - |")
                continue
            ts = l0["ttft"] - l1["ttft"]
            sv = l0["ctx"] - l1["ctx"]
            f = lambda x: f"{x:.1f}" if x > 0 else "回収不能"
            out.append(f"| {model} | {length} | {r['prompt_tokens']}/{r['completion_tokens']} | {r['total']:.0f} | {r['ctx_tokens']} | {r['acc']:.1%} | "
                       f"{ts:.1f} | {f(r['total'] / ts) if ts > 0 else '回収不能'} | "
                       f"{f((r['prompt_tokens'] + r['completion_tokens']) / sv) if sv > 0 else '回収不能'} | "
                       f"{f((r['prompt_tokens'] + 5 * r['completion_tokens']) / sv) if sv > 0 else '回収不能'} |")
    return "\n".join(out)
