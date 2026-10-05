"""Issue #33 report, SeCF part: SCF (existing short run) vs SeCF conditions on the same histories, judged blind by qwen3-8b."""
from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import jpchat, jpdata, terms
from .jpchatreport import JUDGE_KEYS, OUT, boot_ci, jl, load, perm_p
from .jpreport import svg_lines

ADD = ["struct_per_100", "bullet_rate", "field_per_100"]


def main():
    g1, j1 = load("short")
    g2, j2 = load("secf")
    gen, judge = {**g1, **g2}, {**j1, **j2}
    if not g2:
        print("no secf generations")
        return
    for r in gen.values():
        r["style"] = jpchat.style_metrics(r["answer"])
    sids = sorted({s for _, s in g2})
    hist = {h["sid"]: h for h in jpchat.load_histories("short")}
    base = "A_base"
    conds = list(jpchat.cond_table("short")) + list(jpchat.cond_table("secf"))
    L: List[str] = []
    P = L.append
    col = lambda c, f: [f(gen[(c, s)], judge.get((c, s))) for s in sids if (c, s) in gen]
    mean = lambda c, k: st.mean(col(c, lambda g, j: g[k]))
    pt0 = mean(base, "prompt_tokens")
    P("# Issue #33 SeCF-L1 vs SCF-L1: 圧縮 Chat 履歴が日本語回答に与える影響\n")
    P(terms.note() + "(SCF = sudachi-m、SeCF = gemma-4-12b が各発話を意味構造 `主体{...}` に変換。変換は温度 0・キャッシュ済み)\n")
    P(f"シナリオ {len(sids)} 件、8 往復。回答 gemma-4-12b、匿名採点 qwen3-8b。条件の対応: A=A_base、B=B_c*(SCF hybrid/full)、C=C_c*(SeCF hybrid/full、c=8 が E)、D=D_c*(SCF 分離)、G=G_c*(SeCF 分離)、F/G の対照は T0_plain。\n")
    P("## 1. 圧縮と採点\n")
    P("| 条件 | n | tok(圧縮率) | 自然さ | 意味保持 | 会話らしさ | 汚染 | 自然さ差 p | 汚染差 p | 敬体率 | 体言止め | 構造記号/100字 | 箇条書き率 | field名/100字 | 目標言及率 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    needle = lambda ev: [x for x in (jpdata.event_of(ev).np.replace("新しい", "").rstrip("をにがへ"),) if x]
    rows = {}
    for c in conds:
        if not col(c, lambda g, j: 1):
            continue
        js = {s: judge[(c, s)] for s in sids if (c, s) in judge}
        sm = lambda k: st.mean(col(c, lambda g, j: g["style"][k]))
        m = st.mean(any(n in gen[(c, s)]["answer"] for n in needle(hist[s]["event"])) for s in sids if (c, s) in gen)
        dn = [js[s]["自然さ"] - judge[(base, s)]["自然さ"] for s in js if (base, s) in judge]
        dc = [js[s]["L1文体汚染"] - judge[(base, s)]["L1文体汚染"] for s in js if (base, s) in judge]
        sc = [st.mean(j[k] for j in js.values()) for k in JUDGE_KEYS] if js else [float("nan")] * 4
        rows[c] = (sc, sm("polite_rate"), mean(c, "prompt_tokens") / pt0)
        P(f"| {c} | {len(js)} | {mean(c, 'prompt_tokens'):.0f} ({mean(c, 'prompt_tokens') / pt0:.2f}) | " + " | ".join(f"{x:.2f}" for x in sc)
          + f" | {perm_p(dn) if c != base else float('nan'):.3f} | {perm_p(dc) if c != base else float('nan'):.3f} | {sm('polite_rate'):.2f} | {sm('taigen_rate'):.2f} | {sm('struct_per_100'):.2f} | {sm('bullet_rate'):.2f} | {sm('field_per_100'):.2f} | {m:.2f} |")
    P("\n## 2. 比率・直近 NF 窓(SCF vs SeCF)\n")
    P("| c | 比率 | 直近NF | SCF msg(B) 自然/汚染/敬体 | SeCF msg(C) | SCF 分離(D) | SeCF 分離(G) |")
    P("|---|---|---|---|---|---|---|")
    series: Dict[str, List[tuple]] = defaultdict(list)
    for k in (2, 4, 6, 7, 8):
        cells = []
        for pre, nm in (("B_c", "SCF msg"), ("C_c", "SeCF msg"), ("D_c", "SCF 分離"), ("G_c", "SeCF 分離")):
            r = rows.get(f"{pre}{k}")
            cells.append(f"{r[0][0]:.2f}/{r[0][3]:.2f}/{r[1]:.2f}" if r else "-")
            if r:
                series[f"{nm} 自然さ"].append((k / 8 * 100, r[0][0]))
                series[f"{nm} 汚染"].append((k / 8 * 100, r[0][3]))
        P(f"| {k} | {k / 8:.0%} | {8 - k} | " + " | ".join(cells) + " |")
    for nm, c0 in (("SCF msg", base), ("SeCF msg", base), ("SCF 分離", "T0_plain"), ("SeCF 分離", "T0_plain")):
        if c0 in rows:
            series[f"{nm} 自然さ"].append((0, rows[c0][0][0]))
            series[f"{nm} 汚染"].append((0, rows[c0][0][3]))
    if series:
        svg_lines(dict(series), "圧縮比率 vs 自然さ・汚染: SCF と SeCF", "圧縮した古い履歴の割合(%)", "スコア(1-5)", OUT / "chart_secf_ratio.svg")
    P("\n## 3. 配置と指示(c=6)\n")
    P("| 条件 | 自然さ | 意味保持 | 会話らしさ | 汚染 | 敬体率 | 圧縮率 |")
    P("|---|---|---|---|---|---|---|")
    for c in ("D_c6", "D0_noinstr_c6", "P2_recent_then_block_c6", "P3_block_persona_c6", "G_c6", "G0_noinstr_c6", "GP2_recent_then_block_c6", "GP3_block_persona_c6", "T0_plain"):
        if c in rows:
            r = rows[c]
            P(f"| {c} | " + " | ".join(f"{x:.2f}" for x in r[0]) + f" | {r[1]:.2f} | {r[2]:.2f} |")
    P("\n## 4. 状態別の意味保持\n")
    sts = jpdata.STATUSES
    P("| 条件 | " + " | ".join(sts) + " |")
    P("|---|" + "---|" * len(sts))
    for c in (base, "B_c8", "C_c8", "D_c8", "G_c8"):
        v = [[judge[(c, s)]["意味保持"] for s in sids if (c, s) in judge and hist[s]["status"] == x] for x in sts]
        P(f"| {c} | " + " | ".join(f"{st.mean(a):.2f}" if a else "-" for a in v) + " |")
    P("\n## 5. 回答の例(UNDECIDED)\n")
    ex = next((s for s in sids if hist[s]["status"] == "UNDECIDED"), None)
    if ex is not None:
        P(f"正しい状況: {hist[ex]['gold_note']}\n")
        for c in (base, "B_c8", "C_c8", "D_c8", "G_c8"):
            if (c, ex) in gen:
                P(f"- **{c}**: {gen[(c, ex)]['answer'][:300].replace(chr(10), ' ')}")
    P("\n## 6. SeCF 変換の例と変換コスト\n")
    cache = jl(OUT / "secf_cache.jsonl")
    if cache:
        P(f"変換 {len(cache)} 発話、1 発話平均 {st.mean(r['ms'] for r in cache):.0f} ms(LLM 推論。SCF の sudachi-m は約 1 ms 未満)。\n")
        for r in cache[:4]:
            P(f"- {r['text']} → `{r['secf']}`")
    (OUT / "secf_report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "secf_report.md")


if __name__ == "__main__":
    main()
