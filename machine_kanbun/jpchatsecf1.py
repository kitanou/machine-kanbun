"""Issue #33 redo report: NF vs SCF-L1 (#20, sudachi-m) vs SeCF-L1 (#19-style concise natural Japanese) as chat history.
Structured `主体{...}` conditions of the first #33 run are SeCF-L5-like and are NOT part of this comparison (kept only as a reference row set)."""
from __future__ import annotations

import statistics as st
from typing import Dict, List

import numpy as np

from . import jpchat
from .jpchatqareport import rows as qa_rows
from .jpchatreport import JUDGE_KEYS, OUT, boot_ci, jl, load, perm_p
from .jpreport import svg_lines


def block(variant_scf: str, variant_secf: str, ks, label: str, L: List[str]):
    g1, j1 = load(variant_scf)
    g2, j2 = load(variant_secf)
    gen, judge = {**g1, **g2}, {**j1, **j2}
    if not g2:
        return
    for r in gen.values():
        r.setdefault("style", jpchat.style_metrics(r["answer"]))
    sids = sorted({s for _, s in g2})
    base = "A_base"
    col = lambda c, f: [f(gen[(c, s)], judge.get((c, s))) for s in sids if (c, s) in gen]
    mean = lambda c, k: st.mean(col(c, lambda g, j: g[k]))
    pt0 = mean(base, "prompt_tokens")
    pre = [("B_c", "SCF-L1 メッセージ列"), ("C1_c", "SeCF-L1 メッセージ列"), ("D_c", "SCF-L1 分離+指示"), ("G1_c", "SeCF-L1 分離+指示")]
    L += [f"## {label}\n", "| 条件 | 内容 | n | tok(圧縮率) | 自然さ | 意味保持 | 会話らしさ | 汚染 | 自然さ差 p | 汚染差 p | 敬体率 | 体言止め | 平均文長 | 変換 ms/往復(SeCF-L1 はキャッシュ参照。実コストは下の節) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    series: Dict[str, List[tuple]] = {}
    names = [base] + [f"{p}{k}" for k in ks for p, _ in pre] + (["T0_plain", "G10_noinstr_c6"] if "short" in variant_scf else [])
    for c in names:
        if not col(c, lambda g, j: 1):
            continue
        js = {s: judge[(c, s)] for s in sids if (c, s) in judge}
        sm = lambda k: st.mean(col(c, lambda g, j: g["style"][k]))
        d = [js[s]["自然さ"] - judge[(base, s)]["自然さ"] for s in js if (base, s) in judge] if c != base else [0]
        dc = [js[s]["L1文体汚染"] - judge[(base, s)]["L1文体汚染"] for s in js if (base, s) in judge] if c != base else [0]
        sc = [st.mean(j[k] for j in js.values()) for k in JUDGE_KEYS] if js else [float("nan")] * 4
        P = next((nm for p, nm in pre if c.startswith(p)), c)
        L.append(f"| {c} | {P} | {len(js)} | {mean(c, 'prompt_tokens'):.0f} ({mean(c, 'prompt_tokens') / pt0:.2f}) | " + " | ".join(f"{x:.2f}" for x in sc)
                 + f" | {perm_p(d) if c != base else float('nan'):.3f} | {perm_p(dc) if c != base else float('nan'):.3f} | {sm('polite_rate'):.2f} | {sm('taigen_rate'):.2f} | {sm('avg_sentence_chars'):.1f} | {mean(c, 'conv_ms'):.1f} |")
    L.append("")


def qa_block(L: List[str]):
    L += ["## 想起精度(状態 7 択 QA、#49 と同じ方式。決定的採点)\n", "| 履歴 | 条件 | 正答率 | tok(圧縮率) | vs A 差[95%CI], p |", "|---|---|---|---|---|"]
    for v in ("short", "long"):
        R = qa_rows(v)
        from collections import defaultdict
        by, tok = defaultdict(lambda: defaultdict(list)), defaultdict(list)
        for r in R:
            by[r["cond"]][r["sid"]].append(r["pred"] == r["gold"])
            tok[r["cond"]].append(r["prompt_tokens"])
        if "E1_secf1" not in by:
            continue
        acc = {c: {s: float(np.mean(x)) for s, x in d.items()} for c, d in by.items()}
        shown = [c for c in ("A_base", "N_nf_tokmatch", "B_scf", "E1_secf1", "D_scf_tag", "G1_secf1_tag", "E_secf", "G_secf_tag") if c in acc]
        sids = sorted(set.intersection(*[set(acc[c]) for c in shown]))
        for c in ("A_base", "N_nf_tokmatch", "B_scf", "E1_secf1", "D_scf_tag", "G1_secf1_tag", "E_secf", "G_secf_tag"):
            if c not in acc:
                continue
            a = np.array([acc[c][s] for s in sids])
            d = list(a - np.array([acc["A_base"][s] for s in sids]))
            lo, hi = boot_ci(d)
            tag = " (構造化 SeCF-L5 相当・参考)" if c in ("E_secf", "G_secf_tag") else ""
            L.append(f"| {v} | {c}{tag} | {a.mean():.3f} | {st.mean(tok[c]):.0f} ({st.mean(tok[c]) / st.mean(tok['A_base']):.2f}) | " + ("-" if c == "A_base" else f"{np.mean(d) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}], p={perm_p(d):.3f}") + " |")
        L += ["", f"状態別({v}): " + " / ".join(f"{c}: " + ",".join(f"{s[:4]}={np.mean([x['pred'] == x['gold'] for x in R if x['cond'] == c and x['status'] == s]):.2f}"
                                                              for s in ("DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE")) for c in ("A_base", "B_scf", "E1_secf1") if c in by), ""]


def conv_cost(L: List[str]):
    c = jl(OUT / "secf1_cache.jsonl")
    if c:
        L += ["## 変換コスト\n", f"SeCF-L1(LLM 書き直し): {len(c)} 発話、1 発話平均 {st.mean(r['ms'] for r in c):.0f} ms。SCF-L1(sudachi-m): 1 発話 1 ms 未満(#20 で 0.05〜0.07 ms)。\n"]
        for r in c[:6]:
            L.append(f"- {r['text']} → {r['secf1']}")
        L.append("")


def main():
    L = ["# Issue #33 再実行: NF vs SCF-L1 vs SeCF-L1(#19 の簡潔な自然言語)を Chat 履歴として使う\n",
         "SeCF-L1 = gemma-4-12b による簡潔な自然言語への書き直し(温度 0、キャッシュ)。初回の構造化(`主体{…}`)条件は SeCF-L5 相当で、ここでは比較対象にしない。回答 gemma-4-12b、匿名採点 qwen3-8b。\n"]
    block("short", "secf1", (2, 4, 6, 7, 8), "短い履歴(8 往復、35 シナリオ): 比率・窓(c = 圧縮した古い往復数)", L)
    block("long", "secf1long", (16, 24), "長い履歴(24 往復、21 シナリオ)", L)
    qa_block(L)
    conv_cost(L)
    (OUT / "secf1_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
