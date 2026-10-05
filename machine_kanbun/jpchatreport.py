"""Issue #33 report: compression, judged quality, automatic style metrics, sliding window / L1 ratio, placement."""
from __future__ import annotations

import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import re

from . import jpchat, jpdata, terms
from .jpbench import RES
from .jpreport import svg_lines

OUT = RES.parent / "jp3"
JUDGE_KEYS = ["自然さ", "意味保持", "会話らしさ", "L1文体汚染"]
AUTO = ["particle_rate", "avg_sentence_chars", "noun_rate", "verb_rate", "sentence_final_rate", "taigen_rate", "polite_rate", "telegraphic_rate", "l1_label_per_100", "slash_per_100", "chars"]
GEN_MODEL = "google_gemma-4-12b"
JUDGE_MODEL = "qwen_qwen3-8b"


def jl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def perm_p(diffs: List[float], n: int = 20000, seed: int = 0) -> float:
    """Paired sign-flip permutation test (two-sided) on per-scenario differences."""
    d = np.array(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(axis=1)) >= obs - 1e-12).mean()))


def boot_ci(diffs: List[float], n: int = 4000, seed: int = 0):
    d = np.array(diffs, dtype=float)
    if len(d) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def load(variant: str = "short"):
    gen = {(r["cond"], r["sid"]): r for r in jl(OUT / f"{jpchat.variant_files(variant)[1]}{GEN_MODEL}.jsonl") if "error" not in r}
    judge = {(r["cond"], r["sid"]): r["scores"] for r in jl(OUT / f"{jpchat.variant_files(variant)[2]}{JUDGE_MODEL}_on_{GEN_MODEL}.jsonl") if r.get("scores")}
    return gen, judge


def main(variant: str = "short"):
    gen, judge = load(variant)
    if not gen:
        print("no generations")
        return
    for k, r in gen.items():
        if "style" not in r:
            r["style"] = jpchat.style_metrics(r["answer"])
    conds = list(jpchat.cond_table(variant))
    n_ex = len(next(iter(hist_probe))["exchanges"]) if (hist_probe := jpchat.load_histories(variant)) else 8
    long = variant == "long"
    KS = (16, 24) if long else (2, 4, 6, 7, 8)
    MID = 16 if long else 6
    SUF = "long_" if long else ""
    sids = sorted({s for _, s in gen})
    hist = {h["sid"]: h for h in jpchat.load_histories(variant)}
    L: List[str] = []
    P = L.append
    P("# Issue #33 SCF 圧縮チャット履歴が日本語回答の自然さ・文体に与える影響" + (" — 長い履歴(24 往復)" if variant == "long" else "") + "\n")
    P(terms.note() + "(本報告の「L1 化」は SCF 化。履歴の変換は sudachi-m = SCF-L1)\n")
    P(f"シナリオ {len(sids)} 件(状態 7 種 × 各 {len(sids) // 7})、条件 {len(conds)}、生成: gemma-4-12b(温度 0)、採点: qwen3-8b(条件を伏せた匿名採点。生成モデルとは別モデル)。"
      "差は同一シナリオ間のペア比較(符号反転の置換検定、ブートストラップ 95%CI)。\n")
    base = "A_base"

    def col(c, f):
        return [f(gen[(c, s)], judge.get((c, s))) for s in sids if (c, s) in gen]

    # ---- 1 compression / performance
    P("## 1. 圧縮と性能\n")
    P("| 条件 | プロンプトtok | 圧縮率(vs A) | L1変換(ms) | TTFT(s) | 総時間(s) | tok/s | 回答文字数 |")
    P("|---|---|---|---|---|---|---|---|")
    mean = lambda c, k: st.mean(col(c, lambda g, j: g[k]))
    pt0 = mean(base, "prompt_tokens") if (base, sids[0]) in gen else None
    rows_csv = []
    for c in conds:
        if not col(c, lambda g, j: 1):
            continue
        pt = mean(c, "prompt_tokens")
        P(f"| {c} | {pt:.0f} | {pt / pt0:.2f} | {mean(c, 'conv_ms'):.2f} | {mean(c, 'ttft'):.1f} | {mean(c, 'total'):.1f} | {mean(c, 'tok_s'):.0f} | {st.mean(col(c, lambda g, j: len(g['answer']))):.0f} |")
    # ---- 2 judge
    P("\n## 2. 匿名採点(1〜5。L1文体汚染は低いほど良い)\n")
    P("| 条件 | n | 自然さ | 意味保持 | 会話らしさ | L1文体汚染 | 自然さ差(vs A)[95%CI], p | 汚染差(vs A)[95%CI], p | 意味保持差(vs A), p |")
    P("|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        js = {s: judge[(c, s)] for s in sids if (c, s) in judge}
        if not js:
            continue
        cells = [f"{st.mean(j[k] for j in js.values()):.2f}" for k in JUDGE_KEYS]
        extra = ["", "", ""]
        if c != base:
            for i, k in enumerate(("自然さ", "L1文体汚染", "意味保持")):
                d = [js[s][k] - judge[(base, s)][k] for s in js if (base, s) in judge]
                lo, hi = boot_ci(d)
                extra[i] = f"{np.mean(d):+.2f} [{lo:+.2f},{hi:+.2f}], p={perm_p(d):.3f}" if i < 2 else f"{np.mean(d):+.2f}, p={perm_p(d):.3f}"
        P(f"| {c} | {len(js)} | " + " | ".join(cells) + " | " + " | ".join(extra) + " |")
    # ---- 3 auto metrics
    P("\n## 3. 自動指標(回答テキストの形態素統計)\n")
    P("| 条件 | 助詞率 | 平均文長(字) | 名詞率 | 動詞率 | 文末表現率 | 体言止め率 | 敬体率 | 電報文率 | L1記号/100字 | 「/」「→」/100字 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        if not col(c, lambda g, j: 1):
            continue
        m = lambda k: st.mean(col(c, lambda g, j: g["style"][k]))
        P(f"| {c} | {m('particle_rate'):.3f} | {m('avg_sentence_chars'):.1f} | {m('noun_rate'):.3f} | {m('verb_rate'):.3f} | {m('sentence_final_rate'):.2f} | {m('taigen_rate'):.2f} | {m('polite_rate'):.2f} | {m('telegraphic_rate'):.3f} | {m('l1_label_per_100'):.2f} | {m('slash_per_100'):.2f} |")
    # ---- 4 sliding window / ratio
    P("\n## 4. SCF 比率(= SCF にした古い往復の割合)・直近 NF 窓\n")
    P(f"{n_ex} 往復のうち古い c 往復を SCF にする(SCF 比率 = c/{n_ex}、直近 NF = {n_ex}-c 往復)。c=0 は baseline、c={n_ex} は全履歴 SCF。\n")
    P("| c(SCF往復数) | SCF比率 | 直近NF | family | 圧縮率 | 自然さ | 意味保持 | 会話らしさ | 汚染 | 敬体率 | 助詞率 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    series: Dict[str, List[tuple]] = defaultdict(list)
    for fam, prefix in (("role(メッセージ列)", "B_c"), ("tag(分離+指示)", "D_c")):
        for k in KS:
            c = f"{prefix}{k}"
            js = [judge[(c, s)] for s in sids if (c, s) in judge]
            if not js or not col(c, lambda g, j: 1):
                continue
            nat, sem, conv, pol = (st.mean(j[x] for j in js) for x in JUDGE_KEYS)
            pr = st.mean(col(c, lambda g, j: g["style"]["polite_rate"]))
            ptl = st.mean(col(c, lambda g, j: g["style"]["particle_rate"]))
            P(f"| {k} | {k / n_ex:.0%} | {n_ex - k} | {fam} | {mean(c, 'prompt_tokens') / pt0:.2f} | {nat:.2f} | {sem:.2f} | {conv:.2f} | {pol:.2f} | {pr:.2f} | {ptl:.3f} |")
            series[f"{fam} 自然さ"].append((k / n_ex * 100, nat))
            series[f"{fam} 汚染"].append((k / n_ex * 100, pol))
    for fam, c0 in (("role(メッセージ列)", base), ("tag(分離+指示)", "T0_plain")):
        js = [judge[(c0, s)] for s in sids if (c0, s) in judge]
        if js:
            series[f"{fam} 自然さ"].append((0, st.mean(j["自然さ"] for j in js)))
            series[f"{fam} 汚染"].append((0, st.mean(j["L1文体汚染"] for j in js)))
    if series:
        svg_lines(dict(series), "SCF 比率 vs 自然さ・SCF文体汚染(qwen 採点)", "SCF にした古い履歴の割合(%)", "スコア(1-5)", OUT / f"chart_{SUF}l1_ratio.svg")
    # ---- 5 placement / instruction
    P(f"\n## 5. 配置と分離指示(古い履歴の {MID}/{n_ex} 往復を SCF 化)\n")
    P("| 条件 | 内容 | 自然さ | 意味保持 | 会話らしさ | 汚染 | 敬体率 | 圧縮率 |")
    P("|---|---|---|---|---|---|---|---|")
    desc = {"A_base": "全 NF(メッセージ列)", f"B_c{MID}": f"メッセージ列、古い {MID} 往復 SCF、指示なし", "T0_plain": "タグ構造+指示、古い履歴は NF(構造の対照)", f"D_c{MID}": "タグ構造+指示、<compressed_context> を先頭、直近 NF を後",
            "D0_noinstr_c6": "D と同じで指示なし", f"P2_recent_then_block_c{MID}": "直近 NF → SCF ブロック → 質問(SCF が質問に近い)", "P3_block_persona_c6": "L1 ブロックの直後に persona と文体注意"}
    for c, d in desc.items():
        js = [judge[(c, s)] for s in sids if (c, s) in judge]
        if not js or not col(c, lambda g, j: 1):
            continue
        P(f"| {c} | {d} | " + " | ".join(f"{st.mean(j[k] for j in js):.2f}" for k in JUDGE_KEYS) + f" | {st.mean(col(c, lambda g, j: g['style']['polite_rate'])):.2f} | {mean(c, 'prompt_tokens') / pt0:.2f} |")
    # ---- 6 by status
    P("\n## 6. 状態別の意味保持(採点の平均)\n")
    sts = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]
    P("| 条件 | " + " | ".join(sts) + " |")
    P("|---|" + "---|" * len(sts))
    for c in (base, f"B_c{MID}", f"B_c{KS[-1]}", "T0_plain", f"D_c{MID}", f"D_c{KS[-1]}"):
        cells = []
        for s_ in sts:
            v = [judge[(c, s)]["意味保持"] for s in sids if (c, s) in judge and hist[s]["status"] == s_]
            cells.append(f"{st.mean(v):.2f}" if v else "-")
        P(f"| {c} | " + " | ".join(cells) + " |")
    # ---- 6b target-fact mention (heuristic, complements the judge, which is lenient: a generic answer is never "wrong")
    def needles(ev):
        e = jpdata.event_of(ev)
        obj = re.sub(r"[をにがへ]$", "", e.np.replace("新しい", ""))
        verb = e.noun or (re.findall(r"[一-鿿]+", e.verb) or [e.verb[:-1]])[0]
        return [x for x in (obj, verb) if x]

    P("\n### 6b. 回答が目標の事実(出来事)に言及した割合(ヒューリスティック: 出来事の語が回答に含まれるか)\n")
    P("意味保持の採点は寛容(一般論の回答は「誤り」にならない)ので、古い履歴の事実が実際に使われたかを補助的に見る。圧縮された履歴からも事実が取り出せていれば baseline 並みかそれ以上になる。\n")
    P("| 条件 | 言及率 | baseline との差 |")
    P("|---|---|---|")
    mention = {}
    for c in conds:
        v = [any(n in gen[(c, s_)]["answer"] for n in needles(hist[s_]["event"])) for s_ in sids if (c, s_) in gen]
        if v:
            mention[c] = float(np.mean(v))
    for c, m in mention.items():
        P(f"| {c} | {m:.2f} | {m - mention.get(base, m):+.2f} |")
    # ---- 7 success criteria
    P("\n## 7. 成功条件の判定(baseline A_base に対して)\n")
    P("基準: 意味保持の低下 0.3 点未満、自然さの低下が有意でない(p≥0.05)、汚染の増加が 0.3 点未満、敬体率の低下 0.1 未満、圧縮率 < 0.95。\n")
    P("| 条件 | 意味保持 差 | 自然さ 差(p) | 汚染 差 | 敬体率 差 | 圧縮率 | 判定 |")
    P("|---|---|---|---|---|---|---|")
    for c in conds:
        if c == base or not [1 for s in sids if (c, s) in judge]:
            continue
        sem = np.mean([judge[(c, s)]["意味保持"] - judge[(base, s)]["意味保持"] for s in sids if (c, s) in judge and (base, s) in judge])
        dn = [judge[(c, s)]["自然さ"] - judge[(base, s)]["自然さ"] for s in sids if (c, s) in judge and (base, s) in judge]
        dc = np.mean([judge[(c, s)]["L1文体汚染"] - judge[(base, s)]["L1文体汚染"] for s in sids if (c, s) in judge and (base, s) in judge])
        dp = st.mean(col(c, lambda g, j: g["style"]["polite_rate"])) - st.mean(col(base, lambda g, j: g["style"]["polite_rate"]))
        ratio = mean(c, "prompt_tokens") / pt0
        ok = (sem > -0.3) and (np.mean(dn) > -0.3 or perm_p(dn) >= 0.05) and dc < 0.3 and dp > -0.1 and ratio < 0.95
        P(f"| {c} | {sem:+.2f} | {np.mean(dn):+.2f} (p={perm_p(dn):.3f}) | {dc:+.2f} | {dp:+.2f} | {ratio:.2f} | {'満たす' if ok else '満たさない'} |")
        rows_csv.append(dict(cond=c, sem_diff=round(float(sem), 3), nat_diff=round(float(np.mean(dn)), 3), nat_p=round(perm_p(dn), 4), contam_diff=round(float(dc), 3), polite_diff=round(dp, 3),
                             token_ratio=round(ratio, 3), meets_criteria=bool(ok)))
    # ---- examples + blind sheet
    P("\n## 8. 回答の例(同じシナリオ、状態 UNDECIDED)\n")
    ex = next((s for s in sids if hist[s]["status"] == "UNDECIDED"), None)
    if ex is not None:
        P(f"質問: {hist[ex]['question']} / 正しい状況: {hist[ex]['gold_note']}\n")
        for c in (base, f"B_c{MID}", f"B_c{KS[-1]}", f"D_c{MID}", f"D_c{KS[-1]}", f"P2_recent_then_block_c{MID}"):
            if (c, ex) in gen:
                P(f"- **{c}**: {gen[(c, ex)]['answer'][:300].replace(chr(10), ' ')}")
    rng = random.Random(7)
    keys = list(gen)
    rng.shuffle(keys)
    with (OUT / f"{SUF}blind_sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question", "gold_note", "answer", "naturalness_1to5", "meaning_1to5", "conversational_1to5", "l1_contamination_1to5"])
        for i, (c, s) in enumerate(keys[:80]):
            w.writerow([i, gen[(c, s)]["question"], hist[s]["gold_note"], gen[(c, s)]["answer"], "", "", "", ""])
    with (OUT / f"{SUF}blind_key.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "cond", "sid"])
        for i, (c, s) in enumerate(keys[:80]):
            w.writerow([i, c, s])
    if rows_csv:
        with (OUT / f"{SUF}results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    (OUT / f"{SUF}report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / f"{SUF}report.md", len(L), "lines")


if __name__ == "__main__":
    import sys
    main("long" if len(sys.argv) > 1 and sys.argv[1] == "long" else "short")
