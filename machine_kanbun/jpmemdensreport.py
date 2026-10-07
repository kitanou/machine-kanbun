"""Issue #72 report: two-stage compression, extraction coverage, QA per memory condition, same-budget retention. Writes results/jp7/dens_report.md."""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

import numpy as np

from .jpchatreport import boot_ci, perm_p
from .jpmemdens import DAYS, keyword_ok, stale_hit, OUT, block, conv_text, jl, load_conv, o200k, questions, rep_lines, users

NAMES = {"A_raw": "A 生の会話", "c_nf": "B 抽出 NF(統合)", "c_secf1": "C 抽出 SeCF-L1(統合)", "c_scf": "(対照)抽出 SCF-L1(統合)", "i_nf": "B' 抽出 NF(日ごと)", "i_secf1": "C' 抽出 SeCF-L1(日ごと)"}


def cover(u, text, qtype="final"):
    fs = [f for f in questions(u) if f["qtype"] == qtype]
    return sum(1 for f in fs if any(g in text for g in f["gold"])) / max(1, len(fs))


def pair(rows, a, b, key="cond", field="ok"):
    A = {(r["uid"], r["fid"]): float(r[field]) for r in rows if r[key] == a}
    B = {(r["uid"], r["fid"]): float(r[field]) for r in rows if r[key] == b}
    ks = [k for k in A if k in B]
    d = [A[k] - B[k] for k in ks]
    lo, hi = boot_ci(d)
    return f"{np.mean(d) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}]", perm_p(d), len(ks)


def rescore(rows, U):
    """Re-score stored answers with the current gold keyword lists (the lists were widened after the first pilot answers: 未確定, 中止, ...)."""
    F = {(u["uid"], f["id"]): f for u in U for f in u["facts"]}
    for r in rows:
        f = F[(r["uid"], r["fid"])]
        r["ok"], r["stale"] = keyword_ok(r["answer"], f), stale_hit(r["answer"], f)
    return rows


def main() -> None:
    U = users()
    C = load_conv()
    L = [f"# Issue #72 パーソナル AI 長期記憶と Memory Density(ユーザー {len(U)} 人、{DAYS} 日分の会話、gemma-4-12b)\n"]
    # 1. two-stage compression
    L += ["## 1. 圧縮率を 2 段階に分ける(o200k トークン、ユーザー平均)\n", "| 記憶の作り方 | 会話 | NF 記憶 | SeCF-L1 | SCF-L1 | 蒸留(会話→NF 記憶) | 表現圧縮(NF→SeCF-L1) | 会話→SeCF-L1 |", "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for kind in ("consolidated", "incremental"):
        rows = []
        for u in U:
            R = rep_lines(u["uid"], kind)
            conv = o200k(conv_text(C[u["uid"]]))
            t = {r: o200k(block(R[r], kind == "incremental")) for r in R}
            rows.append((conv, t["nf"], t["secf1"], t["scf"]))
        m = [st.mean(x[i] for x in rows) for i in range(4)]
        d, rp = [x[1] / x[0] for x in rows], [x[2] / x[1] for x in rows]
        agg[kind] = (d, rp)
        L.append(f"| {kind} | {m[0]:.0f} | {m[1]:.0f} | {m[2]:.0f} | {m[3]:.0f} | {np.mean(d):.3f} | {np.mean(rp):.3f}(min {min(rp):.2f}, max {max(rp):.2f}) | {np.mean([x[2] / x[0] for x in rows]):.3f} |")
    L.append("\n会話→SeCF-L1 の比は「蒸留 × 表現圧縮」で、SeCF-L1 が会話を圧縮したとは解釈しない。\n")
    # 2. coverage
    L += ["## 2. 記憶に残った gold 事実(キーワードが記憶本文にある割合。QA ではなく決定的な照合)\n", "| 記憶 | 現在の状態(final) | 更新前の状態(history) | 不確実(uncertain) |", "|---|---|---|---|"]
    for kind in ("consolidated", "incremental"):
        for rep in ("nf", "secf1", "scf"):
            if kind == "incremental" and rep == "scf":
                continue
            v = []
            for qt in ("final", "history", "uncertain"):
                v.append(np.mean([cover(u, block(rep_lines(u["uid"], kind)[rep], kind == "incremental"), qt) for u in U]))
            L.append(f"| {kind}/{rep} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |")
    L.append("(history の gold は更新前の値。統合記憶は最新だけ残すので低くなりうる。uncertain は「迷・未定」などの語の有無。)\n")
    # 3. QA
    qa = rescore(jl(OUT / "qa.jsonl"), U)
    if qa:
        g = defaultdict(list)
        for r in qa:
            g[r["cond"]].append(r)
        L += ["## 3. 全文を渡した QA(質問ごとに短い語句で回答、キーワード照合)\n", "| 条件 | n | 正答率 | 古い値を答えた率 | プロンプト tok(平均) | 1k tok あたりの正答 | TTFT 平均(秒) |", "|---|---|---|---|---|---|---|"]
        for c in NAMES:
            v = g.get(c)
            if not v:
                continue
            pt = st.mean(x["prompt_tokens"] for x in v)
            acc = np.mean([x["ok"] for x in v])
            L.append(f"| {NAMES[c]} | {len(v)} | {acc:.3f} | {np.mean([x['stale'] for x in v]):.3f} | {pt:.0f} | {acc / (pt / 1000):.2f} | {st.mean(x['ttft'] for x in v):.2f} |")
        L += ["\n### 質問タイプ別の正答率\n", "| 条件 | final | history | uncertain |", "|---|---|---|---|"]
        for c in NAMES:
            if c in g:
                L.append(f"| {NAMES[c]} | " + " | ".join(f"{np.mean([x['ok'] for x in g[c] if x['qtype'] == qt]):.3f}" for qt in ('final', 'history', 'uncertain')) + " |")
        L += ["\n### ペア比較(質問単位、符号反転の置換検定)\n", "| 比較 | 差[95%CI] | p | n |", "|---|---|---|---|"]
        for a, b in [("c_nf", "A_raw"), ("c_secf1", "c_nf"), ("c_secf1", "A_raw"), ("c_scf", "c_nf"), ("i_secf1", "i_nf"), ("i_nf", "c_nf"), ("c_nf", "i_nf")]:
            if a in g and b in g:
                s, p, n = pair(qa, a, b)
                L.append(f"| {a} − {b} | {s} | {p:.3f} | {n} |")
    # 4. budget
    bd = rescore(jl(OUT / "budget.jsonl"), U)
    if bd:
        L += ["\n## 4. 同一 token budget(日ごと抽出の記憶、古い日から切り捨て。budget = 全 NF 記憶の o200k 数 × 倍率)\n", "| budget | 表現 | 残った行数 | 正答率 | 古い値を答えた率 | プロンプト tok |", "|---|---|---|---|---|---|"]
        gb = defaultdict(list)
        for r in bd:
            gb[(r["frac"], r["rep"])].append(r)
        for (fr, rep), v in sorted(gb.items()):
            per = {}
            for r in v:
                per[r["uid"]] = r["lines"] / r["lines_total"]
            L.append(f"| {fr} | {rep} | {np.mean(list(per.values())):.2f}(行の割合) | {np.mean([x['ok'] for x in v]):.3f} | {np.mean([x['stale'] for x in v]):.3f} | {st.mean(x['prompt_tokens'] for x in v):.0f} |")
        L += ["\n### budget 別のペア比較\n", "| budget | 比較 | 差[95%CI] | p | n |", "|---|---|---|---|---|"]
        for fr in sorted({r["frac"] for r in bd}):
            rows = [r for r in bd if r["frac"] == fr]
            for a, b in [("secf1", "nf"), ("scf", "nf")]:
                s, p, n = pair(rows, a, b, key="rep")
                L.append(f"| {fr} | {a} − {b} | {s} | {p:.3f} | {n} |")
    ca = jl(OUT / "cache.jsonl")
    if ca:
        L += ["\n## 5. Prefix/KV キャッシュ(日ごと抽出の記憶、K=4 スロット、ユーザー数 U を増やして 3 ラウンド巡回。ヒット = TTFT が同じ表現・同じ U のラウンド 0 の中央値の半分未満)\n",
              "| U | 表現 | 総プロンプト tok(U 人分) | ラウンド 0 の TTFT 中央値(秒) | ラウンド 1-2 のヒット率 | ラウンド 1-2 の TTFT 平均(秒) | エラー |", "|---|---|---|---|---|---|---|"]
        for U_ in sorted({r["users"] for r in ca}):
            for rep in ("nf", "secf1"):
                v = [r for r in ca if r["users"] == U_ and r["rep"] == rep]
                if not v:
                    continue
                r0 = [r["ttft"] for r in v if r["round"] == 0 and r["ttft"]]
                rr = [r for r in v if r["round"] > 0 and r["ttft"]]
                thr = 0.5 * st.median(r0) if r0 else 0
                tot = sum(r["prompt_tokens"] for r in v if r["round"] == 0 and r["prompt_tokens"])
                L.append(f"| {U_} | {rep} | {tot} | {st.median(r0) if r0 else 0:.2f} | {np.mean([r['ttft'] < thr for r in rr]) if rr else float('nan'):.2f}(n={len(rr)}) | {st.mean(r['ttft'] for r in rr) if rr else float('nan'):.2f} | {sum(1 for r in v if r['error'])} |")
    (OUT / "dens_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
