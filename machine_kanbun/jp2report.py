"""Report for Issues #25 / #30 (context ablations), plus helpers shared by the other Issue #26-#31 reports."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List

OUT = Path(__file__).parent.parent / "results" / "jp2"


def rows_of(pattern: str) -> List[Dict]:
    out = []
    for f in sorted(OUT.glob(pattern)):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if "error" not in r:
                out.append(r)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def svg_scatter(points, title: str, xlabel: str, ylabel: str, path: Path, hull_label=True):
    """points: [(name, x, y, kind)]; kind picks the colour; the Pareto frontier (max y for min x) is drawn."""
    W, H, M = 860, 520, 66
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x0, x1, y0, y1 = min(xs) - 0.03, max(xs) + 0.03, max(0, min(ys) - 0.05), min(1.0, max(ys) + 0.05)
    sx = lambda v: M + (v - x0) / ((x1 - x0) or 1) * (W - 2 * M - 120)
    sy = lambda v: H - M - (v - y0) / ((y1 - y0) or 1) * (H - 2 * M)
    cols = {"L0": "#c0392b", "surgical": "#e67e22", "l1": "#2471a3", "ablation": "#1e8449", "control": "#7f8c8d"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">', f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
         f'<line x1="{M}" y1="{H - M}" x2="{W - M - 120}" y2="{H - M}" stroke="#444"/><line x1="{M}" y1="{M}" x2="{M}" y2="{H - M}" stroke="#444"/>',
         f'<text x="{(W - 120) / 2}" y="{H - 14}" text-anchor="middle">{xlabel}</text>', f'<text x="16" y="{H / 2}" transform="rotate(-90 16 {H / 2})" text-anchor="middle">{ylabel}</text>']
    for t in range(5):
        o.append(f'<text x="{M + (W - 2 * M - 120) * t / 4:.0f}" y="{H - M + 14}" text-anchor="middle">{x0 + (x1 - x0) * t / 4:.2f}</text>')
        o.append(f'<text x="{M - 6}" y="{H - M - (H - 2 * M) * t / 4 + 4:.0f}" text-anchor="end">{y0 + (y1 - y0) * t / 4:.2f}</text>')
    front, best = [], -1
    for n_, x, y, k in sorted(points, key=lambda p: (p[1], -p[2])):
        if y > best:
            front.append((x, y))
            best = y
    o.append('<polyline fill="none" stroke="#999" stroke-dasharray="5 3" points="' + " ".join(f"{sx(x):.0f},{sy(y):.0f}" for x, y in front) + '"/>')
    for n_, x, y, k in points:
        o.append(f'<circle cx="{sx(x):.0f}" cy="{sy(y):.0f}" r="4" fill="{cols.get(k, "#444")}"/><text x="{sx(x) + 6:.0f}" y="{sy(y) - 5:.0f}" font-size="9">{n_}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def cond_table(rows: List[Dict], group: str):
    by = defaultdict(list)
    for r in rows:
        if r["group"] == group:
            by[r["cond"]].append(r)
    return by


def summarize(rs: List[Dict]) -> Dict:
    st = [r for r in rs if r["kind"] == "status"]
    tm = [r for r in rs if r["kind"] == "time"]
    cold = next(r for r in rs if r["cold"])
    return dict(n_status=len(st), acc=sum(r["ok"] for r in st) / len(st), ci=wilson(sum(r["ok"] for r in st), len(st)), time_acc=sum(r["ok"] for r in tm) / max(1, len(tm)),
                ctx=cold["ctx_tokens"], ttft=cold["ttft"], warm=sum(r["total"] for r in rs if not r["cold"]) / max(1, len(rs) - 1))


def paired(by: Dict[str, List[Dict]], a: str, b: str):
    """questions answered correctly only under a / only under b (status questions, same question set)."""
    ra = {r["qi"]: r["ok"] for r in by[a] if r["kind"] == "status"}
    rb = {r["qi"]: r["ok"] for r in by[b] if r["kind"] == "status"}
    x = sum(1 for i in ra if ra[i] and not rb.get(i, False))
    y = sum(1 for i in ra if rb.get(i, False) and not ra[i])
    return x, y, sign_p(x, y)


def section_ctx(model_key: str, P):
    rows = pool(rows_of(f"ctx_{model_key}.jsonl"))
    if not rows:
        return
    groups = sorted({r["group"] for r in rows})
    csvrows = []
    for g in groups:
        by = cond_table(rows, g)
        base = "L0" if "L0" in by else next((c for c in by if c.startswith("L0@")), None)
        P(f"\n### グループ {g}({model_key})\n")
        P("| 条件 | 文脈tok | トークン比 | 状態QA | 95%CI | 時期QA | コールドTTFT(s) | ウォーム(s/問) | 精度/1k tok | L0に対する符号検定(のみ正解 L1:L0, p) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        s0 = summarize(by[base]) if base else None
        for c, rs in by.items():
            s = summarize(rs)
            sg = ""
            if base and c != base:
                x, y, p = paired(by, c, base)
                sg = f"{x}:{y}, p={p:.3f}"
            ratio = s["ctx"] / s0["ctx"] if s0 else float("nan")
            P(f"| {c} | {s['ctx']} | {ratio:.2f} | {s['acc']:.1%} | {s['ci'][0]:.2f}-{s['ci'][1]:.2f} | {s['time_acc']:.0%} | {s['ttft']:.0f} | {s['warm']:.1f} | {s['acc'] / (s['ctx'] / 1000):.3f} | {sg} |")
            csvrows.append(dict(model=model_key, group=g, cond=c, ctx_tokens=s["ctx"], token_ratio=round(ratio, 3), status_acc=round(s["acc"], 4), time_acc=round(s["time_acc"], 4),
                                cold_ttft=round(s["ttft"], 1), warm_s=round(s["warm"], 2)))
        # by status category and position
        P("\n状態別の正答率(状態QA):\n")
        sts = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]
        P("| 条件 | " + " | ".join(sts) + " |")
        P("|---|" + "---|" * len(sts))
        for c, rs in by.items():
            cells = []
            for s_ in sts:
                q = [r for r in rs if r["kind"] == "status" and r["status"] == s_]
                cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f}" if q else "-")
            P(f"| {c} | " + " | ".join(cells) + " |")
        P("\n正解メモリの位置別の正答率(状態QA, 位置ビン0=先頭〜4=末尾):\n")
        P("| 条件 | " + " | ".join(f"bin{b}" for b in range(5)) + " |")
        P("|---|---|---|---|---|---|")
        for c, rs in by.items():
            cells = []
            for b in range(5):
                q = [r for r in rs if r["kind"] == "status" and r["pos_bin"] == b]
                cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f}" if q else "-")
            P(f"| {c} | " + " | ".join(cells) + " |")
    with (OUT / "ctx_summary.csv").open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csvrows[0]))
        if fh.tell() == 0:
            w.writeheader()
        w.writerows(csvrows)


def pareto(model_key: str):
    rows = [r for r in rows_of(f"ctx_{model_key}.jsonl") if r["group"].startswith("ablation")]
    if not rows:
        return
    by = cond_table(rows, rows[0]["group"])
    base = summarize(by["L0"])
    kind = lambda c: "L0" if c == "L0" else "control" if "control" in c or c == "naive" else "surgical" if c in ("no-filler", "no-polite", "no-particle", "state-normalized", "no-chitchat(oracle)", "surface-stripped", "surface-stripped+state") else "l1" if c in ("sudachi-m", "sudachi-g", "content-only") else "ablation"
    pts = []
    for c, rs in by.items():
        s = summarize(rs)
        pts.append((c, s["ctx"] / base["ctx"], s["acc"], kind(c)))
    svg_scatter(pts, f"トークン比 vs 状態QA正答率({model_key}, N=300)", "トークン比(L0=1.0)", "状態QA正答率", OUT / f"chart_pareto_{model_key}.svg")


def section_natural(model_key: str, P):
    rows = rows_of(f"nat_qa_{model_key}.jsonl")
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["rep"]].append(r)
    base = by.get("L0")
    P(f"\n### 自然会話(LLM生成)での再現: {model_key}\n")
    P("| 表現 | 文脈tok | トークン比 | 正答率 | 95%CI | コールドTTFT(s) | 変換(ms/件) | L0との符号検定(L1のみ正解:L0のみ正解) |")
    P("|---|---|---|---|---|---|---|---|")
    c0 = next(r for r in base if r["i"] == 0)["ctx_tokens"] if base else None
    for rep, rs in by.items():
        n, k = len(rs), sum(r["ok"] for r in rs)
        cold = next(r for r in rs if r["i"] == 0)
        sg = ""
        if base and rep != "L0":
            a = {r["id"]: r["ok"] for r in rs}
            b = {r["id"]: r["ok"] for r in base}
            x = sum(1 for i in a if a[i] and not b[i])
            y = sum(1 for i in a if b[i] and not a[i])
            sg = f"{x}:{y}, p={sign_p(x, y):.3f}"
        lo, hi = wilson(k, n)
        P(f"| {rep} | {cold['ctx_tokens']} | {cold['ctx_tokens'] / c0:.2f} | {k / n:.1%} | {lo:.2f}-{hi:.2f} | {cold['ttft']:.0f} | {cold['conv_ms']:.2f} | {sg} |")
    P("\n失敗カテゴリ別(現象タグ / 状態)の正答率:\n")
    cats = ["subj_omit", "anaphora", "selfcorrect", "fragment", "slang", "multi_turn"]
    P("| 表現 | " + " | ".join(cats) + " | typo注入 | " + " | ".join(["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]) + " |")
    P("|---|" + "---|" * 14)
    for rep, rs in by.items():
        cells = []
        for c in cats:
            q = [r for r in rs if c in r["phenomena"]]
            cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f} ({len(q)})" if q else "-")
        q = [r for r in rs if r["typo"]]
        cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f} ({len(q)})" if q else "-")
        for s_ in ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]:
            q = [r for r in rs if r["status"] == s_]
            cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f} ({len(q)})" if q else "-")
        P(f"| {rep} | " + " | ".join(cells) + " |")


def jl(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def section_tokaware(P):
    d = jl("tokaware.json")
    if not d:
        return
    P("\n## #26 Tokenizer-aware L1\n")
    P("同じ意味(sudachi の語・標識)を、書き方(単語区切り / 文区切り / 標識の表記 / 活用語の表記)だけ変えて 4×4×3×2=96 通り試し、トークン数が最小になるスタイルを tokenizer ごとに選んだ。選択制約は、ルールによる状態復元率 97% 以上かつ「非完了を完了と誤読」0 件。較正は 400 件、下表は別の 600 件(held-out)。\n")
    P("| tokenizer | L0 tok/件 | 汎用 m | 汎用 g | 最小スタイル | 読みやすさ制約付き最小 | 最小スタイル(選択) |")
    P("|---|---|---|---|---|---|---|")
    for k, v in d.items():
        P(f"| {k} | {v['L0_tokens_per_memory']:.1f} | {v['generic_m']['ratio_vs_L0']:.2f} | {v['generic_g']['ratio_vs_L0']:.2f} | {v['best']['ratio_vs_L0']:.2f} | {v['best_readable']['ratio_vs_L0']:.2f} | {v['best']['style']} / 制約付き: {v['best_readable']['style']} |")
    P("\n(列の数値は L1 / L0 のトークン比。\"読みやすさ制約\" = 単語区切りと文区切りを必ず残す。)\n")


def section_dual(P):
    d = jl("dual_index.json")
    if not d:
        return
    P("\n## #28 Dual-Index RAG\n")
    P(f"メモリ {d['stats']['n']} 件、クエリ {d['nq']} 件/規模。融合は RRF(k=60)、重み付きの重みは別の開発クエリ 150 件で選択。\n")
    S = ["bm25_L0", "bm25_L1", "vec_L0", "vec_L1", "hybrid_L0", "dual", "dual_2view", "hybrid_L1", "quad", "weighted_dual", "rerank_bm25L1_by_vecL0"]
    for N in d["scales"]:
        for m in ("text-embedding-nomic-embed-text-v1.5", "text-embedding-qwen3-embedding-0.6b"):
            P(f"\n### N={N}, {m.replace('text-embedding-', '')}\n")
            P("| 方式 | R@1 | R@5 | R@10 | MRR | nDCG@10 | 状態別 R@5 (UNDECIDED/HEARSAY/POSSIBLE) |")
            P("|---|---|---|---|---|---|---|")
            for s in S:
                r = d["results"][f"{m}|{N}|{s}"]
                g = r["recall5_by_status"]
                P(f"| {s} | {r['recall1']:.2f} | {r['recall5']:.2f} | {r['recall10']:.2f} | {r['mrr']:.2f} | {r['ndcg10']:.2f} | {g['UNDECIDED']:.2f}/{g['HEARSAY']:.2f}/{g['POSSIBLE']:.2f} |")
    P("\nコスト:\n")
    st = d["stats"]
    P(f"- テキスト容量: L0 {st['bytes_L0']:,} B / L1 {st['bytes_L1']:,} B({st['bytes_L1'] / st['bytes_L0']:.0%})。BM25 ポスティング: {st['bm25_postings_L0']:,} → {st['bm25_postings_L1']:,}、語彙 {st['bm25_terms_L0']} → {st['bm25_terms_L1']}。変換 {st['convert_s']:.2f} s/{st['n']} 件。")
    for k, v in d["timing"].items():
        P(f"- {k}: BM25(L0) {v['bm25_L0_ms']:.0f} ms、BM25(L1) {v['bm25_L1_ms']:.0f} ms、ベクトル検索 {v['vec_ms']:.2f} ms、クエリ埋め込み {v['query_embed_ms']:.0f} ms")
    qa = rows_of("dual_qa_*.jsonl")
    if qa:
        P("\n最終QA(上位5件を LLM へ、メモリ 10,000 件):\n")
        by = defaultdict(list)
        for r in qa:
            by[(r["model"], r["strategy"], r["feed"])].append(r)
        P("| モデル | 検索 | LLMに渡す表現 | 取得ヒット率 | 正答率 | プロンプトtok |")
        P("|---|---|---|---|---|---|")
        for (m, s, f), rs in by.items():
            P(f"| {m} | {s} | {f} | {sum(r['hit'] for r in rs) / len(rs):.0%} | {sum(r['ok'] for r in rs) / len(rs):.1%} | {sum(r['prompt_tokens'] for r in rs) / len(rs):.0f} |")


def section_ml(P):
    d = jl("ml_tokens.json")
    if not d:
        return
    P("\n## #31 多言語パーサベース L1\n")
    P(f"同じ事実 {d['n_facts']} 件を 4 言語で描画し、言語ごとの既存解析器で決定的に L1 化(LLM なし)。tok/fact の言語間ばらつきは CV(平均の標準偏差/平均)。差の信頼区間は事実のブートストラップ。\n")
    P("| tokenizer | tok/fact L0 (ja/en/ko/zh) | tok/fact L1 (ja/en/ko/zh) | CV L0 | CV L1 | ΔCV 95%CI | P(CV L1<L0) | 事実ごとの言語間CV(L0→L1) |")
    P("|---|---|---|---|---|---|---|---|")
    for k, v in d["tokenizers"].items():
        P(f"| {k} | {'/'.join(f'{x:.1f}' for x in v['tok_per_fact_L0'].values())} | {'/'.join(f'{x:.1f}' for x in v['tok_per_fact_L1'].values())} | {v['cv_L0']:.3f} | {v['cv_L1']:.3f} | "
          f"{v['cv_diff_ci'][0]:+.3f}〜{v['cv_diff_ci'][1]:+.3f} | {v['p_cv_L1_lt_L0']:.2f} | {v['per_fact_cv_L0']:.3f}→{v['per_fact_cv_L1']:.3f} |")
    P("\n感度分析(マーカー設計への依存):\n")
    P("| L1の種類 | " + " | ".join(d["tokenizers"]) + " |")
    P("|---|" + "---|" * len(d["tokenizers"]))
    for vn, t in d["sensitivity"].items():
        P(f"| {vn} | " + " | ".join(f"{t[k]['cv_L0']:.2f}→{t[k]['cv_L1']:.2f}" for k in d["tokenizers"]) + " |")
    P("\n状態復元(規則)と文字数:\n")
    P("| 言語 | 状態復元率 | 完了への誤読 | 文字数 L0→L1 | パース時間(ms/事実) |")
    P("|---|---|---|---|---|")
    for lg, v in d["retention"].items():
        P(f"| {lg} | {v['status_recovered']:.1%} | {v['false_done']} | {v['chars_L0']:.1f}→{v['chars_L1']:.1f} | {d['parse_ms_per_fact'][lg]:.2f} |")
    qa = rows_of("ml_qa_*.jsonl")
    if qa:
        P("\nLLM QA(状態QA、N=300 事実、60 問):\n")
        by = defaultdict(list)
        for r in qa:
            by[(r["model"], r["lang"], r["rep"])].append(r)
        P("| モデル | 言語 | 表現 | 文脈tok | コールドTTFT(s) | 正答率 | 状態別(DONE/NOT/PLAN/WANT/UNDEC/HEAR/POSS) |")
        P("|---|---|---|---|---|---|---|")
        for (m, lg, rep), rs in by.items():
            cold = next(r for r in rs if r["i"] == 0)
            cells = []
            for s_ in ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]:
                q = [r for r in rs if r["status"] == s_]
                cells.append(f"{sum(r['ok'] for r in q)}/{len(q)}")
            P(f"| {m} | {lg} | {rep} | {cold['ctx_tokens']} | {cold['ttft']:.0f} | {sum(r['ok'] for r in rs) / len(rs):.1%} | {' '.join(cells)} |")


def section_attn(P):
    for name in ("attn", "attn_long", "attn2"):
        d = jl(f"{name}.json")
        if not d:
            continue
        import numpy as np
        res = d["results"]
        P(f"\n### #29 {name}(Qwen3-1.7B 4bit, N={d['n']} メモリ, 正解メモリ位置×{d['per_pos']}件/位置)\n")
        P(f"トークン/メモリ: " + ", ".join(f"{k} {v:.1f}" for k, v in d["tokens_per_memory"].items()) + "\n")
        P("| 条件 | プロンプトtok | 正答率 | 正解logit余裕(平均) | 注意量 target(上位層) | person | event | state | question | prefix | 注意エントロピー |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in d["conds"]:
            rs = [r for r in res if r["cond"] == c]
            up = lambda key: np.mean([np.mean(r[key][14:]) for r in rs if key in r])
            P(f"| {c} | {np.mean([r['T'] for r in rs]):.0f} | {np.mean([r['correct'] for r in rs]):.2f} | {np.mean([r['margin'] for r in rs]):+.2f} | {up('mass_target'):.3f} | {up('mass_person'):.4f} | {up('mass_event'):.4f} | {up('mass_state'):.4f} | {up('mass_question'):.3f} | {up('mass_prefix'):.3f} | {np.mean([np.mean(r['entropy'][14:]) for r in rs]):.2f} |")
        if all("maxhead_target" in r for r in res):
            P("\n1つのヘッドが正解メモリ/状態トークンへ向ける注意の最大値(層ごとの最大ヘッドの、層14〜27での平均):\n")
            P("| 条件 | target 最大ヘッド | person | event | state |")
            P("|---|---|---|---|---|")
            for c in d["conds"]:
                rs = [r for r in res if r["cond"] == c]
                mh = lambda key: np.nanmean([np.mean(r[key][14:]) for r in rs if key in r])
                P(f"| {c} | {mh('maxhead_target'):.3f} | {mh('maxhead_person'):.3f} | {mh('maxhead_event'):.3f} | {mh('maxhead_state'):.3f} |")
        P("\n位置別(正解メモリの位置 0=先頭〜1=末尾)の正解logit余裕 / 正答率:\n")
        pos = sorted({r["pos"] for r in res})
        P("| 条件 | " + " | ".join(f"pos {p}" for p in pos) + " |")
        P("|---|" + "---|" * len(pos))
        for c in d["conds"]:
            cells = []
            for p in pos:
                q = [r for r in res if r["cond"] == c and r["pos"] == p]
                cells.append(f"{np.mean([r['margin'] for r in q]):+.1f} / {np.mean([r['correct'] for r in q]):.2f}")
            P(f"| {c} | " + " | ".join(cells) + " |")
        # per-prompt association between attention to the target memory and the answer margin
        from itertools import combinations
        rho = {}
        for c in d["conds"]:
            rs = [r for r in res if r["cond"] == c]
            x = np.array([np.mean(r["mass_target"][14:]) for r in rs])
            y = np.array([r["margin"] for r in rs])
            rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
            rho[c] = float(np.corrcoef(rx, ry)[0, 1])
        P("\nプロンプト単位の Spearman 相関(正解メモリへの注意量 vs 正解logit余裕): " + ", ".join(f"{c} {v:+.2f}" for c, v in rho.items()))
        hp = OUT / f"{name}_hidden.npz"
        if hp.exists():
            z = np.load(hp)
            keys = [k for k in z.files if k.endswith("|last")]
            conds = d["conds"]
            P("\n層ごとの隠れ状態の類似度(最終トークン、同じ質問・同じ位置の L0 との cosine の中央値、層 0〜27):\n")
            P("| 条件 | 層0 | 層7 | 層14 | 層21 | 層27 |")
            P("|---|---|---|---|---|---|")
            for c in conds[1:]:
                sims = defaultdict(list)
                for k in keys:
                    cc, t, p, _ = k.split("|")
                    if cc != c:
                        continue
                    a, b = z[k].astype(np.float32), z[f"L0|{t}|{p}|last"].astype(np.float32)
                    sims_l = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
                    for l, v in enumerate(sims_l):
                        sims[l].append(v)
                if sims:
                    P(f"| {c} | " + " | ".join(f"{np.median(sims[l]):.3f}" for l in (0, 7, 14, 21, 27)) + " |")


def pool(rows):
    """Merge a group with its second question sample (group name + "b"): same conditions, independent questions."""
    out = []
    for r in rows:
        r = dict(r)
        if r["group"].endswith("b") and r["group"][:-1] in {"tm600", "tm300", "ablation300"}:
            r["group"] = r["group"][:-1]
            r["qi"] += 1000
        out.append(r)
    return out


def main():
    lines: List[str] = []
    P = lines.append
    P("# Issue #25〜#31 追加検証(日本語簡易L1)\n")
    P("共通の方法: 条件ごとに文脈を固定して prefix cache を使い、同一の質問集合で全条件を比較する(コールド=1問目のprefill、ウォーム=2問目以降)。差は同じ質問でのペア符号検定。各条件の質問は状態QA 60 問 + 時期QA 20 問。\n")
    (OUT / "ctx_summary.csv").unlink(missing_ok=True)
    P("## #25 / #30 長文脈での精度要因・最小十分表現\n")
    for f in sorted(OUT.glob("ctx_*.jsonl")):
        key = f.stem[4:]

        def _sec(P=P, key=key):
            section_ctx(key, P)
        section_ctx(key, P)
        pareto(key)
    section_tokaware(P)
    section_dual(P)
    P("\n## #27 自然会話での再現\n")
    for f in sorted(OUT.glob("nat_qa_*.jsonl")):
        section_natural(f.stem[7:], P)
    section_ml(P)
    P("\n## #29 内部表現\n")
    section_attn(P)
    charts()
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "report.md", len(lines), "lines")



def svg_bars(groups, series, title: str, ylabel: str, path: Path, ymax=None):
    """groups: x labels; series: {name: [value per group]} (grouped bars)."""
    W, H, M = 900, 440, 60
    n, k = len(groups), len(series)
    ymax = ymax or max(v for s in series.values() for v in s) * 1.1
    cols = ["#c0392b", "#2471a3", "#1e8449", "#e67e22", "#8e44ad"]
    gw = (W - 2 * M) / n
    bw = gw * 0.8 / k
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">', f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
         f'<line x1="{M}" y1="{H - M}" x2="{W - M}" y2="{H - M}" stroke="#444"/><line x1="{M}" y1="{M}" x2="{M}" y2="{H - M}" stroke="#444"/>',
         f'<text x="16" y="{H / 2}" transform="rotate(-90 16 {H / 2})" text-anchor="middle">{ylabel}</text>']
    for t in range(5):
        v = ymax * t / 4
        y = H - M - (H - 2 * M) * t / 4
        o.append(f'<text x="{M - 6}" y="{y + 4:.0f}" text-anchor="end">{v:.2f}</text><line x1="{M}" y1="{y:.0f}" x2="{W - M}" y2="{y:.0f}" stroke="#eee"/>')
    for gi, g in enumerate(groups):
        o.append(f'<text x="{M + gw * (gi + 0.5):.0f}" y="{H - M + 14}" text-anchor="middle">{g}</text>')
        for si, (name, vals) in enumerate(series.items()):
            h = (H - 2 * M) * vals[gi] / ymax
            x = M + gw * gi + gw * 0.1 + bw * si
            o.append(f'<rect x="{x:.0f}" y="{H - M - h:.0f}" width="{bw:.0f}" height="{h:.0f}" fill="{cols[si % 5]}"/>')
    for si, name in enumerate(series):
        o.append(f'<rect x="{W - 190}" y="{34 + 15 * si}" width="10" height="10" fill="{cols[si % 5]}"/><text x="{W - 175}" y="{43 + 15 * si}">{name}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def charts():
    d = jl("ml_tokens.json")
    if d:
        tk = list(d["tokenizers"])
        svg_bars(tk, {"L0": [d["tokenizers"][t]["cv_L0"] for t in tk], "L1": [d["tokenizers"][t]["cv_L1"] for t in tk], "L1(content only)": [d["sensitivity"]["L1_content_words_only"][t]["cv_L1"] for t in tk]},
                 "#31 言語間 tok/fact の CV(小さいほど収束)", "CV", OUT / "chart_ml_cv.svg")
    d = jl("dual_index.json")
    if d:
        m = "text-embedding-qwen3-embedding-0.6b"
        S = ["bm25_L0", "bm25_L1", "vec_L0", "vec_L1", "hybrid_L0", "dual", "dual_2view", "weighted_dual"]
        svg_bars(S, {"qwen3-emb": [d["results"][f"{m}|10000|{s}"]["recall5"] for s in S],
                     "nomic": [d["results"][f"text-embedding-nomic-embed-text-v1.5|10000|{s}"]["recall5"] for s in S]}, "#28 Recall@5(メモリ 10,000 件)", "Recall@5", OUT / "chart_dual_recall5.svg", ymax=1.0)
    d = jl("tokaware.json")
    if d:
        tk = list(d)
        svg_bars(tk, {"汎用m": [d[t]["generic_m"]["ratio_vs_L0"] for t in tk], "汎用g": [d[t]["generic_g"]["ratio_vs_L0"] for t in tk], "最小": [d[t]["best"]["ratio_vs_L0"] for t in tk],
                      "読みやすさ制約付き最小": [d[t]["best_readable"]["ratio_vs_L0"] for t in tk]}, "#26 L1/L0 トークン比(tokenizer別)", "トークン比", OUT / "chart_tokaware_ratio.svg")


if __name__ == "__main__":
    main()
