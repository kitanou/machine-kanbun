"""Issue #20 report: L0 vs simple L1 (existing Japanese analyzers) -- tables, CSV and SVG charts."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import terms

RES = Path(__file__).parent.parent / "results" / "jp"


def jload(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jsonl(pattern):
    rows = []
    for f in sorted(RES.glob(pattern)):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rows.append(json.loads(l))
    return rows


def svg_lines(series: Dict[str, List[tuple]], title: str, xlabel: str, ylabel: str, path: Path, logx=False, logy=False, ymax=None):
    import math
    W, H, M = 760, 440, 62
    pts = [p for s in series.values() for p in s]
    if not pts:
        return
    fx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    fy = (lambda v: math.log10(max(v, 1e-9))) if logy else (lambda v: v)
    x0, x1 = min(fx(p[0]) for p in pts), max(fx(p[0]) for p in pts)
    y0, y1 = min(fy(p[1]) for p in pts), max(fy(p[1]) for p in pts) if ymax is None else fy(ymax)
    sx = lambda v: M + (fx(v) - x0) / ((x1 - x0) or 1) * (W - 2 * M)
    sy = lambda v: H - M - (fy(v) - y0) / ((y1 - y0) or 1) * (H - 2 * M)
    cols = ["#c0392b", "#2471a3", "#1e8449", "#e67e22", "#8e44ad", "#7f8c8d", "#16a085", "#d35400"]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">', f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
         f'<line x1="{M}" y1="{H - M}" x2="{W - M}" y2="{H - M}" stroke="#444"/><line x1="{M}" y1="{M}" x2="{M}" y2="{H - M}" stroke="#444"/>',
         f'<text x="{W / 2}" y="{H - 14}" text-anchor="middle">{xlabel}</text>',
         f'<text x="16" y="{H / 2}" transform="rotate(-90 16 {H / 2})" text-anchor="middle">{ylabel}</text>']
    for t in range(5):
        xv = x0 + (x1 - x0) * t / 4
        yv = y0 + (y1 - y0) * t / 4
        o.append(f'<text x="{M + (W - 2 * M) * t / 4:.0f}" y="{H - M + 14}" text-anchor="middle">{(10 ** xv if logx else xv):.3g}</text>')
        o.append(f'<text x="{M - 6}" y="{H - M - (H - 2 * M) * t / 4 + 4:.0f}" text-anchor="end">{(10 ** yv if logy else yv):.3g}</text>')
    for i, (name, s) in enumerate(series.items()):
        c = cols[i % len(cols)]
        s = sorted(s)
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="' + " ".join(f"{sx(x):.0f},{sy(y):.0f}" for x, y in s) + '"/>')
        for x, y in s:
            o.append(f'<circle cx="{sx(x):.0f}" cy="{sy(y):.0f}" r="3" fill="{c}"/>')
        o.append(f'<rect x="{W - 190}" y="{34 + 15 * i}" width="10" height="10" fill="{c}"/><text x="{W - 175}" y="{43 + 15 * i}">{name}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def main():
    L: List[str] = []
    P = L.append
    csv_rows = []
    P("# Issue #20 日本語 NF(旧 L0)と既存解析器ベースの SCF(旧 簡易 L1)の比較\n")
    P(terms.note() + "(以下、表中の L0 は NF、L1 は SCF-L1、sudachi-m などは SCF の変種)\n")
    P("対象は日本語の会話文のみ。簡易 L1 は既存の形態素解析器(SudachiPy / MeCab(fugashi, unidic-lite) / Janome)と係り受け解析器(GiNZA)の出力を"
      "軽量ルールで写像したもので、LLM は使わない。変種: **m** 形態素のみ(空白区切り)、**g** m を空白なしで連結、**p** 格助詞を残す、**c** 接続を残す、"
      "**d** GiNZA の係り受けで節に分け省略された主語を補う、**naive** 解析器なしでひらがな連を削るだけ。\n")
    from . import jl1
    ex = "昨日、田中さんと久しぶりに話したんだけど、どうも会社を辞めようか迷っているらしい。ただ、まだ決めたわけではないみたい。"
    P("## 変換例(課題文の例)\n")
    P("```text\nL0       : " + ex)
    for an, var in [("sudachi", "m"), ("sudachi", "g"), ("sudachi", "p"), ("mecab", "m"), ("janome", "m"), ("ginza", "d"), ("naive", "m")]:
        try:
            c = jl1.Converter(an, var)
            P(f"{c.name:9}: {c(ex)}")
        except Exception as e:  # analyzers missing in this interpreter
            P(f"{an}-{var}: (解析器が未導入: {type(e).__name__})")
    P("```\n")
    # E1
    c1 = jload("conversion_cost.json")
    if c1:
        P("## 実験1: 変換コスト(2,000 メッセージ, 1 メッセージ平均 約32文字)\n")
        P("| 変換 | 起動(s) | 平均(ms/通) | p95(ms) | スループット(通/s) | CPU使用率 | メモリ増(MB) | 文字比 | トークン比(o200k) |")
        P("|---|---|---|---|---|---|---|---|---|")
        for k, c in c1["configs"].items():
            P(f"| {k} | {c['load_s']:.2f} | {c['mean_ms']:.3f} | {c['p95_ms']:.3f} | {c['throughput_per_s']:,.0f} | {c['cpu_util']:.0%} | {c['rss_delta_mb']:.0f} | {c['char_ratio']:.2f} | {c['token_ratio']:.2f} |")
            csv_rows.append(dict(exp="E1", config=k, mean_ms=round(c["mean_ms"], 4), throughput=round(c["throughput_per_s"]), token_ratio=round(c["token_ratio"], 3), char_ratio=round(c["char_ratio"], 3)))
        P(f"\n入力: {c1['chars_in']} 文字 / {c1['tokens_in']} トークン(o200k)。\n")
    if c1:
        P("## 実験5(容量): 同じコンテキスト窓に入るメモリ件数\n")
        P("コンテキスト窓 19,456 トークン(gemma-4-12b の実効値)からプロンプトの固定部分 300 トークンを引いた分に、メモリを詰められる件数。\n")
        tpm = c1["tokens_in"] / c1["n"]
        P(f"L0 は 1 件あたり {tpm:.1f} トークン。\n")
        P("| 表現 | トークン/件 | 入る件数 | L0 比 |")
        P("|---|---|---|---|")
        cap0 = int((19456 - 300) / tpm)
        P(f"| L0 | {tpm:.1f} | {cap0:,} | 1.00× |")
        for k, c in c1["configs"].items():
            t = tpm * c["token_ratio"]
            cap = int((19456 - 300) / t)
            P(f"| {k} | {t:.1f} | {cap:,} | {cap / cap0:.2f}× |")
        P("")
        try:
            from .tokcross import counters
            from . import jl1, jpdata
            cnt = counters()
            texts = [m.text for m in jpdata.make_memories(500, seed=7)]
            convs = {n: jl1.Converter(a_, v_) for n, (a_, v_) in {"sudachi-m": ("sudachi", "m"), "sudachi-g": ("sudachi", "g"), "sudachi-p": ("sudachi", "p"),
                                                                    "ginza-d": ("ginza", "d"), "naive": ("naive", "m")}.items()}
            l1 = {n: [c(t) for t in texts] for n, c in convs.items()}
            P("### トークナイザ依存(L1 / L0 のトークン比, 500 件)\n")
            P("| トークナイザ | L0 tok/件 | " + " | ".join(convs) + " |")
            P("|---|---|" + "---|" * len(convs))
            for name, f in cnt.items():
                base = sum(f(t) for t in texts)
                P(f"| {name} | {base / len(texts):.1f} | " + " | ".join(f"{sum(f(x) for x in l1[n]) / base:.2f}" for n in convs) + " |")
            P("")
        except Exception as e:
            P(f"(トークナイザ比較は省略: {type(e).__name__})\n")
    # E4
    r4 = jload("retention.json")
    if r4:
        P("## 実験4: 情報保持(ルールベース。LLM による確認は実験2)\n")
        P(f"原文に同じ規則を当てた状態判定の精度(較正): {r4['_calibration_l0_status_accuracy']:.1%}。\n")
        P("| 変換 | 人名 | 出来事 | 時間表現 | 数値 | 状態を復元できた割合 | 非完了を「完了」と誤読 |")
        P("|---|---|---|---|---|---|---|")
        for k, c in r4.items():
            if k.startswith("_"):
                continue
            P(f"| {k} | {c['person']:.0%} | {c['event']:.0%} | {c.get('time', 0):.0%} | {c.get('nums', 0):.0%} | {c['status']:.0%} | {c['not_done_as_done']:.1%} |")
        P("\n状態別の復元率(代表: sudachi-m / janome-m / naive):\n")
        sts = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]
        P("| 変換 | " + " | ".join(sts) + " |")
        P("|---|" + "---|" * len(sts))
        for k in ("sudachi-m", "sudachi-g", "janome-m", "ginza-d", "naive"):
            if k in r4:
                P(f"| {k} | " + " | ".join(f"{r4[k]['by_status'][s]:.0%}" for s in sts) + " |")
        P("")
    # E3
    r3 = jload("rag.json")
    if r3:
        P("## 実験3: RAG / メモリ検索(クエリ 1 件につき正解メモリ 1 件, 紛らわしい似たメモリを含む)\n")
        P("Recall@k / MRR / nDCG@10。`raw` = クエリを自然文のまま、`l1` = クエリも同じ L1 変換。\n")
        models = [k for k in next(iter(r3["results"].values())) if k not in ("bm25", "by_group_recall5")]
        for N in r3["scales"]:
            P(f"\n### メモリ {N:,} 件\n")
            P("| 表現 | クエリ | " + " | ".join(f"{m}: R@1 / R@5 / MRR / nDCG10" for m in ["bm25"] + [m.replace("text-embedding-", "") for m in models]) + " |")
            P("|---|---|" + "---|" * (1 + len(models)))
            for key, e in r3["results"].items():
                n_, name, qm = key.split("|")
                if int(n_) != N:
                    continue
                cells = [f"{e['bm25']['recall1']:.2f} / {e['bm25']['recall5']:.2f} / {e['bm25']['mrr']:.2f} / {e['bm25']['ndcg10']:.2f}"]
                for m in models:
                    cells.append(f"{e[m]['recall1']:.2f} / {e[m]['recall5']:.2f} / {e[m]['mrr']:.2f} / {e[m]['ndcg10']:.2f}")
                P(f"| {name} | {qm} | " + " | ".join(cells) + " |")
                for m in ["bm25"] + models:
                    csv_rows.append(dict(exp="E3", config=f"{name}|{qm}", scale=N, retriever=m.replace("text-embedding-", ""), recall1=round(e[m]["recall1"], 4),
                                         recall5=round(e[m]["recall5"], 4), mrr=round(e[m]["mrr"], 4), ndcg10=round(e[m]["ndcg10"], 4)))
        P("\n### 日本語固有のケース別 Recall@5(メモリ 1,000 件, nomic, クエリ raw)\n")
        m0 = [m for m in models if "nomic" in m] or models[:1]
        grp_keys = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE", "omit_subject", "pronoun", "polite", "casual"]
        P("| 表現 | " + " | ".join(grp_keys) + " |")
        P("|---|" + "---|" * len(grp_keys))
        for key, e in r3["results"].items():
            n_, name, qm = key.split("|")
            if int(n_) == 1000 and qm == "raw" and m0:
                g = e["by_group_recall5"][m0[0]]
                P(f"| {name} | " + " | ".join(f"{g.get(k, float('nan')):.2f}" for k in grp_keys) + " |")
        rt = (RES / "embed_retime.json")
        rt = json.loads(rt.read_text(encoding="utf-8"))["models"] if rt.exists() else None
        P("\n埋め込み・変換コスト" + ("(埋め込み速度は、対象の埋め込みモデルだけをロードした状態で再計測した値。検索実行中の値は他モデル常駐の影響を受けるため不使用)" if rt else "(注意: 検索実行中の計測で、他モデルの影響を受けうる)") + ":\n")
        P("| 表現 | 平均文字数 | 変換(ms/件) | " + " | ".join(m.replace("text-embedding-", "") + " 埋め込み(件/s)" for m in models) + " |")
        P("|---|---|---|" + "---|" * len(models))
        for name, ch in r3["avg_chars"].items():
            def _tps(m):
                if rt is not None:
                    return rt.get(m, {}).get(name, {}).get("texts_per_s", 0)
                return r3["embed_texts_per_s"].get(m + "|" + name, 0)
            P(f"| {name} | {ch:.1f} | {r3['convert_ms_per_text'].get(name, 0):.3f} | " + " | ".join(f"{_tps(m):.0f}" for m in models) + " |")
        P("")
        # chart: recall@5 (nomic) vs scale per representation
        ser = {}
        for key, e in r3["results"].items():
            n_, name, qm = key.split("|")
            if qm == "raw" and m0:
                ser.setdefault(name, []).append((int(n_), e[m0[0]]["recall5"]))
        svg_lines(ser, "Recall@5 vs memories (nomic, raw query)", "memories", "Recall@5", RES / "chart_recall.svg", logx=True)
    # E2 / E5
    ll = jsonl("llm_*.jsonl")
    if ll:
        P("## 実験2: LLM エンドツーエンド(全メモリを文脈に投入。確定/否定/未確定/不明の3値判定)\n")
        P("コールド = prefix cache なしの 1 問目(prefill 時間)、ウォーム = 2 問目以降の平均。解析・変換時間は書き込み時に1回(メモリ1件あたり)払う想定で、"
          "全メモリをその場で変換する場合の時間も併記。\n")
        by = defaultdict(list)
        for r in ll:
            if "error" not in r:
                by[(r["model"], r["n"], r["rep"])].append(r)
        P("| モデル | N | 表現 | 精度 | 文脈tok | コールドTTFT(s) | コールド総(s) | ウォーム総(s/問) | 変換(その場, ms) | メモリΔ(MB) | swap Δ(MB) | tokens/s |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (m, n, rep), rs in sorted(by.items()):
            cold = next(r for r in rs if r["cold"])
            warm = [r["total"] for r in rs if not r["cold"]]
            conv_all = cold["conv_ms_per_memory"] * n
            tps = cold["prompt_tokens"] / cold["ttft"]
            P(f"| {m.split('/')[-1]} | {n} | {rep} | {sum(r['ok'] for r in rs) / len(rs):.1%} | {cold['prompt_tokens'] - cold['base_tokens']} | {cold['ttft']:.1f} | {cold['total']:.1f} | "
              f"{st.mean(warm) if warm else 0:.1f} | {conv_all:.1f} | {cold['used_delta_mb']:.0f} | {cold['swap_after_mb'] - cold['swap_before_mb']:+.0f} | {tps:.0f} |")
            csv_rows.append(dict(exp="E2", model=m, scale=n, config=rep, acc=round(sum(r["ok"] for r in rs) / len(rs), 4), ctx_tokens=cold["prompt_tokens"] - cold["base_tokens"],
                                 cold_ttft=round(cold["ttft"], 2), e2e_cold_s=round(cold["total"] + conv_all / 1000, 3)))
        ser = {}
        for (m, n, rep), rs in by.items():
            if "gemma" in m:
                cold = next(r for r in rs if r["cold"])
                ser.setdefault(rep, []).append((n, cold["ttft"] + cold["conv_ms_per_memory"] * n / 1000))
        svg_lines(ser, "gemma-4-12b cold TTFT incl. on-the-fly conversion", "memories in context", "seconds", RES / "chart_ttft.svg", logx=True)
        P("\n(精度は 3 値判定: 「はい/いいえ/未確定/不明」の先頭語が正解と一致した割合。)\n")
    rg = jsonl("rag_llm_*.jsonl")
    if rg:
        P("## 実験5: 長文メモリ(RAG モード: BM25 上位 5 件 → LLM)\n")
        P("全件はコンテキストに入らないため、メモリ 1,000 / 10,000 件は検索で 5 件に絞って LLM に渡す。エンドツーエンド = 検索 + LLM(変換は書き込み時)。\n")
        by = defaultdict(list)
        for r in rg:
            by[(r["model"], r["n"], r["rep"])].append(r)
        P("| モデル | N | 表現 | 正解メモリを取得 | 回答精度 | プロンプトtok | 検索(ms) | LLM総(s) |")
        P("|---|---|---|---|---|---|---|---|")
        for (m, n, rep), rs in sorted(by.items()):
            P(f"| {m.split('/')[-1]} | {n:,} | {rep} | {sum(r['hit'] for r in rs) / len(rs):.0%} | {sum(r['ok'] for r in rs) / len(rs):.1%} | "
              f"{st.mean(r['prompt_tokens'] for r in rs):.0f} | {st.mean(r['retrieval_ms'] for r in rs):.1f} | {st.mean(r['total'] for r in rs):.2f} |")
        P("")
    # E6
    b6 = jload("breakeven.json")
    if b6:
        P("## 実験6: 損益分岐\n")
        P("変換時間を `a + b·文字数`、L1 化によるトークン削減を `s·文字数` として実測から回帰し、LLM の prefill 速度ごとに"
          "**その記憶を何回プロンプトに入れれば変換コストを回収できるか**(R_be = 変換時間 / prefill 短縮量。1 以下なら最初の 1 回で回収)を求めた。"
          "変換コストも節約も文字数にほぼ比例するため、損益は「長さの閾値」ではなく**prefill 速度の条件**で決まる。\n")
        P("| 変換 | a(ms) | b(µs/文字) | s(削減tok/文字) | " + " | ".join(f"R_be @{r}tok/s" for r in b6["rates"]) + " |")
        P("|---|---|---|---|" + "---|" * len(b6["rates"]))
        for k, f in b6["fits"].items():
            cells = []
            for r in b6["rates"]:
                v = b6["table"][k][str(r)]["r_be"]
                cells.append("回収不能" if v is None else f"{v:.3g}")
            P(f"| {k} | {f['a_ms']:.3f} | {f['b_ms_per_char'] * 1000:.2f} | {f['saved_tokens_per_char']:.3f} | " + " | ".join(cells) + " |")
        P("\n測定した prefill 速度: " + ", ".join(f"{k} {v} tok/s" for k, v in b6["measured_rates"].items()) + "。")
        ser = {}
        for k in b6["fits"]:
            if k in ("sudachi-m", "mecab-m", "janome-m", "ginza-m", "naive"):
                ser[k] = [(r, b6["table"][k][str(r)]["r_be"]) for r in b6["rates"] if b6["table"][k][str(r)]["r_be"]]
        svg_lines(ser, "Break-even reuse count vs LLM prefill speed (lower is better)", "prefill tokens/s", "R_be (uses to repay)", RES / "chart_breakeven.svg", logx=True, logy=True)
        P("\nグラフ: [chart_breakeven.svg](chart_breakeven.svg)、[chart_recall.svg](chart_recall.svg)、[chart_ttft.svg](chart_ttft.svg)\n")
    text = "\n".join(L)
    (RES / "report.md").write_text(text + "\n", encoding="utf-8")
    if csv_rows:
        for r in csv_rows:
            r["form"] = terms.form_of(str(r.get("config", "")).split("|")[0])
        keys = sorted({k for r in csv_rows for k in r})
        with (RES / "results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
    print(text)


if __name__ == "__main__":
    main()
