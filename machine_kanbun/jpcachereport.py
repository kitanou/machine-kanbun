"""Issue #69 report: TTFT of NF vs SeCF-L1 long-term memory under prefix-cache states (hit / partial / miss) and by number of cached prefixes (parallel slots)."""
from __future__ import annotations

import statistics as st
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .jpcache import OUT, jl
from .jpreport import svg_lines

STATES = ("BUILD", "HIT", "PARTIAL", "MISS")


def main() -> None:
    H = jl(OUT / "hit.jsonl")
    S = jl(OUT / "slots.jsonl")
    L = ["# Issue #69 SeCF-L1 長期記憶 × Prefix / KV キャッシュ\n",
         "gemma-4-12b(MLX、LM Studio)。読み込まれたコンテキストは 19,456 トークンで固定(`-c` も REST の context_length も無視される)。キャッシュは TTFT だけで観測(使用量にキャッシュ済みトークン数は出ない)。"
         "**LM Studio ではキャッシュを OFF にできない**ので、OFF = MISS(先頭が既存のキャッシュと一致しないプロンプトで、全トークンを再計算)として扱う。\n"]
    g = defaultdict(list)
    for r in H:
        g[(r["size"], r["rep"], r["state"])].append(r)
    sizes = sorted({r["size"] for r in H})
    med = lambda v, k="ttft": st.median(x[k] for x in v)
    L += ["## 1. 記憶量 × 表現 × キャッシュ状態(TTFT 秒の中央値。HIT は 5 回、PARTIAL・MISS は 3 回)\n",
          "| 記憶量(目標 NF tok) | 表現 | プロンプト tok | BUILD(初回) | HIT | HIT の範囲 | PARTIAL(先頭半分が共通) | MISS(= OFF) | HIT の短縮倍率 | 全再計算 tok/s |", "|---|---|---|---|---|---|---|---|---|---|"]
    series: Dict[str, List[tuple]] = defaultdict(list)
    for size in sizes:
        for rep in ("nf", "secf1"):
            if not all(g[(size, rep, s)] for s in STATES):
                continue
            tok = med(g[(size, rep, "HIT")], "prompt_tokens")
            m = {s: med(g[(size, rep, s)]) for s in STATES}
            hits = [x["ttft"] for x in g[(size, rep, "HIT")]]
            L.append(f"| {size} | {rep} | {tok:.0f} | {m['BUILD']:.1f} | {m['HIT']:.2f} | {min(hits):.2f}〜{max(hits):.2f} | {m['PARTIAL']:.1f} | {m['MISS']:.1f} | {m['MISS'] / m['HIT']:.0f}× | {tok / m['MISS']:.0f} |")
            for s in ("HIT", "PARTIAL", "MISS"):
                series[f"{rep} {s}"].append((tok, m[s]))
    L += ["\n## 2. SeCF-L1 / NF の比(同じ記憶内容)\n", "| 記憶量 | tok 比 | HIT の TTFT 比 | PARTIAL の TTFT 比 | MISS の TTFT 比 | MISS の差(秒) | HIT の差(秒) |", "|---|---|---|---|---|---|---|"]
    for size in sizes:
        if not all(g[(size, r, s)] for r in ("nf", "secf1") for s in STATES):
            continue
        t = {r: med(g[(size, r, "HIT")], "prompt_tokens") for r in ("nf", "secf1")}
        m = {(r, s): med(g[(size, r, s)]) for r in ("nf", "secf1") for s in STATES}
        L.append(f"| {size} | {t['secf1'] / t['nf']:.2f} | {m[('secf1', 'HIT')] / m[('nf', 'HIT')]:.2f} | {m[('secf1', 'PARTIAL')] / m[('nf', 'PARTIAL')]:.2f} | {m[('secf1', 'MISS')] / m[('nf', 'MISS')]:.2f} | "
                 f"{m[('secf1', 'MISS')] - m[('nf', 'MISS')]:+.1f} | {m[('secf1', 'HIT')] - m[('nf', 'HIT')]:+.2f} |")
    sw = [r["swap_mb"] for r in H]
    mf = [r.get("memfree_pct", -1) for r in H if r.get("memfree_pct", -1) >= 0]
    L += ["\n## 3. メモリ(システム全体。MLX の KV キャッシュ単体の使用量は取得できない)\n", f"- スワップ使用量(各測定後): 最小 {min(sw):.0f} MB、最大 {max(sw):.0f} MB。空きメモリ率: 最小 {min(mf):.0f}%、最大 {max(mf):.0f}%。"]
    if S:
        L += ["\n## 4. 並列スロット数(同時に保持できる別々の記憶 = 別々のキャッシュの数)× 4 利用者の巡回\n",
              "4 利用者がそれぞれ別の記憶(同じ大きさ)を持ち、順に問い合わせる。2 巡目以降の TTFT が 1 巡目(全再計算)の 35% 未満ならヒットとする。\n",
              "| スロット数 | 表現 | プロンプト tok | 1 巡目 TTFT(中央) | 2 巡目以降 TTFT(中央) | ヒット率 | エラー |", "|---|---|---|---|---|---|---|"]
        for K in sorted({r["slots"] for r in S}):
            for rep in ("nf", "secf1"):
                v = [r for r in S if r["slots"] == K and r["rep"] == rep]
                if not v:
                    continue
                first = {r["user"]: r["ttft"] for r in v if r["round"] == 0 and r["ttft"]}
                later = [r for r in v if r["round"] >= 1 and r["ttft"]]
                hit = [r["ttft"] < 0.35 * first[r["user"]] for r in later if r["user"] in first]
                if not first or not later:
                    continue  # this cell is still running
                L.append(f"| {K} | {rep} | {st.mean(r['prompt_tokens'] for r in v if r['prompt_tokens']):.0f} | {st.median(first.values()):.1f} | {st.median(r['ttft'] for r in later):.1f} | "
                         f"{np.mean(hit):.2f} | {sum(1 for r in v if r['error'])} |")
    import glob
    from pathlib import Path

    def summarize(path):
        rows = jl(Path(path))
        out = {}
        for rep in ("nf", "secf1"):
            v = [r for r in rows if r["rep"] == rep]
            if not v:
                continue
            first = {r["user"]: r["ttft"] for r in v if r["round"] == 0 and r["ttft"]}
            later = [r for r in v if r["round"] >= 1 and r["ttft"]]
            if not first or not later:
                continue
            ratio = [r["ttft"] / first[r["user"]] for r in later if r["user"] in first]
            tok = st.mean(r["prompt_tokens"] for r in v if r["prompt_tokens"])
            out[rep] = dict(users=v[0]["users"], size=v[0]["size"], tok=tok, cold=st.median(first.values()), later=st.median(r["ttft"] for r in later), ratio=st.median(ratio),
                            hit=sum(1 for x in ratio if x < 0.35) / len(ratio), swap=max(r["swap_mb"] for r in v))
        return out

    files = sorted(glob.glob(str(OUT / "slots_*R*_S*.jsonl")))
    if files:
        L += ["\n## 5. キャッシュの上限を探る(利用者 = 別々の記憶の数 R、1 本の大きさ S。総トークン数の昇順)\n",
              "ヒット率 = 2 巡目以降の TTFT が 1 巡目(全再計算)の 35% 未満の割合。「2 巡目以降 / 1 巡目」が 1.00 なら完全なミス、小さいほどヒット、0.4〜0.6 は部分的なヒット。`clean` は、SeCF-L1 を先に測り、各表現の前にモデルを再読み込みしてキャッシュを空にした対照(NF を先に測り、キャッシュを空にしない元の条件との比較で、順序・キャッシュの汚染の影響を見る)。\n",
              "| 条件 | 利用者 R | S | 表現 | 1 本のトークン | 総トークン | 1 巡目 TTFT | 2 巡目以降 TTFT | 2 巡目以降 / 1 巡目 | ヒット率 | スワップ MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        recs = []
        for f in files:
            name = Path(f).stem
            for rep, d in summarize(f).items():
                recs.append((d["users"] * d["tok"], "clean" if "clean" in name else "元の条件", rep, d))
        for tot, kind, rep, d in sorted(recs, key=lambda x: (x[0], x[1], x[2])):
            L.append(f"| {kind} | {d['users']} | {d['size']} | {rep} | {d['tok']:.0f} | {tot:.0f} | {d['cold']:.1f} | {d['later']:.1f} | {d['ratio']:.2f} | {d['hit']:.2f} | {d['swap']:.0f} |")
    (OUT / "cache_report.md").write_text("\n".join(L), encoding="utf-8")
    if series:
        svg_lines(dict(series), "記憶のトークン数 vs TTFT(キャッシュ状態別)", "プロンプトのトークン数", "TTFT(秒)", OUT / "chart_cache_ttft.svg", logx=True, logy=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
