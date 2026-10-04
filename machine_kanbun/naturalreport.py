"""Issue #19 C/D/B report: natural sentences -> IR -> QA, round trip, core coverage."""
from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import tiktoken

from . import core
from .ablreport import diff_ci
from .longreport import sign_test
from .natural_data import ITEMS, LANGS, PHENOMENA

RES = Path(__file__).parent.parent / "results"
EXPECT = {"否定": {"不", "無", "未", "非"}, "時制": {"過", "今", "将", "既", "継"}, "様相": {"禁", "必", "可", "願", "意", "不"},
          "不確実性": {"疑"}, "伝聞": {"伝", "推"}, "因果": {"故", "然"}, "条件": {"若"}, "比較": {">", "="}, "意図・中止": {"止", "延", "予", "意"}}
CTX = ("orig", "ir", "rt")
CTXN = {"orig": "原文", "ir": "IRのみ", "rt": "往復(IR→自然文)"}


def load():
    recs = []
    for f in sorted((RES / "natural").glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                recs.append(json.loads(l))
    return recs


def main():
    recs = load()
    enc = tiktoken.get_encoding("o200k_base")
    cnt = lambda s: len(enc.encode(s))
    L: List[str] = []
    P = L.append
    P("# Issue #19 B/C/D: 自然文 → Semantic IR → QA / 往復\n")
    P(f"30 文 × 4 言語(各現象 3 文)。LLM が**原文から直接** IR(コア演算子 {len(core.CORE)} 個, `{core.CORE_VERSION}`)を生成し、"
      "(1) IR だけで QA、(2) IR から自然文に戻して(往復)QA、(3) 原文での QA(上限)を比較する。"
      "内容語は原言語のまま、IR 変換に日本語の翻訳は挟まない。QA は現象に依存する問い(否定・時制・様相・不確実性・伝聞・因果・条件・比較・照応・意図)。\n")
    for model in sorted({r["model"] for r in recs}):
        R = [r for r in recs if r["model"] == model]
        P(f"\n## {model}({len(R)} 件)\n")
        # accuracy by language x ctx
        P("**言語別精度**\n")
        P("| 言語 | 問題数 | 原文 | IRのみ | 往復 | IR − 原文 | 往復 − 原文 |")
        P("|---|---|---|---|---|---|---|")
        def acc(rs, c):
            q = [x for r in rs for x in r["qa"] if x["ctx"] == c]
            return sum(x["ok"] for x in q) / len(q) if q else 0, len(q)
        def paired(rs, a, b):
            bb = cc = n = 0
            for r in rs:
                da = {x["q"]: x["ok"] for x in r["qa"] if x["ctx"] == a}
                db = {x["q"]: x["ok"] for x in r["qa"] if x["ctx"] == b}
                for q in da:
                    n += 1
                    bb += da[q] and not db[q]
                    cc += db[q] and not da[q]
            lo, hi = diff_ci(bb, cc, n)
            return f"{(bb - cc) / max(n, 1) * 100:+.1f}pt [{lo:+.0f},{hi:+.0f}] p={sign_test(bb, cc):.2f}"
        for lg in list(LANGS) + ["ALL"]:
            rs = [r for r in R if lg == "ALL" or r["lang"] == lg]
            if not rs:
                continue
            a = {c: acc(rs, c) for c in CTX}
            P(f"| {lg} | {a['orig'][1]} | {a['orig'][0]:.1%} | {a['ir'][0]:.1%} | {a['rt'][0]:.1%} | {paired(rs, 'ir', 'orig')} | {paired(rs, 'rt', 'orig')} |")
        P("\n**意味現象別精度(4言語プール)**\n")
        P("| 現象 | 問題数 | 原文 | IRのみ | 往復 | 期待演算子の使用率 | 自由記述(~)あり | 未知演算子あり |")
        P("|---|---|---|---|---|---|---|---|")
        for ph in PHENOMENA:
            rs = [r for r in R if r["ph"] == ph]
            if not rs:
                continue
            a = {c: acc(rs, c) for c in CTX}
            exp = EXPECT.get(ph)
            use = f"{sum(1 for r in rs if exp & set(r['parse']['used'])) / len(rs):.0%}" if exp else "(該当なし)"
            P(f"| {ph} | {a['orig'][1]} | {a['orig'][0]:.1%} | {a['ir'][0]:.1%} | {a['rt'][0]:.1%} | {use} | "
              f"{sum(1 for r in rs if r['parse']['free_text']) / len(rs):.0%} | {sum(1 for r in rs if r['parse']['unknown']) / len(rs):.0%} |")
        # core coverage (B)
        P("\n**コア演算子のカバレッジ(段階B)**\n")
        n = len(R)
        free = sum(1 for r in R if r["parse"]["free_text"])
        unk = sum(1 for r in R if r["parse"]["unknown"])
        bad = sum(1 for r in R if r["parse"]["malformed"])
        P(f"- 変換 {n} 件のうち、自由記述(`~`)フォールバックを使った件 **{free}/{n} ({free / n:.0%})**、コア外の演算子を出力した件 {unk}/{n} ({unk / n:.0%})、"
          f"形式が崩れた行を含む件 {bad}/{n} ({bad / n:.0%})。")
        clean = [r for r in R if not r["parse"]["free_text"] and not r["parse"]["unknown"]]
        P(f"- コアだけで表現できた件(自由記述・未知演算子なし)は {len(clean)}/{n} ({len(clean) / n:.0%})。その QA 精度(IRのみ) {acc(clean, 'ir')[0]:.1%}、"
          f"それ以外 {acc([r for r in R if r not in clean], 'ir')[0]:.1%}。")
        used = Counter()
        for r in R:
            used.update(r["parse"]["used"])
        P("- 演算子の使用頻度: " + ", ".join(f"{o['symbol']}({o['name']}) {used.get(o['symbol'], 0)}" for o in core.CORE))
        P(f"- 未使用の演算子: {', '.join(o['symbol'] for o in core.CORE if used.get(o['symbol'], 0) == 0) or 'なし'}。"
          f"1文あたりの平均文数 {st.mean(r['parse']['statements'] for r in R):.1f}、1文の演算子数の最大の平均 {st.mean(r['parse']['max_ops_per_statement'] for r in R):.1f}。")
        # tokens + latency
        P("\n**トークンと変換時間**\n")
        P("| 言語 | 原文 tok(o200k) | IR tok | IR/原文 | 変換時間(s) | 往復時間(s) |")
        P("|---|---|---|---|---|---|")
        for lg in LANGS:
            rs = [r for r in R if r["lang"] == lg]
            if rs:
                o, i = st.mean(cnt(r["text"]) for r in rs), st.mean(cnt(r["ir"]) for r in rs)
                P(f"| {lg} | {o:.1f} | {i:.1f} | {i / o:.2f} | {st.mean(r['conv_s'] for r in rs):.1f} | {st.mean(r['rt_s'] for r in rs):.1f} |")
        # failures
        P("\n**IR のみで誤答した例(最大 8 件)**\n")
        shown = 0
        for r in R:
            for x in r["qa"]:
                if x["ctx"] == "ir" and not x["ok"] and shown < 8:
                    shown += 1
                    P(f"- [{r['lang']} {r['item']} {r['ph']}] 原文: {r['text']}\n  - IR: `{r['ir'].replace(chr(10), ' / ')}`\n  - 問: {x['q']} → 答: {x['answer'][:60]}")
    text = "\n".join(L)
    (RES / "synthesis" / "natural_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
