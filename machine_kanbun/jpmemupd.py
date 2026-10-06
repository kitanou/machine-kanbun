"""Issue #62 (Meta #56 Priority 2/3): memories whose state changes over time -- latest-state recall, raw memory list vs SeCF-L1 consolidation.

Stream: 60 (person, event) pairs with 2-3 dated state updates (answer changes along the chain) + ~600 single memories (similar-episode distractors).
Conditions (latest-state question, 3-way yes/no/undecided, gemma temperature 0): A raw NF, B raw SCF-L1, C raw SeCF-L1, D consolidated SeCF-L1 note per pair
(an LLM rewrites the dated updates into one concise timeline note ending with the current state), E oracle (latest memory only, NF).
Retrieval: per-representation Hybrid (BM25 + qwen3-embedding, RRF), top-6, shown oldest first with the date.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
import statistics as st
import sys
import uuid
from collections import Counter
from typing import Dict, List

import numpy as np

from . import jl1, jlvar, jpdata
from .jpbench import RES
from .jpchat import secf1_text
from .jpdual import ranks_of_scores, rrf
from .jpllm import INSTR, SYSTEM, score
from .jpmem import EMB, lms
from .jprag import BM25, embed
from .lmstudio import chat

OUT = RES.parent / "jp4"
SEED, N_POOL, N_UPD, K = 62, 640, 60, 6
CHAINS = [("UNDECIDED", "DONE"), ("HEARSAY", "DONE"), ("POSSIBLE", "DONE"), ("UNDECIDED", "PLANNED"), ("WANTED", "DONE"), ("PLANNED", "DONE"), ("PLANNED", "UNDECIDED"),
          ("UNDECIDED", "PLANNED", "DONE"), ("HEARSAY", "UNDECIDED", "DONE"), ("WANTED", "PLANNED", "DONE")]
CONSOL_SYSTEM = """同じ人物・同じ出来事についての、日付つきの記憶メモを、1 つの簡潔な経緯メモに統合します(SeCF-L1)。出力は経緯メモのみ(1〜2 文)。
- 日付の古い順に状態の変化を書き、最後に「現在: …」と最新の状態を明記する。
- 迷い・伝聞・可能性・予定・願望・実行済み・していない、などの状態を落とさない。助詞と活用は残して普通の日本語にする。
例:
入力:
[2026-02-03] 田中さんは会社を辞めようか迷っているらしい。まだ決めたわけではないみたい。
[2026-04-10] 田中さんは来月会社を辞める予定です。
[2026-06-01] 田中さん、この前会社を辞めたよ。
出力: 田中さんの退職: 2/3 迷っていた → 4/10 来月辞める予定 → 6/1 辞めた。現在: 退職済み。"""


def build():
    rng = random.Random(SEED)
    pool = jpdata.make_memories(N_POOL, seed=SEED)
    base = dt.date(2026, 1, 1)
    items, upd = [], []
    for k in range(N_UPD):
        m = pool[k]
        ev = jpdata.event_of(m.event)
        chain = CHAINS[k % len(CHAINS)]
        d = rng.randint(0, 120)
        steps = []
        for j, status in enumerate(chain):
            text = jpdata.render(rng, m.person, ev, status, rng.random() < 0.3, False, False)
            steps.append(dict(id=len(items), person=m.person, event=m.event, status=status, text=text, date=str(base + dt.timedelta(days=d)), pair=k, step=j))
            items.append(steps[-1])
            d += rng.randint(20, 60)
        upd.append(dict(pair=k, person=m.person, event=m.event, steps=steps, gold=jpdata.ANSWER[chain[-1]], stale={jpdata.ANSWER[s] for s in chain[:-1]} - {jpdata.ANSWER[chain[-1]]},
                        query=jpdata.make_queries([m], 1, seed=k)[0].text))
    for m in pool[N_UPD:]:
        items.append(dict(id=len(items), person=m.person, event=m.event, status=m.status, text=m.text, date=str(base + dt.timedelta(days=rng.randint(0, 300))), pair=None, step=0))
    return items, upd


def consolidate(steps: List[Dict], model: str, cache: Dict[str, str]) -> str:
    inp = "\n".join(f"[{s['date']}] {s['text']}" for s in steps)
    if inp in cache:
        return cache[inp]
    out = ""
    for mt in (200, 400, 400):
        try:
            r = chat(model, CONSOL_SYSTEM, f"入力:\n{inp}\n出力:", max_tokens=mt, extra={"reasoning_effort": "none"}, deadline=120, timeout=140)
        except ValueError:
            continue
        out = re.sub(r"^出力[:：]\s*", "", r.text.strip().split("\n")[0].strip())
        if out:
            break
    cache[inp] = out or inp.replace("\n", " ")
    with (OUT / "consol_cache.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(inp=inp, out=cache[inp], fallback=not out), ensure_ascii=False) + "\n")
    return cache[inp]


def load_consol() -> Dict[str, str]:
    p = OUT / "consol_cache.jsonl"
    return {r["inp"]: r["out"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())} if p.exists() else {}


def prep(model: str, log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items, upd = build()
    n = 0
    for it in items:
        secf1_text(it["text"])
        n += 1
        if n % 100 == 0:
            log(f"secf1 {n}/{len(items)}")
    for u in upd:
        secf1_text(u["query"])
    cache = load_consol()
    for u in upd:
        consolidate(u["steps"], model, cache)
    log("consolidated")


def docs_by_condition(items, upd, cache):
    """Document texts (retrieval + feeding) per condition. Raw conditions index every memory; D replaces each update chain with one consolidated note."""
    conv = jl1.Converter("sudachi", "m")
    D = {"A": [(it["date"], it["text"]) for it in items], "B": [(it["date"], conv(it["text"])) for it in items], "C": [(it["date"], secf1_text(it["text"])) for it in items]}
    chain_ids = {s["id"] for u in upd for s in u["steps"]}
    notes = [(u["steps"][-1]["date"], consolidate(u["steps"], "", cache)) for u in upd]
    D["D"] = [(it["date"], secf1_text(it["text"])) for it in items if it["id"] not in chain_ids] + notes
    return D


def retrieval_and_qa(model: str, log=print) -> None:
    items, upd = build()
    cache = load_consol()
    D = docs_by_condition(items, upd, cache)
    chain_latest = {u["pair"]: u["steps"][-1]["id"] for u in upd}
    sud = jlvar.sud()
    tok_nf = lambda t: [x.surface for x in sud.analyze(t) if x.pos not in ("補助記号", "空白")]
    qrep = {"A": lambda u: u["query"], "B": lambda u: jl1.Converter("sudachi", "m")(u["query"]), "C": lambda u: secf1_text(u["query"]), "D": lambda u: secf1_text(u["query"])}
    conv = jl1.Converter("sudachi", "m")
    qrep["B"] = lambda u: conv(u["query"])
    tk = {"A": tok_nf, "B": lambda t: t.split(), "C": tok_nf, "D": tok_nf}
    lms(EMB)
    emb_d = {c: embed(EMB, [t for _, t in D[c]]) for c in "ABCD"}
    bm = {c: BM25([tk[c](t) for _, t in D[c]]) for c in "ABCD"}
    # index -> (condition-specific) id of the relevant document for the latest state
    latest_idx = {}
    for c in "ABCD":
        texts = [t for _, t in D[c]]
        for u in upd:
            if c == "D":
                latest_idx[(c, u["pair"])] = len(texts) - N_UPD + u["pair"]
            else:
                latest_idx[(c, u["pair"])] = chain_latest[u["pair"]]
    tops = {}
    for c in "ABCD":
        qv = embed(EMB, [qrep[c](u) for u in upd])
        for i, u in enumerate(upd):
            s_b = bm[c].scores(tk[c](qrep[c](u)))
            s_v = emb_d[c] @ qv[i]
            sc = rrf(ranks_of_scores(s_b), ranks_of_scores(s_v))
            tops[(c, i)] = [int(x) for x in np.argsort(-sc, kind="stable")[:K]]
    lms(model, 20000)
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    path = OUT / f"upd_{model.replace('/', '_')}.jsonl"
    done = {(r["cond"], r["pair"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    instr = INSTR + "記憶メモには日付があり、同じ人物・出来事について複数あるときは、日付が最新の状態を答えてください。"
    for c in ("A", "B", "C", "D", "E"):
        for i, u in enumerate(upd):
            if (c, u["pair"]) in done:
                continue
            if c == "E":
                ctx_items = [(u["steps"][-1]["date"], u["steps"][-1]["text"])]
                hit, nupd = True, 1
            else:
                idxs = tops[(c, i)]
                ctx_items = sorted((D[c][j] for j in idxs), key=lambda x: x[0])
                hit = latest_idx[(c, u["pair"])] in idxs
                nupd = sum(1 for s in u["steps"] if c != "D" and s["id"] in idxs) if c != "D" else 1
            ctx = "\n".join(f"[{d}] {t}" for d, t in ctx_items)
            r = chat(model, f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}", f"記憶メモ:\n{ctx}\n\n質問: {u['query']}\n{instr}", base_url="http://localhost:1234/v1", extra=extra, deadline=240)
            m = re.match(r"\s*[「『]?(はい|いいえ|未確定|不明)", r.text)
            pred = m.group(1) if m else None
            rec = dict(cond=c, pair=u["pair"], gold=u["gold"], pred=pred, ok=pred == u["gold"], stale=pred in u["stale"], hit=hit, n_updates_in_context=nupd, steps=len(u["steps"]),
                       prompt_tokens=r.prompt_tokens, ttft=r.ttft, answer=r.text[:30])
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"cond {c} done")


def report() -> None:
    from .jpchatreport import boot_ci, perm_p
    rows = [json.loads(l) for l in (OUT / "upd_google_gemma-4-12b.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["cond"], {})[r["pair"]] = r
    cons = [json.loads(l) for l in (OUT / "consol_cache.jsonl").read_text(encoding="utf-8").splitlines()]
    names = {"A": "A 生の記憶(NF)", "B": "B 生の記憶(SCF-L1)", "C": "C 生の記憶(SeCF-L1)", "D": "D 統合メモ(SeCF-L1)", "E": "E 上限(最新の記憶のみ・NF)"}
    L = ["# Issue #62 状態が変わる記憶: 最新状態の取り出しと stale 回答\n", f"60 組(2〜3 回更新)+ 単発 約 580 件、検索 top-{K}(各表現の Hybrid)、gemma 温度 0。\n",
         "| 条件 | 正答率 | stale 率 | 最新が検索に入った率 | 文脈内の更新数(平均) | プロンプトtok | vs A 差[95%CI], p | vs D 差[95%CI], p |", "|---|---|---|---|---|---|---|---|"]
    A = {p: float(r["ok"]) for p, r in by["A"].items()}
    Dd = {p: float(r["ok"]) for p, r in by["D"].items()}
    for c in "ABCDE":
        if c not in by:
            continue
        v = by[c]
        cell = []
        for ref in (A, Dd):
            d = [float(v[p]["ok"]) - ref[p] for p in v]
            lo, hi = boot_ci(d)
            cell.append("-" if (c == "A" and ref is A) or (c == "D" and ref is Dd) else f"{np.mean(d) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}], p={perm_p(d):.3f}")
        L.append(f"| {names[c]} | {np.mean([r['ok'] for r in v.values()]):.3f} | {np.mean([r['stale'] for r in v.values()]):.3f} | {np.mean([r['hit'] for r in v.values()]):.2f} | "
                 f"{np.mean([r['n_updates_in_context'] for r in v.values()]):.2f} | {st.mean(r['prompt_tokens'] for r in v.values()):.0f} | {cell[0]} | {cell[1]} |")
    L += ["\n## 更新回数別の正答率(2 回更新 / 3 回更新)\n", "| 条件 | 2 ステップ | 3 ステップ |", "|---|---|---|"]
    for c in "ABCDE":
        if c in by:
            L.append(f"| {names[c]} | " + " | ".join(f"{np.mean([r['ok'] for r in by[c].values() if r['steps'] == s]):.3f}" for s in (2, 3)) + " |")
    L += ["\n## 誤答の内訳(条件、正解 → 回答)\n"]
    for c in "ABCD":
        if c in by:
            conf = Counter((r["gold"], r["pred"]) for r in by[c].values() if not r["ok"])
            L.append(f"- {names[c]}: " + ", ".join(f"{g}→{p}:{n}" for (g, p), n in conf.most_common(5)))
    L += ["\n## 統合のコスト\n", f"統合 {len(cons)} 組、フォールバック {sum(1 for x in cons if x['fallback'])} 組。例:\n"] + [f"- {x['out']}" for x in cons[:5]]
    (OUT / "upd_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep":
        prep(sys.argv[2], log=lambda s: print(s, flush=True))
    elif mode == "run":
        retrieval_and_qa(sys.argv[2], log=lambda s: print(s, flush=True))
    elif mode == "report":
        report()
