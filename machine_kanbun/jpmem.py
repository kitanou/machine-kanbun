"""Issue #59 (Meta #56 Priority 2): SeCF-L1 as a long-term memory representation vs NF vs SCF-L1.

Memories: jpdata benchmark (similar-episode distractors: same event / other person, same person / other event), N=1000. Representations: NF, SCF-L1 (sudachi-m),
SeCF-L1 (gemma rewrite via jpchat.secf1_text; queries rewritten the same way). Retrieval: BM25 and dense (qwen3-embedding) per representation, RRF fusions, and
the top-1 confusion breakdown. Answering: top-5 memories fed in each representation to gemma (3-way yes/no/undecided, deterministic scoring).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jl1, jlvar, jpdata
from .jpbench import RES
from .jpchat import secf1_text
from .jpdual import ranks_of_scores, rrf
from .jpllm import INSTR, SYSTEM, score
from .jprag import BM25, embed, metrics, rank_of
from .lmstudio import chat

OUT = RES.parent / "jp4"
EMB = "text-embedding-qwen3-embedding-0.6b"
N, NQ, NQA, SEED = 1000, 150, 60, 28
REPS = ("nf", "scf", "secf1")


def lms(model: str, ctx: int = 0):
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    subprocess.run(["lms", "load", model] + (["-c", str(ctx), "--parallel", "1"] if ctx else []) + ["-y"], capture_output=True)


def data():
    mems = jpdata.make_memories(N, seed=SEED)
    qs = jpdata.make_queries(mems, NQ, seed=SEED + 5)
    return mems, qs


def prep(log=print) -> None:
    """SeCF-L1 of every memory text and every query (gemma, cached); timing recorded."""
    OUT.mkdir(parents=True, exist_ok=True)
    mems, qs = data()
    t0 = time.perf_counter()
    n = 0
    for text in [m.text for m in mems] + [q.text for q in qs]:
        secf1_text(text)
        n += 1
        if n % 100 == 0:
            log(f"secf1 {n}/{len(mems) + len(qs)}")
    (OUT / "prep_secf1_seconds.json").write_text(json.dumps(dict(n=n, seconds=time.perf_counter() - t0)))


def reps_of(texts: List[str]) -> Dict[str, List[str]]:
    conv = jl1.Converter("sudachi", "m")
    return {"nf": list(texts), "scf": [conv(t) for t in texts], "secf1": [secf1_text(t) for t in texts]}


def retrieval(log=print) -> Dict:
    mems, qs = data()
    D = reps_of([m.text for m in mems])
    Q = reps_of([q.text for q in qs])
    sud = jlvar.sud()
    tok = {"nf": lambda t: [x.surface for x in sud.analyze(t) if x.pos not in ("補助記号", "空白")], "scf": lambda t: t.split(),
           "secf1": lambda t: [x.surface for x in sud.analyze(t) if x.pos not in ("補助記号", "空白")]}
    bm = {r: BM25([tok[r](t) for t in D[r]]) for r in REPS}
    lms(EMB)
    emb_d = {r: embed(EMB, D[r]) for r in REPS}
    emb_q = {r: embed(EMB, Q[r]) for r in REPS}
    sigs = []
    for i in range(len(qs)):
        s = {}
        for r in REPS:
            s[f"bm25_{r}"] = bm[r].scores(tok[r](Q[r][i]))
            s[f"vec_{r}"] = emb_d[r] @ emb_q[r][i]
        rk = {k: ranks_of_scores(v) for k, v in s.items()}
        for r in ("scf", "secf1"):
            s[f"dual_{r}"] = rrf(rk[f"bm25_{r}"], rk["vec_nf"])      # lexical side in representation r, dense side on NF
        for r in REPS:
            s[f"hybrid_{r}"] = rrf(rk[f"bm25_{r}"], rk[f"vec_{r}"])  # both sides in representation r
        sigs.append(s)
    names = list(sigs[0])
    out = {}
    for k in names:
        ranks = [rank_of(s[k], q.relevant) for s, q in zip(sigs, qs)]
        conf = Counter()
        for s, q in zip(sigs, qs):
            top = int(np.argmax(s[k]))
            if top == q.relevant:
                conf["ok"] += 1
            else:
                m = mems[top]
                conf["same_event_other_person" if (m.event == q.event and m.person != q.person) else "same_person_other_event" if (m.person == q.person and m.event != q.event) else "other"] += 1
        out[k] = dict(metrics(ranks), ranks=ranks, confusion={c: conf[c] / len(qs) for c in ("ok", "same_event_other_person", "same_person_other_event", "other")})
    stats = {r: dict(bytes=sum(len(t.encode()) for t in D[r]), postings=sum(len(c) for c in bm[r].tf)) for r in REPS}
    (OUT / "retrieval.json").write_text(json.dumps(dict(n=N, nq=NQ, metrics=out, stats=stats), ensure_ascii=False, indent=1))
    np.save(OUT / "top5.npy", np.array([[np.argsort(-s[k], kind="stable")[:5] for k in names] for s in sigs]))
    (OUT / "top5_names.json").write_text(json.dumps(names))
    for k in names:
        print(f"{k:14s} R@1={out[k]['recall1']:.3f} R@5={out[k]['recall5']:.3f} MRR={out[k]['mrr']:.3f} conf={ {c: round(v, 3) for c, v in out[k]['confusion'].items()} }", flush=True)
    return out


def qa(model: str, log=print) -> None:
    mems, qs = data()
    D = reps_of([m.text for m in mems])
    top5 = np.load(OUT / "top5.npy")
    names = json.loads((OUT / "top5_names.json").read_text())
    lms(model, 20000)
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    path = OUT / f"qa_{model.replace('/', '_')}.jsonl"
    done = {(r["retrieval"], r["feed"], r["i"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    # (retrieval strategy, feed representation): end-to-end per representation, and a fixed retrieval (dual_scf) fed in each representation
    plan = [("hybrid_nf", "nf"), ("hybrid_scf", "scf"), ("hybrid_secf1", "secf1")] + [("dual_scf", f) for f in REPS]
    for strat, feed in plan:
        j = names.index(strat)
        rows = []
        system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}"
        for i in range(NQA):
            if (strat, feed, i) in done:
                continue
            ctx = "\n".join(D[feed][int(x)] for x in top5[i][j])
            r = chat(model, system, f"記憶メモ:\n{ctx}\n\n質問: {qs[i].text}\n{INSTR}", base_url="http://localhost:1234/v1", extra=extra, deadline=240)
            rows.append(dict(retrieval=strat, feed=feed, i=i, gold=qs[i].answer, answer=r.text, ok=score(r.text, qs[i].answer), hit=qs[i].relevant in set(int(x) for x in top5[i][j]),
                             prompt_tokens=r.prompt_tokens, ttft=r.ttft))
        with path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if rows:
            print(f"QA {strat} feed={feed}: acc={np.mean([r['ok'] for r in rows]):.3f} hit={np.mean([r['hit'] for r in rows]):.3f} prompt={np.mean([r['prompt_tokens'] for r in rows]):.0f}tok", flush=True)


def report() -> None:
    import statistics as st
    from collections import defaultdict
    from .jpchatreport import boot_ci, perm_p
    d = json.loads((OUT / "retrieval.json").read_text(encoding="utf-8"))
    M = d["metrics"]
    L = [f"# Issue #59 SeCF-L1 を Long-term Memory 表現にしたときの検索と回答(N={d['n']}、クエリ {d['nq']}、QA {NQA} 問、gemma-4-12b / qwen3-embedding-0.6b)\n",
         "## 1. 検索(R@1 は rank、top-1 の内訳は argmax(同点では R@1 より低く出る))\n", "| 戦略 | R@1 | R@5 | MRR | top-1 正解 | 同出来事・別人物 | 同人物・別出来事 | その他 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in M.items():
        c = v["confusion"]
        L.append(f"| {k} | {v['recall1']:.3f} | {v['recall5']:.3f} | {v['mrr']:.3f} | {c['ok']:.3f} | {c['same_event_other_person']:.3f} | {c['same_person_other_event']:.3f} | {c['other']:.3f} |")
    L += ["\n### R@1 のペア比較(クエリ単位、符号反転の置換検定)\n", "| 比較 | 差[95%CI] | p |", "|---|---|---|"]
    for a, b in [("vec_secf1", "vec_nf"), ("vec_secf1", "vec_scf"), ("bm25_secf1", "bm25_scf"), ("bm25_secf1", "bm25_nf"), ("dual_secf1", "dual_scf"), ("hybrid_secf1", "hybrid_nf"), ("dual_scf", "vec_nf")]:
        x = np.array(M[a]["ranks"]) <= 1
        y = np.array(M[b]["ranks"]) <= 1
        dd = list(x.astype(float) - y.astype(float))
        lo, hi = boot_ci(dd)
        L.append(f"| {a} − {b} | {np.mean(dd) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}] | {perm_p(dd):.3f} |")
    rows = [json.loads(l) for l in (OUT / "qa_google_gemma-4-12b.jsonl").read_text(encoding="utf-8").splitlines()]
    g = defaultdict(list)
    for r in rows:
        g[(r["retrieval"], r["feed"])].append(r)
    L += ["\n## 2. 検索結果を渡した回答(3 択、60 問)\n", "| 検索 | 渡す表現 | 正答率 | 取得ヒット | プロンプトtok |", "|---|---|---|---|---|"]
    for (s_, f), v in g.items():
        L.append(f"| {s_} | {f} | {np.mean([x['ok'] for x in v]):.3f} | {np.mean([x['hit'] for x in v]):.3f} | {st.mean(x['prompt_tokens'] for x in v):.0f} |")
    acc = lambda s_, f: {r["i"]: float(r["ok"]) for r in rows if r["retrieval"] == s_ and r["feed"] == f}
    L += ["\n### 回答のペア比較\n", "| 比較 | 差[95%CI] | p |", "|---|---|---|"]
    for a, b in [(("hybrid_secf1", "secf1"), ("hybrid_nf", "nf")), (("hybrid_secf1", "secf1"), ("hybrid_scf", "scf")), (("hybrid_scf", "scf"), ("hybrid_nf", "nf")),
                 (("dual_scf", "secf1"), ("dual_scf", "nf")), (("dual_scf", "scf"), ("dual_scf", "nf")), (("dual_scf", "secf1"), ("dual_scf", "scf"))]:
        A, B = acc(*a), acc(*b)
        dd = [A[i] - B[i] for i in A]
        lo, hi = boot_ci(dd)
        L.append(f"| {a[0]}/{a[1]} − {b[0]}/{b[1]} | {np.mean(dd) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}] | {perm_p(dd):.3f} |")
    cache = [json.loads(l) for l in (Path(RES.parent) / "jp3" / "secf1_cache.jsonl").read_text(encoding="utf-8").splitlines()]
    prep = json.loads((OUT / "prep_secf1_seconds.json").read_text())
    L += ["\n## 3. コスト\n", f"- SeCF-L1 変換: 1 件平均 {st.mean(x['ms'] for x in cache if x['ms']):.0f} ms(LLM、{len(cache)} 件、うち原文へのフォールバック {sum(1 for x in cache if x.get('fallback'))} 件)。SCF-L1(sudachi-m)は 1 ms 未満。",
          "- 保存バイト(1,000 件): " + ", ".join(f"{r} {v['bytes']:,}" for r, v in d["stats"].items()) + "。BM25 の postings: " + ", ".join(f"{r} {v['postings']:,}" for r, v in d["stats"].items()) + "。"]
    (OUT / "mem_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep":
        prep(log=lambda s: print(s, flush=True))
    elif mode == "retr":
        retrieval()
    elif mode == "qa":
        qa(sys.argv[2])
    elif mode == "report":
        report()
