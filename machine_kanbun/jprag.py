"""Issue #20 experiment 3: retrieval over memories stored as L0 or simple L1 (BM25 + dense embeddings).

Memory sets are nested prefixes of one generated list (so embeddings are computed once per representation).
Queries are natural-language status questions ("田中花子さんは会社を辞めた？"); each has exactly one relevant memory.
The query is embedded raw and, as a second condition, after the same L1 conversion as the memories.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from . import jl1, jpdata
from .jpbench import RES

EMB_URL = "http://localhost:1234/v1/embeddings"
SCALES = [10, 100, 1000, 5000, 10000]
REPS = [("sudachi", "m"), ("sudachi", "g"), ("mecab", "m"), ("sudachi", "p"), ("ginza", "d"), ("naive", "m")]


def embed(model: str, texts: List[str], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": model, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(EMB_URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out += [d["embedding"] for d in json.load(r)["data"]]
    a = np.asarray(out, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.inv: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w in c:
                self.inv[w].append(i)

    def scores(self, q: List[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for w in set(q):
            if w not in self.idf:
                continue
            for i in self.inv[w]:
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def metrics(ranks: List[int]) -> Dict[str, float]:
    n = len(ranks)
    return dict(recall1=sum(r <= 1 for r in ranks) / n, recall3=sum(r <= 3 for r in ranks) / n, recall5=sum(r <= 5 for r in ranks) / n,
                recall10=sum(r <= 10 for r in ranks) / n, mrr=sum(1 / r for r in ranks) / n,
                ndcg10=sum((1 / math.log2(r + 1)) if r <= 10 else 0 for r in ranks) / n)


def rank_of(scores: np.ndarray, rel: int) -> int:
    return int((scores > scores[rel]).sum()) + 1


def run(models=("text-embedding-nomic-embed-text-v1.5", "text-embedding-qwen3-embedding-0.6b"), scales=SCALES, nq: int = 200, seed: int = 11) -> Dict:
    mems = jpdata.make_memories(max(scales), seed=seed)
    sud = jl1.make("sudachi")
    tokz = lambda t: [x.surface for x in sud.analyze(t) if x.pos not in ("補助記号", "空白")]
    out: Dict = {"scales": scales, "nq": nq, "results": {}}
    reps: Dict[str, List[str]] = {"L0": [m.text for m in mems]}
    conv_cost = {}
    for an, var in REPS:
        c = jl1.Converter(an, var)
        t0 = time.perf_counter()
        reps[c.name] = [c(m.text) for m in mems]
        conv_cost[c.name] = (time.perf_counter() - t0) / len(mems) * 1000
        reps[c.name + "|conv"] = c  # converter kept to convert queries
    convs = {k.split("|")[0]: v for k, v in reps.items() if k.endswith("|conv")}
    reps = {k: v for k, v in reps.items() if not k.endswith("|conv")}
    # embeddings once per (model, representation) over all memories
    emb: Dict[tuple, np.ndarray] = {}
    emb_s: Dict[tuple, float] = {}
    for model in models:
        for name, texts in reps.items():
            t0 = time.perf_counter()
            emb[(model, name)] = embed(model, texts)
            emb_s[(model, name)] = time.perf_counter() - t0
            print(f"embedded {model[-24:]} {name:12} {emb_s[(model, name)]:6.1f}s", flush=True)
    for N in scales:
        sub = mems[:N]
        qs = jpdata.make_queries(sub, min(nq, N), seed=seed + N)
        qtexts = [q.text for q in qs]
        status = [next(m.status for m in sub if m.id == q.relevant) for q in qs]
        meta = [(sub[q.relevant].features["omit_subject"], sub[q.relevant].features["pronoun"], sub[q.relevant].register) for q in qs]
        for name in reps:
            bm_docs = BM25([tokz(t) if name != "L0" and " " not in t else (t.split() if name != "L0" else tokz(t)) for t in reps[name][:N]])
            for qmode in ("raw", "l1"):
                if name == "L0" and qmode == "l1":
                    continue
                qt = qtexts if qmode == "raw" else [convs[name](q) for q in qtexts]
                ranks_bm = []
                t0 = time.perf_counter()
                for q, qq in zip(qs, qt):
                    toks = tokz(qq) if (qmode == "raw" or name in ("sudachi-g",)) else qq.split()
                    ranks_bm.append(rank_of(bm_docs.scores(toks), q.relevant))
                bm_ms = (time.perf_counter() - t0) / len(qs) * 1000
                entry = {"bm25": dict(metrics(ranks_bm), ms=bm_ms)}
                allr = {"bm25": ranks_bm}
                for model in models:
                    D = emb[(model, name)][:N]
                    t0 = time.perf_counter()
                    qe = embed(model, qt)
                    qe_s = (time.perf_counter() - t0) / len(qs)
                    t0 = time.perf_counter()
                    sims = qe @ D.T
                    ranks = [rank_of(sims[i], q.relevant) for i, q in enumerate(qs)]
                    ret_ms = (time.perf_counter() - t0) / len(qs) * 1000
                    entry[model] = dict(metrics(ranks), retrieve_ms=ret_ms, query_embed_ms=qe_s * 1000)
                    allr[model] = ranks
                # by memory status / form (recall@5), embedding model 1 and bm25
                grp = {}
                for key, ranks in allr.items():
                    g = defaultdict(list)
                    for r, s_, m_ in zip(ranks, status, meta):
                        g[s_].append(r <= 5)
                        g["omit_subject" if m_[0] else "has_subject"].append(r <= 5)
                        if m_[1]:
                            g["pronoun"].append(r <= 5)
                        g[m_[2]].append(r <= 5)
                    grp[key] = {k: sum(v) / len(v) for k, v in g.items()}
                entry["by_group_recall5"] = grp
                out["results"][f"{N}|{name}|{qmode}"] = entry
        print(f"scale {N} done", flush=True)
    out["embed_seconds"] = {f"{m}|{n}": s for (m, n), s in emb_s.items()}
    out["embed_texts_per_s"] = {f"{m}|{n}": len(mems) / s for (m, n), s in emb_s.items()}
    out["convert_ms_per_text"] = conv_cost
    out["avg_chars"] = {n: sum(len(t) for t in v) / len(v) for n, v in reps.items()}
    return out


def main(**kw):
    RES.mkdir(parents=True, exist_ok=True)
    res = run(**kw)
    (RES / "rag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
