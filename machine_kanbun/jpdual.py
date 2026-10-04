"""Issue #28: L0/L1 dual-index retrieval -- BM25 over L1 (lexical) + dense vectors over L0 (semantic), fused.

Documents are stored in both views (L0 text, simple-L1 text). Retrieval strategies compared on the same queries:
  single index:  bm25_L0  bm25_L1  vec_L0  vec_L1 (query also L1-converted)
  fusion:        hybrid_L0 = RRF(bm25_L0, vec_L0)    dual = RRF(bm25_L1, vec_L0)    hybrid_L1 = RRF(bm25_L1, vec_L1)
                 quad = RRF of all four              weighted (z-score, weight picked on dev queries)
                 rerank = bm25_L1 top-50 re-ordered by vec_L0 (an embedding re-ranker; no cross-encoder is available offline)
  two-view query: vec_L0_2view = (q_raw + q_L1) embedding against L0 documents
Embedding models: nomic-embed-text-v1.5 and qwen3-embedding-0.6b (LM Studio). Embeddings are cached in results/jp2/cache.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jl1, jpdata, jlvar
from .jpbench import RES
from .jpllm import INSTR, SYSTEM, score
from .jprag import BM25, embed, metrics, rank_of
from .lmstudio import chat

OUT = RES.parent / "jp2"
CACHE = OUT / "cache"
MODELS = ["text-embedding-nomic-embed-text-v1.5", "text-embedding-qwen3-embedding-0.6b"]
SEED = 28
RRF_K = 60


def _emb(model: str, tag: str, texts: List[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{model.split('-')[2]}_{tag}_{len(texts)}.npy"
    if f.exists():
        return np.load(f)
    t0 = time.perf_counter()
    a = embed(model, texts)
    np.save(f, a)
    (CACHE / (f.name + ".sec")).write_text(str(time.perf_counter() - t0))
    return a


def lms_load(model: str) -> None:
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    subprocess.run(["lms", "load", model, "-y"], capture_output=True)


def rrf(*rank_lists: np.ndarray) -> np.ndarray:
    """Reciprocal rank fusion over per-document ranks (1 = best) -> fused score."""
    return sum(1.0 / (RRF_K + r) for r in rank_lists)


def ranks_of_scores(s: np.ndarray) -> np.ndarray:
    order = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.int32)
    r[order] = np.arange(1, len(s) + 1)
    return r


def znorm(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-9)


class Index:
    def __init__(self, n: int):
        self.mems = jpdata.make_memories(n, seed=SEED)
        self.l0 = [m.text for m in self.mems]
        self.conv = jl1.Converter("sudachi", "m")
        t0 = time.perf_counter()
        self.l1 = [self.conv(t) for t in self.l0]
        self.convert_s = time.perf_counter() - t0
        sud = jlvar.sud()
        self.tok0 = lambda t: [x.surface for x in sud.analyze(t) if x.pos not in ("補助記号", "空白")]
        self.bm0 = BM25([self.tok0(t) for t in self.l0])
        self.bm1 = BM25([t.split() for t in self.l1])
        self.d = {}

    def load(self, model: str) -> None:
        """Only this embedding model resident in LM Studio (clean timings); document vectors are cached on disk."""
        lms_load(model)
        if model not in self.d:
            self.d[model] = {"L0": _emb(model, "L0", self.l0), "L1": _emb(model, "L1", self.l1)}

    def stats(self) -> Dict:
        postings = lambda bm: sum(len(c) for c in bm.tf)
        return dict(n=len(self.l0), bytes_L0=sum(len(t.encode()) for t in self.l0), bytes_L1=sum(len(t.encode()) for t in self.l1),
                    bm25_postings_L0=postings(self.bm0), bm25_postings_L1=postings(self.bm1), bm25_terms_L0=len(self.bm0.idf), bm25_terms_L1=len(self.bm1.idf),
                    convert_s=self.convert_s, vector_bytes={m: int(v["L0"].nbytes) for m, v in self.d.items()})


def queries(idx: Index, n: int, seed: int, pool: int):
    qs = jpdata.make_queries(idx.mems[:pool], n, seed=seed)
    raw = [q.text for q in qs]
    return qs, raw, [idx.conv(t) for t in raw]


def score_all(idx: Index, model: str, N: int, raw: List[str], l1q: List[str]):
    """Per-query score vectors (length N) for the five base signals, plus timing."""
    t0 = time.perf_counter()
    q0 = embed(model, raw)
    emb_ms = (time.perf_counter() - t0) / len(raw) * 1000
    q1 = embed(model, l1q)
    d0, d1 = idx.d[model]["L0"][:N], idx.d[model]["L1"][:N]
    sub0, sub1 = idx.bm0, idx.bm1
    sig = []
    t_bm0 = t_bm1 = t_vec = 0.0
    for i in range(len(raw)):
        a = time.perf_counter()
        b0 = sub0.scores(idx.tok0(raw[i]))[:N]
        t_bm0 += time.perf_counter() - a
        a = time.perf_counter()
        b1 = sub1.scores(l1q[i].split())[:N]
        t_bm1 += time.perf_counter() - a
        a = time.perf_counter()
        v0 = d0 @ q0[i]
        t_vec += time.perf_counter() - a
        v1 = d1 @ q1[i]
        q2 = q0[i] + q1[i]
        v2 = d0 @ (q2 / np.linalg.norm(q2))
        sig.append(dict(bm25_L0=b0, bm25_L1=b1, vec_L0=v0, vec_L1=v1, vec_L0_2view=v2))
    n = len(raw)
    return sig, dict(bm25_L0_ms=t_bm0 / n * 1000, bm25_L1_ms=t_bm1 / n * 1000, vec_ms=t_vec / n * 1000, query_embed_ms=emb_ms)


def strategies(sig: Dict[str, np.ndarray], w: float) -> Dict[str, np.ndarray]:
    rk = {k: ranks_of_scores(v) for k, v in sig.items()}
    out = {k: v for k, v in sig.items()}
    out["hybrid_L0"] = rrf(rk["bm25_L0"], rk["vec_L0"])
    out["dual"] = rrf(rk["bm25_L1"], rk["vec_L0"])
    out["hybrid_L1"] = rrf(rk["bm25_L1"], rk["vec_L1"])
    out["quad"] = rrf(rk["bm25_L0"], rk["bm25_L1"], rk["vec_L0"], rk["vec_L1"])
    out["dual_2view"] = rrf(rk["bm25_L1"], rk["vec_L0_2view"])
    out["weighted_dual"] = w * znorm(sig["bm25_L1"]) + (1 - w) * znorm(sig["vec_L0"])
    cand = np.argsort(-sig["bm25_L1"], kind="stable")[:50]
    rr = np.full(len(sig["vec_L0"]), -1e9, dtype=np.float32)
    rr[cand] = sig["vec_L0"][cand]
    out["rerank_bm25L1_by_vecL0"] = rr
    return out


def pick_weight(idx: Index, model: str, N: int, nq: int = 150) -> float:
    qs, raw, l1q = queries(idx, nq, seed=SEED + 1000 + N, pool=N)
    sig, _ = score_all(idx, model, N, raw, l1q)
    best, bw = -1, 0.5
    for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        mrr = np.mean([1 / rank_of(w * znorm(s["bm25_L1"]) + (1 - w) * znorm(s["vec_L0"]), q.relevant) for s, q in zip(sig, qs)])
        if mrr > best:
            best, bw = mrr, w
    return bw


def run(scales=(1000, 10000), nq: int = 300) -> Dict:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = Index(max(scales))
    res = {"stats": idx.stats(), "scales": list(scales), "nq": nq, "results": {}, "weights": {}, "timing": {}}
    for model in MODELS:
        idx.load(model)
        for N in scales:
            w = pick_weight(idx, model, N)
            res["weights"][f"{model}|{N}"] = w
            qs, raw, l1q = queries(idx, nq, seed=SEED + N, pool=N)
            sig, tm = score_all(idx, model, N, raw, l1q)
            res["timing"][f"{model}|{N}"] = tm
            per: Dict[str, List[int]] = {}
            for s, q in zip(sig, qs):
                for name, sc in strategies(s, w).items():
                    per.setdefault(name, []).append(rank_of(sc, q.relevant))
            status = [idx.mems[q.relevant].status for q in qs]
            for name, ranks in per.items():
                grp = {}
                for st in jpdata.STATUSES:
                    rs = [r for r, s_ in zip(ranks, status) if s_ == st]
                    grp[st] = sum(r <= 5 for r in rs) / max(1, len(rs))
                res["results"][f"{model}|{N}|{name}"] = dict(metrics(ranks), recall5_by_status=grp, ranks=ranks)
            print(f"{model[-24:]} N={N} done (w={w})", flush=True)
    (OUT / "dual_index.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


STRATS = ["bm25_L0", "bm25_L1", "vec_L0", "vec_L1", "hybrid_L0", "dual", "dual_2view", "hybrid_L1", "quad", "weighted_dual", "rerank_bm25L1_by_vecL0"]


def run_qa(model: str, N: int = 10000, nq: int = 60, k: int = 5, emb_model: str = MODELS[1], base_url="http://localhost:1234/v1") -> None:
    """End-to-end: strategy top-k -> LLM, feeding the original L0 text or the L1 text of the retrieved memories."""
    import uuid
    idx = Index(N)
    w = json.loads((OUT / "dual_index.json").read_text())["weights"][f"{emb_model}|{N}"]
    qs, raw, l1q = queries(idx, nq, seed=SEED + 77, pool=N)
    idx.load(emb_model)
    sig, _ = score_all(idx, emb_model, N, raw, l1q)
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    subprocess.run(["lms", "load", model, "-c", "20000", "--parallel", "1", "-y"], capture_output=True)
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    path = OUT / f"dual_qa_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["strategy"], r["feed"]))
    for name in ["bm25_L0", "bm25_L1", "vec_L0", "hybrid_L0", "dual", "quad"]:
        for feed in ("L0", "L1"):
            if (name, feed) in done:
                continue
            rows = []
            system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}"
            for i, (s, q) in enumerate(zip(sig, qs)):
                sc = strategies(s, w)[name]
                top = list(np.argsort(-sc, kind="stable")[:k])
                ctx = "\n".join((idx.l0 if feed == "L0" else idx.l1)[j] for j in top)
                r = chat(model, system, f"記憶メモ:\n{ctx}\n\n質問: {q.text}\n{INSTR}{nt}", base_url=base_url, extra=extra, deadline=240)
                rows.append(dict(model=model, strategy=name, feed=feed, i=i, gold=q.answer, answer=r.text, ok=score(r.text, q.answer), hit=q.relevant in set(int(x) for x in top),
                                 prompt_tokens=r.prompt_tokens, ttft=r.ttft, total=r.total))
            with path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"QA {model} {name} feed={feed}: acc={sum(r['ok'] for r in rows) / len(rows):.1%} hit={sum(r['hit'] for r in rows) / len(rows):.0%} "
                  f"prompt={sum(r['prompt_tokens'] for r in rows) / len(rows):.0f}tok", flush=True)


if __name__ == "__main__":
    run()
