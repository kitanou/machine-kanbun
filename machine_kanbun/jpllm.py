"""Issue #20 experiments 2 and 5: LLM end-to-end with L0 vs simple-L1 memories (full context and RAG mode).

Question: "{person}さんは{event}た？" -> the model must answer はい (done) / いいえ (not done, planned, wished, negated) /
未確定 (undecided, hearsay, possible) / 不明 (no such memory) from the memory notes only. The first question of each
(model, representation, N) is a cold request (random nonce defeats the prefix cache) = prefill time; the rest are warm.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List

from . import jl1, jpdata
from .jpbench import RES
from .lmstudio import Stalled, chat
from .sysmem import Peak

SYSTEM = "あなたは記憶メモだけを根拠に質問へ答えるアシスタントです。"
INSTR = ("上の記憶メモだけを根拠に、質問の出来事が確定しているかを答えてください。出来事が既に完了していると確定している場合は「はい」、"
         "していない・予定や希望の段階・否定されている場合は「いいえ」、迷っている・聞いた話で未確認・可能性にとどまる場合は「未確定」と、最初の語だけで答えてください。"
         "該当する記憶メモがない場合は「不明」と答えてください。")
REPS = [("L0", None), ("sudachi", "m"), ("sudachi", "g"), ("sudachi", "p"), ("ginza", "d"), ("naive", "m")]
_FIRST = re.compile(r"\s*[「『]?(はい|いいえ|未確定|不明)")


def swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def score(answer: str, gold: str) -> bool:
    m = _FIRST.match(answer)
    return bool(m) and m.group(1) == gold


def rep_name(an: str, var) -> str:
    return "L0" if an == "L0" else (f"{an}-{var}" if an != "naive" else "naive")


def contexts(n_max: int, seed: int):
    mems = jpdata.make_memories(n_max, seed=seed)
    out = {"L0": [m.text for m in mems]}
    conv_ms = {}
    for an, var in REPS[1:]:
        c = jl1.Converter(an, var)
        t0 = time.perf_counter()
        out[c.name] = [c(m.text) for m in mems]
        conv_ms[c.name] = (time.perf_counter() - t0) / n_max * 1000
    return mems, out, conv_ms


def run(model: str, ns: List[int], nq: int = 40, seed: int = 21, base_url: str = "http://localhost:1234/v1", out_dir: Path = RES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"llm_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["n"], r["rep"]))
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    mems, ctxs, conv_ms = contexts(max(ns), seed)
    for N in ns:
        sub = mems[:N]
        qs = jpdata.make_queries(sub, min(nq, N), seed=seed + N)
        for name, texts in ctxs.items():
            if (N, name) in done:
                continue
            ctx = "\n".join(texts[:N])
            system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}"
            rows = []
            try:
                base = chat(model, system, f"記憶メモ:\n\n\n質問: x\n{INSTR}{nt}", max_tokens=5, base_url=base_url, extra=extra, deadline=240)
                for i, q in enumerate(qs):
                    prompt = f"記憶メモ:\n{ctx}\n\n質問: {q.text}\n{INSTR}{nt}"
                    if i == 0:
                        s0 = swap_mb()
                        with Peak() as pk:
                            r = chat(model, system, prompt, base_url=base_url, extra=extra, deadline=1800, timeout=1800)
                        extra_fields = dict(base_tokens=base.prompt_tokens, used_delta_mb=pk.peak["used_mb"] - pk.base["used_mb"],
                                            rss_delta_mb=pk.peak["rss_mb"] - pk.base["rss_mb"], swap_before_mb=s0, swap_after_mb=swap_mb())
                    else:
                        r = chat(model, system, prompt, base_url=base_url, extra=extra, deadline=240)
                        extra_fields = {}
                    rows.append(dict(model=model, n=N, rep=name, i=i, cold=(i == 0), q=q.text, gold=q.answer, answer=r.text, ok=score(r.text, q.answer),
                                     prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, ttft=r.ttft, total=r.total,
                                     ctx_chars=len(ctx), conv_ms_per_memory=conv_ms.get(name, 0.0), **extra_fields))
            except Stalled:
                raise
            except Exception as e:  # context overflow etc.
                rows = [dict(model=model, n=N, rep=name, error=f"{type(e).__name__}: {e}"[:300])]
                print(f"ERROR {model} N={N} {name}: {rows[0]['error'][:160]}", flush=True)
            with path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if "error" not in rows[0]:
                print(f"{model} N={N} {name}: acc={sum(r['ok'] for r in rows) / len(rows):.1%} ctx={rows[0]['prompt_tokens'] - rows[0]['base_tokens']}tok "
                      f"cold_ttft={rows[0]['ttft']:.1f}s", flush=True)


def run_rag(model: str, ns: List[int], k: int = 5, nq: int = 40, seed: int = 21, base_url: str = "http://localhost:1234/v1", out_dir: Path = RES) -> None:
    """RAG mode: BM25 top-k memories (stored as L0 or L1) -> LLM. End-to-end = retrieval + LLM (conversion happened at write time)."""
    from .jprag import BM25
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rag_llm_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["n"], r["rep"]))
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    mems, ctxs, conv_ms = contexts(max(ns), seed)
    sud = jl1.make("sudachi")
    tokz = lambda t: [x.surface for x in sud.analyze(t) if x.pos not in ("補助記号", "空白")]
    convs = {rep_name(a, v): jl1.Converter(a, v) for a, v in REPS[1:]}
    for N in ns:
        sub = mems[:N]
        qs = jpdata.make_queries(sub, min(nq, N), seed=seed + N)
        for name in ("L0", "sudachi-m", "sudachi-g", "ginza-d", "naive"):
            if (N, name) in done:
                continue
            texts = ctxs[name][:N]
            bm = BM25([tokz(t) if name in ("L0", "sudachi-g") else t.split() for t in texts])
            system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}"
            rows = []
            for i, q in enumerate(qs):
                qq = q.text if name == "L0" else convs[name](q.text)
                t0 = time.perf_counter()
                sc = bm.scores(tokz(qq) if name in ("L0", "sudachi-g") else qq.split())
                top = sc.argsort()[::-1][:k]
                ret_ms = (time.perf_counter() - t0) * 1000
                ctx = "\n".join(texts[j] for j in top)
                r = chat(model, system, f"記憶メモ:\n{ctx}\n\n質問: {q.text}\n{INSTR}{nt}", base_url=base_url, extra=extra, deadline=240)
                rows.append(dict(model=model, n=N, rep=name, i=i, gold=q.answer, answer=r.text, ok=score(r.text, q.answer), hit=q.relevant in set(int(x) for x in top),
                                 prompt_tokens=r.prompt_tokens, ttft=r.ttft, total=r.total, retrieval_ms=ret_ms))
            with path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"RAG {model} N={N} {name}: acc={sum(r['ok'] for r in rows) / len(rows):.1%} hit={sum(r['hit'] for r in rows) / len(rows):.0%} "
                  f"prompt={sum(r['prompt_tokens'] for r in rows) / len(rows):.0f}tok total={sum(r['total'] for r in rows) / len(rows):.2f}s", flush=True)
