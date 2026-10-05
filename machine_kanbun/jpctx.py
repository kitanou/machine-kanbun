"""Issues #25 / #30: controlled long-context QA over Japanese chat memories with representation variants.

One *condition* = (representation variant, number of memories). Its context is fixed, so the prefill is paid once (cold
request, random nonce) and the questions are served from the prefix cache. All conditions of a group share the SAME
question set: targets are drawn only from memories that are present in every condition of the group, stratified over
position bins (so L0-token-matched never loses a target). Two question kinds: status (はい/いいえ/未確定/不明) and
time ("いつ?" -> the time expression of a DONE memory).
"""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import jl1, jlvar, jpdata
from .jpbench import RES
from .jpllm import INSTR, SYSTEM, score
from .lmstudio import Stalled, chat

OUT = RES.parent / "jp2"
SEED = 25
NMAX = 720
TIME_INSTR = ("上の記憶メモだけを根拠に、その出来事が起きた時期を、メモにある表現のまま短く答えてください(例: 先月)。メモに時期がない場合は「不明」と答えてください。")
_TIMES = [t.rstrip("に") for t in jpdata.TIMES]


def memories():
    return jpdata.make_memories(NMAX, seed=SEED)


def gold_time(text: str) -> Optional[str]:
    for t in jpdata.TIMES:
        if t in text:
            return t.rstrip("に")
    return None


def make_questions(mems, universe: int, nq: int = 60, ntq: int = 20, bins: int = 5, seed: int = 1):
    """Status questions stratified over `bins` position bins of the first `universe` memories + time questions on DONE memories."""
    rng = random.Random(seed)
    ev = {e.key: e for e in jpdata.EVENTS}
    per = nq // bins
    qs = []
    for b in range(bins):
        lo, hi = b * universe // bins, (b + 1) * universe // bins
        for m in rng.sample(mems[lo:hi], per):
            e = ev[m.event]
            f = jpdata.conj(e)
            qs.append(dict(kind="status", mem=m.id, pos_bin=b, text=f"{m.person}さんは{e.np}{f['ta']}？", gold=jpdata.ANSWER[m.status], status=m.status))
    done = [m for m in mems[:universe] if m.status == "DONE" and gold_time(m.text)]
    for m in rng.sample(done, min(ntq, len(done))):
        e = ev[m.event]
        f = jpdata.conj(e)
        qs.append(dict(kind="time", mem=m.id, pos_bin=m.id * bins // universe, text=f"{m.person}さんが{e.np}{f['ta']}のはいつ？", gold=gold_time(m.text), status="DONE"))
    return qs


def score_q(q, answer: str) -> bool:
    if q["kind"] == "status":
        return score(answer, q["gold"])
    return q["gold"] in answer.replace(" ", "")


def representations(names: List[str], n: int):
    texts = [m.text for m in memories()[:n]]
    out = {}
    for nm in names:
        if nm == "ginza-d":
            c = jl1.Converter("ginza", "d")
            out[nm] = [c(t) for t in texts]
        else:
            out[nm] = [jlvar.variant(nm)(t) for t in texts]
    return out


def run_group(model: str, group: str, conds: Dict[str, Tuple[str, int]], universe: int, nq: int = 60, ntq: int = 20, qseed: int = 1,
              base_url: str = "http://localhost:1234/v1", log=print) -> None:
    """conds: {cond_name: (variant, n_memories)}; results appended to results/jp2/ctx_<model>.jsonl (resumable)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ctx_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["group"], r["cond"]))
    mems = memories()
    qs = make_questions(mems, universe, nq, ntq, seed=qseed)
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    variants = sorted({v for v, _ in conds.values()})
    reps = representations(variants, max(n for _, n in conds.values()))
    for cond, (variant, n) in conds.items():
        if (group, cond) in done:
            continue
        ctx = "\n".join(reps[variant][:n])
        system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}"
        rows = []
        try:
            base = chat(model, f"[base:{uuid.uuid4().hex[:8]}] {SYSTEM}", f"記憶メモ:\n\n\n質問: x\n{INSTR}{nt}", max_tokens=12, base_url=base_url, extra=extra, deadline=240)
            for i, q in enumerate(qs):
                instr = INSTR if q["kind"] == "status" else TIME_INSTR
                prompt = f"記憶メモ:\n{ctx}\n\n質問: {q['text']}\n{instr}{nt}"
                r = chat(model, system, prompt, max_tokens=30, base_url=base_url, extra=extra, deadline=400 if i == 0 else 240, timeout=420 if i == 0 else 300)
                rows.append(dict(model=model, group=group, cond=cond, variant=variant, n_mem=n, qi=i, cold=(i == 0), kind=q["kind"], pos_bin=q["pos_bin"],
                                 status=q["status"], q=q["text"], gold=q["gold"], answer=r.text, ok=score_q(q, r.text), prompt_tokens=r.prompt_tokens,
                                 ctx_tokens=(r.prompt_tokens - base.prompt_tokens) if i == 0 else None, ttft=r.ttft, total=r.total, ctx_chars=len(ctx)))
        except Stalled:
            raise
        except Exception as e:  # context overflow etc.
            rows = [dict(group=group, cond=cond, error=f"{type(e).__name__}: {e}"[:300])]
            log(f"ERROR {group} {cond}: {rows[0]['error'][:160]}")
        with path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if "error" not in rows[0]:
            st = [r for r in rows if r["kind"] == "status"]
            tm = [r for r in rows if r["kind"] == "time"]
            log(f"{model} {group} {cond}: status={sum(r['ok'] for r in st) / len(st):.1%} time={sum(r['ok'] for r in tm) / max(1, len(tm)):.1%} "
                f"ctx={rows[0]['ctx_tokens']}tok cold_ttft={rows[0]['ttft']:.1f}s")


# ---- condition groups ------------------------------------------------------------------------------------------------
def token_matched_group(n: int, ratio: float):
    """L0 and sudachi-m at n memories, plus L0 shrunk to L1's token budget and L1 grown to L0's token budget.
    `ratio` = tokens(L1 memory) / tokens(L0 memory) under the target model's tokenizer."""
    return {f"L0@{n}": ("L0", n), f"m@{n}": ("sudachi-m", n), f"L0-tm@{int(n * ratio)}": ("L0", int(n * ratio)), f"m-tm@{int(n / ratio)}": ("sudachi-m", int(n / ratio))}


ABLATION = ["L0", "no-filler", "no-polite", "no-particle", "state-normalized", "no-chitchat(oracle)", "surface-stripped", "surface-stripped+state",
            "content-only", "sudachi-m", "sudachi-g", "naive",
            "m+case-particles", "m+connectors", "m+conjugation", "m+polite", "m+filler", "m-sentence-sep", "m-negation", "m-tense", "m-desire", "m-volition",
            "m-uncertainty(かも)", "m-evidential(らしい/みたい)", "m-word-order-rev", "m-word-order-shuffle", "m-subject(control)", "ginza-d"]


def ablation_group(n: int):
    return {v: (v, n) for v in ABLATION}


def taw_group(model: str, n: int = 300):
    """Issue #26: generic L1 vs tokenizer-aware L1 styles (own tokenizer and a foreign one)."""
    own = "qwen3" if "qwen" in model else "gemma2"
    foreign = "gemma2" if own == "qwen3" else "qwen3"
    c = {"L0": ("L0", n), "sudachi-m": ("sudachi-m", n), "sudachi-g": ("sudachi-g", n)}
    for tk, label in ((own, "own"), (foreign, "foreign")):
        for kind in ("best_readable", "best"):
            c[f"taw[{label}:{tk}]-{kind}"] = (f"taw:{tk}:{kind}", n)
    return c
