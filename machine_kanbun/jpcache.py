"""Issue #69: SeCF-L1 long-term memory x prefix / KV cache (LM Studio, gemma-4-12b).

Prompt = System -> Long-term Memory (NF or SeCF-L1 of the same memories) -> Recent conversation -> Current query. LM Studio reuses the KV cache of a common prefix (single slot or
several slots over one shared context); the cache is observed through TTFT only (no cached-token count in the usage). Cache OFF == MISS here: LM Studio cannot switch the cache off, so
"OFF" is a prompt whose first tokens (a unique id) differ from everything cached, i.e. a full prefill.
Experiment B (hit): per memory size and representation: BUILD (cold prime), HIT x3 (same memory, other queries), PARTIAL x3 (first half of the memory shared), MISS x3.
Experiment A (slots): R=4 users, each with its own memory of the same size, requests round-robin; the number of parallel slots (= distinct prefixes that can stay cached) is swept; hit = later requests with a TTFT far below the cold one.
The loaded context of this MLX model is fixed by LM Studio (19,456 tokens, `-c` and the REST context_length are ignored), so cache capacity cannot be set through the context length;
the largest single prefix is therefore the loaded context (~19k tokens), and the cache count is controlled by --parallel. A guard stops the run when the swap grows too large (16 GB machine).
"""
from __future__ import annotations

import json
import re
import statistics as st
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jpdata
from .jpbench import RES
from .jpchat import secf1_text
from .jpllm import swap_mb
from .lmstudio import Stalled, chat

OUT = RES.parent / "jp6"
MODEL = "google/gemma-4-12b"
SYSTEM = "あなたは記憶メモだけを根拠に答えるアシスタントです。"
MEM_PER_1K = 50   # ~20 gemma tokens per memory
SEED = 28
_mems = None
_qs = None


def mems():
    global _mems
    if _mems is None:
        _mems = jpdata.make_memories(1000, seed=SEED)
    return _mems


def queries():
    global _qs
    if _qs is None:
        _qs = [q.text for q in jpdata.make_queries(mems(), 200, seed=SEED + 9)]
    return _qs


def memory_text(rep: str, idx: List[int]) -> str:
    texts = [mems()[i].text for i in idx]
    return "\n".join(texts if rep == "nf" else [secf1_text(t) for t in texts])


SWAP_LIMIT_MB = 9000


def memfree_pct() -> float:
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else -1.0


def guard() -> None:
    sw = swap_mb()
    if sw > SWAP_LIMIT_MB:
        raise RuntimeError(f"swap {sw:.0f} MB exceeds the guard ({SWAP_LIMIT_MB} MB): stopping to protect the machine")


def load(ctx: int, parallel: int = 1) -> int:
    """Load the model (the requested context is ignored by this MLX model); return the loaded context length reported by LM Studio."""
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    subprocess.run(["lms", "load", MODEL, "-c", str(ctx), "--parallel", str(parallel), "-y"], capture_output=True)
    import urllib.request
    try:
        d = json.load(urllib.request.urlopen("http://localhost:1234/api/v0/models/google/gemma-4-12b", timeout=10))
        return int(d.get("loaded_context_length", 0))
    except Exception:
        return 0


def call(memory: str, query: str, nonce: str = "", recent: str = "") -> Dict:
    user = f"{nonce}記憶メモ:\n{memory}\n\n直近の会話:\n{recent or 'ユーザー: 続きをお願いします。'}\n\n質問: {query}\n「了解」とだけ答えてください。"
    r = chat(MODEL, SYSTEM, user, max_tokens=4, extra={"reasoning_effort": "none"}, deadline=900, timeout=920)
    return dict(ttft=r.ttft, total=r.total, prompt_tokens=r.prompt_tokens)


def jl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def exp_hit(sizes, ctx: int, repeats: int = 3, hits: int = 5, tag: str = "", log=print) -> None:
    import random
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"hit{tag}.jsonl"
    done = {(r["rep"], r["size"], r["state"], r["i"]) for r in jl(path)}
    eff = load(ctx)
    qs = queries()
    call(memory_text("nf", list(range(20))), "ウォームアップ", nonce=f"[run:{uuid.uuid4().hex[:8]}]\n")  # first call after a load is slow; discard
    for size in sizes:
        k = int(size / 1000 * MEM_PER_1K)
        A = list(range(k // 2))
        for rep in ("nf", "secf1"):
            mem = memory_text(rep, list(range(k)))
            n0 = uuid.uuid4().hex[:8]
            plan = [("BUILD", 0, dict(memory=mem, nonce=f"[run:{n0}]\n", query=qs[0]))]
            plan += [("HIT", i, dict(memory=mem, nonce=f"[run:{n0}]\n", query=qs[1 + i])) for i in range(hits)]
            for i in range(repeats):  # first half shared; the second half is the same memories in another order (token mismatch right after the shared half)
                rest = list(range(k // 2, k))
                random.Random(100 + i).shuffle(rest)
                plan.append(("PARTIAL", i, dict(memory=memory_text(rep, A + rest), nonce=f"[run:{n0}]\n", query=qs[10 + i])))
            plan += [("MISS", i, dict(memory=mem, nonce=f"[run:{uuid.uuid4().hex[:8]}]\n", query=qs[20 + i])) for i in range(repeats)]
            for state, i, kw in plan:
                if (rep, size, state, i) in done:
                    continue
                guard()
                s0 = swap_mb()
                r = call(**kw)
                rec = dict(rep=rep, size=size, state=state, i=i, ctx=eff, n_memories=k, swap_mb=swap_mb(), swap_before=s0, memfree_pct=memfree_pct(), **r)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log(f"hit {rep} {size} {state}{i} tokens={r['prompt_tokens']} ttft={r['ttft']:.2f}s swap={rec['swap_mb']:.0f}MB free={rec['memfree_pct']:.0f}%")


def exp_slots(slots, users: int = 4, size: int = 2500, rounds: int = 4, tag: str = "", log=print, reps=("nf", "secf1"), reload: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"slots{tag}.jsonl"
    done = {(r["rep"], r["slots"], r["round"], r["user"]) for r in jl(path)}
    k = int(size / 1000 * MEM_PER_1K)
    qs = queries()
    for K in slots:
        eff = None
        for rep in reps:
            if users * k <= len(mems()):
                idx = {u: list(range(u * k, (u + 1) * k)) for u in range(users)}
            else:  # not enough distinct memories: each user gets another random subset in another order (different from the first token on)
                import random
                idx = {u: random.Random(1000 + u).sample(range(len(mems())), k) for u in range(users)}
            sets = {u: memory_text(rep, idx[u]) for u in range(users)}
            if all((rep, K, rd, u) in done for rd in range(rounds) for u in range(users)):
                continue
            if eff is None or reload:  # reload = a clean cache for every representation (otherwise the first representation leaves its prefixes in the cache of the second)
                eff = load(24576, parallel=K)
                call(memory_text("nf", list(range(20))), "ウォームアップ", nonce=f"[run:{uuid.uuid4().hex[:8]}]\n")
            nonces = {u: f"[run:{uuid.uuid4().hex[:8]}]\n" for u in range(users)}
            for rd in range(rounds):
                for u in range(users):
                    if (rep, K, rd, u) in done:
                        continue
                    guard()
                    try:
                        r = call(sets[u], qs[(rd * users + u) % len(qs)], nonce=nonces[u])
                        err = None
                    except (ValueError, Stalled) as e:
                        r, err = dict(ttft=None, total=None, prompt_tokens=None), str(e)[:120]
                    rec = dict(rep=rep, slots=K, eff_ctx=eff, round=rd, user=u, users=users, size=size, n_memories=k, swap_mb=swap_mb(), memfree_pct=memfree_pct(), error=err, **r)
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    log(f"slots {rep} K={K} round={rd} user={u} tokens={r['prompt_tokens']} ttft={(r['ttft'] or 0):.2f}s {err or ''}")


if __name__ == "__main__":
    mode = sys.argv[1]
    pilot = "--pilot" in sys.argv
    lg = lambda s: print(s, flush=True)
    try:
        if mode == "hit":
            exp_hit((1000, 4000) if pilot else (1000, 2000, 4000, 8000, 12000, 16000, 18000), ctx=24576, tag="_pilot" if pilot else "", log=lg)
        elif mode == "stress":
            # A2: 4 users x ~5,500-token NF memories (NF total > the 19,456 loaded context, SeCF-L1 total fits); A3: many small memories (cache count)
            for K, R, S in ((4, 4, 5500), (4, 8, 1000), (4, 16, 1000), (4, 32, 1000)):
                exp_slots((K,), users=R, size=S, rounds=3, tag=f"_R{R}_S{S}", log=lg)
        elif mode == "stress2":
            # where is the threshold, and is it the total cached tokens or the number of cached prefixes? (R x S: total tokens = R x S x ~1.0 for NF)
            for K, R, S in ((4, 20, 1000), (4, 24, 1000), (4, 28, 1000), (4, 32, 500), (4, 64, 500)):
                exp_slots((K,), users=R, size=S, rounds=3, tag=f"_R{R}_S{S}", log=lg)
        elif mode == "stress3":
            # control for the order/pollution confound: secf1 first, model reloaded (empty cache) before every representation
            for K, R, S in ((4, 28, 1000), (4, 32, 1000), (4, 32, 500)):
                exp_slots((K,), users=R, size=S, rounds=3, tag=f"_clean_R{R}_S{S}", reps=("secf1", "nf"), reload=True, log=lg)
        elif mode == "slots":
            exp_slots((1, 4) if pilot else (1, 2, 4, 8), tag="_pilot" if pilot else "", rounds=3 if pilot else 4, log=lg)
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)
