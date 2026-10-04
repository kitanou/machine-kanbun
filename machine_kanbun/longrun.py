"""Long-context runner (Issue #4): Model x Representation x Legend x Context length.

Resumable: a (model, length, fmt, legend, seed) variant already present in the output
file is skipped. The first question of each variant is the *cold* request (a random
nonce in the system prompt defeats prefix caching), so its TTFT is the prefill time;
the remaining questions reuse the cached context ("warm").
"""
from __future__ import annotations

import json
import time
import urllib.error
import uuid
from pathlib import Path
from typing import List, Optional

from . import legend as lg
from . import qa as qamod
from .encoder import POLICIES, encode_doc
from . import mlenc
from .gen import generate
from .lmstudio import chat
from .sysmem import Peak
from .tokens import get_counters

OUT = Path(__file__).parent.parent / "results" / "long"

DEFAULT_VARIANTS = ["json", "L0", "L1", "L3:none", "L3:full", "L5:none", "L5:full", "L5:minimal",
                    "L5:category", "adaptive:none", "adaptive:minimal"]


def policy_for(model: str) -> dict:
    return POLICIES["gemma" if "gemma" in model else "qwen"]


def parse_variant(v: str):
    fmt, _, leg = v.partition(":")
    return fmt, (leg or "none")


def run(model: str, lengths: List[int], variants: List[str], seed: int, n_questions: int,
        base_url: str = "http://localhost:1234/v1", out_dir: Path = OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = get_counters()["o200k_base"]
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    no_think = "qwen3" in model and "qwen3." not in model
    pol = policy_for(model)
    for length in lengths:
        path = out_dir / f"{model.replace('/', '_')}__{length}.jsonl"
        done = set()
        if path.exists():
            for l in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                done.add((r["fmt"], r["legend"], r["seed"]))
        entities, questions = generate(length, count, seed=seed, n_questions=n_questions)
        n_facts = sum(len(e.facts) for e in entities)
        for v in variants:
            fmt, cond = parse_variant(v)
            if fmt == "adaptive" and cond == "default":
                cond = pol["legend"]
            if (fmt, cond, seed) in done:
                continue
            ml = mlenc.parse_variant(v)  # multilingual variant such as EN-L0 / KO-MKW / EN-MKW@JA (Issue #11)
            if ml:
                clang, rep_, qlang = ml
                ctx = mlenc.encode_ml(entities, clang, rep_)
                fmt, cond = v, "none"
            else:
                clang = rep_ = None
                qlang = "JA"
                ctx = encode_doc(entities, fmt, pol)
            sys_legend = lg.legend_text(cond) if cond in ("full", "minimal") else None
            system = f"[run:{uuid.uuid4().hex[:8]}] {qamod.SYSTEMS[qlang]}" + (f"\n{sys_legend}" if sys_legend else "")
            use_ml = bool(ml) and (qlang != "JA" or clang != "JA")
            qtext = lambda q: mlenc.question_text(q, qlang)
            mkprompt = (lambda c, q: qamod.build_prompt_lang(c, qtext(q), qlang, no_think)) if ml else None
            rows, err = [], None
            try:
                # baseline: same prompt with an empty context -> context tokens = cold - base
                b = chat(model, system, (mkprompt("", questions[0]) if ml else qamod.build_prompt_v2("", questions[0], None, no_think)),
                         base_url=base_url, max_tokens=5, extra=extra, deadline=240)
                for i, q in enumerate(questions):
                    el = lg.legend_text("category", q.cat) if cond == "category" else None
                    prompt = mkprompt(ctx, q) if ml else qamod.build_prompt_v2(ctx, q, el, no_think)
                    if i == 0:
                        with Peak() as pk:
                            r = chat(model, system, prompt, base_url=base_url, extra=extra, timeout=900, deadline=900)
                    else:
                        r = chat(model, system, prompt, base_url=base_url, extra=extra, timeout=600, deadline=240)
                    rec = dict(model=model, length=length, seed=seed, fmt=fmt, legend=cond, i=i, cold=(i == 0),
                               cat=q.cat, op=q.op, q=qtext(q) if ml else q.q, ok=qamod.score(q, r.text, ml=use_ml), answer=r.text,
                               lang=clang, rep=rep_, qlang=qlang if ml else None,
                               prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                               ttft=r.ttft, total=r.total, n_facts=n_facts, n_entities=len(entities),
                               ctx_chars=len(ctx), legend_chars=len(sys_legend or ""))
                    if i == 0:
                        rec.update(base_tokens=b.prompt_tokens, rss_base_mb=pk.base["rss_mb"], rss_peak_mb=pk.peak["rss_mb"],
                                   used_base_mb=pk.base["used_mb"], used_peak_mb=pk.peak["used_mb"])
                    rows.append(rec)
            except (urllib.error.URLError, OSError, ValueError) as e:  # context overflow, model unloaded, timeout
                err = f"{type(e).__name__}: {e}"
                try:
                    err += " " + e.read().decode()[:200]  # HTTPError body
                except Exception:
                    pass
            if err:
                rows = [dict(model=model, length=length, seed=seed, fmt=fmt, legend=cond, error=err, n_facts=n_facts)]
                print(f"ERROR {model} {length} {v}: {err[:200]}", flush=True)
            with path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not err:
                acc = sum(r["ok"] for r in rows) / len(rows)
                print(f"{model} len={length} {v}: acc={acc:.1%} ctx={rows[0]['prompt_tokens'] - rows[0]['base_tokens']}tok "
                      f"cold_ttft={rows[0]['ttft']:.1f}s", flush=True)
