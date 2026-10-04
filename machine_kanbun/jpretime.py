"""Issue #20: clean re-measurement of embedding timings (only the embedding model loaded).

The retrieval run's embedding timings for a model can be contaminated by other models resident in LM Studio.
This unloads everything, loads one embedding model alone, and times a fixed sample per representation (best of 3).
"""
from __future__ import annotations

import json
import subprocess
import time

from . import jl1, jpdata
from .jpbench import RES
from .jprag import REPS, embed

MODELS = ["text-embedding-qwen3-embedding-0.6b", "text-embedding-nomic-embed-text-v1.5"]


def main(n: int = 1500, reps_n: int = 3, seed: int = 11):
    mems = jpdata.make_memories(10000, seed=seed)[:n]
    reps = {"L0": [m.text for m in mems]}
    for an, var in REPS:
        c = jl1.Converter(an, var)
        reps[c.name] = [c(m.text) for m in mems]
    qs = [q.text for q in jpdata.make_queries(mems, 200, seed=5)]
    out = {"n_texts": n, "repeats": reps_n, "models": {}}
    for model in MODELS:
        subprocess.run(["lms", "unload", "--all"], capture_output=True)
        subprocess.run(["lms", "load", model, "-y"], capture_output=True)
        ps = subprocess.run(["lms", "ps"], capture_output=True, text=True).stdout
        embed(model, reps["L0"][:64])  # warm-up
        res = {"loaded": [l.split()[0] for l in ps.splitlines()[1:] if l.strip()]}
        for name, texts in reps.items():
            best = min(_t(model, texts) for _ in range(reps_n))
            res[name] = {"s": best, "texts_per_s": len(texts) / best}
            print(f"{model[-24:]} {name:12} {len(texts) / best:7.1f} texts/s", flush=True)
        res["query_embed_ms"] = min(_t(model, qs) for _ in range(reps_n)) / len(qs) * 1000
        out["models"][model] = res
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    (RES / "embed_retime.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def _t(model, texts):
    t0 = time.perf_counter()
    embed(model, texts)
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
