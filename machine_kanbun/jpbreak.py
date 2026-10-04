"""Issue #20 experiment 6: break-even of the analyzer-based L1 against the LLM prefill it saves.

cost(chars)    = a + b*chars          conversion time, fitted per converter from concatenated chat text
saved(chars)   = s*chars tokens       token reduction of L1 vs L0 for the same text (fitted)
benefit        = s*chars / prefill_rate   per use of the memory in a prompt
break-even     R_be = (a + b*chars) / (s*chars / rate)   uses of the memory needed to repay the conversion
               (R_be <= 1: repaid already by the first prompt that contains it)
The prefill rate is the unknown hardware variable, so the table sweeps it (measured: ~100 tok/s gemma-4-12b, ~167 tok/s qwen3-8b).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jl1, jpdata
from .jpbench import CONFIGS, RES

RATES = [30, 100, 167, 500, 1000, 5000, 20000]  # prefill tokens/s
MEASURED = {"gemma-4-12b (測定)": 100, "qwen3-8b (測定)": 167}


def fit(conv: jl1.Converter, texts: List[str], enc, ks=(1, 2, 5, 10, 20, 50, 100), reps: int = 30) -> Dict[str, float]:
    xs, ts, ds = [], [], []
    for k in ks:
        chunk = [" ".join(texts[i * k:(i + 1) * k]) for i in range(max(2, min(reps, len(texts) // k)))]
        times = []
        for c in chunk:
            t0 = time.perf_counter()
            out = conv(c)
            times.append((time.perf_counter() - t0) * 1000)
        l0 = np.mean([len(enc.encode(c)) for c in chunk])
        l1 = np.mean([len(enc.encode(conv(c))) for c in chunk])
        xs.append(np.mean([len(c) for c in chunk]))
        ts.append(np.mean(times))
        ds.append(l0 - l1)
    b, a = np.polyfit(xs, ts, 1)
    s = float(np.polyfit(xs, ds, 1)[0])
    return dict(a_ms=float(a), b_ms_per_char=float(b), saved_tokens_per_char=s, chars=xs, ms=ts, saved_tokens=ds)


def run(seed: int = 7) -> Dict:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    texts = [m.text for m in jpdata.make_memories(6000, seed=seed)]
    out: Dict = {"rates": RATES, "measured_rates": MEASURED, "fits": {}, "table": {}}
    for an, var in CONFIGS:
        conv = jl1.Converter(an, var)
        conv(texts[0])
        f = fit(conv, texts, enc)
        out["fits"][conv.name] = f
        tab = {}
        for rate in RATES:
            save_ms_per_char = f["saved_tokens_per_char"] / rate * 1000
            net = save_ms_per_char - f["b_ms_per_char"]
            tab[rate] = dict(save_ms_per_char=save_ms_per_char, r_be=(f["b_ms_per_char"] / save_ms_per_char) if save_ms_per_char > 0 else None,
                             breakeven_chars_R1=(f["a_ms"] / net) if net > 0 else None)
        out["table"][conv.name] = tab
        print(f"{conv.name:12} a={f['a_ms']:.3f}ms b={f['b_ms_per_char'] * 1000:.2f}us/char saved={f['saved_tokens_per_char']:.4f}tok/char "
              f"R_be@100tok/s={tab[100]['r_be']}", flush=True)
    # embedding path: nomic ~ measured throughput; savings proportional to tokens (filled in by the report from rag.json)
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    res = run()
    (RES / "breakeven.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
