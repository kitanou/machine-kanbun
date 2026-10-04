"""Issue #20 experiment 1: conversion cost of L0 -> simple L1 (time, CPU, memory, throughput, compression)."""
from __future__ import annotations

import json
import os
import resource
import statistics as st
import time
from pathlib import Path
from typing import Dict, List

from . import jl1, jpdata

RES = Path(__file__).parent.parent / "results" / "jp"
CONFIGS = [("sudachi", "m"), ("sudachi", "g"), ("mecab", "m"), ("janome", "m"), ("sudachi", "p"), ("sudachi", "c"), ("ginza", "m"), ("ginza", "d"), ("naive", "m")]


def rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1048576 if os.uname().sysname == "Darwin" else r / 1024


def run(n: int = 2000, seed: int = 7) -> Dict:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    tok = lambda s: len(enc.encode(s))
    texts = [m.text for m in jpdata.make_memories(n, seed=seed)]
    chars_in = sum(len(t) for t in texts)
    tok_in = sum(tok(t) for t in texts)
    out: Dict[str, Dict] = {"n": n, "chars_in": chars_in, "tokens_in": tok_in, "configs": {}}
    for an, var in CONFIGS:
        r0 = rss_mb()
        t0 = time.perf_counter()
        conv = jl1.Converter(an, var)
        load_s = time.perf_counter() - t0
        conv(texts[0])  # warm-up
        lat = []
        cpu0 = time.process_time()
        wall0 = time.perf_counter()
        outs = []
        for t in texts:
            a = time.perf_counter()
            outs.append(conv(t))
            lat.append((time.perf_counter() - a) * 1000)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        lat.sort()
        out["configs"][conv.name] = dict(
            load_s=load_s, rss_delta_mb=rss_mb() - r0, rss_peak_mb=rss_mb(), mean_ms=st.mean(lat), p50_ms=lat[len(lat) // 2], p95_ms=lat[int(len(lat) * 0.95)],
            p99_ms=lat[int(len(lat) * 0.99)], throughput_per_s=len(texts) / wall, cpu_util=cpu / wall, chars_out=sum(len(o) for o in outs),
            tokens_out=sum(tok(o) for o in outs), sample=outs[0])
        c = out["configs"][conv.name]
        c["char_ratio"] = c["chars_out"] / chars_in
        c["token_ratio"] = c["tokens_out"] / tok_in
        print(f"{conv.name:12} load {load_s:5.2f}s  mean {c['mean_ms']:7.3f}ms  p95 {c['p95_ms']:7.3f}ms  {c['throughput_per_s']:8.0f} msg/s  "
              f"tok {c['token_ratio']:.2f}  rssΔ {c['rss_delta_mb']:6.1f}MB", flush=True)
    return out


def main(n: int = 2000):
    RES.mkdir(parents=True, exist_ok=True)
    res = run(n)
    (RES / "conversion_cost.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
