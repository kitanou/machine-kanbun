"""CLI for the Issue #25/#30 context experiments (stall-safe: exit 75 -> caller reloads the model and resumes)."""
import argparse
import sys

from . import jpctx
from .lmstudio import Stalled

RATIO = {"google/gemma-4-12b": 0.89}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["tm600", "tm300", "tm100", "ablation300", "ablation600", "taw300", "tm600b", "tm300b", "ablation300b", "taw100"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ratio", type=float)
    a = ap.parse_args()
    r = a.ratio or RATIO.get(a.model, 0.89)
    try:
        if a.group in ("tm600b", "tm300b"):  # second, independent question sample on the same contexts (pooled with the first in the report)
            n = int(a.group[2:5])
            jpctx.run_group(a.model, a.group, jpctx.token_matched_group(n, r), universe=int(0.83 * n), qseed=2, log=lambda s: print(s, flush=True))
        elif a.group == "ablation300b":
            jpctx.run_group(a.model, a.group, jpctx.ablation_group(300), universe=250, qseed=2, log=lambda s: print(s, flush=True))
        elif a.group == "taw100":  # qwen3-8b in LM Studio hangs on prompts above ~4k tokens, so its replication runs at N=100
            jpctx.run_group(a.model, a.group, jpctx.taw_group(a.model, 100), universe=80, log=lambda s: print(s, flush=True))
        elif a.group == "taw300":
            jpctx.run_group(a.model, a.group, jpctx.taw_group(a.model, 300), universe=250, log=lambda s: print(s, flush=True))
        elif a.group.startswith("tm"):
            n = int(a.group[2:])
            jpctx.run_group(a.model, a.group, jpctx.token_matched_group(n, r), universe=int(0.83 * n), log=lambda s: print(s, flush=True))
        else:
            n = int(a.group[8:])
            jpctx.run_group(a.model, a.group, jpctx.ablation_group(n), universe=int(0.83 * n), log=lambda s: print(s, flush=True))
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)


if __name__ == "__main__":
    main()
