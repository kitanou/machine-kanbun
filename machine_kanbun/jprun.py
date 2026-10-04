"""CLI for the Issue #20 LLM stages (stall-safe: exit 75 -> the caller reloads the model and resumes)."""
import argparse
import sys

from . import jpllm
from .lmstudio import Stalled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["llm", "rag"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ns", nargs="+", type=int, required=True)
    a = ap.parse_args()
    try:
        (jpllm.run if a.mode == "llm" else jpllm.run_rag)(a.model, a.ns)
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)


if __name__ == "__main__":
    main()
