"""CLI for Issue #27: generate natural chats, then QA over them (stall-safe, exit 75)."""
import sys

from . import jpnat
from .lmstudio import Stalled

if __name__ == "__main__":
    try:
        if sys.argv[1] == "gen":
            jpnat.generate(sys.argv[2], n=int(sys.argv[3]) if len(sys.argv) > 3 else 90)
        else:
            jpnat.run_qa(sys.argv[2], log=lambda s: print(s, flush=True), max_chats=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)
