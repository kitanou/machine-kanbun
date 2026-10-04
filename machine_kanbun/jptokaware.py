"""Issue #26 (offline part): for every tokenizer, the token-optimal style under the retention constraint, with and without a
readability constraint (words must stay separated), vs generic sudachi-m / sudachi-g. Writes results/jp2/tokaware.json."""
from __future__ import annotations

import json

from . import jpdata, tokaware, tokcross
from .jpbench import RES

OUT = RES.parent / "jp2"


def main(n_cal: int = 400, n_test: int = 600):
    cnt = tokcross.counters()
    cal = jpdata.make_memories(n_cal, seed=26)
    test = jpdata.make_memories(n_test, seed=27)
    out = {}
    for k, c in cnt.items():
        rows = tokaware.compile_search(cal, c)
        feasible = [r for r in rows if r["retention"] >= 0.97 and r["false_done"] == 0]
        best = feasible[0]
        readable = next(r for r in feasible if r["style"].sep != "" and r["style"].stop != "")
        parsed = [tokaware.words_of(m.text) for m in test]  # held-out memories
        l0 = sum(c(m.text) for m in test) / len(test)
        def tok(st):
            return sum(c(tokaware.render(s, v, st)) for s, v in parsed) / len(parsed)
        ev = {}
        for name, st in dict(generic_m=tokaware.GENERIC_M, generic_g=tokaware.GENERIC_G, best=best["style"], best_readable=readable["style"]).items():
            ret, fd = tokaware.retention_ok(test, parsed, st)
            ev[name] = dict(style=st.label(), style_obj=dict(sep=st.sep, stop=st.stop, marker=st.marker, verbform=st.verbform), tokens_per_memory=tok(st), ratio_vs_L0=tok(st) / l0, retention=ret, false_done=fd)
        out[k] = dict(L0_tokens_per_memory=l0, **ev, n_styles=len(rows))
        print(k, f"L0={l0:.1f}", {n: f"{v['ratio_vs_L0']:.2f}" for n, v in ev.items()}, flush=True)
    (OUT / "tokaware.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
