"""CLI: `tokens` (offline token comparison), `qa` (run LM Studio models), `report`."""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from .encoder import FORMATS, encode
from .model import load_profiles
from . import qa as qamod
from .tokens import get_counters

RESULTS = Path(__file__).parent.parent / "results"


def cmd_tokens(a):
    profiles = load_profiles()
    counters = get_counters(a.hf)
    n_facts = sum(len(p.facts) for p in profiles)
    for cname, count in counters.items():
        base = sum(count(encode(p, "L0")) for p in profiles)
        print(f"\n## {cname}  (facts={n_facts})")
        print(f"{'format':7} {'chars':>6} {'tokens':>7} {'vs L0':>7} {'tok/fact':>9}")
        for fmt in FORMATS:
            chars = sum(len(encode(p, fmt)) for p in profiles)
            tok = sum(count(encode(p, fmt)) for p in profiles)
            print(f"{fmt:7} {chars:6d} {tok:7d} {tok / base - 1:+7.1%} {tok / n_facts:9.2f}")
    print(f"\nlegend (L2-L5 only): {len(qamod.LEGEND)} chars; "
          + ", ".join(f"{k}={c(qamod.LEGEND)} tok" for k, c in counters.items()))


def cmd_qa(a):
    from .lmstudio import chat

    profiles = load_profiles()
    RESULTS.mkdir(exist_ok=True)
    tag = a.tag or ("legend" if a.legend else "nolegend")
    out = RESULTS / f"{a.model.replace('/', '_')}__{tag}.jsonl"
    extra = {"reasoning_effort": "none"} if "gemma" in a.model else None  # disable thinking
    no_think = "qwen3" in a.model and "qwen3." not in a.model or a.no_think
    chat(a.model, qamod.SYSTEM, "こんにちは", base_url=a.base_url, max_tokens=5, extra=extra)  # warm-up (model load)
    with out.open("w", encoding="utf-8") as fh:
        for fmt in a.formats:
            for p in profiles:
                ctx = encode(p, fmt)
                for q in p.questions:
                    prompt = qamod.build_prompt(ctx, q, legend=a.legend and fmt not in ("json", "L0", "L1"),
                                                no_think=no_think)
                    r = chat(a.model, qamod.SYSTEM, prompt, base_url=a.base_url, extra=extra)
                    ok = qamod.score(q, r.text)
                    rec = dict(model=a.model, fmt=fmt, profile=p.id, cat=q.cat, q=q.q, ok=ok,
                               answer=r.text, prompt_tokens=r.prompt_tokens, ttft=r.ttft,
                               total=r.total, completion_tokens=r.completion_tokens)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
            print(f"done {a.model} {fmt}", flush=True)
    print("wrote", out)


def _pareto(points):
    """points: [(name, tokens, acc)] -> names not dominated (fewer tokens, higher acc)."""
    return [n for n, t, ac in points
            if not any((t2 <= t and ac2 >= ac) and (t2 < t or ac2 > ac) for _, t2, ac2 in points)]


def cmd_report(a):
    gold = {(p.id, q.q): q for p in load_profiles() for q in p.questions}
    files = sorted(RESULTS.glob("*.jsonl")) if not a.files else [Path(f) for f in a.files]
    for f in files:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        print(f"\n## {f.stem}")
        by = defaultdict(list)
        for r in rows:
            r["ok"] = qamod.score(gold[(r["profile"], r["q"])], r["answer"])  # re-score with current scorer
            by[r["fmt"]].append(r)
        base_tok = sum(r["prompt_tokens"] or 0 for r in by.get("L0", [])) or None
        print(f"{'fmt':6} {'acc':>6} {'n':>4} {'prompt_tok/q':>13} {'vs L0':>7} {'TTFT(s)':>8} {'total(s)':>9}")
        pts = []
        for fmt in FORMATS:
            rs = by.get(fmt)
            if not rs:
                continue
            acc = sum(r["ok"] for r in rs) / len(rs)
            tok = sum(r["prompt_tokens"] or 0 for r in rs)
            rel = f"{tok / base_tok - 1:+.1%}" if base_tok else "-"
            print(f"{fmt:6} {acc:6.1%} {len(rs):4d} {tok / len(rs):13.1f} {rel:>7} "
                  f"{st.mean(r['ttft'] for r in rs):8.3f} {st.mean(r['total'] for r in rs):9.3f}")
            pts.append((fmt, tok / len(rs), acc))
        print("Pareto frontier:", ", ".join(_pareto(pts)))
        cats = sorted({r["cat"] for r in rows})
        print("\naccuracy by category:")
        print(f"{'cat':10}" + "".join(f"{fm:>7}" for fm in FORMATS if fm in by))
        for c in cats:
            line = f"{c:10}"
            for fm in FORMATS:
                if fm in by:
                    rs = [r for r in by[fm] if r["cat"] == c]
                    line += f"{(sum(r['ok'] for r in rs) / len(rs) if rs else float('nan')):7.0%}"
            print(line)


def cmd_longqa(a):
    import sys

    from . import longrun
    from .lmstudio import Stalled
    try:
        longrun.run(a.model, a.lengths, a.variants or longrun.DEFAULT_VARIANTS, a.seed, a.n_questions, a.base_url,
                    **({"out_dir": Path(a.out)} if a.out else {}))
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)  # EX_TEMPFAIL: caller should reload the model and resume


def cmd_mlreport(a):
    from . import mlreport
    mlreport.main()


def cmd_ablreport(a):
    from . import ablreport
    ablreport.main()


def cmd_convcost(a):
    from . import convcost
    convcost.run(a.model, a.lengths, a.llm_lengths)


def cmd_mlconv(a):
    import sys

    from . import mlconv
    from .lmstudio import Stalled
    try:
        mlconv.run(a.model, a.langs, a.length, a.n_questions)
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)


def cmd_longreport(a):
    from . import longreport
    longreport.main(a.reuse)


def main():
    ap = argparse.ArgumentParser(prog="machine_kanbun")
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("tokens"); t.add_argument("--hf", nargs="*", default=[]); t.set_defaults(fn=cmd_tokens)
    q = sub.add_parser("qa")
    q.add_argument("--model", required=True)
    q.add_argument("--formats", nargs="*", default=FORMATS)
    q.add_argument("--legend", action="store_true", help="prepend operator legend for L2-L5")
    q.add_argument("--no-think", action="store_true")
    q.add_argument("--tag"); q.add_argument("--base-url", default="http://localhost:1234/v1")
    q.set_defaults(fn=cmd_qa)
    l = sub.add_parser("longqa", help="Issue #4: long-context grid")
    l.add_argument("--model", required=True)
    l.add_argument("--lengths", nargs="*", type=int, default=[2000, 8000, 32000])
    l.add_argument("--variants", nargs="*", default=None, help="fmt[:legend], e.g. L5:full adaptive:minimal")
    l.add_argument("--seed", type=int, default=1); l.add_argument("--n-questions", type=int, default=64)
    l.add_argument("--base-url", default="http://localhost:1234/v1")
    l.add_argument("--out", default=None, help="output dir (default results/long)")
    l.set_defaults(fn=cmd_longqa)
    ab = sub.add_parser("ablreport"); ab.set_defaults(fn=cmd_ablreport)
    mr = sub.add_parser("mlreport", help="Issue #11 report"); mr.set_defaults(fn=cmd_mlreport)
    cv = sub.add_parser("convcost", help="Issue #10: conversion cost (rule-based timing + LLM L0->L1)")
    cv.add_argument("--model", required=True)
    cv.add_argument("--lengths", nargs="*", type=int, default=[2000, 8000, 16000, 32000])
    cv.add_argument("--llm-lengths", nargs="*", type=int, default=[2000, 8000])
    cv.set_defaults(fn=cmd_convcost)
    mc = sub.add_parser("mlconv", help="Issue #11: direct EN/KO/JA -> MKW conversion by an LLM")
    mc.add_argument("--model", required=True)
    mc.add_argument("--langs", nargs="*", default=["JA", "EN", "KO"])
    mc.add_argument("--length", type=int, default=2000)
    mc.add_argument("--n-questions", type=int, default=8)
    mc.set_defaults(fn=cmd_mlconv)
    lr = sub.add_parser("longreport"); lr.add_argument("--reuse", nargs="*", type=int, default=[1, 10, 100])
    lr.set_defaults(fn=cmd_longreport)
    r = sub.add_parser("report"); r.add_argument("files", nargs="*"); r.set_defaults(fn=cmd_report)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
