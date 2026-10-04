"""Unseen natural sentences -> Semantic IR -> (QA | natural language again). Issue #19 C and D.

For every language and item: (1) QA on the original text (upper bound), (2) the LLM writes the IR directly
from the source-language text (core-1.0 operators, content words kept in the source language), QA on the IR only,
(3) round trip: the LLM rewrites the IR as natural sentences from the IR alone, QA on the regenerated text.
Also reports operator usage, free-text escapes, unknown operators and malformed statements.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

from . import core
from . import qa as qamod
from .lmstudio import Stalled, chat
from .model import Question
from .natural_data import ITEMS, LANGS

OUT = Path(__file__).parent.parent / "results" / "natural"
LANG_NAME = {"JA": "Japanese", "EN": "English", "KO": "Korean", "ZH": "Chinese (Simplified)"}
SYS = "You are a precise semantic parser. You never add information that is not in the input."


def conv_prompt(lang: str, text: str, nt: str) -> str:
    ex = "\n\n".join(f"Input: {a}\nOutput:\n{b}" for a, b in core.EXAMPLES[lang])
    return f"{core.spec()}\n\nExamples ({LANG_NAME[lang]}):\n{ex}\n\nInput: {text}\nOutput:{nt}"


def rt_prompt(lang: str, ir: str, nt: str) -> str:
    return (f"Rewrite the following semantic IR as natural {LANG_NAME[lang]} sentences. Use ONLY the information in the IR; do not add, "
            f"guess or omit facts, and keep negation, tense, modality, uncertainty, sources and conditions exactly as encoded. "
            f"Output only the sentences.\n\nIR notation {core.CORE_VERSION}; operators: " +
            "; ".join(f"{o['symbol']}={o['meaning']}" for o in core.CORE) + f"\n\nIR:\n{ir}\n\n{LANG_NAME[lang]} sentences:{nt}")


def _question(q: dict, lang: str, ph: str) -> Question:
    return Question(q["q"][lang], ph, yn=q["yn"], answer_ml=[q["ans"]] if q["ans"] else [])


def run(model: str, langs: List[str], base_url: str = "http://localhost:1234/v1", out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["lang"], r["item"]))
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    for lang in langs:
        system = f"[run:{uuid.uuid4().hex[:8]}] {qamod.SYSTEMS[lang]}"
        for it in ITEMS:
            if (lang, it["id"]) in done:
                continue
            text = it["t"][lang]
            r1 = chat(model, SYS, conv_prompt(lang, text, nt), max_tokens=700, extra=extra, deadline=300)
            ir = r1.text.strip().strip("`").strip()
            r2 = chat(model, SYS, rt_prompt(lang, ir, nt), max_tokens=700, extra=extra, deadline=300)
            rt = r2.text.strip()
            rec = dict(model=model, lang=lang, item=it["id"], ph=it["ph"], text=text, ir=ir, rt=rt, parse=core.parse(ir),
                       conv_s=r1.total, conv_prompt_tokens=r1.prompt_tokens, conv_out_tokens=r1.completion_tokens,
                       rt_s=r2.total, qa=[])
            for qd in it["qs"]:
                q = _question(qd, lang, it["ph"])
                for ctx_name, ctx in (("orig", text), ("ir", ir), ("rt", rt)):
                    a = chat(model, system, qamod.build_prompt_lang(ctx, q.q, lang, bool(nt)), extra=extra, deadline=120)
                    rec["qa"].append(dict(q=q.q, ctx=ctx_name, ok=qamod.score(q, a.text, ml=True), answer=a.text,
                                          prompt_tokens=a.prompt_tokens))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            acc = {c: sum(x["ok"] for x in rec["qa"] if x["ctx"] == c) / max(1, sum(1 for x in rec["qa"] if x["ctx"] == c)) for c in ("orig", "ir", "rt")}
            print(f"{model} {lang} {it['id']}: orig={acc['orig']:.0%} ir={acc['ir']:.0%} rt={acc['rt']:.0%} "
                  f"free={rec['parse']['free_text']} unk={len(rec['parse']['unknown'])} conv={r1.total:.0f}s", flush=True)


def run_legend(model: str, base_url: str = "http://localhost:1234/v1", out: Path = OUT) -> None:
    """QA over the *saved* IR with the operator legend in the system prompt (separates information loss from
    the model's ability to read the operators: the round-trip prompt already contains the legend)."""
    src = out / f"{model.replace('/', '_')}.jsonl"
    dst = out / f"{model.replace('/', '_')}__irlegend.jsonl"
    done = set()
    if dst.exists():
        for l in dst.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["lang"], r["item"]))
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    items = {it["id"]: it for it in ITEMS}
    for line in src.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (r["lang"], r["item"]) in done:
            continue
        system = f"[run:{uuid.uuid4().hex[:8]}] {qamod.SYSTEMS[r['lang']]}\n{core.legend_text()}"
        qa = []
        for qd in items[r["item"]]["qs"]:
            q = _question(qd, r["lang"], r["ph"])
            a = chat(model, system, qamod.build_prompt_lang(r["ir"], q.q, r["lang"], bool(nt)), extra=extra, deadline=120)
            qa.append(dict(q=q.q, ctx="irL", ok=qamod.score(q, a.text, ml=True), answer=a.text, prompt_tokens=a.prompt_tokens))
        with dst.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(model=model, lang=r["lang"], item=r["item"], qa=qa), ensure_ascii=False) + "\n")
        print(f"{model} {r['lang']} {r['item']}: irL={sum(x['ok'] for x in qa) / len(qa):.0%}", flush=True)
