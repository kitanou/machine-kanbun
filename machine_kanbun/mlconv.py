"""Direct {EN,KO,JA} natural text -> MKW conversion by an LLM (Issue #11).

No pivot through Japanese: the model reads the source-language text and writes the IR.
The oracle IR (mlenc.encode_ml(..., "MKW")) is the reference; we report conversion time/tokens,
item-level F1 against the oracle, and QA accuracy when the converted context is used.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List

from . import mlenc
from . import qa as qamod
from .gen import generate
from .lmstudio import chat
from .tokens import get_counters

OUT = Path(__file__).parent.parent / "results" / "ml" / "conversion.json"
SYSTEM = "You are a precise information-extraction engine that rewrites text into a compact semantic notation."
SPEC = """Rewrite the INPUT into Machine Kanbun (MKW), a compact semantic notation. Rules:
- One line per entity: <class><name>{<item>;<item>;...;<group>{<item>;...};...}. Keep every fact; never invent facts.
- Relations, operators and concept words are written in kanji, whatever the input language. Proper names (people, nicknames, cities, venues, owners, project names) keep the input language's script.
- Attributes are key:value (e.g. 生:1972 職:<job in kanji> 住:<city> 愛称:<nickname>); timeline items are 過:x 今:x 将:x (past / present / future).
- A plain statement is object+relation (e.g. 鍋好 = likes hot pot). Negation operators: 不=not, 無=absent, 未=not yet, 非=is not, 禁=prohibited (e.g. 酒不飲, 駐車場無, Kindle購入未).
- 疑:<fact> marks an uncertain/unconfirmed fact. 雨?散歩不行 = if rain then no walk. A故B = A because-of B. 好:A>B = prefers A to B.
- Output only the notation, no explanations."""


def items(text: str) -> set:
    return {x.strip() for x in re.split(r"[;{}\n]", re.sub(r"\s+", "", text)) if x.strip()}


def f1(pred: str, gold: str) -> float:
    p, g = items(pred), items(gold)
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def convert(model: str, lang: str, length: int, base_url: str, extra: dict, seed: int = 1, n_questions: int = 8) -> Dict:
    count = get_counters()["o200k_base"]
    ents, qs = generate(length, count, seed=seed, n_questions=n_questions)
    src = mlenc.encode_ml(ents, lang, "L0")
    oracle = mlenc.encode_ml(ents, lang, "MKW")
    ex, _ = generate(300, count, seed=99, n_questions=4)
    example = f"EXAMPLE INPUT:\n{mlenc.encode_ml(ex[:1], lang, 'L0')}\nEXAMPLE OUTPUT:\n{mlenc.encode_ml(ex[:1], lang, 'MKW')}\n\n"
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""  # qwen3 thinks by default
    r = chat(model, SYSTEM, f"{SPEC}\n\n{example}INPUT:\n{src}\nOUTPUT:{nt}", max_tokens=8000, extra=extra, timeout=3600, deadline=3000)
    converted = r.text
    system = f"[run:{uuid.uuid4().hex[:8]}] {qamod.SYSTEMS[lang]}"
    mk = lambda c, q: qamod.build_prompt_lang(c, mlenc.question_text(q, lang), lang, no_think=bool(nt))
    base = chat(model, system, mk("", qs[0]), max_tokens=5, extra=extra, deadline=240)
    oks, ctx_tokens = [], None
    for i, q in enumerate(qs):
        a = chat(model, system, mk(converted, q), extra=extra, deadline=1800 if i == 0 else 240)
        if i == 0:
            ctx_tokens = a.prompt_tokens - base.prompt_tokens
        oks.append(qamod.score(q, a.text, ml=(lang != "JA")))
    return dict(lang=lang, length=length, src_tokens_o200k=count(src), oracle_tokens_o200k=count(oracle),
                prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, ttft=r.ttft, total=r.total,
                decode_tps=r.completion_tokens / max(r.total - r.ttft, 1e-6), ctx_tokens=ctx_tokens,
                acc=sum(oks) / len(oks), n=len(oks), f1=f1(converted, oracle), out_chars=len(converted), sample=converted[:300])


def run(model: str, langs: List[str], length: int, n_questions: int, base_url: str = "http://localhost:1234/v1") -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    for lang in langs:
        key = f"{model}|{lang}|{length}"
        if key in data:
            continue
        res = convert(model, lang, length, base_url, extra, n_questions=n_questions)
        data[key] = res
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{model} {lang} len={length}: convert {res['total']:.0f}s ({res['prompt_tokens']}+{res['completion_tokens']} tok, "
              f"{res['decode_tps']:.1f} tok/s) F1={res['f1']:.2f} acc={res['acc']:.1%}", flush=True)
