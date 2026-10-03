"""Conversion cost (Issue #10): rule-based rendering time and LLM-based L0 -> L1 conversion.

TotalCost = ConversionCost + PrefillCost + DecodeCost. The rule-based numbers start from
structured facts (the generator's ground truth), NOT from raw prose, so they are a lower
bound: a real system would first need a parser/extractor. The LLM conversion is the honest
end-to-end path from natural text, and its output is QA-evaluated.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List

from . import ablate
from . import qa as qamod
from .encoder import POLICIES, encode_doc
from .gen import generate
from .lmstudio import chat
from .tokens import get_counters

OUT = Path(__file__).parent.parent / "results" / "conversion_cost.json"
CONVERT_SYSTEM = "あなたは日本語の文章を、事実を落とさずに短く言い換える編集者です。"
CONVERT_INSTR = ("次の文章を、事実を一つも落とさず、できるだけ短い日本語に要約してください。"
                 "各人物・案件・催事の名前は行頭に必ず残し、1つの対象につき1行にしてください。"
                 "数値・固有名詞・否定(〜ない/まだ〜ない/禁止)・時制(以前/現在/将来)・条件・因果・比較・不確実(未確認)は変えないでください。"
                 "説明や前置きは書かず、変換結果だけを出力してください。")


def rule_timing(lengths: List[int], seed: int = 1, reps: int = 20) -> Dict[str, Dict[str, float]]:
    count = get_counters()["o200k_base"]
    out: Dict[str, Dict[str, float]] = {}
    variants = ["L1", "L5", "adaptive"] + ablate.single_variants() + ablate.ladder_variants()[-1:]
    for length in lengths:
        ents, _ = generate(length, count, seed=seed, n_questions=8)
        row = {}
        for v in variants:
            t0 = time.perf_counter()
            for _ in range(reps):
                encode_doc(ents, v, POLICIES["gemma"])
            row[v] = (time.perf_counter() - t0) / reps * 1000
        out[str(length)] = row
    return out


def llm_convert(model: str, length: int, base_url: str, extra: dict, seed: int = 1, n_questions: int = 48) -> dict:
    count = get_counters()["o200k_base"]
    ents, qs = generate(length, count, seed=seed, n_questions=n_questions)
    l0 = encode_doc(ents, "L0")
    # one-shot example from a different seed so the output format is stable
    ex_ents, _ = generate(300, count, seed=99, n_questions=4)
    ex = (f"例:\n入力:\n{encode_doc(ex_ents[:1], 'L0')}\n出力:\n{encode_doc(ex_ents[:1], 'L1')}\n\n")
    r = chat(model, CONVERT_SYSTEM, f"{ex}{CONVERT_INSTR}\n\n入力:\n{l0}\n出力:", max_tokens=6000, extra=extra,
             timeout=3600, deadline=3000)
    converted = r.text
    system = f"[run:{uuid.uuid4().hex[:8]}] {qamod.SYSTEM}"
    base = chat(model, system, qamod.build_prompt_v2("", qs[0]), max_tokens=5, extra=extra, deadline=240)
    oks, ctx_tokens = [], None
    for i, q in enumerate(qs):
        a = chat(model, system, qamod.build_prompt_v2(converted, q), extra=extra, deadline=600 if i else 1800)
        if i == 0:
            ctx_tokens = a.prompt_tokens - base.prompt_tokens
        oks.append(qamod.score(q, a.text))
    return dict(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, ttft=r.ttft, total=r.total,
                ctx_tokens=ctx_tokens, acc=sum(oks) / len(oks), n=len(oks), out_chars=len(converted),
                l0_chars=len(l0), sample=converted[:400])


def run(model: str, lengths: List[int], llm_lengths: List[int], base_url: str = "http://localhost:1234/v1") -> None:
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"rule": {}, "llm": {}}
    data["rule"] = rule_timing(lengths)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    for length in llm_lengths:
        if str(length) in data["llm"].get(model, {}):
            continue
        res = llm_convert(model, length, base_url, extra)
        data["llm"].setdefault(model, {})[str(length)] = res
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{model} len={length}: convert {res['total']:.0f}s ({res['prompt_tokens']}+{res['completion_tokens']} tok) "
              f"ctx={res['ctx_tokens']} acc={res['acc']:.1%}", flush=True)
