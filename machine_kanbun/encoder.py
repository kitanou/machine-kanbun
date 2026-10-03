"""KCR encoder: renders a Profile at compression Levels 0-5 (plus a JSON baseline).

L0 natural Japanese        L3 pseudo-kanbun (grouped, 、-joined)
L1 Japanese summary        L4 structured kanbun (group{...} blocks)
L2 flat kanji compression  L5 aggressive (one line, ';', '?' for 若)

Operators: 不 not / 無 absence / 未 not yet / 非 is-not / 禁 prohibited,
過 今 将 for tense, 若 if, 故 because, 疑 uncertain, ">" comparison.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List

from .model import Fact, Profile

FORMATS = ["json"] + [f"L{n}" for n in range(6)]


def _phrase(s: Dict[str, str]) -> str:
    return s.get("obj", "") + s.get("adv", "") + s.get("neg", "") + s.get("pred", "")


def _core(f: Fact, level: int) -> str:
    sep = ":" if level >= 3 else ""
    if f.kind == "simple":
        out = f.obj + f.adv + f.neg + f.pred
    elif f.kind == "attr":
        out = f"{f.key}{sep}{f.value}"
    elif f.kind == "time":
        out = f"{f.tense}{sep}{f.value}"
    elif f.kind == "compare":
        out = f"{f.pred}{sep}{f.a}>{f.b}"
    elif f.kind == "cause":
        out = f"{_phrase(f.cause)}故{_phrase(f.effect)}"
    elif f.kind == "cond":
        if level >= 5:
            out = f"{_phrase(f.if_)}?{_phrase(f.then)}"
        else:
            out = f"若{_phrase(f.if_)}→{_phrase(f.then)}"
    else:
        raise ValueError(f"unknown kind {f.kind}")
    return f"疑{sep}{out}" if f.unc else out


def _grouped(p: Profile) -> "OrderedDict[str, List[Fact]]":
    g: "OrderedDict[str, List[Fact]]" = OrderedDict()
    for f in p.facts:
        g.setdefault(f.group, []).append(f)
    return g


def to_json(p: Profile) -> str:
    names = {"simple": "statement", "attr": "attribute", "time": "timeline",
             "cond": "condition", "cause": "causality", "compare": "comparison"}
    items = []
    for f in p.facts:
        d: Dict[str, object] = {"type": names[f.kind], "category": f.group}
        if f.kind == "simple":
            d.update(object=f.obj, predicate=f.pred, adverb=f.adv, negation=f.neg)
        elif f.kind == "attr":
            d.update(key=f.key, value=f.value)
        elif f.kind == "time":
            d.update(tense=f.tense, value=f.value)
        elif f.kind == "compare":
            d.update(predicate=f.pred, greater=f.a, lesser=f.b)
        elif f.kind == "cause":
            d.update(cause=f.cause, effect=f.effect)
        elif f.kind == "cond":
            d.update({"if": f.if_, "then": f.then})
        if f.unc:
            d["uncertain"] = True
        items.append({k: v for k, v in d.items() if v not in ("", None)})
    return json.dumps({"subject": p.label, "facts": items}, ensure_ascii=False, separators=(",", ":"))


def encode(p: Profile, fmt: str) -> str:
    if fmt == "json":
        return to_json(p)
    n = int(fmt[1:])
    if n == 0:
        return "\n".join(f.ja for f in p.facts)
    if n == 1:
        return "".join(f.summary for f in p.facts)
    groups = _grouped(p)
    if n == 2:
        return p.label + "。" + "".join(f.group + _core(f, 2) + "。" for f in p.facts)
    if n == 3:
        return p.label + "：" + "".join(
            (g + "・" if g else "") + "、".join(_core(f, 3) for f in fs) + "。"
            for g, fs in groups.items()
        )
    if n == 4:
        lines = [p.label + "{"]
        for g, fs in groups.items():
            if g:
                lines.append(f" {g}{{")
                lines += [f"  {_core(f, 4)}" for f in fs]
                lines.append(" }")
            else:
                lines += [f" {_core(f, 4)}" for f in fs]
        lines.append("}")
        return "\n".join(lines)
    if n == 5:
        parts = []
        for g, fs in groups.items():
            body = ";".join(_core(f, 5) for f in fs)
            parts.append(f"{g}{{{body}}}" if g else body)
        return p.label + "{" + ";".join(parts) + "}"
    raise ValueError(fmt)
