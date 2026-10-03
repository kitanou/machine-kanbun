"""L1 ablation (Issue #10): change ONE linguistic element of the L1 summary at a time.

L1 is the natural-Japanese summary (`Fact.summary`). An *op* swaps the representation of
one semantic class of facts (or one surface property) while everything else stays L1:

  particle  strip particles (は が を に の で と へ も より から) from the natural phrases
  kanji     orthographic kanji substitution on a small closed vocabulary (コーヒー→珈琲 ...)
  subj      repeat the subject label in front of every fact (the reverse ablation)
  attr      attribute facts       -> key:value             (生:1976)
  tense     past/present/future   -> 過:x 今:x 将:x
  simple    plain statements      -> kanji core            (鍋好)
  neg       不 無 未 非 禁 facts   -> operator core          (酒不飲, 駐車場無)
  unc       uncertain facts       -> 疑:core
  cond      conditions            -> 雨?散歩不行
  cause     causes                -> 雨故催事止
  cmp       comparisons           -> 好:A>B
  order     relation-first order for simple/neg/unc cores   (好:鍋, 不飲:酒)
  struct    group{...;...} nesting with ';' separators instead of 。-joined sentences

Variant names: ``ab=<op>[+<op>...]`` (independent) and ``abL=<op>+<op>...`` (cumulative ladder).
With every semantic op plus struct (and no order) the output is identical to L5.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

from .encoder import _core, _grouped
from .model import Fact, Profile

SEMANTIC = ["attr", "tense", "simple", "neg", "unc", "cond", "cause", "cmp"]
SURFACE = ["particle", "kanji", "subj", "order", "struct"]
OPS = set(SEMANTIC + SURFACE)
LADDER = ["particle", "kanji", "attr", "tense", "simple", "neg", "unc", "cond", "cause", "cmp", "struct", "order"]

_PARTICLE = re.compile(r"(?<![぀-ゟ])(?:より|から|[はがをにのとへも]|で(?![はな]))")
_KANJI = [("コーヒー", "珈琲"), ("うどん", "饂飩"), ("好き", "好"), ("飲む", "飲"), ("使う", "使"), ("生まれ", "生")]


def is_ablation(name: str) -> bool:
    return name.startswith("ab=") or name.startswith("abL=")


def parse(name: str) -> Set[str]:
    assert is_ablation(name), name
    ops = set(name.split("=", 1)[1].split("+"))
    bad = ops - OPS
    if bad:
        raise ValueError(f"unknown ablation ops: {sorted(bad)}")
    return ops


def single_variants() -> List[str]:
    return [f"ab={o}" for o in ("particle", "kanji", "subj", "attr", "tense", "simple", "neg", "unc",
                                "cond", "cause", "cmp", "struct")] + ["ab=simple+neg+order"]


def ladder_variants() -> List[str]:
    return [f"abL={'+'.join(LADDER[:k])}" for k in range(2, len(LADDER) + 1)]


def strip_particles(s: str) -> str:
    return _PARTICLE.sub("", s)


def kanjify(s: str) -> str:
    for a, b in _KANJI:
        s = s.replace(a, b)
    return s


def fact_class(f: Fact) -> str:
    """Semantic class used by the ops (note: differs from the adaptive encoder's coarser classes)."""
    if f.unc:
        return "unc"
    if f.kind == "simple":
        return "neg" if (f.neg or f.pred in ("禁", "無")) else "simple"
    return {"attr": "attr", "time": "tense", "cond": "cond", "cause": "cause", "compare": "cmp"}[f.kind]


def _ordered_core(f: Fact) -> str:
    """Relation-first form for simple facts: 好:鍋 / 不飲:酒 / 必飲:蕎麦湯 / 未:iPad購入."""
    out = f"{f.adv}{f.neg}{f.pred}:{f.obj}"
    return f"疑:{out}" if f.unc else out


def _fact_text(f: Fact, ops: Set[str]) -> str:
    cls = fact_class(f)
    if cls in ops:
        if "order" in ops and f.kind == "simple":
            return _ordered_core(f)
        return _core(f, 5)
    t = f.summary.rstrip("。")
    if "particle" in ops:
        t = strip_particles(t)
    if "kanji" in ops:
        t = kanjify(t)
    return t


def encode_ablation(profiles: Iterable[Profile], name: str) -> str:
    ops = parse(name)
    lines = []
    for p in profiles:
        if "struct" in ops:
            parts = []
            for g, fs in _grouped(p).items():
                body = ";".join(_with_subj(p, f, ops) for f in fs)
                parts.append(f"{g}{{{body}}}" if g else body)
            lines.append(p.label + "{" + ";".join(parts) + "}")
        else:
            lines.append(p.label + "：" + "".join(_with_subj(p, f, ops) + "。" for f in p.facts))
    return "\n".join(lines)


def _with_subj(p: Profile, f: Fact, ops: Set[str]) -> str:
    t = _fact_text(f, ops)
    return f"{p.label}：{t}" if "subj" in ops else t


OP_JA = {"particle": "助詞削除", "kanji": "漢字置換", "subj": "主語反復(逆アブレーション)", "attr": "属性→key:value",
         "tense": "時制→過今将", "simple": "平叙→漢字核", "neg": "否定→不無未非禁", "unc": "不確実→疑:", "cond": "条件→若/?",
         "cause": "因果→故", "cmp": "比較→A>B", "order": "関係先頭の語順", "struct": "group{}構造化"}


def describe(name: str) -> str:
    ops = parse(name)
    return "+".join(OP_JA[o] for o in LADDER + ["subj"] if o in ops)
