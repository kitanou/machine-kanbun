"""Multilingual contexts (Issue #11): {EN, KO, JA} x {L0, L1, MKW} from the same canonical facts.

MKW across languages: relations, operators and concept words stay in kanji (language-independent);
only proper names (person / alias / city / venue / owner / event place) keep the source-language
script. So "EN->MKW" and "JA->MKW" differ only in those names, by construction.
"""
from __future__ import annotations

import copy
from typing import Iterable, List

from . import i18n, i18n_zh
from .encoder import encode_doc
from .model import Profile

LANGS = ("JA", "EN", "KO")
ALL_LANGS = LANGS + ("ZH",)  # ZH added in Issue #19; reports keep the 3-language CV unless asked
REPS = ("L0", "L1", "MKW")
_IDX = {"EN": 0, "KO": 1}


def label_of(p: Profile, lang: str) -> str:
    m = p.meta
    if lang == "ZH":
        return i18n_zh.labels(m["cls"], m["name"], **({k: m[k] for k in ("place", "season") if k in m}))
    ja, en, ko = i18n.labels(m["cls"], m["name"], **({k: m[k] for k in ("place", "season") if k in m}))
    return {"JA": ja, "EN": en, "KO": ko}[lang]


def localize(p: Profile, lang: str) -> Profile:
    """Copy of `p` with proper names in the source language's script (kanji concepts untouched)."""
    if lang == "JA":
        return p
    if lang == "ZH":
        return _localize_zh(p)
    i = _IDX[lang]
    q = copy.deepcopy(p)
    m = q.meta
    if m["cls"] == "人物":
        q.label = "人物" + i18n.SUR[m["name"]][i]
    elif m["cls"] == "催事":
        q.label = "催事" + i18n.PLACE[m["place"]][i] + m["season"]
    for f in q.facts:
        if f.kind == "attr":
            if f.key == "愛称":
                f.value = i18n.ALIAS[f.value][i]
            elif f.key == "住":
                f.value = i18n.CITY[f.value][i]
            elif f.key == "担当":
                f.value = i18n.SUR[f.value][i]
            elif f.key == "場":
                f.value = i18n.PLACE[f.value[:-2]][i] + "公園"
    return q


def _localize_zh(p: Profile) -> Profile:
    q = copy.deepcopy(p)
    m = q.meta
    if m["cls"] == "人物":
        q.label = "人物" + i18n_zh.SUR[m["name"]]
    elif m["cls"] == "催事":
        q.label = "催事" + i18n_zh.PLACE[m["place"]] + m["season"]
    for f in q.facts:
        if f.kind == "attr":
            if f.key == "愛称":
                f.value = i18n_zh.ALIAS[f.value]
            elif f.key == "住":
                f.value = i18n_zh.CITY[f.value]
            elif f.key == "担当":
                f.value = i18n_zh.SUR[f.value]
            elif f.key == "場":
                f.value = i18n_zh.PLACE[f.value[:-2]] + "公園"
    return q


def encode_ml(profiles: Iterable[Profile], lang: str, rep: str) -> str:
    ps: List[Profile] = list(profiles)
    if rep in ("IRC", "IRL", "IRID", "IRIDL", "IRCS", "IREN", "IRSYM"):  # label-localisation ablation (Issue #17)
        from . import irlabel
        return irlabel.render(ps, lang, rep)
    if rep == "MKW":
        return encode_doc([localize(p, lang) for p in ps], "L5")
    if rep == "L0":
        attr = {"JA": "ja", "EN": "en", "KO": "ko", "ZH": "zh"}[lang]
        return "\n".join(getattr(f, attr) for p in ps for f in p.facts)
    if rep == "L1":
        if lang == "JA":
            return encode_doc(ps, "L1")
        attr, sep = {"EN": ("en1", " "), "KO": ("ko1", " "), "ZH": ("zh1", "")}[lang]
        return "\n".join(f"{label_of(p, lang)}: " + sep.join(getattr(f, attr) for f in p.facts) for p in ps)
    raise ValueError(rep)


def question_text(q, lang: str) -> str:
    return {"JA": q.q, "EN": q.q_en, "KO": q.q_ko, "ZH": q.q_zh}[lang]


def parse_variant(v: str):
    """'EN-L0' / 'KO-MKW' / 'EN-MKW@JA' -> (ctx_lang, rep, question_lang) or None."""
    import re
    m = re.fullmatch(r"(JA|EN|KO|ZH)-(L0|L1|MKW|IRC|IRL|IRID|IRIDL|IRCS|IREN|IRSYM)(?:@(JA|EN|KO|ZH))?", v)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3) or m.group(1)
