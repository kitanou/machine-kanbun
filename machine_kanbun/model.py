"""Data model: a Profile holds structured facts plus the QA benchmark for them."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Fact:
    id: str
    kind: str  # simple | attr | time | cond | cause | compare
    ja: str  # Level 0: natural Japanese
    summary: str  # Level 1: Japanese summary
    group: str = ""  # semantic domain, e.g. 食 / 機 / 規 (empty = top level)
    unc: bool = False  # uncertain fact
    obj: str = ""
    adv: str = ""
    neg: str = ""  # 不 無 未 非 禁 or ""
    pred: str = ""
    key: str = ""
    value: str = ""
    tense: str = ""  # 過 今 将 既
    if_: Optional[Dict[str, str]] = None
    then: Optional[Dict[str, str]] = None
    cause: Optional[Dict[str, str]] = None
    effect: Optional[Dict[str, str]] = None
    a: str = ""
    b: str = ""
    # other-language renderings of the same canonical fact (Issue #11): L0 / L1 text
    en: str = ""
    en1: str = ""
    ko: str = ""
    ko1: str = ""


@dataclass
class Question:
    q: str
    cat: str
    yn: Optional[bool] = None  # gold for yes/no questions
    answer: List[str] = field(default_factory=list)  # all must appear (span questions)
    reject: List[str] = field(default_factory=list)  # none may appear
    op: str = ""  # semantic operator exercised (不 無 未 非 禁 疑 故 若 過今将 ...)
    answer_ml: List[str] = field(default_factory=list)  # "ja|en|ko" alternatives for non-Japanese settings
    q_en: str = ""  # same question in English / Korean (Issue #11)
    q_ko: str = ""


@dataclass
class Profile:
    id: str
    label: str
    facts: List[Fact]
    questions: List[Question]
    meta: Dict[str, str] = field(default_factory=dict)  # entity class / proper names for localisation


def _fact(d: Dict[str, Any]) -> Fact:
    d = dict(d)
    if "if" in d:
        d["if_"] = d.pop("if")
    return Fact(**d)


def load_profiles(path: Optional[Path] = None) -> List[Profile]:
    raw = json.loads((path or DATA_DIR / "profiles.json").read_text(encoding="utf-8"))
    return [
        Profile(
            id=p["id"],
            label=p["label"],
            facts=[_fact(f) for f in p["facts"]],
            questions=[Question(**q) for q in p["questions"]],
        )
        for p in raw
    ]
