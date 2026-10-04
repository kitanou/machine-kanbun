"""Issue #26: target-tokenizer-aware simple L1.

L1 words come from jl1 (Sudachi mode C, simple mapping rules). A *style* then decides HOW the same words are written:

  sep      word separator             " "  ""  "・"  ","
  stop     sentence separator         " / "  "/"  "。"  ""
  marker   style of the 7 meaning markers (negation, past, hearsay, ... ): kana (as in #20) | kanji (1 char) | ascii (3 letters)
  verbform conjugation-free verb/adjective/adverb written as lemma or in hiragana reading (names and nouns are never rewritten)

`compile_l1(text, tokenizer)` picks the style that minimises tokens for that tokenizer on a calibration set, subject to a
retention constraint (every status is still recovered by the rule classifier from the compiled text alone, and no
non-DONE status is ever read as DONE). The search is exhaustive over the small style grid (4*4*3*2 = 96 styles).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional

from . import jl1, jlvar, jpdata
from .jpretention import status_from_l1

MARKERS = ["ない", "た", "らしい", "たい", "かも", "みたい", "う"]
MARKER_STYLES = {
    "kana": dict(zip(MARKERS, MARKERS)),
    "kanji": dict(zip(MARKERS, "否済伝欲疑様意")),
    "ascii": dict(zip(MARKERS, ["NOT", "PST", "HRS", "WNT", "MAY", "SEM", "VOL"])),
}
SEPS = [" ", "", "・", ","]
STOPS = [" / ", "/", "。", ""]
VERBFORMS = ["lemma", "kana"]


@dataclass(frozen=True)
class Style:
    sep: str = " "
    stop: str = " / "
    marker: str = "kana"
    verbform: str = "lemma"

    def label(self) -> str:
        return f"sep={self.sep!r} stop={self.stop!r} marker={self.marker} verb={self.verbform}"


GENERIC_M = Style()
GENERIC_G = Style(sep="", stop="/")


@lru_cache(maxsize=None)
def _reading_hira(word: str) -> str:
    ms = jlvar.sud()._t.tokenize(word, jlvar.sud()._m)
    if len(ms) != 1:
        return word
    kata = ms[0].reading_form()
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kata) or word


def words_of(text: str):
    """Structured L1: list of sentences, each a list of (word, kind) with kind in marker|verb|other."""
    toks = jlvar.sud().analyze(text)
    ws = jl1.to_l1(toks).split(" ")
    sents, cur = [], []
    for w in ws:
        if w == "/":
            sents.append(cur)
            cur = []
        elif w:
            cur.append(w)
    if cur:
        sents.append(cur)
    verbs = {t.lemma for t in toks if t.pos in ("動詞", "形容詞", "副詞")}
    return sents, verbs


def render(sents, verbs, st: Style) -> str:
    mk = MARKER_STYLES[st.marker]
    out = []
    for s in sents:
        ws = []
        for w in s:
            if w in mk:
                ws.append(mk[w])
            elif st.verbform == "kana" and w in verbs:
                ws.append(_reading_hira(w))
            else:
                ws.append(w)
        out.append(st.sep.join(ws))
    return st.stop.join(out)


def decode(text: str, st: Style) -> str:
    """Undo the marker style (for the rule-based retention check)."""
    for k, v in MARKER_STYLES[st.marker].items():
        text = text.replace(v, k) if st.marker != "kana" else text
    return text


def retention_ok(mems, parsed, st: Style) -> float:
    """Fraction of memories whose status is recovered exactly from the compiled text; also returns false-DONE count."""
    ok = bad = 0
    for m, (sents, verbs) in zip(mems, parsed):
        pred = status_from_l1(decode(render(sents, verbs, st), st))
        ok += pred == m.status
        bad += pred == "DONE" and m.status != "DONE"
    return ok / len(mems), bad


def all_styles() -> List[Style]:
    return [Style(*x) for x in itertools.product(SEPS, STOPS, MARKER_STYLES, VERBFORMS)]


def compile_search(mems, counter: Callable[[str], int], min_retention: float = 0.97, styles: Optional[List[Style]] = None):
    """Evaluate every style on the calibration memories. Returns rows sorted by tokens (retention-feasible first)."""
    parsed = [words_of(m.text) for m in mems]
    rows = []
    for st in styles or all_styles():
        texts = [render(s, v, st) for s, v in parsed]
        tok = sum(counter(t) for t in texts) / len(texts)
        ret, bad = retention_ok(mems, parsed, st)
        rows.append(dict(style=st, tokens=tok, retention=ret, false_done=bad, chars=sum(len(t) for t in texts) / len(texts)))
    rows.sort(key=lambda r: (r["retention"] < min_retention or r["false_done"] > 0, r["tokens"]))
    return rows


_compiled: Dict[str, Style] = {}


def compile_l1(text: str, target_tokenizer: str, counters: Optional[Dict[str, Callable[[str], int]]] = None) -> str:
    """Target-aware L1: the best style for `target_tokenizer` (searched once on a calibration set, cached)."""
    if target_tokenizer not in _compiled:
        from .tokcross import counters as _counters
        counters = counters or _counters()
        cal = jpdata.make_memories(400, seed=26)
        _compiled[target_tokenizer] = compile_search(cal, counters[target_tokenizer])[0]["style"]
    sents, verbs = words_of(text)
    return render(sents, verbs, _compiled[target_tokenizer])
