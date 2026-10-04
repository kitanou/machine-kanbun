"""Representation variants for the Japanese L0 -> L1 ablations (Issues #25 and #30).

Two families:
  surgical  keep the natural sentence and delete / normalise ONE kind of surface material (surfaces glued, no spaces)
  l1        telegraphic content words + markers (jl1.to_l1) with one switch flipped relative to sudachi-m
"""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List

from . import jl1, jpdata

FILLERS = jl1.FILLER_ADV | {"そういえば", "ところで"}
DROP_PARTICLES = {"が", "を", "は", "も", "に", "で", "へ", "よ", "ね", "わ"}
POLITE_DROP = {"ます", "まし", "ませ"}
POLITE_MAP = {"です": "だ", "でし": "だっ"}
CHAT_SENTENCES = tuple(jpdata.CHAT)

_sud = None


def sud():
    global _sud
    if _sud is None:
        _sud = jl1.make("sudachi")
    return _sud


_FILLER_RE = re.compile(r"(?:なんか|ちょっと|まあ|えっと|そういえば|ところで)、?")
_GODAN = jpdata._GODAN


def plain(tok: jl1.Tok, form: str) -> str:
    """Plain (casual) form of the verb token `tok`: form in ta / nai / dict. Falls back to the surface."""
    lem, ct = tok.lemma, tok.ctype
    suf = {"ta": "た", "nai": "ない", "dict": ""}[form]
    if form == "dict":
        return lem
    if ct.startswith("サ行変格"):
        return lem[:-2] + ("した" if form == "ta" else "しない") if lem.endswith("する") else ("した" if form == "ta" else "しない")
    if ct.startswith("カ行変格"):
        return "来" + suf
    if ct.startswith("上一段") or ct.startswith("下一段"):
        return lem[:-1] + suf
    if ct.startswith("五段") and lem:
        last = lem[-1]
        if last in _GODAN:
            a, i, te, ta, o = _GODAN[last]
            if lem.endswith("行く") or lem == "行く":
                ta = "った"
            return lem[:-1] + (ta if form == "ta" else a + "ない")
    return tok.surface


def surgical(text: str, ops: frozenset, sep: str = "") -> str:
    """ops subset of {filler, polite, particle, chitchat, state}."""
    if "chitchat" in ops:  # oracle: remove the generator's off-topic sentences
        for c in CHAT_SENTENCES:
            text = text.replace(c, "")
    if "filler" in ops:
        text = _FILLER_RE.sub("", text)
    toks = jl1.merge_kamo(sud().analyze(text))
    out: List[str] = []
    toks_out: List[jl1.Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        s = t.surface
        if "polite" in ops and t.pos == "助動詞":
            if s in POLITE_DROP and toks_out:
                prev = toks_out[-1]
                nxt = toks[i + 1].surface if i + 1 < len(toks) else ""
                if s == "ませ" and nxt == "ん":
                    form, step = "nai", 2
                elif s == "まし" and nxt == "た":
                    form, step = "ta", 2
                else:
                    form, step = "dict", 1
                out[-1] = plain(prev, form)
                i += step
                continue
            if s in POLITE_MAP:
                out.append(POLITE_MAP[s])
                toks_out.append(t)
                i += 1
                continue
        if "particle" in ops and t.pos == "助詞" and s in DROP_PARTICLES:
            i += 1
            continue
        if "state" in ops:  # canonical marker words instead of conjugated auxiliaries
            mk = jl1._marker(t, toks[i + 1:i + 4]) if s != "ん" and not (out and out[-1] == "ない" and s == "ない") else None
            if mk:
                out.append(mk)
                toks_out.append(t)
                i += 1
                continue
            if s == "かも":
                out.append("かも")
                toks_out.append(t)
                i += 1
                continue
        out.append(s)
        toks_out.append(t)
        i += 1
    return sep.join(out)


def l1(text: str, sep: str = " ", **kw) -> str:
    return jl1.to_l1(sud().analyze(text), sep=sep, **kw)


def reorder(text: str, mode: str, seed: int = 0) -> str:
    """Word-order ablation on an L1 string: reverse / shuffle the words inside each '/'-separated sentence."""
    rng = random.Random(seed)
    parts = []
    for seg in text.split(" / "):
        w = seg.split()
        if mode == "rev":
            w = w[::-1]
        else:
            rng.shuffle(w)
        parts.append(" ".join(w))
    return " / ".join(parts)


def _marker_drop(*m):
    return lambda t: l1(t, drop=frozenset(m))


VARIANTS: Dict[str, Callable[[str], str]] = {
    "L0": lambda t: t,
    # --- surgical edits of the natural sentence
    "no-filler": lambda t: surgical(t, frozenset({"filler"})),
    "no-polite": lambda t: surgical(t, frozenset({"polite"})),
    "no-particle": lambda t: surgical(t, frozenset({"particle"})),
    "state-normalized": lambda t: surgical(t, frozenset({"state"})),
    "no-chitchat(oracle)": lambda t: surgical(t, frozenset({"chitchat"})),
    "surface-stripped": lambda t: surgical(t, frozenset({"filler", "polite", "particle"})),
    "surface-stripped+state": lambda t: surgical(t, frozenset({"filler", "polite", "particle", "state"})),
    # --- L1 family
    "sudachi-m": lambda t: l1(t),
    "sudachi-g": lambda t: l1(t, sep=""),
    "naive": jl1.naive_l1,
    "content-only": lambda t: l1(t, drop=frozenset({"ない", "た", "らしい", "たい", "かも", "みたい", "う"})),
    # --- #30: one switch relative to sudachi-m
    "m+case-particles": lambda t: jl1.to_l1(sud().analyze(t), keep_case=True),
    "m+connectors": lambda t: jl1.to_l1(sud().analyze(t), keep_connectors=True),
    "m+conjugation": lambda t: l1(t, surface=True),
    "m+polite": lambda t: l1(t, keep_polite=True),
    "m+filler": lambda t: l1(t, keep_filler=True),
    "m-sentence-sep": lambda t: l1(t, no_slash=True),
    "m-negation": _marker_drop("ない"),
    "m-tense": _marker_drop("た"),
    "m-desire": _marker_drop("たい"),
    "m-volition": _marker_drop("う"),
    "m-uncertainty(かも)": _marker_drop("かも"),
    "m-evidential(らしい/みたい)": _marker_drop("らしい", "みたい"),
    "m-word-order-rev": lambda t: reorder(l1(t), "rev"),
    "m-word-order-shuffle": lambda t: reorder(l1(t), "shuffle"),
    "m-subject(control)": lambda t: l1(t, drop_person=True),
}


def variant(name: str) -> Callable[[str], str]:
    """VARIANTS lookup + tokenizer-aware styles "taw:<tokenizer>:<best|best_readable|generic_m|generic_g>" (Issue #26, from results/jp2/tokaware.json)."""
    if name in VARIANTS:
        return VARIANTS[name]
    if name.startswith("taw:"):
        import json
        from pathlib import Path
        from . import tokaware
        _, tk, kind = name.split(":")
        so = json.loads((Path(__file__).parent.parent / "results" / "jp2" / "tokaware.json").read_text())[tk][kind]["style_obj"]
        st = tokaware.Style(**so)

        def f(t, st=st):
            sents, verbs = tokaware.words_of(t)
            return tokaware.render(sents, verbs, st)
        return f
    raise KeyError(name)


def build(texts: List[str], names: List[str]) -> Dict[str, List[str]]:
    return {n: [VARIANTS[n](t) for t in texts] for n in names}


if __name__ == "__main__":
    ex = ["なんか、田中さんは会社を辞めようか迷っているらしい。まだ決めたわけではないみたい。",
          "最近ちょっと寒いよね。まあ、佐藤花子さんは去年新しい車を買いました。", "えっと、鈴木さんは大阪に引っ越していません。"]
    for n, f in VARIANTS.items():
        print(f"{n:28}", " | ".join(f(t) for t in ex))
