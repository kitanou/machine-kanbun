import re

import pytest
import tiktoken

from machine_kanbun import irlabel, mlenc
from machine_kanbun.gen import generate

_enc = tiktoken.get_encoding("o200k_base")
count = lambda s: len(_enc.encode(s))
CJK = re.compile(r"[぀-ヿ一-鿿]")
ENTS = {seed: generate(8000, count, seed=seed, n_questions=8)[0] for seed in (1, 2, 3)}


@pytest.mark.parametrize("lang", mlenc.LANGS)
def test_irc_is_exactly_the_existing_mkw(lang):
    for ents in ENTS.values():
        assert irlabel.render(ents, lang, "IRC") == mlenc.encode_ml(ents, lang, "MKW")


def test_japanese_irl_has_the_same_vocabulary_as_irc():
    # kanji already are Japanese's own words; only separators differ
    for ents in ENTS.values():
        assert irlabel.render(ents, "JA", "IRL").replace(" ", "") == irlabel.render(ents, "JA", "IRC").replace(" ", "")


@pytest.mark.parametrize("lang", ["EN", "KO"])
@pytest.mark.parametrize("cond", ["IRL", "IRID"])
def test_no_japanese_left_in_localized_conditions(lang, cond):
    for ents in ENTS.values():
        assert not CJK.findall(irlabel.render(ents, lang, cond))


@pytest.mark.parametrize("lang", mlenc.LANGS)
def test_skeleton_is_identical_across_conditions(lang):
    # the IR-ID entity label is `E1:name` (one extra ':' per entity); strip that prefix before comparing
    skel = lambda t: re.sub(r"[^{};:?>\n]", "", re.sub(r"(?m)^E\d:", "", t))
    for ents in ENTS.values():
        base = skel(irlabel.render(ents, lang, "IRC"))
        assert skel(irlabel.render(ents, lang, "IRID")) == base
        assert skel(irlabel.render(ents, lang, "IRL")) == base


def test_id_codes_are_language_independent_and_unique():
    flat = [c for d in irlabel.CODES.values() for c in d.values()]
    assert len(flat) == len(set(flat))
    ja, en, ko = (irlabel.render(ENTS[1][:3], l, "IRID") for l in mlenc.LANGS)
    codes = lambda t: re.findall(r"\b[EKGTOR]\d+\b", t)
    assert codes(ja) == codes(en) == codes(ko)


def test_legend_covers_every_code_and_is_localized():
    for lang in mlenc.LANGS:
        leg = irlabel.legend(lang)
        for d in irlabel.CODES.values():
            for code in d.values():
                assert f"{code}=" in leg
    assert "occupation" in irlabel.legend("EN") and "직업" in irlabel.legend("KO") and "職" in irlabel.legend("JA")


def test_variant_parsing_and_rendering_via_mlenc():
    assert mlenc.parse_variant("EN-IRID") == ("EN", "IRID", "EN") and mlenc.parse_variant("KO-IRIDL@JA") == ("KO", "IRIDL", "JA")
    assert mlenc.encode_ml(ENTS[1][:1], "EN", "IRL").startswith("Person ")
