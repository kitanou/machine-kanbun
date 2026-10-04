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


@pytest.mark.parametrize("lang", ["EN", "KO"])
def test_label_families_have_expected_vocabulary(lang):
    ents = ENTS[1]
    cs, en, sym = (irlabel.render(ents, lang, c) for c in ("IRCS", "IREN", "IRSYM"))
    assert "生:" in cs and "職:" in cs and "born" not in cs      # kanji structural words
    assert "born:" in en and "occupation:" in en and not CJK.findall(en)  # English structural words, local content
    assert not CJK.findall(sym) and "⌼" in sym or "E" not in sym  # opaque rare symbols
    assert len({v for d in irlabel.SYMBOLS.values() for v in d.values()}) == sum(len(d) for d in irlabel.SYMBOLS.values())


@pytest.mark.parametrize("cond", ["IRCS", "IREN", "IRSYM"])
@pytest.mark.parametrize("lang", mlenc.LANGS)
def test_label_families_share_the_skeleton(lang, cond):
    skel = lambda t: re.sub(r"[^{};?>\n]", "", t)
    for ents in ENTS.values():
        assert skel(irlabel.render(ents, lang, cond)) == skel(irlabel.render(ents, lang, "IRC"))


def test_new_variants_parse():
    assert mlenc.parse_variant("EN-IRSYM") == ("EN", "IRSYM", "EN") and mlenc.parse_variant("KO-IREN@JA") == ("KO", "IREN", "JA")


KANA = re.compile(r"[぀-ヿ]")


def test_chinese_lexicons_cover_the_generator():
    from machine_kanbun import gen, i18n_zh
    for pool, lex in ((gen.SURNAMES, i18n_zh.SUR), (gen.ALIASES, i18n_zh.ALIAS), (gen.CITIES, i18n_zh.CITY), (gen.PLACES, i18n_zh.PLACE),
                      (gen.JOBS, i18n_zh.JOB), (gen.FOODS, i18n_zh.FOOD), (gen.UNC_FOODS, i18n_zh.UNC)):
        assert set(pool) <= set(lex), set(pool) - set(lex)
    assert len(set(i18n_zh.SUR.values())) == len(i18n_zh.SUR) and len(set(i18n_zh.ALIAS.values())) == len(i18n_zh.ALIAS)
    assert {x for pair in gen.COMPARE_PAIRS for x in pair} <= set(i18n_zh.PAIR) and {w for w, _, _ in gen.WEATHER} == set(i18n_zh.WEATHER)
    for kind, voc in irlabel.VOC.items():  # every structural word has a Chinese word
        assert set(voc) == set(irlabel.VOC_ZH[kind])
    assert set(irlabel.NOUN) == set(irlabel.NOUN_ZH)


@pytest.mark.parametrize("rep", ["L0", "L1", "IRL", "IRID", "IREN", "IRSYM"])
def test_chinese_contexts_have_no_kana_and_every_fact_has_text(rep):
    for ents in ENTS.values():
        for e in ents:
            for f in e.facts:
                assert f.zh and f.zh1, (e.label, f.id)
        text = mlenc.encode_ml(ents, "ZH", rep)
        assert not KANA.findall(text) or rep in ("L0", "L1") and False, rep
        assert all(q.q_zh for q in generate(2000, count, seed=1, n_questions=48)[1])


def test_chinese_irl_uses_simplified_words_and_differs_from_irc():
    irl, irc = (irlabel.render(ENTS[1][:3], "ZH", c) for c in ("IRL", "IRC"))
    assert "职业:" in irl and "出生:" in irl and "職:" in irc and irl != irc
    assert mlenc.parse_variant("ZH-IRL@JA") == ("ZH", "IRL", "JA")
