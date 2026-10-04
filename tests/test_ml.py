import pytest
import tiktoken

from machine_kanbun import i18n, mlenc
from machine_kanbun import qa as qamod
from machine_kanbun.gen import generate
from machine_kanbun.mlconv import f1
from machine_kanbun.model import Question

_enc = tiktoken.get_encoding("o200k_base")
count = lambda s: len(_enc.encode(s))
ENTS, QS = generate(8000, count, seed=1, n_questions=64)


def test_lexicons_cover_generator_pools_and_are_unique_per_language():
    from machine_kanbun import gen
    for pool, lex in ((gen.SURNAMES, i18n.SUR), (gen.ALIASES, i18n.ALIAS), (gen.CITIES, i18n.CITY), (gen.PLACES, i18n.PLACE),
                      (gen.JOBS, i18n.JOB), (gen.FOODS, i18n.FOOD), (gen.UNC_FOODS, i18n.UNC)):
        assert set(pool) <= set(lex), set(pool) - set(lex)
    for lex in (i18n.SUR, i18n.ALIAS, i18n.CITY, i18n.PLACE):  # unique names, otherwise questions are ambiguous
        assert len({v[0] for v in lex.values()}) == len(lex) and len({v[1] for v in lex.values()}) == len(lex)
    assert {x for pair in gen.COMPARE_PAIRS for x in pair} <= set(i18n.PAIR)
    assert {w for w, _, _ in gen.WEATHER} == set(i18n.WEATHER)
    assert set(gen.SEASONS) == set(i18n.SEASON)


def test_every_fact_and_question_has_english_and_korean():
    for e in ENTS:
        for f in e.facts:
            assert f.en and f.en1 and f.ko and f.ko1, (e.label, f.id)
    for q in QS:
        assert q.q_en and q.q_ko


def test_japanese_generation_is_unchanged_by_translations():
    # regression guard: seed 1 / 2k must keep producing the document used in the earlier experiments
    ents, _ = generate(2000, count, seed=1, n_questions=48)
    assert len(ents) == 9 and count(mlenc.encode_ml(ents, "JA", "L0")) == 2230 and count(mlenc.encode_ml(ents, "JA", "L1")) == 998


@pytest.mark.parametrize("w,exp", [("이시다", False), ("나고야", False), ("산", True), ("MacBook", True), ("iPad", False),
                                   ("Kindle", True), ("PostgreSQL", True), ("Atlas", False), ("v1", True), ("v2", False)])
def test_korean_batchim(w, exp):
    assert i18n.has_batchim(w) is exp


def test_korean_particles():
    assert i18n.topic("이시다") == "이시다는" and i18n.topic("산") == "산은"
    assert i18n.obj("Kindle") == "Kindle을" and i18n.with_("Go") == "Go와"


def test_mkw_differs_across_languages_only_in_proper_names():
    ja, en, ko = (mlenc.encode_ml(ENTS[:3], l, "MKW") for l in mlenc.LANGS)
    assert ja != en != ko
    # every kanji operator / concept appears in all three; only names change
    for op in ("不飲", "疑:", "若", "故", "過:", "今:", "将:"):
        assert (op in ja) == (op in en) == (op in ko)
    assert "Ishida" in mlenc.encode_ml(ENTS[:1], "EN", "MKW") or "石田" not in mlenc.encode_ml(ENTS[:1], "EN", "MKW")


def test_scoring_multilingual():
    yn = Question("q", "c", yn=False)
    assert qamod.score(yn, "No, he does not.") and qamod.score(yn, "아니요, 마시지 않습니다.") and not qamod.score(yn, "Yes")
    assert not qamod.score(yn, "疑")
    span = Question("q", "c", answer=["10月12日"], answer_ml=["10月12日|October 12|10월 12일"])
    assert qamod.score(span, "October 12", ml=True) and qamod.score(span, "10월 12일", ml=True)
    assert not qamod.score(span, "October 1", ml=True) and not qamod.score(span, "October 123", ml=True)
    assert qamod.score(span, "10月12日")  # Japanese setting unchanged


def test_variant_parsing():
    assert mlenc.parse_variant("EN-L0") == ("EN", "L0", "EN")
    assert mlenc.parse_variant("EN-MKW@JA") == ("EN", "MKW", "JA")
    assert mlenc.parse_variant("L1") is None and mlenc.parse_variant("ab=particle") is None


def test_f1():
    assert f1("a;b{c}", "a;b{c}") == 1.0 and f1("a;x", "a;b") == 0.5 and f1("", "a") == 0.0


@pytest.mark.parametrize("ans,gold", [("주차장이 없습니다.", False), ("주차장 없음.", False), ("미실시", False), ("아닙니다", False),
                                       ("주차장이 있습니다.", True), ("The parking lot is not available.", False),
                                       ("There is no parking.", False), ("疑", None), ("확인되지 않았습니다", False)])
def test_free_form_polarity(ans, gold):
    q = Question("q", "c", yn=True if gold else False)
    got = qamod._polarity(qamod._norm(ans))
    assert got == gold
