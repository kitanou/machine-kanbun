"""Tests for the Issue #25-#31 helpers (representation variants, tokenizer-aware styles, multilingual L1)."""
import pytest

pytest.importorskip("sudachipy")
from machine_kanbun import jlvar, jpctx, jpdata, tokaware  # noqa: E402


def test_filler_and_polite_surgery():
    assert jlvar.VARIANTS["no-filler"]("なんか、田中さんは会社を辞めた。") == "田中さんは会社を辞めた。"
    assert jlvar.VARIANTS["no-polite"]("佐藤花子さんは去年新しい車を買いました。") == "佐藤花子さんは去年新しい車を買った。"
    assert jlvar.VARIANTS["no-polite"]("鈴木さんは大阪に引っ越していません。") == "鈴木さんは大阪に引っ越していない。"


def test_marker_ablations_remove_only_that_marker():
    t = "田中さんは会社を辞めるかもしれない。"
    assert "かも" in jlvar.VARIANTS["sudachi-m"](t)
    assert "かも" not in jlvar.VARIANTS["m-uncertainty(かも)"](t)
    assert "ない" in jlvar.VARIANTS["sudachi-m"]("鈴木さんは大阪に引っ越していない。")
    assert "ない" not in jlvar.VARIANTS["m-negation"]("鈴木さんは大阪に引っ越していない。")


def test_word_order_variants_keep_the_same_words():
    base = jlvar.VARIANTS["sudachi-m"]("田中さんは去年会社を辞めた。")
    rev = jlvar.VARIANTS["m-word-order-rev"]("田中さんは去年会社を辞めた。")
    assert sorted(base.replace("/", "").split()) == sorted(rev.replace("/", "").split())


def test_questions_share_targets_across_token_matched_conditions():
    mems = jpctx.memories()
    qs = jpctx.make_questions(mems, 250)
    assert len([q for q in qs if q["kind"] == "status"]) == 60
    assert all(q["mem"] < 250 for q in qs)
    g = jpctx.token_matched_group(300, 0.89)
    assert g["L0-tm@267"] == ("L0", 267) and g["m-tm@337"] == ("sudachi-m", 337)


def test_tokaware_styles_round_trip_status():
    mems = jpdata.make_memories(60, seed=3)
    parsed = [tokaware.words_of(m.text) for m in mems]
    for st in (tokaware.GENERIC_M, tokaware.GENERIC_G, tokaware.Style(sep="・", stop="/", marker="kanji")):
        ret, false_done = tokaware.retention_ok(mems, parsed, st)
        assert ret > 0.9 and false_done == 0


def test_multilingual_l1_recovers_status():
    pytest.importorskip("kiwipiepy")
    pytest.importorskip("jieba")
    from machine_kanbun import jpml
    facts = jpml.make_facts(120)
    for lg in jpml.LANGS:
        ok = sum(jpml.status_from_l1(jpml.L1[lg](jpml.render(f, lg)), lg) == f.status for f in facts)
        assert ok / len(facts) > 0.95
