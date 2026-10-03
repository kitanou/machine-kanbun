from machine_kanbun.encoder import FORMATS, encode
from machine_kanbun.model import Question, load_profiles
from machine_kanbun.qa import score

P = {p.id: p for p in load_profiles()}


def test_issue_examples_appear_in_l3_to_l5():
    a = P["A"]
    assert "蕎麦湯必飲" in encode(a, "L3")
    assert "酒不飲" in encode(a, "L5")
    assert "若雨→散歩不行" in encode(a, "L4")
    assert "雨?散歩不行" in encode(a, "L5")
    assert "好:蕎麦>うどん" in encode(a, "L3")
    assert "過:ThinkPad、今:MacBook、将:iPad" in encode(a, "L3")


def test_every_fact_is_encoded_at_every_level():
    for p in P.values():
        for fmt in FORMATS:
            assert encode(p, fmt)
        for f in p.facts:
            if f.kind == "attr":
                assert f.value in encode(p, "L5")


def test_negation_operators_are_distinct():
    ops = {f.neg for p in P.values() for f in p.facts} | {f.pred for p in P.values() for f in p.facts}
    assert {"不", "未", "非", "禁", "無"} <= ops


def test_score_yn_and_span():
    yn = Question("q", "c", yn=False)
    assert score(yn, "いいえ、飲みません") and not score(yn, "はい")
    assert not score(yn, "疑")
    span = Question("q", "c", answer=["MacBook"], reject=["ThinkPadを使"])
    assert score(span, "MacBookです") and not score(span, "ThinkPad")
