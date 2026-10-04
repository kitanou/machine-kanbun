from machine_kanbun import core, qa as qamod
from machine_kanbun.natural import _question, conv_prompt, rt_prompt
from machine_kanbun.natural_data import ITEMS, LANGS, PHENOMENA


def test_benchmark_is_complete_and_balanced():
    assert len(ITEMS) == 30 and len({i["id"] for i in ITEMS}) == 30
    for ph in PHENOMENA:
        assert sum(i["ph"] == ph for i in ITEMS) == 3, ph
    for it in ITEMS:
        assert set(it["t"]) == set(LANGS) and all(it["t"].values())
        for q in it["qs"]:
            assert set(q["q"]) == set(LANGS) and all(q["q"].values())
            assert (q["yn"] is None) == bool(q["ans"])  # exactly one of yn / span answer


def test_core_is_frozen_at_27_operators_with_unique_symbols():
    assert len(core.CORE) == 27 and len(set(core.SYMBOLS)) == 27 and "~" in core.SYMBOLS
    assert 20 <= len(core.CORE) <= 30  # the issue asks for a 20-30 operator core


def test_parse_statistics():
    r = core.parse("過 不: 田中 出席 昨日の会議\n伝(社長): [予算 削減]\n~: \"他\"\n未知 foo: x\nfoo")
    assert r["statements"] == 5 and r["free_text"] == 1 and r["malformed"] == 1 and "未知" in r["unknown"] and r["used"]["伝"] == 1


def test_prompts_contain_spec_examples_and_language():
    for lang in LANGS:
        p = conv_prompt(lang, ITEMS[0]["t"][lang], "")
        assert core.spec() in p and ITEMS[0]["t"][lang] in p and core.EXAMPLES[lang][0][0] in p
        assert "IR:" in rt_prompt(lang, "過: x", "")


def test_scoring_of_natural_questions_in_every_language():
    it = next(i for i in ITEMS if i["id"] == "t1")
    for lang, ans in (("JA", "京都です"), ("EN", "Kyoto."), ("KO", "교토입니다"), ("ZH", "京都")):
        assert qamod.score(_question(it["qs"][0], lang, "時制"), ans, ml=True)
        assert not qamod.score(_question(it["qs"][0], lang, "時制"), "Osaka", ml=True)
    n1 = _question(next(i for i in ITEMS if i["id"] == "n1")["qs"][0], "ZH", "否定")
    assert qamod.score(n1, "没有", ml=True) and not qamod.score(n1, "是的", ml=True)


def test_parser_handles_spacing_and_spelling_variants():
    r = core.parse("伝(Mr. Yamada): [予算 削減]\n將 不: 申請 処理\n>=: 太郎 > 次郎")
    assert not r["unknown"] and r["used"].get("伝") == 1 and r["used"].get("将") == 1 and r["used"].get(">") == 1
    assert core.parse("<比較>: x")["unknown"] == ["<比較>"]
