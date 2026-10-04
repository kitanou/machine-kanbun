import pytest

jl1 = pytest.importorskip("machine_kanbun.jl1")
pytest.importorskip("sudachipy")
pytest.importorskip("fugashi")
pytest.importorskip("janome")

SENT = {
    "undecided": ("森田結衣さんは家を売ろうか迷っているようです。まだ決めていません。", {"迷う", "まだ", "ない"}),
    "hearsay": ("山崎麻衣さんが犬を飼うって聞いたけど、本当かどうか分かりません。", {"聞く", "ない"}),
    "possible": ("原田瑞希さんは手術を受けるかもしれません。", {"かも"}),
    "past": ("遠藤拓也さん、先月沖縄に行ったよ。", {"た", "沖縄", "先月"}),
    "negated": ("森田真理さんは運転免許を取っていません。", {"ない"}),
    "rumor": ("田中花子さんは会社を辞めようか迷っているらしい。", {"らしい", "迷う"}),
}


@pytest.mark.parametrize("analyzer", ["sudachi", "mecab", "janome"])
@pytest.mark.parametrize("key", list(SENT))
def test_meaning_carrying_morphemes_survive(analyzer, key):
    text, must = SENT[key]
    toks = set(jl1.Converter(analyzer, "m")(text).replace("/", " ").split())
    assert must <= toks, (analyzer, key, toks)


@pytest.mark.parametrize("analyzer", ["sudachi", "mecab", "janome"])
def test_particles_politeness_and_fillers_are_dropped(analyzer):
    out = jl1.Converter(analyzer, "m")("えっと、田中さんは会社を辞めました。").split()
    assert not {"は", "を", "です", "ます", "ました", "えっと", "さん"} & set(out) and "田中" in out and "会社" in out


def test_naive_baseline_loses_negation_and_tense():
    out = jl1.naive_l1("森田真理さんは運転免許を取っていません。")
    assert "ない" not in out and "運転免許" in out


def test_case_particles_are_kept_only_after_nouns():
    out = jl1.Converter("sudachi", "p")("田中さんは会社を辞めなかった。")
    assert "田中は" in out and "会社を" in out and "ないを" not in out


def test_compression_on_the_issue_example():
    text = "昨日、田中さんと久しぶりに話したんだけど、どうも会社を辞めようか迷っているらしい。ただ、まだ決めたわけではないみたい。"
    out = jl1.Converter("sudachi", "m")(text)
    assert len(out.replace(" ", "")) < len(text) * 0.8 and {"迷う", "らしい", "まだ", "ない"} <= set(out.replace("/", " ").split())
