import pytest
import tiktoken

from machine_kanbun import ablate
from machine_kanbun.encoder import encode_doc
from machine_kanbun.gen import generate

_enc = tiktoken.get_encoding("o200k_base")
count = lambda s: len(_enc.encode(s))
ENTS, _ = generate(2000, count, seed=1, n_questions=8)


def test_all_semantic_ops_plus_struct_is_exactly_l5():
    name = "ab=" + "+".join(ablate.SEMANTIC + ["struct"])
    assert ablate.encode_ablation(ENTS, name) == encode_doc(ENTS, "L5")


def test_no_ops_applied_means_l1_like_text_without_changes():
    # "subj" is the only op that adds text; particle/kanji alone must not touch semantic classes
    t = ablate.encode_ablation(ENTS, "ab=particle")
    assert all(e.label in t for e in ENTS)
    assert t.count("\n") == len(ENTS) - 1


@pytest.mark.parametrize("src,dst", [
    ("海より山が好き", "海山好き"),
    ("酒は飲まない", "酒飲まない"),
    ("雨なら散歩に行かない", "雨なら散歩行かない"),
    ("障害原因はPostgreSQLではない", "障害原因PostgreSQLではない"),  # copula では must survive
    ("睡眠不足で日中眠い", "睡眠不足日中眠い"),
])
def test_particle_stripping(src, dst):
    assert ablate.strip_particles(src) == dst


def test_kanji_substitution():
    assert ablate.kanjify("コーヒーは飲む") == "珈琲は飲"
    assert ablate.kanjify("うどんより蕎麦が好き") == "饂飩より蕎麦が好"


@pytest.mark.parametrize("name", ablate.single_variants() + ablate.ladder_variants())
def test_every_variant_keeps_entities_and_hard_values(name):
    t = ablate.encode_ablation(ENTS, name)
    for e in ENTS:
        assert e.label in t
        for f in e.facts:
            if f.kind in ("attr", "time"):  # numbers / names must never be lost
                assert f.value.replace("-", "") in t.replace("-", ""), (name, f.value)


def test_ladder_is_cumulative_and_ends_with_order():
    ladder = ablate.ladder_variants()
    assert len(ladder) == len(ablate.LADDER) - 1
    sets = [ablate.parse(v) for v in ladder]
    assert all(a < b for a, b in zip(sets, sets[1:])) and sets[-1] == set(ablate.LADDER)


def test_unknown_op_rejected():
    with pytest.raises(ValueError):
        ablate.parse("ab=nonsense")


def test_order_is_relation_first():
    t = ablate.encode_ablation(ENTS, "ab=simple+neg+order")
    assert "不飲:酒" in t or "飲:酒" in t
