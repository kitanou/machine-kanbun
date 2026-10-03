import tiktoken

from machine_kanbun import legend as lg
from machine_kanbun.encoder import POLICIES, encode_doc
from machine_kanbun.gen import generate
from machine_kanbun.longreport import pareto

_enc = tiktoken.get_encoding("o200k_base")
count = lambda s: len(_enc.encode(s))


def test_generation_is_deterministic_and_reaches_target():
    a, qa = generate(2000, count, seed=3, n_questions=40)
    b, qb = generate(2000, count, seed=3, n_questions=40)
    assert [e.label for e in a] == [e.label for e in b] and [q.q for q in qa] == [q.q for q in qb]
    assert count(encode_doc(a, "L0")) >= 2000
    assert len({e.label for e in a}) == len(a)  # entity labels unique -> questions unambiguous


def test_every_entity_is_named_in_every_format():
    ents, _ = generate(2000, count, seed=1, n_questions=10)
    for fmt in ("json", "L0", "L1", "L2", "L3", "L4", "L5", "adaptive"):
        text = encode_doc(ents, fmt, POLICIES["gemma"])
        assert all(e.label in text for e in ents), fmt


def test_questions_cover_operators_and_cross_fact():
    _, qs = generate(8000, count, seed=1, n_questions=64)
    assert {"横断推論", "参照解決"} <= {q.cat for q in qs}
    assert {"不", "未", "禁", "疑", "若", "故"} <= {q.op for q in qs}


def test_adaptive_policy_differs_by_model():
    ents, _ = generate(2000, count, seed=1, n_questions=10)
    g, q = encode_doc(ents, "adaptive", POLICIES["gemma"]), encode_doc(ents, "adaptive", POLICIES["qwen"])
    assert "疑:" in g and "疑:" not in q and "若" in q and "?" in g  # qwen: 疑 as prose, cond as 若→


def test_legends():
    assert lg.legend_text("none") is None and "疑" in lg.legend_text("minimal")
    assert lg.legend_text("category", "不確実性") == "凡例: 疑=不確実(未確定)"
    assert lg.legend_text("category", "属性") is None
    assert count(lg.MINIMAL) < count(lg.FULL)


def test_pareto():
    assert pareto([("a", 10, 0.9), ("b", 5, 0.9), ("c", 5, 0.8)]) == ["b"]
