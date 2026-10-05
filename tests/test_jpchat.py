"""Tests for Issue #33 (chat history compression / style contamination)."""
import pytest

pytest.importorskip("sudachipy")
from machine_kanbun import jpchat  # noqa: E402


def _scenario():
    sc = jpchat.build_scenarios(14)[0]
    sc = dict(sc)
    sc["exchanges"] = [[u, "そうなんですね。それは大変ですね。"] for u in sc["user_turns"]]
    return sc


def test_scenarios_cover_all_statuses_evenly():
    sc = jpchat.build_scenarios(35)
    assert len(sc) == 35 and {s["status"] for s in sc} == set(jpchat.jpdata.STATUSES)
    assert all(s["person"] in s["user_turns"][0] for s in sc)  # the target fact is in the oldest exchange


def test_role_family_replaces_only_the_old_exchanges():
    sc = _scenario()
    msgs, ms = jpchat.build_messages(sc, jpchat.cond_table()["B_c2"], "n")
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users[0] != sc["exchanges"][0][0] and users[1] != sc["exchanges"][1][0]   # L1
    assert users[2] == sc["exchanges"][2][0] and users[-1] == sc["question"]          # L0, natural final question
    assert msgs[0]["role"] == "system" and ms >= 0


def test_tag_family_separates_compressed_context_and_adds_rules():
    sc = _scenario()
    msgs, _ = jpchat.build_messages(sc, jpchat.cond_table()["D_c6"], "n")
    assert "Do not imitate its grammar" in msgs[0]["content"]
    u = msgs[1]["content"]
    assert u.index("<compressed_context>") < u.index("<recent_conversation>") and u.rstrip().endswith("</recent_conversation>")
    m0, _ = jpchat.build_messages(sc, jpchat.cond_table()["D0_noinstr_c6"], "n")
    assert "Do not imitate" not in m0[0]["content"]
    p2, _ = jpchat.build_messages(sc, jpchat.cond_table()["P2_recent_then_block_c6"], "n")
    assert p2[1]["content"].index("<recent_conversation>") < p2[1]["content"].index("<compressed_context>")


def test_style_metrics_separate_natural_from_telegraphic():
    nat = jpchat.style_metrics("田中さんには、まず無理に結論を求めず、話を聞く姿勢がいいと思います。")
    tel = jpchat.style_metrics("田中 退職 検討。決定 未。不安 有。結論要求せず、話聞く姿勢 推奨。")
    assert nat["particle_rate"] > tel["particle_rate"] and nat["polite_rate"] == 1.0
    assert tel["l1_label_per_100"] > 0 and tel["taigen_rate"] >= nat["taigen_rate"]
