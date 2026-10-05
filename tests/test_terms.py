"""Terminology (Issue #35): NF / SCF / SeCF mapping."""
from machine_kanbun import terms


def test_nf_scf_secf_mapping():
    assert terms.form_of("L0") == "NF" and terms.form_of("L0@600") == "NF" and terms.form_of("L0-tm@534") == "NF"
    for n in ("sudachi-m", "sudachi-g", "mecab-m", "ginza-d", "naive", "refined-r2", "m+conjugation", "m-negation", "taw[own:gemma2]-best"):
        assert terms.form_of(n) == "SCF", n
    assert terms.form_of("L5") == "SeCF" and terms.form_of("IR-C") == "SeCF"
    assert terms.form_of("A_base") == "" and terms.form_of("D_c6") == ""      # experiment ids are not representations


def test_bare_l1_depends_on_the_era():
    assert terms.form_of("L1", "jp") == "SCF" and terms.form_of("L1", "legacy") == "SeCF"
    assert terms.display("L1") == "SCF-L1" and terms.display("L1", era="legacy") == "SeCF-L1" and terms.display("L5") == "SeCF-L5"


def test_display_names_and_legacy_note():
    assert terms.display("L0@600") == "NF@600" and terms.display("sudachi-m") == "SCF/sudachi-m"
    assert terms.display("L0", legacy=True) == "NF (formerly L0)"
    assert terms.profile_of("sudachi-m") == "L1" and terms.profile_of("L0") == "L0" and terms.profile_of("L5") == "L5"
    assert "NF" in terms.note() and "SCF" in terms.note() and "SeCF" in terms.note()
