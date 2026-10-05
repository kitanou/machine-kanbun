"""Terminology (Issue #35): representation names NF / SCF / SeCF and the old L0 / L1 names.

  NF   Natural Form            original natural language (formerly L0)
  SCF  Structural Compact Form deterministic, analyzer-based compact form (formerly "lightweight L1" / "Parser L1")
  SeCF Semantic Compact Form   semantic compact form (formerly L1)

Representation (NF / SCF / SeCF) and Compression Profile (L0 / L1 / L5 ...) are separate axes: SCF-L1, SeCF-L5, ...
Result files keep their stable data keys ("L0", "sudachi-m", ...); this module only maps them to the standard names for tables, legends and CSV.
Context matters for the bare name "L1": in the Japanese experiments since Issue #20 it always means the parser-based form (SCF); in the
experiments up to Issue #19 it is the semantic telegraphic form (SeCF).
"""
from __future__ import annotations

import re
from typing import Dict

NF, SCF, SECF = "NF", "SCF", "SeCF"
FULL: Dict[str, str] = {NF: "Natural Form", SCF: "Structural Compact Form", SECF: "Semantic Compact Form"}
LEGACY: Dict[str, str] = {NF: "L0", SCF: "軽量L1 / Parser L1", SECF: "L1"}
LEGACY_TO_NEW: Dict[str, str] = {"L0": NF, "軽量L1": SCF, "Parser L1": SCF, "L1": SECF}

# deterministic, analyzer-based variants used since Issue #20 (all SCF)
_SCF_PREFIX = ("sudachi-", "mecab-", "janome-", "ginza-", "naive", "refined-", "taw", "m+", "m-", "m@", "no-", "surface-", "state-normalized", "content-only")
_SCF_EXACT = {"m", "g", "p", "d", "r1", "r2"}


def form_of(name: str, era: str = "jp") -> str:
    """Standard representation name of a variant / representation key.

    era="jp" (Issue #20 onwards): bare "L1" / "L1n" = parser-based => SCF.   era="legacy" (up to #19): bare "L1" => SeCF, "L5"/IR => SeCF.
    Returns "" when the key is not a representation (e.g. an experiment id).
    """
    n = name.strip()
    if re.match(r"^(L0|NF)(\b|[@\-_(])", n) or n in ("L0", "NF"):
        return NF
    if re.match(r"^(L5|IR)", n):
        return SECF
    if re.match(r"^L1", n):
        return SCF if era == "jp" else SECF
    if n.startswith(("SCF", "SeCF")):
        return SECF if n.startswith("SeCF") else SCF
    if n.startswith(_SCF_PREFIX) or n in _SCF_EXACT:
        return SCF
    return ""


def profile_of(name: str) -> str:
    """Compression profile implied by the key: L0 (none), L1 (light surface removal), L5 (strong semantic re-encoding)."""
    f = form_of(name, "legacy")
    if re.match(r"^(L5|IR)", name):
        return "L5"
    if f == NF:
        return "L0"
    return "L1" if form_of(name) else ""


def display(name: str, era: str = "jp", legacy: bool = False) -> str:
    """Name for tables / legends: L0@600 -> NF@600, sudachi-m -> SCF/sudachi-m, L1 -> SCF (era jp). legacy=True appends the old name."""
    f = form_of(name, era)
    if not f:
        return name
    if re.match(r"^(L0|NF)", name):
        out = re.sub(r"^(L0|NF)", NF, name, count=1)
    elif name in ("L1", "L5"):
        out = f"{f}-{name}"                      # Representation-Profile notation, e.g. SCF-L1, SeCF-L5
    elif name == "L1n":
        out = f"{f}-L1(native markers)"
    else:
        out = f"{f}/{name}"
    return f"{out} (formerly {LEGACY[f]})" if legacy and f == NF and out == NF else out


def note() -> str:
    """One-line legend placed at the top of generated reports."""
    return ("用語(Issue #35): NF = Natural Form(旧 L0)、SCF = Structural Compact Form(旧 軽量L1 / Parser L1。本リポジトリの sudachi-m 等)、"
            "SeCF = Semantic Compact Form(旧 L1)。データのキー(L0, sudachi-m, ...)は互換のため据え置き、表・凡例で標準名に読み替える。")
