"""Legend conditions (Issue #4): none / full / minimal / category-specific."""
from __future__ import annotations

from typing import Optional

GLOSS = {
    "不": "不=否定", "無": "無=不在", "未": "未=まだ〜でない", "非": "非=〜ではない", "禁": "禁=禁止",
    "疑": "疑=不確実(未確定)", "過": "過=過去", "今": "今=現在", "将": "将=将来",
    "若": "若=もし(?も同じ)", "故": "故=ゆえに", "{}": "{}=グループ", ">": "A>B=AはBより上",
}
FULL = ("凡例: 不=否定 無=不在 未=まだ〜でない 非=〜ではない 禁=禁止 過=過去 今=現在 将=将来 "
        "若=もし 故=ゆえに 疑=不確実 {}=グループ ?=もし〜なら")
MINIMAL = "凡例: 疑=不確実(未確定)"
# category -> glosses relevant to questions of that category
BY_CAT = {
    "否定(不)": ["不"], "否定(無)": ["無"], "否定(未)": ["未"], "否定(非)": ["非"], "否定(禁)": ["禁"],
    "不確実性": ["疑"], "時制": ["過", "今", "将"], "条件": ["若"], "因果": ["故"], "比較": [">"],
}

CONDITIONS = ["none", "full", "minimal", "category"]


def legend_text(cond: str, cat: str = "") -> Optional[str]:
    """Static text (full/minimal) or per-question text (category)."""
    if cond == "full":
        return FULL
    if cond == "minimal":
        return MINIMAL
    if cond == "category":
        ops = BY_CAT.get(cat)
        return "凡例: " + " ".join(GLOSS[o] for o in ops) if ops else None
    return None
