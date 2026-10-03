"""Semantic QA harness: answer questions from the encoded context only, then score."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .model import Question

SYSTEM = "あなたは与えられたコンテキストだけを根拠に質問へ答えるアシスタントです。"
LEGEND = ("凡例: 不=否定 無=不在 未=まだ〜でない 非=〜ではない 禁=禁止 過=過去 今=現在 将=将来 "
          "若=もし 故=ゆえに 疑=不確実 {}=グループ ?=もし〜なら")


def build_prompt(context: str, q: Question, legend: bool = False, no_think: bool = False) -> str:
    head = f"{LEGEND}\n" if legend else ""
    tail = "\n/no_think" if no_think else ""
    return (f"{head}コンテキスト:\n{context}\n\n質問: {q.q}\n"
            "コンテキストに基づき簡潔に答えてください。質問が「〜ですか」のように"
            "はい/いいえで答える形式の場合のみ、最初の語を「はい」または「いいえ」にしてください。"
            f"それ以外の質問には、答えとなる語句だけを述べてください。{tail}")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


def score(q: Question, answer: str) -> bool:
    a = _norm(answer)
    if q.yn is not None:
        m = re.match(r"\W*(はい|いいえ|yes|no)", a)
        if m:
            return (m.group(1) in ("はい", "yes")) == q.yn
        # no leading はい/いいえ: accept an explicit natural-language denial only.
        # Bare operator echoes such as "疑" are NOT accepted (that is a real failure).
        if re.search(r"(ません|いない|ではない|ない)[。\s]*$", a):
            return q.yn is False
        return False
    if not all(_norm(x) in a for x in q.answer):
        return False
    return not any(_norm(x) in a for x in q.reject)
