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


YES = ("はい", "yes", "yeah", "예", "네", "是", "对", "是的")
NO = ("いいえ", "no", "nope", "아니요", "아니오", "아니에요", "不是", "否", "不")
_YN_RE = re.compile(r"\W*(はい|いいえ|yeah|yes|nope|no|예|네|아니요|아니오|아니에요|是的|不是|是|对|否)")


def _has(alt: str, a: str) -> bool:
    """Substring match; an alternative ending in a digit must not be followed by another digit
    ('October 1' must not match 'October 12')."""
    alt = _norm(alt)
    if alt[-1:].isdigit():
        return re.search(re.escape(alt) + r"(?!\d)", a) is not None
    return alt in a


_NEG_JA = re.compile(r"(ません|いない|ではない|ない)[。\s]*$")
_NEG_KO = re.compile(r"(없|않|아닙니다|아니|못\s?(하|했)|불가|금지|미(실시|구매|완료|확정|확인|정))")
_POS_KO = re.compile(r"(있습니다|있다|합니다|했습니다|마십니다|좋아합니다|입니다|됩니다|확정되었|완료되었|실시되었|구매했)")
_NEG_EN = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|hasn't|haven't|wasn't|none|unconfirmed|pending|prohibited|forbidden)\b")
_POS_EN = re.compile(r"\b(is|are|does|has|have|can|may|was|will)\b")


_NEG_ZH = re.compile(r"(没有|没|并非|不是|不能|不可|不会|不允许|未|无|禁止|尚未|不)")
_POS_ZH = re.compile(r"(有|是|会|可以|能|已经|已)")


def _polarity(a: str):
    """Yes/no polarity of a free-form answer that does not start with an explicit yes/no word."""
    if _NEG_JA.search(a) or _NEG_KO.search(a) or _NEG_EN.search(a) or _NEG_ZH.search(a):
        return False
    if _POS_KO.search(a) or _POS_EN.search(a) or _POS_ZH.search(a):
        return True
    return None


def score(q: Question, answer: str, ml: bool = False) -> bool:
    """ml=True: non-Japanese setting; uses `answer_ml` ("ja|en|ko" alternatives) when present."""
    a = _norm(answer)
    if q.yn is not None:
        m = _YN_RE.match(a)
        if m:
            return (m.group(1) in YES) == q.yn
        # no leading yes/no: accept explicit natural-language polarity in the answer (Japanese / Korean / English).
        # Bare operator echoes such as "疑" are NOT accepted (that is a real failure).
        pol = _polarity(a)
        return pol is not None and pol == q.yn
    groups = q.answer_ml if (ml and q.answer_ml) else q.answer
    if not all(any(_has(alt, a) for alt in g.split("|")) for g in groups):
        return False
    return not any(_norm(x) in a for x in q.reject)


INSTR = ("コンテキストに基づき簡潔に答えてください。質問が「〜ですか」のようにはい/いいえで答える形式の場合のみ、"
         "最初の語を「はい」または「いいえ」にしてください。それ以外の質問には、答えとなる語句だけを述べてください。")


def build_prompt_v2(context: str, q: Question, extra_legend: Optional[str] = None, no_think: bool = False) -> str:
    """Long-context prompt. `extra_legend` (category legend) sits after the context so the
    context prefix stays identical across questions (prefix-cache friendly)."""
    leg = f"{extra_legend}\n" if extra_legend else ""
    tail = "\n/no_think" if no_think else ""
    return f"コンテキスト:\n{context}\n\n{leg}質問: {q.q}\n{INSTR}{tail}"


# ---- multilingual prompts (Issue #11): instruction, headers and system prompt follow the question language
SYSTEMS = {"ZH": "你是一个只根据给定上下文回答问题的助手。", "JA": SYSTEM, "EN": "You are an assistant that answers questions using only the given context.",
           "KO": "당신은 주어진 컨텍스트만을 근거로 질문에 답하는 어시스턴트입니다."}
INSTRS = {"ZH": "请根据上下文简洁作答。只有当问题可以用“是”或“否”回答时，才以“是”或“否”开头；其他问题只回答关键词语。", "JA": INSTR,
          "EN": ("Answer concisely based on the context. Only if the question can be answered with yes or no, start your answer "
                 "with the single word \"Yes\" or \"No\". Otherwise state only the answer phrase."),
          "KO": ("컨텍스트에 근거하여 간단히 답하세요. 예/아니요로 답할 수 있는 질문인 경우에만 첫 단어를 \"예\" 또는 \"아니요\"로 하세요. "
                 "그 외의 질문에는 답이 되는 어구만 말하세요.")}
HEADS = {"ZH": ("上下文：", "问题："), "JA": ("コンテキスト:", "質問:"), "EN": ("Context:", "Question:"), "KO": ("컨텍스트:", "질문:")}


def build_prompt_lang(context: str, question: str, lang: str, no_think: bool = False) -> str:
    c, qh = HEADS[lang]
    tail = "\n/no_think" if no_think else ""
    return f"{c}\n{context}\n\n{qh} {question}\n{INSTRS[lang]}{tail}"
