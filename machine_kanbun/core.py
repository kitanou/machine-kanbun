"""Frozen Core Semantic Primitives (Issue #19 B): 27 operators, versioned, never extended ad hoc.

Operators carry the *functional* meaning (polarity, modality, time, epistemic status, logic, comparison, event
status). Content words (entities, relations, quantities) are open-class and written in the source language.
Anything an operator cannot express goes into the `~` free-text escape hatch, whose rate is the coverage metric.
"""
from __future__ import annotations

import re
from typing import Dict, List

CORE_VERSION = "core-1.0"

# symbol, name, family, meaning
_OPS = [
    ("不", "NEG", "極性", "predicate is negated (not)"), ("無", "ABSENT", "極性", "does not exist / none"),
    ("未", "NOT_YET", "極性", "has not happened yet"), ("非", "IS_NOT", "極性", "is not (identity / class negation)"),
    ("禁", "PROHIBIT", "モダリティ", "prohibited / must not"), ("必", "MUST", "モダリティ", "required / must"),
    ("可", "CAN", "モダリティ", "possible / permitted / able"), ("願", "WANT", "モダリティ", "desired / wanted"),
    ("意", "INTEND", "モダリティ", "intended / planned by the agent"),
    ("過", "PAST", "時制", "past"), ("今", "PRESENT", "時制", "present"), ("将", "FUTURE", "時制", "future"),
    ("既", "PERFECT", "時制", "already done / completed (resultant state)"), ("継", "PROGRESSIVE", "時制", "ongoing / continuing"),
    ("疑", "DOUBT", "認識", "uncertain / possible / unconfirmed"), ("確", "CERTAIN", "認識", "certain / confirmed"),
    ("伝", "HEARSAY", "認識", "reported by a source: 伝(source)"), ("推", "INFER", "認識", "inferred / forecast / appears to be"),
    ("故", "BECAUSE", "論理", "cause → effect: [cause] → [effect]"), ("若", "IF", "論理", "condition → consequence: [if] → [then]"),
    ("然", "CONCESSION", "論理", "concession: [although A] → [yet B]"),
    (">", "GREATER", "比較", "left is greater than right in the stated dimension"), ("=", "EQUAL", "比較", "equal in the stated dimension"),
    ("止", "CANCEL", "事象状態", "canceled / stopped"), ("延", "POSTPONE", "事象状態", "postponed"), ("予", "SCHEDULED", "事象状態", "scheduled / planned event"),
    ("~", "FREE_TEXT", "退避", "escape hatch: quoted free text for what no operator expresses"),
]
CORE: List[Dict[str, str]] = [dict(symbol=s, name=n, family=f, meaning=m) for s, n, f, m in _OPS]
SYMBOLS = [o["symbol"] for o in CORE]
FAMILIES = sorted({o["family"] for o in CORE}, key=[o["family"] for o in CORE].index)


def spec() -> str:
    """Prompt text describing the notation and the 27 operators."""
    lines = [f"IR notation ({CORE_VERSION}). One statement per line: `<ops>: <content>`.",
             "- <ops> = zero or more operator symbols separated by spaces (from the list below); <content> = a short phrase of the original-language words "
             "(subject, predicate, object; no grammatical particles). Resolve pronouns to the entity they refer to.",
             "- Compound statements use brackets: `故: [cause] → [effect]`, `若: [condition] → [consequence]`, `然: [A] → [B]` (A, yet B), "
             "`伝(source): [statement]`, `>: X > Y (dimension)`.",
             "- Use `~: \"text\"` ONLY if no operator can express it. Never invent facts. Output only the statements.",
             "Operators:"]
    for fam in FAMILIES:
        lines.append(f"  [{fam}] " + "; ".join(f"{o['symbol']}={o['meaning']}" for o in CORE if o["family"] == fam))
    return "\n".join(lines)


def legend_text() -> str:
    """Compact operator table for QA over an IR (what the round-trip prompt already contains)."""
    return (f"IR notation {CORE_VERSION}: each line is `<ops>: <content>`; the operators mean: "
            + "; ".join(f"{o['symbol']}={o['meaning']}" for o in CORE if o["symbol"] != "~") + ". `~` marks free text.")


EXAMPLES = {  # per source language; not part of the benchmark
    "EN": [("The bakery stays closed on Mondays, but it opens early on Sundays.", "今 不: bakery open Monday\n今: bakery open Sunday early"),
           ("If the form is not signed, the request will not be processed. A new rule, they say.", "若: [不: form signed] → [将 不: request processed]\n伝(unknown): [new_rule]"),
           ("The old pump failed because it overheated, so the plant postponed the restart.", "過 故: [old_pump overheat] → [old_pump fail]\n過 延: plant restart")],
    "JA": [("そのパン屋は月曜日は休みだが、日曜日は早く開く。", "今 不: パン屋 営業 月曜\n今: パン屋 開店 日曜 早朝"),
           ("書類に署名がなければ、申請は処理されない。新しい規則だそうだ。", "若: [不: 書類 署名] → [将 不: 申請 処理]\n伝(不明): [新しい規則]"),
           ("古いポンプは過熱して故障したため、工場は再稼働を延期した。", "過 故: [古いポンプ 過熱] → [古いポンプ 故障]\n過 延: 工場 再稼働")],
    "KO": [("그 빵집은 월요일에는 문을 닫지만 일요일에는 일찍 연다.", "今 不: 빵집 영업 월요일\n今: 빵집 개점 일요일 이른_아침"),
           ("서류에 서명이 없으면 신청은 처리되지 않는다. 새 규칙이라고 한다.", "若: [不: 서류 서명] → [将 不: 신청 처리]\n伝(불명): [새_규칙]"),
           ("오래된 펌프는 과열되어 고장 났기 때문에 공장은 재가동을 연기했다.", "過 故: [오래된_펌프 과열] → [오래된_펌프 고장]\n過 延: 공장 재가동")],
    "ZH": [("那家面包店周一不营业，但周日很早开门。", "今 不: 面包店 营业 周一\n今: 面包店 开门 周日 早"),
           ("如果表格没有签字，申请就不会被处理。据说这是新规定。", "若: [不: 表格 签字] → [将 不: 申请 处理]\n伝(不明): [新规定]"),
           ("旧水泵因为过热而损坏，所以工厂推迟了重启。", "過 故: [旧水泵 过热] → [旧水泵 损坏]\n過 延: 工厂 重启")],
}
_ALIAS = {"將": "将", "==": "=", ">=": ">", "≥": ">"}  # spelling variants of core operators, not new operators
_STMT = re.compile(r"^\s*([^:：]*?)\s*[:：]\s*(.*)$")
_OPTOK = re.compile(r"[^\s()（）]+(?:\([^)]*\))?")


def parse(ir: str) -> Dict[str, object]:
    """Statement-level statistics: operators used, unknown operators, free-text escapes, malformed lines."""
    used: Dict[str, int] = {}
    unknown: List[str] = []
    free = bad = n = 0
    depth = 0
    for line in ir.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        n += 1
        m = _STMT.match(line)
        if not m:
            bad += 1
            continue
        head = re.sub(r"\([^)]*\)", lambda g: "(" + g.group(0)[1:-1].replace(" ", "_") + ")", m.group(1))  # 伝(Mr. Yamada) is one token
        ops = [_ALIAS.get(re.sub(r"\(.*\)$", "", t), re.sub(r"\(.*\)$", "", t)) for t in head.split()]
        for o in ops:
            if o in SYMBOLS:
                used[o] = used.get(o, 0) + 1
            elif o:
                unknown.append(o)
        if "~" in ops:
            free += 1
        depth = max(depth, len([o for o in ops if o in SYMBOLS]))
    return dict(statements=n, used=used, unknown=unknown, free_text=free, malformed=bad, max_ops_per_statement=depth)
