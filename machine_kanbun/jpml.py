"""Issue #31: parser-based (deterministic, no LLM) L1 for Japanese / English / Korean / Chinese -- does L1 converge in tokens per fact?

The SAME facts (person x event x status x time) are rendered in four languages with plain templates (no fillers or chit-chat:
a clean parallel corpus; this is the main limitation). Each language is converted with ITS OWN existing analyzer:
  ja  SudachiPy (jl1)            ko  Kiwi (kiwipiepy)           zh  jieba.posseg
  en  spaCy tokenizer + lookup lemmatizer (spacy-lookups-data) + closed-class word lists  [no statistical tagger available offline]
L1 = content words (lemmas) + native meaning markers (negation, past, hearsay, wish, possibility); function words, particles,
auxiliaries, honorifics and punctuation are dropped -- the same recipe as the Japanese simple L1 of Issue #20.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

from . import i18n, jl1, jpdata

LANGS = ["ja", "en", "ko", "zh"]
EVENTS12 = ["結婚", "転職", "留学", "起業", "入院", "昇進", "出張", "就職", "退院", "優勝", "転勤", "合格"]
EN = {  # base / past / participle / gerund
    "結婚": ("get married", "got married", "gotten married", "getting married"), "転職": ("change jobs", "changed jobs", "changed jobs", "changing jobs"),
    "留学": ("study abroad", "studied abroad", "studied abroad", "studying abroad"), "起業": ("start a company", "started a company", "started a company", "starting a company"),
    "入院": ("go into the hospital", "went into the hospital", "gone into the hospital", "going into the hospital"),
    "昇進": ("get promoted", "got promoted", "gotten promoted", "getting promoted"),
    "出張": ("go on a business trip", "went on a business trip", "gone on a business trip", "going on a business trip"),
    "就職": ("get a job", "got a job", "gotten a job", "getting a job"), "退院": ("leave the hospital", "left the hospital", "left the hospital", "leaving the hospital"),
    "優勝": ("win the championship", "won the championship", "won the championship", "winning the championship"),
    "転勤": ("be transferred abroad", "was transferred abroad", "been transferred abroad", "being transferred abroad"),
    "合格": ("pass the exam", "passed the exam", "passed the exam", "passing the exam")}
KO = {"結婚": "결혼", "転職": "이직", "留学": "유학", "起業": "창업", "入院": "입원", "昇進": "승진", "出張": "출장", "就職": "취직", "退院": "퇴원", "優勝": "우승", "転勤": "전근", "合格": "합격"}
ZH = {"結婚": "结婚", "転職": "跳槽", "留学": "留学", "起業": "创业", "入院": "住院", "昇進": "升职", "出張": "出差", "就職": "就业", "退院": "出院", "優勝": "夺冠", "転勤": "调动", "合格": "及格"}
TIME = {"en": ["last month", "last year", "three years ago", "in May", "recently", "last Tuesday", "a month ago"],
        "ko": ["지난달", "작년", "3년 전", "5월에", "최근", "지난주 화요일", "한 달 전"],
        "zh": ["上个月", "去年", "三年前", "五月", "最近", "上周二", "一个月前"]}
FUT = {"en": ["next month", "next year", "in two weeks", "this spring", "within the year"], "ko": ["다음 달", "내년", "2주 뒤", "봄쯤", "올해 안에"],
       "zh": ["下个月", "明年", "两周后", "春天", "年内"]}
ANS = {"ja": ("はい", "いいえ", "未確定", "不明"), "en": ("Yes", "No", "Undetermined", "Unknown"), "ko": ("예", "아니요", "미확정", "모름"), "zh": ("是", "否", "未确定", "不知道")}
GOLD = {"DONE": 0, "NOT_DONE": 1, "PLANNED": 1, "WANTED": 1, "UNDECIDED": 2, "HEARSAY": 2, "POSSIBLE": 2}
def _zh_single(s: str) -> bool:
    import jieba.posseg as pseg
    return [w.word for w in pseg.cut(s + "结婚了")][0] == s


# surnames that jieba keeps as one token (Japanese kanji names are not Chinese names; the others would be split arbitrarily)
SURNAMES = [s for s in i18n.SUR if _zh_single(s)]


@dataclass
class Fact:
    id: int
    sur: str
    ev: str
    status: str
    ti: int
    fi: int


def make_facts(n: int, seed: int = 31) -> List[Fact]:
    rng = random.Random(seed)
    pairs = [(s, e) for s in SURNAMES for e in EVENTS12]
    rng.shuffle(pairs)
    return [Fact(i, s, e, jpdata.STATUSES[rng.randrange(7)], rng.randrange(7), rng.randrange(5)) for i, (s, e) in enumerate(pairs[:n])]


def ko_josa(w: str, kind: str) -> str:
    return {"topic": i18n.topic, "subj": i18n.subj, "obj": i18n.obj}[kind](w)


def render(f: Fact, lang: str) -> str:
    en, ko, zh = EN[f.ev], KO[f.ev], ZH[f.ev]
    s = f.status
    if lang == "ja":
        ev = jpdata.event_of(f.ev)
        return jpdata.render(random.Random(f.id), f.sur, ev, s, polite=False, omit_subject=False, pronoun=False).split("。")[-2].split("、")[-1] + "。" if False else _ja(f)
    P = i18n.SUR[f.sur]
    if lang == "en":
        p = P[0]
        t, fu = TIME["en"][f.ti], FUT["en"][f.fi]
        return {"DONE": f"{p} {en[1]} {t}.", "NOT_DONE": f"{p} hasn't {en[2]} yet.", "PLANNED": f"{p} is planning to {en[0]} {fu}.", "WANTED": f"{p} wants to {en[0]}.",
                "UNDECIDED": f"{p} is thinking about {en[3]}, but hasn't decided yet.", "HEARSAY": f"I heard that {p} {en[1]}, but I'm not sure if it's true.",
                "POSSIBLE": f"{p} might {en[0]}."}[s]
    if lang == "ko":
        p = P[1]
        t, fu = TIME["ko"][f.ti], FUT["ko"][f.fi]
        return {"DONE": f"{ko_josa(p, 'topic')} {t} {ko}했어요.", "NOT_DONE": f"{ko_josa(p, 'topic')} 아직 {ko}하지 않았어요.", "PLANNED": f"{ko_josa(p, 'topic')} {fu} {ko}할 예정이에요.",
                "WANTED": f"{ko_josa(p, 'topic')} {ko}하고 싶어 해요.", "UNDECIDED": f"{ko_josa(p, 'topic')} {ko}할지 고민하고 있어요. 아직 정하지 않았어요.",
                "HEARSAY": f"{ko_josa(p, 'subj')} {ko}했다고 들었는데 사실인지 모르겠어요.", "POSSIBLE": f"{ko_josa(p, 'topic')} {ko}할지도 몰라요."}[s]
    t, fu = TIME["zh"][f.ti], FUT["zh"][f.fi]
    p = f.sur
    return {"DONE": f"{p}{t}{zh}了。", "NOT_DONE": f"{p}还没有{zh}。", "PLANNED": f"{p}打算{fu}{zh}。", "WANTED": f"{p}想{zh}。", "UNDECIDED": f"{p}在考虑要不要{zh}，还没决定。",
            "HEARSAY": f"听说{p}{zh}了，不知道是不是真的。", "POSSIBLE": f"{p}可能会{zh}。"}[s]


def _ja(f: Fact) -> str:
    ev = jpdata.event_of(f.ev)
    c = jpdata.conj(ev)
    who = f.sur + "さん"
    t, fu = jpdata.TIMES[f.ti], jpdata.FUT[f.fi]
    return {"DONE": f"{who}は{t}{c['ta']}。", "NOT_DONE": f"{who}はまだ{c['te']}いない。", "PLANNED": f"{who}は{fu}{c['dict']}予定だ。", "WANTED": f"{who}は{c['stem']}たがっている。",
            "UNDECIDED": f"{who}は{c['vol']}か迷っている。まだ決めていない。", "HEARSAY": f"{who}が{c['ta']}って聞いたけど、本当かどうか分からない。",
            "POSSIBLE": f"{who}は{c['dict']}かもしれない。"}[f.status]


def question(f: Fact, lang: str) -> str:
    en = EN[f.ev]
    if lang == "ja":
        return f"{f.sur}さんは{jpdata.conj(jpdata.event_of(f.ev))['ta']}？"
    if lang == "en":
        return f"Did {i18n.SUR[f.sur][0]} {en[0]}?"
    if lang == "ko":
        return f"{ko_josa(i18n.SUR[f.sur][1], 'topic')} {KO[f.ev]}했어요?"
    return f"{f.sur}{ZH[f.ev]}了吗？"


# --------------------------------------------------------------------------------------------- analyzers / L1
EN_STOP = set("a an the to of in on at for with by from as is are was were be been being am has have had do does did will would shall that this these those it its i you he she we they me him her us them my your his our their and or but if so than then there here".split())
EN_PAST_IRREG = {"got", "went", "won", "left", "was", "were", "quit", "found", "gone", "gotten", "been", "heard"}
EN_MARK = {"not": "NOT", "n't": "NOT", "never": "NOT", "might": "MAY", "may": "MAY", "maybe": "MAY", "perhaps": "MAY", "want": "WANT", "wants": "WANT", "wanted": "WANT", "heard": "HEAR"}


@lru_cache(maxsize=None)
def _en_nlp():
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
    nlp.initialize()
    return nlp


@lru_cache(maxsize=None)
def _kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


def l1_en(text: str) -> str:
    out = []
    for t in _en_nlp()(text):
        w = t.text.lower()
        if t.is_punct or t.is_space:
            continue
        if w in EN_MARK and not (w == "may" and t.text != "may"):
            if not out or out[-1] != EN_MARK[w]:
                out.append(EN_MARK[w])
            continue
        if w in EN_STOP:
            continue
        lem = t.lemma_.lower()
        if (w != lem and (w.endswith("ed") or w in EN_PAST_IRREG)) and "PST" not in out[-1:]:
            out.append(lem)
            out.append("PST")
            continue
        out.append(lem)
    return " ".join(out)


KO_DROP = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "EF", "EC", "ETN", "ETM", "SF", "SP", "SS", "SE", "SO", "SW", "XSV", "XSA", "NNB", "VCP", "XSN"}


def l1_ko(text: str) -> str:
    out = []
    toks = _kiwi().tokenize(text)
    for i, t in enumerate(toks):
        f, tag = t.form, t.tag.split("-")[0]
        if tag == "EC" and f == "ᆯ지" and i + 1 < len(toks) and toks[i + 1].form == "도":
            out.append("MAY")
            continue
        if tag == "NNB" and f in ("달", "주", "년", "월", "일"):
            out.append(f)
            continue
        if tag in KO_DROP:
            continue
        if tag == "EP" and f in ("었", "았", "였"):
            if not out or out[-1] != "PST":
                out.append("PST")
            continue
        if f in ("않", "못", "안") or (tag == "VX" and f == "않"):
            out.append("NOT")
            continue
        if f == "싶" and tag == "VX":
            out.append("WANT")
            continue
        if f == "듣" and tag == "VV" and any(x.form == "다고" for x in toks[max(0, i - 3):i]):
            out.append("HEAR")
            continue
        if tag in ("NNG", "NNP", "NR", "SN", "SL", "MAG", "MM", "XR", "NP"):
            if f in ("아직", "정말", "진짜", "아주"):
                continue
            out.append(f)
        elif tag in ("VV", "VA", "VX"):
            if f in ("있", "하", "되", "싶", "이") and tag in ("VX", "VV"):
                continue
            out.append(f + "다")
    return " ".join(out)


def l1_zh(text: str) -> str:
    import jieba.posseg as pseg
    out = []
    for w in pseg.cut(text):
        f, fl = w.word, w.flag
        if fl.startswith("x") or fl in ("p", "c", "y", "e", "o", "m", "q", "r") or f in ("要", "是", "会", "的", "在", "能"):
            continue
        if f in ("没有", "没", "不", "不要") and fl in ("v", "d", "df"):
            if not out or out[-1] != "NOT":
                out.append("NOT")
            continue
        if f == "了":
            if not out or out[-1] != "PST":
                out.append("PST")
            continue
        if f in ("听说",):
            out.append("HEAR")
            continue
        if f in ("可能", "也许", "或许"):
            out.append("MAY")
            continue
        if f in ("想", "想要"):
            out.append("WANT")
            continue
        if f in ("还", "还没", "已经"):
            continue
        out.append(f)
    return " ".join(out)


_ja_conv = None


def l1_ja(text: str) -> str:
    global _ja_conv
    if _ja_conv is None:
        _ja_conv = jl1.Converter("sudachi", "m")
    return _ja_conv(text)


L1 = {"ja": l1_ja, "en": l1_en, "ko": l1_ko, "zh": l1_zh}

# L1n: same words, but the meaning markers are written as native words instead of ASCII labels (NOT/PST/HEAR/WANT/MAY)
NATIVE = {"en": {"NOT": "not", "PST": "-ed", "HEAR": "heard", "WANT": "want", "MAY": "might"},
          "ko": {"NOT": "않", "PST": "었", "HEAR": "들었", "WANT": "싶", "MAY": "ᆯ지도"},
          "zh": {"NOT": "没", "PST": "了", "HEAR": "听说", "WANT": "想", "MAY": "可能"}}


def l1n(text: str, lang: str) -> str:
    base = L1[lang](text)
    if lang == "ja":
        return base
    return " ".join(NATIVE[lang].get(w, w) for w in base.split())

# status recovery by native rules (marker + content cue) from the L1 text alone
CUES = {"ja": dict(MAY="かも", HEAR=None, UND=("迷う", "決める"), PLAN=("予定",), WANT=("たい",), NOT="ない", PST="た"),
        "en": dict(MAY="MAY", HEAR="HEAR", UND=("think", "decide"), PLAN=("plan",), WANT="WANT", NOT="NOT", PST="PST"),
        "ko": dict(MAY="MAY", HEAR="HEAR", UND=("고민",), PLAN=("예정",), WANT="WANT", NOT="NOT", PST="PST"),
        "zh": dict(MAY="MAY", HEAR="HEAR", UND=("考虑",), PLAN=("打算",), WANT="WANT", NOT="NOT", PST="PST")}


def status_from_l1(text: str, lang: str) -> str:
    c = CUES[lang]
    n = text.replace(" ", "")
    has = lambda k: (c[k] in text.split() if lang != "ja" else c[k] in n) if isinstance(c[k], str) else any(w in n for w in c[k])
    if has("MAY"):
        return "POSSIBLE"
    if (c["HEAR"] and has("HEAR")) or (lang == "ja" and "聞く" in n and "ない" in n):
        return "HEARSAY"
    if has("UND"):
        return "UNDECIDED"
    if has("PLAN"):
        return "PLANNED"
    if has("WANT"):
        return "WANTED"
    if has("NOT"):
        return "NOT_DONE"
    return "DONE" if has("PST") else "UNKNOWN"


if __name__ == "__main__":
    fs = make_facts(14)
    for lang in LANGS:
        print("==", lang)
        for f in fs[:7]:
            t = render(f, lang)
            print(f"  {f.status:9} {t}  ->  {L1[lang](t)}   [{status_from_l1(L1[lang](t), lang)}]  Q: {question(f, lang)}")
