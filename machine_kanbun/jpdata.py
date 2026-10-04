"""Japanese chat-memory benchmark (Issue #20): colloquial memories, similar-memory distractors, status questions.

A *memory* is 1-3 chat-like sentences about one (person, event) pair with a certainty/polarity *status*:

  DONE        done (past, certain)             -> question "…した？" = はい
  NOT_DONE    not done (negated)               -> いいえ
  PLANNED     planned / intended               -> いいえ (not done yet)
  WANTED      wished for                       -> いいえ (not done yet)
  UNDECIDED   undecided ("迷っているらしい")    -> 未確定
  HEARSAY     heard but unconfirmed            -> 未確定
  POSSIBLE    "…かもしれない"                    -> 未確定

Each (person, event) pair appears at most once in a memory set, so a query has exactly one relevant memory; all other
memories are distractors (same event / other person, same person / other event). Surface forms vary in register
(plain, polite), filler words, subject omission across two sentences, and pronouns (彼/彼女).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .gen import SURNAMES

GIVEN = ("太郎 花子 健太 美咲 大輔 愛 翔 結衣 拓也 陽子 直樹 恵 悠斗 彩 誠 優子 亮 千尋 和也 真理 駿 沙織 剛 由美 蓮 麻衣 康介 菜々子 "
         "隼人 瑞希 慎吾 香織 涼 里奈 達也 綾 洋平 舞").split()
STATUSES = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]
ANSWER = {"DONE": "はい", "NOT_DONE": "いいえ", "PLANNED": "いいえ", "WANTED": "いいえ", "UNDECIDED": "未確定", "HEARSAY": "未確定", "POSSIBLE": "未確定"}
FILLERS = ["なんか、", "ちょっと、", "まあ、", "えっと、", "そういえば、", "ところで、", ""]
CHAT = ["昨日のランチは美味しかったなあ。", "最近ちょっと寒いよね。", "週末は映画に行ってきた。", "今日はずっと雨だった。", "新しいカフェを見つけたよ。",
        "駅前が工事中で大変。", "最近ぜんぜん眠れなくて。", "来週は忙しくなりそう。"]
TIMES = ["先月", "去年", "3年前", "5月に", "この前", "先週の火曜日", "1か月前"]
FUT = ["来月", "来年", "再来週", "春ごろ", "年内に"]


@dataclass
class Event:
    key: str
    np: str          # object / complement phrase incl. particle, e.g. "会社を"
    verb: str        # dictionary form
    kind: str        # ichidan | godan | suru
    noun: str = ""   # for suru verbs (verb = noun + する)


EVENTS: List[Event] = [
    Event("辞職", "会社を", "辞める", "ichidan"), Event("引越", "大阪に", "引っ越す", "godan"), Event("結婚", "", "結婚する", "suru", "結婚"),
    Event("転職", "", "転職する", "suru", "転職"), Event("旅行", "沖縄に", "行く", "godan"), Event("合格", "試験に", "合格する", "suru", "合格"),
    Event("車購入", "新しい車を", "買う", "godan"), Event("犬", "犬を", "飼う", "godan"), Event("手術", "手術を", "受ける", "ichidan"),
    Event("留学", "", "留学する", "suru", "留学"), Event("起業", "", "起業する", "suru", "起業"), Event("免許", "運転免許を", "取る", "godan"),
    Event("家建築", "家を", "建てる", "ichidan"), Event("入院", "", "入院する", "suru", "入院"), Event("昇進", "", "昇進する", "suru", "昇進"),
    Event("優勝", "", "優勝する", "suru", "優勝"), Event("新事業", "新しい仕事を", "始める", "ichidan"), Event("家売却", "家を", "売る", "godan"),
    Event("ピアノ", "ピアノを", "習う", "godan"), Event("出産", "子供が", "生まれる", "ichidan"), Event("転勤", "海外に", "転勤する", "suru", "転勤"),
    Event("就職", "", "就職する", "suru", "就職"), Event("退院", "", "退院する", "suru", "退院"), Event("出張", "大阪に", "出張する", "suru", "出張"),
]
_GODAN = {"う": ("わ", "い", "って", "った", "おう"), "く": ("か", "き", "いて", "いた", "こう"), "ぐ": ("が", "ぎ", "いで", "いだ", "ごう"),
          "す": ("さ", "し", "して", "した", "そう"), "つ": ("た", "ち", "って", "った", "とう"), "ぬ": ("な", "に", "んで", "んだ", "のう"),
          "ぶ": ("ば", "び", "んで", "んだ", "ぼう"), "む": ("ま", "み", "んで", "んだ", "もう"), "る": ("ら", "り", "って", "った", "ろう")}


def conj(ev: Event) -> Dict[str, str]:
    """Verb forms: dict, ta, nai, nakatta, te, vol, stem (masu stem), plus polite past/negative."""
    v = ev.verb
    if ev.kind == "suru":
        n = ev.noun
        return dict(dict=v, ta=n + "した", nai=n + "しない", nakatta=n + "しなかった", te=n + "して", vol=n + "しよう", stem=n + "し",
                    pta=n + "しました", pneg=n + "していません")
    if ev.kind == "ichidan":
        s = v[:-1]
        return dict(dict=v, ta=s + "た", nai=s + "ない", nakatta=s + "なかった", te=s + "て", vol=s + "よう", stem=s, pta=s + "ました", pneg=s + "ていません")
    s, last = v[:-1], v[-1]
    a, i, te, ta, o = _GODAN[last]
    if v == "行く":
        te, ta = "って", "った"
        s_te = s
    return dict(dict=v, ta=s + ta, nai=s + a + "ない", nakatta=s + a + "なかった", te=s + te, vol=s + o, stem=s + i, pta=s + i + "ました", pneg=s + te + "いません")


@dataclass
class Memory:
    id: int
    person: str
    event: str
    status: str
    text: str
    register: str
    features: Dict[str, object] = field(default_factory=dict)


@dataclass
class Query:
    id: int
    text: str
    person: str
    event: str
    relevant: int
    answer: str


def render(rng: random.Random, name: str, ev: Event, status: str, polite: bool, omit_subject: bool, pronoun: bool) -> str:
    f = conj(ev)
    np = ev.np
    who = name + "さん"
    t = rng.choice(TIMES)
    fut = rng.choice(FUT)
    fill = rng.choice(FILLERS)
    if status == "DONE":
        core = f"{np}{f['pta']}" if polite else f"{np}{f['ta']}"
        s = (f"{who}は{t}{core}。" if polite else f"{who}、{t}{core}よ。")
    elif status == "NOT_DONE":
        s = (f"{who}は{np}{f['pneg']}。" if polite else f"{who}は{np}{f['te']}いない。")
    elif status == "PLANNED":
        s = (f"{who}は{fut}{np}{f['dict']}予定です。" if polite else f"{who}は{fut}{np}{f['dict']}つもりだ。")
    elif status == "WANTED":
        s = (f"{who}は{np}{f['stem']}たいと言っていました。" if polite else f"{who}は{np}{f['stem']}たがっている。")
    elif status == "UNDECIDED":
        s = (f"{who}は{np}{f['vol']}か迷っているようです。まだ決めていません。" if polite
             else f"{who}は{np}{f['vol']}か迷っているらしい。まだ決めたわけではないみたい。")
    elif status == "HEARSAY":
        s = (f"{who}が{np}{f['dict']}と聞きましたが、本当かどうか分かりません。" if polite
             else f"{who}が{np}{f['dict']}って聞いたけど、本当かどうか分からない。")
    else:  # POSSIBLE
        s = (f"{who}は{np}{f['dict']}かもしれません。" if polite else f"{who}は{np}{f['dict']}かもしれない。")
    if omit_subject and "。" in s:
        # two-sentence memory: sentence 1 introduces the person, sentence 2 omits the subject
        body = s.split("、", 1)[1] if (not polite and status == "DONE") else s
        core2 = body.replace(f"{who}は", "").replace(f"{who}が", "").replace(who, "")
        return f"{who}と話した。{fill}{core2}" if core2 != body else s
    if pronoun:
        pr = "彼" if rng.random() < 0.5 else "彼女"
        s2 = s.replace(who + "は", pr + "は").replace(who + "が", pr + "が").replace(who + "、", pr + "、")
        if s2 != s:
            return f"{who}に会った。{fill}{s2}"
    if rng.random() < 0.35:
        return f"{rng.choice(CHAT)}{fill}{s}"
    return f"{fill}{s}"


def make_memories(n: int, seed: int = 0) -> List[Memory]:
    """n memories over distinct (person, event) pairs. Person names are surname+given name, unique."""
    rng = random.Random(seed)
    pairs = [(f"{s}{g}", ev) for s in SURNAMES for g in GIVEN for ev in EVENTS]
    rng.shuffle(pairs)
    if n > len(pairs):
        raise ValueError(f"at most {len(pairs)} distinct memories")
    mems = []
    for i, (name, ev) in enumerate(pairs[:n]):
        status = STATUSES[rng.randrange(len(STATUSES))]
        polite = rng.random() < 0.3
        omit = rng.random() < 0.15
        pron = (not omit) and rng.random() < 0.10
        text = render(rng, name, ev, status, polite, omit, pron)
        mems.append(Memory(i, name, ev.key, status, text, "polite" if polite else "casual",
                           dict(omit_subject=omit, pronoun=pron)))
    return mems


def make_queries(mems: List[Memory], n: int, seed: int = 1) -> List[Query]:
    rng = random.Random(seed)
    ev = {e.key: e for e in EVENTS}
    qs = []
    for k, m in enumerate(rng.sample(mems, min(n, len(mems)))):
        e = ev[m.event]
        f = conj(e)
        polite = rng.random() < 0.3
        q = f"{m.person}さんは{e.np}{f['pta'] if polite else f['ta']}{'か' if polite else ''}？"
        qs.append(Query(k, q, m.person, m.event, m.id, ANSWER[m.status]))
    return qs


def event_of(key: str) -> Event:
    return next(e for e in EVENTS if e.key == key)
