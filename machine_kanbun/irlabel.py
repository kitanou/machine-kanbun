"""Label-localisation ablation of the Semantic IR (Issue #17).

Same skeleton (`class name{key:value;group{...};...}`) rendered with three vocabularies:

  IR-C   canonical kanji labels (the current MKW; kanji are a shared vocabulary across languages)
  IR-L   every structural word and concept word in the input language (EN / KO); JA is identical
         to IR-C because Japanese's own words *are* kanji
  IR-ID  structural words (class, keys, groups, tense, operators, relation verbs) replaced by opaque
         codes (E1, K2, G3, T1, O1, R5 ...) that are the same in every language; content words
         (names, concept nouns, numbers, units) stay as in IR-L
  IR-IDL IR-ID plus a legend that maps the codes back to words (cost counted as a fixed overhead)

Label families (Issue #19 E) keep content words (names, concept nouns, numbers) as in IR-L and vary
only the structural words:
  IR-CS  kanji structural labels (like IR-C) + local content words (the "pure label" version of IR-C)
  IR-EN  English structural labels in every language (an English "universal" label set)
  IR-SYM rare Unicode symbols, one per structural word, language independent (opaque *and* token-expensive)

Symbols (`{ } ; : > ?`) are language-independent and kept in every condition.
"""
from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Tuple

from . import i18n, mlenc
from .encoder import _grouped
from .model import Fact, Profile

CONDS = ("IRC", "IRL", "IRID", "IRIDL")
EXTRA_CONDS = ("IRCS", "IREN", "IRSYM")  # Issue #19 (E): label families with the *content words* held at IR-L


def _t(s: str) -> Dict[str, Tuple[str, str]]:
    return {a: (b, c) for a, b, c in (x.split(":") for x in s.split())}


# ------------------------------------------------------------------ structural vocabulary (kind -> JA token -> (EN, KO))
CLASS = _t("人物:Person:인물 案件:Project:프로젝트 催事:Event:행사")
KEY = _t("生:born:출생 職:occupation:직업 住:residence:거주 愛称:nickname:별명 日:date:날짜 場:venue:장소 定員:capacity:정원 予算:budget:예산 "
         "期限:due:기한 担当:owner:담당 人数:team_size:인원 言語:language:언어 DB:db:DB")
GROUP = _t("食:food:식사 医:medical:의료 機:devices:기기 習:habits:습관 運:operations:운영 規:rules:규정 屋台:stalls:노점 版:version:버전 工:engineering:개발")
TENSE = _t("過:past:과거 今:now:현재 将:future:미래")
OP = _t("不:not:안 未:not_yet:아직안 非:is_not:아님 禁:prohibited:금지 無:none:없음 有:exists:있음 疑:uncertain:불확실 必:always:반드시 故:so:그래서")
REL = _t("好:like:좋아함 飲:drink:마심 食:eat:먹음 行:go:감 速:faster:빠름 要:needed:필요 止:cancel:취소 設:set_up:설치 実施:held:실시 超過:exceeded:초과 可:allowed:가능")
# Simplified-Chinese words (Issue #19 A). Chinese uses hanzi natively, so the canonical kanji labels (IR-C) are partly native
# and IR-L is the simplified-hanzi version (職->职业, 医療->医疗 ...).
def _z(s: str) -> Dict[str, str]:
    return dict(x.split(":") for x in s.split())


VOC_ZH = {
    "class": _z("人物:人物 案件:项目 催事:活动"),
    "key": _z("生:出生 職:职业 住:居住 愛称:昵称 日:日期 場:场地 定員:容量 予算:预算 期限:截止 担当:负责人 人数:团队人数 言語:语言 DB:数据库"),
    "group": _z("食:饮食 医:医疗 機:设备 習:习惯 運:运营 規:规定 屋台:摊位 版:版本 工:开发"),
    "tense": _z("過:过去 今:现在 将:将来"),
    "op": _z("不:不 未:尚未 非:并非 禁:禁止 無:无 有:有 疑:存疑 必:必定 故:因此"),
    "rel": _z("好:喜欢 飲:喝 食:吃 行:去 速:更快 要:需要 止:取消 設:设置 実施:举办 超過:超出 可:允许"),
}
NOUN_ZH = _z("鍋:火锅 十割蕎麦:纯荞麦面 カレー:咖喱 寿司:寿司 焼き鳥:烤鸡肉串 ラーメン:拉面 天ぷら:天妇罗 お好み焼き:大阪烧 餃子:饺子 うなぎ:鳗鱼 "
             "納豆:纳豆 ヨーグルト:酸奶 チーズ:奶酪 パクチー:香菜 生卵:生鸡蛋 レバー:肝脏 酒:酒 夜間珈琲:夜间咖啡 火気:明火 花火:烟花 "
             "警察届出:警方报备 負荷試験:压力测试 本番直接変更:直接修改生产环境 外注:外包 障害原因:故障原因 AWS移行:AWS迁移 駐車場:停车场 "
             "雨:下雨 強風:大风 雪:下雪 予算:预算 散歩:散步 催事:活动 屋台:摊位 釣り:钓鱼 部長承認:部长批准 ランニング:跑步 仕様変更:需求变更 "
             "睡眠不足:睡眠不足 運動不足:运动不足 残業過多:加班过多 日中眠:白天犯困 体重増:体重增加 体調悪化:身体状况恶化 "
             "前回催事止:上次活动取消 開発遅延:开发延迟 蕎麦:荞麦面 うどん:乌冬面 珈琲:咖啡 紅茶:红茶 犬:狗 猫:猫 山:山 海:大海 朝:早晨 "
             "夜:夜晚 電車:电车 バス:公交车")
# opaque, language-independent codes
CODES: Dict[str, Dict[str, str]] = {
    "class": {k: f"E{i}" for i, k in enumerate(CLASS, 1)}, "key": {k: f"K{i}" for i, k in enumerate(KEY, 1)},
    "group": {k: f"G{i}" for i, k in enumerate(GROUP, 1)}, "tense": {k: f"T{i}" for i, k in enumerate(TENSE, 1)},
    "op": {k: f"O{i}" for i, k in enumerate(OP, 1)}, "rel": {k: f"R{i}" for i, k in enumerate(REL, 1)},
}
_sym = random.Random(2026).sample([chr(c) for c in range(0x2300, 0x2400) if chr(c).isprintable()], 60)
SYMBOLS: Dict[str, Dict[str, str]] = {}
_i = 0
for _kind in ("class", "key", "group", "tense", "op", "rel"):
    SYMBOLS[_kind] = {}
    for _k in {"class": CLASS, "key": KEY, "group": GROUP, "tense": TENSE, "op": OP, "rel": REL}[_kind]:
        SYMBOLS[_kind][_k] = _sym[_i]
        _i += 1
VOC = {"class": CLASS, "key": KEY, "group": GROUP, "tense": TENSE, "op": OP, "rel": REL}

# ------------------------------------------------------------------ content nouns (kept as words in IR-L and IR-ID)
NOUN = _t("鍋:hot_pot:전골 十割蕎麦:soba:소바 カレー:curry:카레 寿司:sushi:초밥 焼き鳥:yakitori:야키토리 ラーメン:ramen:라멘 天ぷら:tempura:튀김 "
          "お好み焼き:okonomiyaki:오코노미야키 餃子:gyoza:만두 うなぎ:eel:장어 納豆:natto:낫토 ヨーグルト:yogurt:요거트 チーズ:cheese:치즈 "
          "パクチー:cilantro:고수 生卵:raw_egg:날계란 レバー:liver:간 酒:alcohol:술 夜間珈琲:night_coffee:밤커피 火気:open_flame:화기 "
          "花火:fireworks:불꽃놀이 警察届出:police_report:경찰신고 負荷試験:load_test:부하테스트 本番直接変更:direct_prod_change:운영직접변경 "
          "外注:outsourcing:외주 障害原因:incident_cause:장애원인 AWS移行:AWS_migration:AWS이전 駐車場:parking:주차장 "
          "雨:rain:비 強風:strong_wind:강풍 雪:snow:눈 予算:budget:예산 散歩:walk:산책 催事:event:행사 屋台:stalls:노점 釣り:fishing:낚시 "
          "部長承認:head_approval:부장승인 ランニング:running:달리기 仕様変更:spec_change:사양변경 "
          "睡眠不足:sleep_deprivation:수면부족 運動不足:lack_of_exercise:운동부족 残業過多:excess_overtime:야근과다 "
          "日中眠:daytime_sleepiness:낮졸림 体重増:weight_gain:체중증가 体調悪化:health_decline:건강악화 "
          "前回催事止:last_event_canceled:지난행사취소 開発遅延:dev_delay:개발지연 "
          "蕎麦:soba:소바 うどん:udon:우동 珈琲:coffee:커피 紅茶:tea:홍차 犬:dogs:개 猫:cats:고양이 山:mountains:산 海:sea:바다 朝:mornings:아침 "
          "夜:nights:밤 電車:trains:전철 バス:buses:버스")
UNIT = {"人": ("{n} people", "{n}명"), "名": ("{n} members", "{n}명"), "軒": ("{n} stalls", "{n}개")}


def _noun(t: str, lang: str) -> str:
    """Content word in the input language (underscores keep multi-word nouns together)."""
    if lang == "JA":
        return t
    if lang == "ZH":
        if t in NOUN_ZH:
            return NOUN_ZH[t]
        return t[:-2] + "购买" if t.endswith("購入") else t
    if t in NOUN:
        return NOUN[t][0 if lang == "EN" else 1]
    if t.endswith("購入"):  # "Kindle購入" -> "Kindle_purchase"
        return t[:-2] + ("_purchase" if lang == "EN" else "_구매")
    return t  # Latin names / numbers


def _struct(kind: str, t: str, lang: str, cond: str) -> str:
    if cond in ("IRC", "IRCS") or (lang == "JA" and cond == "IRL"):
        return t
    if cond == "IREN":
        return VOC[kind][t][0]
    if cond == "IRSYM":
        return SYMBOLS[kind][t]
    if cond == "IRID" or cond == "IRIDL":
        return CODES[kind][t]
    if lang == "ZH":
        return VOC_ZH[kind][t]
    return VOC[kind][t][0 if lang == "EN" else 1]


def _word(kind: str, t: str, lang: str, cond: str) -> str:
    """kind in structural kinds -> label vocabulary; 'noun' -> content word."""
    if kind == "noun":
        return t if cond == "IRC" else _noun(t, lang)
    return _struct(kind, t, lang, cond)


def _value(key: str, v: str, lang: str, cond: str) -> str:
    """Attribute value: names are already localised; concept values / units follow the condition."""
    if cond == "IRC" or lang == "JA":
        return v
    if lang == "ZH":
        from . import i18n_zh
        if key == "職":
            return i18n_zh.JOB[v]
        if key == "日":
            return v  # "11月26日" is already Chinese
        if key == "予算":
            return f"{int(v[:-2])}万日元"
        if key in ("定員", "人数"):
            return f"{v[:-1]}人"
        if key == "場":
            return v[:-2] + "公园" if v.endswith("公園") else v
        return v
    i = 0 if lang == "EN" else 1
    if key == "職":
        return i18n.JOB[v][i]
    if key == "日":
        m, d = (int(x) for x in re.match(r"(\d+)月(\d+)日", v).groups())
        return f"{i18n.MONTHS[m - 1][:3]} {d}" if lang == "EN" else f"{m}월 {d}일"
    if key == "予算":
        n = int(v[:-2])
        return f"{n * 10000:,} yen" if lang == "EN" else f"{n}만 엔"
    if key in ("定員", "人数"):
        return UNIT[v[-1]][i].format(n=v[:-1])
    if key == "場":
        return v[:-2] + (" Park" if lang == "EN" else "공원") if v.endswith("公園") else v
    return v


def _phrase(spec: Dict[str, str], lang: str, cond: str) -> str:
    """obj + adv + neg + pred of a cond / cause / effect spec."""
    parts: List[str] = []
    joined = spec.get("obj", "") + spec.get("neg", "") + spec.get("pred", "")
    if joined in NOUN:  # whole cause/effect expression is one content word (睡眠不足, 開発遅延 ...)
        return _word("noun", joined, lang, cond)
    if spec.get("obj"):
        parts.append(_word("noun", spec["obj"], lang, cond))
    if spec.get("adv"):
        parts.append(_word("op", spec["adv"], lang, cond))
    if spec.get("neg"):
        parts.append(_word("op", spec["neg"], lang, cond))
    if spec.get("pred"):
        p = spec["pred"]
        parts.append(_word("rel", p, lang, cond) if p in REL else _word("noun", p, lang, cond))
    return _join(parts, cond)


def _join(parts: List[str], cond: str) -> str:
    # IR-C concatenates kanji; word vocabularies need a separator
    return "".join(parts) if cond == "IRC" else " ".join(parts)


def _core(f: Fact, lang: str, cond: str) -> str:
    w = lambda k, t: _word(k, t, lang, cond)
    if f.kind == "simple":
        parts = [w("noun", f.obj)] if f.obj else []
        if f.adv:
            parts.append(w("op", f.adv))
        if f.neg:
            parts.append(w("op", f.neg))
        if f.pred:
            parts.append(w("rel", f.pred) if f.pred in REL else w("op", f.pred) if f.pred in OP else w("noun", f.pred))
        out = _join(parts, cond)
    elif f.kind == "attr":
        out = f"{w('key', f.key)}:{_value(f.key, f.value, lang, cond)}"
    elif f.kind == "time":
        v = f.value
        if v.endswith("軒") and cond != "IRC" and lang != "JA":
            v = f"{v[:-1]}个摊位" if lang == "ZH" else UNIT["軒"][0 if lang == "EN" else 1].format(n=v[:-1])
        out = f"{w('tense', f.tense)}:{v}"
    elif f.kind == "compare":
        out = f"{w('rel', f.pred)}:{_noun(f.a, lang) if cond != 'IRC' else f.a}>{_noun(f.b, lang) if cond != 'IRC' else f.b}"
    elif f.kind == "cause":
        sep = "" if cond == "IRC" else " "
        out = f"{_phrase(f.cause, lang, cond)}{sep}{w('op', '故')}{sep}{_phrase(f.effect, lang, cond)}"
    elif f.kind == "cond":
        out = f"{_phrase(f.if_, lang, cond)}?{_phrase(f.then, lang, cond)}"
    else:
        raise ValueError(f.kind)
    return f"{w('op', '疑')}:{out}" if f.unc else out


def _label(p: Profile, lang: str, cond: str) -> str:
    cls = p.meta["cls"]
    q = mlenc.localize(p, lang)  # proper names in the source script (all conditions)
    name = q.label[len(cls):]
    if cond in ("IRC", "IRCS") or (lang == "JA" and cond == "IRL"):
        return q.label
    if cond == "IRSYM":
        return f"{SYMBOLS['class'][cls]}:{_event_name(p, lang) if cls == '催事' else name}"
    if cond == "IREN":
        return f"{CLASS[cls][0]} {_event_name(p, lang) if cls == '催事' else name}"
    if cond in ("IRID", "IRIDL"):
        sep = " " if p.meta["cls"] == "催事" and lang != "JA" else ""
        name = _event_name(p, lang) if cls == "催事" else name
        return f"{CODES['class'][cls]}:{name}"
    c = (VOC_ZH["class"][cls] if lang == "ZH" else CLASS[cls][0 if lang == "EN" else 1]) if lang != "JA" else cls
    return f"{c} {_event_name(p, lang) if cls == '催事' else name}"


def _event_name(p: Profile, lang: str) -> str:
    """'Chuo Harvest Festival' / '주오 수확제' (season words are content words, localised)."""
    m = p.meta
    if lang == "JA":
        return m["place"] + m["season"]
    if lang == "ZH":
        from . import i18n_zh
        return i18n_zh.PLACE[m["place"]] + i18n_zh.SEASON[m["season"]]
    i = 0 if lang == "EN" else 1
    return f"{i18n.PLACE[m['place']][i]} {i18n.SEASON[m['season']][i]}"


def render(profiles: Iterable[Profile], lang: str, cond: str) -> str:
    out = []
    for p in profiles:
        parts = []
        lp = mlenc.localize(p, lang)
        for g, fs in _grouped(lp).items():
            body = ";".join(_core(f, lang, cond) for f in fs)
            gname = _word("group", g, lang, cond) if g else ""
            parts.append(f"{gname}{{{body}}}" if g else body)
        out.append(_label(p, lang, cond) + "{" + ";".join(parts) + "}")
    return "\n".join(out)


def legend(lang: str) -> str:
    """Code -> word table (IR-IDL), in the question language."""
    i = 0 if lang == "EN" else 1
    items = []
    for kind in ("class", "key", "group", "tense", "op", "rel"):
        for ja, code in CODES[kind].items():
            word = ja if lang == "JA" else VOC_ZH[kind][ja] if lang == "ZH" else VOC[kind][ja][i]
            items.append(f"{code}={word}")
    head = {"JA": "対応表", "EN": "Code legend", "KO": "코드 대응표", "ZH": "对照表"}[lang]
    return f"{head}: " + " ".join(items) + " | ?=if-then >=greater-than"
