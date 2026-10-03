"""Deterministic long-context benchmark generator (Issue #4).

A *document* is a list of entity Profiles (人物 / 案件 / 催事) with randomised values.
The same facts are rendered in every format, so context length is defined by the
natural-Japanese (L0) token count and compressed formats simply come out shorter.
Questions are sampled per document, stratified by semantic category and operator,
and include cross-fact (two-hop / cross-entity) and reference-resolution questions.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from .model import Fact, Profile, Question

SURNAMES = ("佐藤 鈴木 高橋 田中 伊藤 渡辺 山本 中村 小林 加藤 吉田 山田 佐々木 山口 松本 井上 木村 林 斎藤 清水 "
            "山崎 森 池田 橋本 阿部 石川 山下 中島 石井 小川 前田 岡田 長谷川 藤田 後藤 近藤 村上 遠藤 青木 坂本 "
            "斉藤 福田 太田 西村 藤井 金子 岡本 藤原 三浦 中野 原田 松田 竹内 小野 田村 中山 和田 石田 上田 森田 柴田").split()
CODENAMES = ("Atlas Borea Cedar Delta Ember Fjord Glint Harbor Iris Juno Kilo Lumen Mosaic Nova Onyx Pylon Quartz "
             "Raven Sigma Tundra Umber Vertex Willow Xenon Yonder Zephyr Amber Birch Cobalt Dune Echo Flint Garnet "
             "Hazel Indigo Jade Kestrel Lotus Maple Nimbus Opal Pebble Quill Ridge Sable Topaz Ultra Violet Wren "
             "Alder Basil Coral Drift Elm Fern Gale Heron Ivy Jasper").split()
PLACES = "桜ヶ丘 中央 港町 緑地 駅前 河原 城址 丘上 湖畔 旧市街 北口 南公園 東通り 西浜 山手 若葉".split()
SEASONS = "春祭 夏祭 秋祭 冬祭 花火大会 収穫祭".split()
ALIASES = ("ポン タロ ミケ ハチ ゴン コタ マメ ソラ ムギ クロ シロ チビ モモ ラム レオ ココ ナナ ユキ ハル ケン "
           "ジロ サブ ヨシ トラ ブン ダイ ノリ リク ヒロ マル ミツ").split()
JOBS = "IT 医療 教育 建設 金融 農業 運輸 製造 小売 出版 法務 設計".split()
CITIES = "大阪 京都 神戸 名古屋 福岡 札幌 仙台 広島 横浜 金沢 熊本 新潟".split()
FOODS = "十割蕎麦 カレー 寿司 焼き鳥 ラーメン 天ぷら お好み焼き 餃子 うなぎ 鍋".split()
UNC_FOODS = "納豆 ヨーグルト チーズ パクチー 生卵 レバー".split()
DEVICES = "ThinkPad MacBook iPad Surface Chromebook Pixel iPhone Kindle Galaxy Xperia Lumix GoPro".split()
LANGS = "Go Rust Java Kotlin Swift TypeScript Python Ruby".split()
DBS = "PostgreSQL MySQL Redis MongoDB SQLite Oracle".split()
COMPARE_PAIRS = [("蕎麦", "うどん"), ("珈琲", "紅茶"), ("犬", "猫"), ("山", "海"), ("朝", "夜"), ("電車", "バス")]
WEATHER = [("雨", "散歩", "散歩に行かない"), ("雪", "ランニング", "ランニングをしない"), ("強風", "釣り", "釣りに行かない")]
CAUSES = [(("睡眠", "不", "足"), ("日中", "", "眠"), "睡眠不足のため日中眠くなる。", "睡眠不足で日中眠い。"),
          (("運動", "不", "足"), ("体重", "", "増"), "運動不足のため体重が増えた。", "運動不足で体重増。"),
          (("残業", "", "過多"), ("体調", "", "悪化"), "残業が多すぎて体調が悪化した。", "残業過多で体調悪化。")]

_YN = lambda b: "はい" if b else "いいえ"


def _fa(i, **kw):
    return Fact(id=i, **kw)


def build_person(rng: random.Random, name: str, alias: str) -> Tuple[Profile, List[Question]]:
    L = f"人物{name}"
    year = rng.randint(1950, 2002)
    job, city = rng.choice(JOBS), rng.choice(CITIES)
    food = rng.choice(FOODS)
    unc_food = rng.choice(UNC_FOODS)
    drinks = rng.random() < 0.4
    dev = rng.sample(DEVICES, 3)
    w, act, act_ja = rng.choice(WEATHER)
    c = rng.choice(CAUSES)
    cmp_a, cmp_b = rng.choice(COMPARE_PAIRS)
    facts = [
        _fa("生", kind="attr", key="生", value=str(year), ja=f"{L}は{year}年に生まれた。", summary=f"{year}年生まれ。"),
        _fa("職", kind="attr", key="職", value=job, ja=f"{L}の職業は{job}関係である。", summary=f"職業は{job}。"),
        _fa("住", kind="attr", key="住", value=city, ja=f"{L}は{city}に住んでいる。", summary=f"{city}在住。"),
        _fa("愛", kind="attr", key="愛称", value=alias, ja=f"{L}は周囲から「{alias}」と呼ばれている。", summary=f"愛称は{alias}。"),
        _fa("好", kind="simple", group="食", obj=food, pred="好", ja=f"{L}は{food}が好きである。", summary=f"{food}が好き。"),
        _fa("酒", kind="simple", group="食", obj="酒", neg="" if drinks else "不", pred="飲",
            ja=f"{L}は酒を{'飲む' if drinks else '飲まない'}。", summary=f"酒は{'飲む' if drinks else '飲まない'}。"),
        _fa("疑", kind="simple", group="食", unc=True, obj=unc_food, neg="不", pred="食",
            ja=f"{L}は{unc_food}を食べないらしいが、確認は取れていない。", summary=f"{unc_food}は食べないらしい(未確認)。"),
        _fa("比", kind="compare", group="食", pred="好", a=cmp_a, b=cmp_b, ja=f"{L}は{cmp_b}より{cmp_a}の方が好きである。",
            summary=f"{cmp_b}より{cmp_a}が好き。"),
        _fa("禁", kind="simple", group="医", obj="夜間珈琲", pred="禁", ja=f"{L}は医師から夜間にコーヒーを飲むことを禁じられている。",
            summary="夜のコーヒーは禁止。"),
        _fa("過", kind="time", group="機", tense="過", value=dev[0], ja=f"{L}は以前{dev[0]}を使っていた。", summary=f"以前は{dev[0]}。"),
        _fa("今", kind="time", group="機", tense="今", value=dev[1], ja=f"{L}は現在{dev[1]}を使っている。", summary=f"現在は{dev[1]}。"),
        _fa("将", kind="time", group="機", tense="将", value=dev[2], ja=f"{L}は将来{dev[2]}を使う予定である。", summary=f"将来は{dev[2]}を使う予定。"),
        _fa("未", kind="simple", group="機", obj=f"{dev[2]}購入", neg="未", ja=f"{dev[2]}はまだ購入していない。", summary=f"{dev[2]}は未購入。"),
        _fa("若", kind="cond", group="習", if_={"obj": w}, then={"obj": act, "neg": "不", "pred": "行"},
            ja=f"{w}の場合、{L}は{act_ja}。", summary=f"{w}なら{act_ja}。"),
        _fa("因", kind="cause", group="習", cause=dict(zip(("obj", "neg", "pred"), c[0])), effect=dict(zip(("obj", "neg", "pred"), c[1])),
            ja=f"{L}は{c[2]}", summary=c[3]),
    ]
    qs = [
        Question(f"{L}の生まれた年は？", "数値", answer=[str(year)], op="数"),
        Question(f"{L}の職業は？", "属性", answer=[job]),
        Question(f"{L}はどこに住んでいますか？", "属性", answer=[city]),
        Question(f"{L}は{food}を好みますか？", "嗜好", yn=True),
        Question(f"{L}は酒を飲みますか？", "否定(不)", yn=drinks, op="不"),
        Question(f"{L}は{unc_food}を食べないと確定していますか？", "不確実性", yn=False, op="疑"),
        Question(f"{L}は{cmp_a}と{cmp_b}のどちらが好きですか？", "比較", answer=[cmp_a], op=">"),
        Question(f"{L}は夜にコーヒーを飲んでもよいですか？", "否定(禁)", yn=False, op="禁"),
        Question(f"{L}が現在使っている機器は？", "時制", answer=[dev[1]], op="今"),
        Question(f"{L}が将来使う予定の機器は？", "時制", answer=[dev[2]], op="将"),
        Question(f"{L}が以前使っていた機器は？", "時制", answer=[dev[0]], op="過"),
        Question(f"{L}は{dev[2]}をすでに購入しましたか？", "否定(未)", yn=False, op="未"),
        Question(f"{w}の日、{L}は{act}をしますか？", "条件", yn=False, op="若"),
        Question(f"{L}の{c[1][0]}に関する問題の原因は？", "因果", answer=[c[0][0]], op="故"),
        Question(f"「{alias}」と呼ばれている人物の職業は？", "参照解決", answer=[job]),
    ]
    return Profile(L, L, facts, []), qs


def build_project(rng: random.Random, code: str, owner: str) -> Tuple[Profile, List[Question]]:
    L = f"案件{code}"
    y, m, d = rng.randint(2026, 2027), rng.randint(1, 12), rng.randint(1, 28)
    due = f"{y}-{m:02d}-{d:02d}"
    size, lang, db = rng.randint(3, 15), rng.choice(LANGS), rng.choice(DBS)
    vers = [f"v{n}" for n in rng.sample(range(1, 9), 3)]
    vers.sort()
    non_db = rng.choice([x for x in DBS if x != db])
    fa, fb = rng.sample(LANGS, 2)
    facts = [
        _fa("期", kind="attr", key="期限", value=due, ja=f"{L}の納期は{y}年{m}月{d}日である。", summary=f"納期{due}。"),
        _fa("担", kind="attr", key="担当", value=owner, ja=f"{L}の担当者は{owner}さんである。", summary=f"担当は{owner}。"),
        _fa("人", kind="attr", key="人数", value=f"{size}名", ja=f"{L}のチームは{size}名で構成されている。", summary=f"{size}名体制。"),
        _fa("言", kind="attr", key="言語", value=lang, ja=f"{L}の実装言語は{lang}である。", summary=f"言語は{lang}。"),
        _fa("D", kind="attr", key="DB", value=db, ja=f"{L}のデータベースには{db}を使っている。", summary=f"DBは{db}。"),
        _fa("過", kind="time", group="版", tense="過", value=vers[0], ja=f"{L}は以前の稼働版が{vers[0]}だった。", summary=f"旧版{vers[0]}。"),
        _fa("今", kind="time", group="版", tense="今", value=vers[1], ja=f"{L}の現在の稼働版は{vers[1]}である。", summary=f"現行{vers[1]}。"),
        _fa("将", kind="time", group="版", tense="将", value=vers[2], ja=f"{L}の次期稼働版は{vers[2]}になる予定である。", summary=f"次期{vers[2]}予定。"),
        _fa("未", kind="simple", group="工", obj="負荷試験", neg="未", ja=f"{L}の負荷試験はまだ実施していない。", summary="負荷試験は未実施。"),
        _fa("禁", kind="simple", group="工", obj="本番直接変更", pred="禁", ja=f"{L}では本番環境を直接変更することは禁止されている。", summary="本番直接変更は禁止。"),
        _fa("不", kind="simple", group="工", obj="外注", neg="不", pred="可", ja=f"{L}では外注は認められていない。", summary="外注不可。"),
        _fa("非", kind="simple", group="工", obj="障害原因", neg="非", pred=non_db, ja=f"{L}の先日の障害の原因は{non_db}ではなかった。", summary=f"障害原因は{non_db}ではない。"),
        _fa("因", kind="cause", group="工", cause={"obj": "仕様変更"}, effect={"obj": "開発", "pred": "遅延"},
            ja=f"{L}は仕様変更のため開発が遅れている。", summary="仕様変更で遅延。"),
        _fa("若", kind="cond", group="工", if_={"obj": "予算", "pred": "超過"}, then={"obj": "部長承認", "pred": "要"},
            ja=f"{L}は予算を超過する場合は部長の承認が必要である。", summary="予算超過なら部長承認要。"),
        _fa("比", kind="compare", group="工", pred="速", a=fa, b=fb, ja=f"{L}では処理速度は{fb}より{fa}の方が速い。", summary=f"{fa}は{fb}より速い。"),
        _fa("疑", kind="simple", group="工", unc=True, obj="AWS移行", ja=f"{L}はAWSへの移行を検討中だが、確定ではない。", summary="AWS移行は検討中(未確定)。"),
    ]
    qs = [
        Question(f"{L}の納期は？", "時系列", answer=[str(y), str(m), str(d)]),
        Question(f"{L}の担当者は誰ですか？", "人物", answer=[owner]),
        Question(f"{L}のチームは何名ですか？", "数値", answer=[str(size)], op="数"),
        Question(f"{L}の実装言語は？", "属性", answer=[lang]),
        Question(f"{L}で使用しているデータベースは？", "属性", answer=[db]),
        Question(f"{L}の現在稼働しているバージョンは？", "時制", answer=[vers[1]], op="今"),
        Question(f"{L}の次期稼働バージョンは？", "時制", answer=[vers[2]], op="将"),
        Question(f"{L}の以前稼働していたバージョンは？", "時制", answer=[vers[0]], op="過"),
        Question(f"{L}の負荷試験は実施済みですか？", "否定(未)", yn=False, op="未"),
        Question(f"{L}で本番環境を直接変更してもよいですか？", "否定(禁)", yn=False, op="禁"),
        Question(f"{L}の開発を外注できますか？", "否定(不)", yn=False, op="不"),
        Question(f"{L}の先日の障害の原因は{non_db}ですか？", "否定(非)", yn=False, op="非"),
        Question(f"{L}の開発が遅れている原因は？", "因果", answer=["仕様変更"], op="故"),
        Question(f"{L}で予算を超過する場合、何が必要ですか？", "条件", answer=["部長", "承認"], op="若"),
        Question(f"{L}で{fa}と{fb}のどちらの処理速度が速いですか？", "比較", answer=[fa], op=">"),
        Question(f"{L}のAWSへの移行は確定していますか？", "不確実性", yn=False, op="疑"),
    ]
    return Profile(L, L, facts, []), qs


def build_event(rng: random.Random, name: str) -> Tuple[Profile, List[Question]]:
    L = f"催事{name}"
    day = f"{rng.randint(9, 11)}月{rng.randint(1, 28)}日"
    place = rng.choice(PLACES) + "公園"
    cap = rng.choice(range(100, 1001, 50))
    bud = rng.choice(range(50, 301, 10))
    stalls = sorted(rng.sample(range(10, 80), 3))
    parking = rng.random() < 0.5
    facts = [
        _fa("日", kind="attr", key="日", value=day, ja=f"{L}の開催日は{day}である。", summary=f"開催日は{day}。"),
        _fa("場", kind="attr", key="場", value=place, ja=f"{L}の会場は{place}である。", summary=f"会場は{place}。"),
        _fa("定", kind="attr", key="定員", value=f"{cap}人", ja=f"{L}の来場者の定員は{cap}人である。", summary=f"定員{cap}人。"),
        _fa("予", kind="attr", key="予算", value=f"{bud}万円", ja=f"{L}の予算は{bud}万円である。", summary=f"予算{bud}万円。"),
        _fa("雨", kind="cond", group="運", if_={"obj": "雨"}, then={"obj": "催事", "pred": "止"}, ja=f"{L}は雨が降った場合は中止する。", summary="雨天中止。"),
        _fa("風", kind="cond", group="運", if_={"obj": "強風"}, then={"obj": "屋台", "neg": "不", "pred": "設"}, ja=f"{L}は強風の場合は屋台を設置しない。", summary="強風なら屋台は設置しない。"),
        _fa("因", kind="cause", group="運", cause={"obj": "雨"}, effect={"obj": "前回催事", "pred": "止"}, ja=f"{L}の前回は雨のため中止になった。", summary="前回は雨で中止。"),
        _fa("禁", kind="simple", group="規", obj="火気", pred="禁", ja=f"{L}の会場内での火気の使用は禁止されている。", summary="火気禁止。"),
        _fa("駐", kind="simple", group="規", obj="駐車場", pred="有" if parking else "無", ja=f"{L}の会場に駐車場は{'ある' if parking else 'ない'}。", summary=f"駐車場{'あり' if parking else 'なし'}。"),
        _fa("未", kind="simple", group="規", obj="警察届出", neg="未", ja=f"{L}の警察への届出はまだ行っていない。", summary="警察届出は未了。"),
        _fa("過", kind="time", group="屋台", tense="過", value=f"{stalls[0]}軒", ja=f"{L}の昨年の屋台は{stalls[0]}軒だった。", summary=f"昨年{stalls[0]}軒。"),
        _fa("今", kind="time", group="屋台", tense="今", value=f"{stalls[1]}軒", ja=f"{L}の現在の屋台の申込は{stalls[1]}軒である。", summary=f"現在申込{stalls[1]}軒。"),
        _fa("将", kind="time", group="屋台", tense="将", value=f"{stalls[2]}軒", ja=f"{L}の屋台は最終的に{stalls[2]}軒になる予定である。", summary=f"将来{stalls[2]}軒予定。"),
        _fa("疑", kind="simple", group="運", unc=True, obj="花火", pred="実施", ja=f"{L}は花火を実施する可能性があるが、まだ確定していない。", summary="花火は実施かも(未確定)。"),
    ]
    qs = [
        Question(f"{L}の開催日は？", "属性", answer=[day]),
        Question(f"{L}の会場はどこですか？", "属性", answer=[place]),
        Question(f"{L}の来場者の定員は何人ですか？", "数値", answer=[str(cap)], op="数"),
        Question(f"{L}で雨が降ったら催事は中止になりますか？", "条件", yn=True, op="若"),
        Question(f"{L}で強風のとき屋台は設置されますか？", "条件", yn=False, op="若"),
        Question(f"{L}の前回催事が中止になった理由は？", "因果", answer=["雨"], op="故"),
        Question(f"{L}の会場内で火気を使用してよいですか？", "否定(禁)", yn=False, op="禁"),
        Question(f"{L}の会場に駐車場はありますか？", "否定(無)", yn=parking, op="無"),
        Question(f"{L}の警察への届出は完了していますか？", "否定(未)", yn=False, op="未"),
        Question(f"{L}の屋台の現在の申込数は？", "時制", answer=[str(stalls[1])], op="今"),
        Question(f"{L}の屋台の昨年の軒数は？", "時制", answer=[str(stalls[0])], op="過"),
        Question(f"{L}の屋台の将来の予定軒数は？", "時制", answer=[str(stalls[2])], op="将"),
        Question(f"{L}の花火の実施は確定していますか？", "不確実性", yn=False, op="疑"),
    ]
    return Profile(L, L, facts, []), qs


def generate(target_tokens: int, count: Callable[[str], int], seed: int = 0, n_questions: int = 40) -> Tuple[List[Profile], List[Question]]:
    """Add entities until the L0 rendering reaches `target_tokens`; return (entities, questions)."""
    rng = random.Random(seed)
    sn = rng.sample(SURNAMES, len(SURNAMES))
    cn = rng.sample(CODENAMES, len(CODENAMES))
    al = rng.sample(ALIASES, len(ALIASES))
    ev = [f"{p}{s}" for p in PLACES for s in SEASONS]
    rng.shuffle(ev)
    entities: List[Profile] = []
    pools: Dict[str, List[Question]] = {}
    persons: List[str] = []
    n = 0
    while True:
        # persons first (projects reference them), then rotate project / event / person
        kind = ("person", "project", "event")[n % 3] if len(persons) >= 2 else "person"
        n += 1
        if kind == "person":
            if not sn or not al:
                kind = "event"
            else:
                name = sn.pop()
                prof, qs = build_person(rng, name, al.pop())
                persons.append(name)
        if kind == "project":
            if not cn:
                kind = "event"
            else:
                prof, qs = build_project(rng, cn.pop(), rng.choice(persons))
        if kind == "event":
            if not ev:
                break
            prof, qs = build_event(rng, ev.pop())
        entities.append(prof)
        pools[prof.label] = qs
        if count("\n".join(f.ja for e in entities for f in e.facts)) >= target_tokens:
            break
    questions = _cross_questions(rng, entities, persons) + _sample_questions(rng, pools, n_questions)
    rng.shuffle(questions)
    return entities, questions[:n_questions]


def _cross_questions(rng, entities, persons):
    by = {e.label: e for e in entities}
    out: List[Question] = []
    projects = [e for e in entities if e.label.startswith("案件")]
    events = [e for e in entities if e.label.startswith("催事")]
    for pr in rng.sample(projects, min(3, len(projects))):
        owner = next(f.value for f in pr.facts if f.key == "担当")
        person = by.get(f"人物{owner}")
        if person:  # two-hop: project -> owner -> owner's attribute
            job = next(f.value for f in person.facts if f.key == "職")
            city = next(f.value for f in person.facts if f.key == "住")
            out.append(Question(f"{pr.label}の担当者の職業は？", "横断推論", answer=[job]))
            out.append(Question(f"{pr.label}の担当者はどこに住んでいますか？", "横断推論", answer=[city]))
    if len(events) >= 2:
        for _ in range(2):
            a, b = rng.sample(events, 2)
            ca = int(next(f.value for f in a.facts if f.key == "定員")[:-1])
            cb = int(next(f.value for f in b.facts if f.key == "定員")[:-1])
            if ca != cb:
                win = a if ca > cb else b
                out.append(Question(f"{a.label}と{b.label}では、定員が多いのはどちらですか？", "横断推論", answer=[win.label[2:]], op=">"))
    return out


def _sample_questions(rng, pools: Dict[str, List[Question]], n: int) -> List[Question]:
    """Round-robin over categories so every semantic category / operator is covered."""
    by_cat: Dict[str, List[Question]] = {}
    for qs in pools.values():
        for q in qs:
            by_cat.setdefault(q.cat, []).append(q)
    for v in by_cat.values():
        rng.shuffle(v)
    out: List[Question] = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop())
    rng.shuffle(out)
    return out
