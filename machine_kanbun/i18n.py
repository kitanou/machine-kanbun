"""English / Korean renderings of the canonical facts (Issue #11).

The Japanese generator (gen.py) stays untouched so that existing seeds reproduce exactly.
This module adds, for every fact and question, the English and Korean wording built from the
same parameters, plus the lexicon needed to localise proper names inside the machine-kanbun IR.

Design of the IR across languages: relations, operators and *concept* words stay in kanji
(language-independent), while *proper names* (people, aliases, cities, venues, owners) keep
the source language's script. So "JA/EN/KO -> MKW" differ only in those proper names.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------- lexicon
def _t(s: str) -> Dict[str, Tuple[str, str]]:
    out = {}
    for item in s.split():
        ja, en, ko = item.split(":")
        out[ja] = (en, ko)
    return out


SUR = _t("佐藤:Sato:사토 鈴木:Suzuki:스즈키 高橋:Takahashi:다카하시 田中:Tanaka:다나카 伊藤:Ito:이토 渡辺:Watanabe:와타나베 "
         "山本:Yamamoto:야마모토 中村:Nakamura:나카무라 小林:Kobayashi:고바야시 加藤:Kato:가토 吉田:Yoshida:요시다 山田:Yamada:야마다 "
         "佐々木:Sasaki:사사키 山口:Yamaguchi:야마구치 松本:Matsumoto:마쓰모토 井上:Inoue:이노우에 木村:Kimura:기무라 林:Hayashi:하야시 "
         "斎藤:Saito:사이토 清水:Shimizu:시미즈 山崎:Yamazaki:야마자키 森:Mori:모리 池田:Ikeda:이케다 橋本:Hashimoto:하시모토 "
         "阿部:Abe:아베 石川:Ishikawa:이시카와 山下:Yamashita:야마시타 中島:Nakajima:나카지마 石井:Ishii:이시이 小川:Ogawa:오가와 "
         "前田:Maeda:마에다 岡田:Okada:오카다 長谷川:Hasegawa:하세가와 藤田:Fujita:후지타 後藤:Goto:고토 近藤:Kondo:곤도 "
         "村上:Murakami:무라카미 遠藤:Endo:엔도 青木:Aoki:아오키 坂本:Sakamoto:사카모토 斉藤:Saitoh:사이토오 福田:Fukuda:후쿠다 "
         "太田:Ota:오타 西村:Nishimura:니시무라 藤井:Fujii:후지이 金子:Kaneko:가네코 岡本:Okamoto:오카모토 藤原:Fujiwara:후지와라 "
         "三浦:Miura:미우라 中野:Nakano:나카노 原田:Harada:하라다 松田:Matsuda:마쓰다 竹内:Takeuchi:다케우치 小野:Ono:오노 "
         "田村:Tamura:다무라 中山:Nakayama:나카야마 和田:Wada:와다 石田:Ishida:이시다 上田:Ueda:우에다 森田:Morita:모리타 柴田:Shibata:시바타")
ALIAS = _t("ポン:Pon:폰 タロ:Taro:타로 ミケ:Mike:미케 ハチ:Hachi:하치 ゴン:Gon:곤 コタ:Kota:코타 マメ:Mame:마메 ソラ:Sora:소라 ムギ:Mugi:무기 "
           "クロ:Kuro:쿠로 シロ:Shiro:시로 チビ:Chibi:치비 モモ:Momo:모모 ラム:Ramu:라무 レオ:Leo:레오 ココ:Koko:코코 ナナ:Nana:나나 "
           "ユキ:Yuki:유키 ハル:Haru:하루 ケン:Ken:켄 ジロ:Jiro:지로 サブ:Sabu:사부 ヨシ:Yoshi:요시 トラ:Tora:토라 ブン:Bun:분 ダイ:Dai:다이 "
           "ノリ:Nori:노리 リク:Riku:리쿠 ヒロ:Hiro:히로 マル:Maru:마루 ミツ:Mitsu:미쓰")
CITY = _t("大阪:Osaka:오사카 京都:Kyoto:교토 神戸:Kobe:고베 名古屋:Nagoya:나고야 福岡:Fukuoka:후쿠오카 札幌:Sapporo:삿포로 仙台:Sendai:센다이 "
          "広島:Hiroshima:히로시마 横浜:Yokohama:요코하마 金沢:Kanazawa:가나자와 熊本:Kumamoto:구마모토 新潟:Niigata:니가타")
PLACE = _t("桜ヶ丘:Sakuragaoka:사쿠라가오카 中央:Chuo:주오 港町:Minatomachi:미나토마치 緑地:Ryokuchi:료쿠치 駅前:Ekimae:에키마에 河原:Kawara:가와라 "
           "城址:Joshi:조시 丘上:Okaue:오카우에 湖畔:Kohan:고한 旧市街:Kyushigai:규시가이 北口:Kitaguchi:기타구치 南公園:Minamikoen:미나미코엔 "
           "東通り:Higashidori:히가시도리 西浜:Nishihama:니시하마 山手:Yamanote:야마노테 若葉:Wakaba:와카바")
SEASON = {"春祭": ("Spring Festival", "봄 축제"), "夏祭": ("Summer Festival", "여름 축제"), "秋祭": ("Autumn Festival", "가을 축제"),
          "冬祭": ("Winter Festival", "겨울 축제"), "花火大会": ("Fireworks Festival", "불꽃놀이 축제"), "収穫祭": ("Harvest Festival", "수확제")}
JOB = _t("IT:IT:IT 医療:healthcare:의료 教育:education:교육 建設:construction:건설 金融:finance:금융 農業:agriculture:농업 運輸:transportation:운송 "
         "製造:manufacturing:제조 小売:retail:소매 出版:publishing:출판 法務:legal:법무 設計:design:설계")
FOOD = _t("十割蕎麦:100%-buckwheat-soba:메밀-100%-소바 カレー:curry:카레 寿司:sushi:초밥 焼き鳥:yakitori:야키토리 ラーメン:ramen:라멘 "
          "天ぷら:tempura:튀김 お好み焼き:okonomiyaki:오코노미야키 餃子:gyoza:만두 うなぎ:eel:장어 鍋:hot-pot:전골")
UNC = _t("納豆:natto:낫토 ヨーグルト:yogurt:요거트 チーズ:cheese:치즈 パクチー:cilantro:고수 生卵:raw-eggs:날계란 レバー:liver:간")
PAIR = _t("蕎麦:soba:소바 うどん:udon:우동 珈琲:coffee:커피 紅茶:tea:홍차 犬:dogs:개 猫:cats:고양이 山:mountains:산 海:the-sea:바다 "
          "朝:mornings:아침 夜:nights:밤 電車:trains:전철 バス:buses:버스")
# (JA cond word, JA act, EN cond clause, EN act, KO cond clause, KO act (neg), KO act (question))
WEATHER = {"雨": ("it rains", "go for a walk", "비가 오면", "산책을 하지 않는다", "산책을 합니까"),
           "雪": ("it snows", "go running", "눈이 오면", "달리기를 하지 않는다", "달리기를 합니까"),
           "強風": ("the wind is strong", "go fishing", "바람이 강하면", "낚시를 가지 않는다", "낚시를 갑니까")}
# index == position in gen.CAUSES
CAUSES_I18N = [
    dict(en0="{L} feels sleepy during the day because of lack of sleep.", en1="Lack of sleep causes daytime sleepiness.",
         ko0="{L}{topic} 수면 부족 때문에 낮에 졸리다.", ko1="수면 부족으로 낮에 졸림.",
         prob_en="daytime sleepiness", prob_ko="낮 졸림", ans="睡眠|sleep|수면"),
    dict(en0="{L} gained weight because of lack of exercise.", en1="Lack of exercise: weight gain.",
         ko0="{L}{topic} 운동 부족 때문에 체중이 늘었다.", ko1="운동 부족으로 체중 증가.",
         prob_en="weight gain", prob_ko="체중 증가", ans="運動|exercise|운동"),
    dict(en0="{L}'s health worsened because of excessive overtime.", en1="Excessive overtime worsens health.",
         ko0="{L}{topic} 야근이 너무 많아 건강이 나빠졌다.", ko1="야근 과다로 건강 악화.",
         prob_en="health problems", prob_ko="건강 문제", ans="残業|overtime|야근"),
]
MONTHS = "January February March April May June July August September October November December".split()
# Korean reading of Latin words ending in a consonant sound (devices, languages, databases, codenames)
_KO_BATCHIM = {"MacBook", "Chromebook", "Pixel", "iPhone", "Kindle", "Kotlin", "Python", "PostgreSQL", "MySQL", "Oracle",
               "Garnet", "Violet"}
_KO_NO_BATCHIM = {"ThinkPad", "iPad", "Surface", "Galaxy", "Xperia", "Lumix", "GoPro", "Go", "Rust", "Java", "Swift",
                  "TypeScript", "Ruby", "Redis", "MongoDB", "SQLite"}
_DIGIT = {"0": True, "1": True, "2": False, "3": True, "4": False, "5": False, "6": True, "7": True, "8": True, "9": False}


def has_batchim(word: str) -> bool:
    """Does the Korean reading of `word` end in a final consonant (batchim)?"""
    w = word.rstrip(".,!?)\"」』 ")
    if w in _KO_BATCHIM:
        return True
    if w in _KO_NO_BATCHIM:
        return False
    c = w[-1]
    if "가" <= c <= "힣":
        return (ord(c) - 0xAC00) % 28 != 0
    if c.isdigit():
        return _DIGIT[c]
    lw = w.lower()  # Latin heuristic for loan words: ... -le / -l / -m / -n / -ng read with a final consonant
    return lw.endswith(("le", "l", "m", "n", "ng"))


def topic(w: str) -> str:
    return w + ("은" if has_batchim(w) else "는")


def subj(w: str) -> str:
    return w + ("이" if has_batchim(w) else "가")


def obj(w: str) -> str:
    return w + ("을" if has_batchim(w) else "를")


def with_(w: str) -> str:
    return w + ("과" if has_batchim(w) else "와")


def _sp(s: str) -> str:
    return s.replace("-", " ")


# ----------------------------------------------------------------------------- labels
def labels(cls: str, name_ja: str, **kw) -> Tuple[str, str, str]:
    """(ja, en, ko) display label of an entity, e.g. 人物石田 / Person Ishida / 인물 이시다."""
    if cls == "人物":
        e, k = SUR[name_ja]
        return f"人物{name_ja}", f"Person {e}", f"인물 {k}"
    if cls == "案件":
        return f"案件{name_ja}", f"Project {name_ja}", f"프로젝트 {name_ja}"
    if cls == "催事":
        place, season = kw["place"], kw["season"]
        pe, pk = PLACE[place]
        se, sk = SEASON[season]
        return f"催事{place}{season}", f"Event {pe} {se}", f"행사 {pk} {sk}"
    raise ValueError(cls)


def name_forms(cls: str, name_ja: str, **kw) -> Tuple[str, str, str]:
    """(ja, en, ko) bare name (label without class word)."""
    ja, en, ko = labels(cls, name_ja, **kw)
    n = {"人物": 2, "案件": 2, "催事": 2}[cls]
    return ja[n:], en.split(" ", 1)[1], ko.split(" ", 1)[1]


# ----------------------------------------------------------------------------- texts
Row = Tuple[str, str, str, str]  # en0, en1, ko0, ko1


def person_texts(L: Tuple[str, str, str], p: dict) -> Tuple[Dict[str, Row], List[Tuple[str, str, Optional[str]]]]:
    """p keys: name alias year job city food drinks unc dev(3) w cmp(a,b) ci (cause index)."""
    _, le, lk = L
    job_e, job_k = JOB[p["job"]]
    city_e, city_k = CITY[p["city"]]
    alias_e, alias_k = ALIAS[p["alias"]]
    food_e, food_k = _sp(FOOD[p["food"]][0]), _sp(FOOD[p["food"]][1])
    unc_e, unc_k = UNC[p["unc"]]
    a_e, a_k = PAIR[p["cmp"][0]]
    b_e, b_k = PAIR[p["cmp"][1]]
    a_e, b_e = _sp(a_e), _sp(b_e)
    d = p["dev"]
    w = WEATHER[p["w"]]
    c = CAUSES_I18N[p["ci"]]
    y = p["year"]
    T: Dict[str, Row] = {
        "生": (f"{le} was born in {y}.", f"Born {y}.", f"{topic(lk)} {y}년에 태어났다.", f"{y}년생."),
        "職": (f"{le} works in {job_e}.", f"Works in {job_e}.", f"{lk}의 직업은 {job_k} 관련이다.", f"직업은 {job_k}."),
        "住": (f"{le} lives in {city_e}.", f"Lives in {city_e}.", f"{topic(lk)} {city_k}에 살고 있다.", f"{city_k} 거주."),
        "愛": (f'People call {le} "{alias_e}".', f"Nickname: {alias_e}.", f'{topic(lk)} 주변에서 "{alias_k}"라고 불린다.', f"별명은 {alias_k}."),
        "好": (f"{le} likes {food_e}.", f"Likes {food_e}.", f"{topic(lk)} {obj(food_k)} 좋아한다.", f"{obj(food_k)} 좋아함."),
        "酒": ((f"{le} drinks alcohol.", "Drinks alcohol.", f"{topic(lk)} 술을 마신다.", "술은 마심.") if p["drinks"] else
               (f"{le} does not drink alcohol.", "No alcohol.", f"{topic(lk)} 술을 마시지 않는다.", "술은 안 마심.")),
        "疑": (f"{le} reportedly does not eat {unc_e}, but this has not been confirmed.", f"Reportedly doesn't eat {unc_e} (unconfirmed).",
               f"{topic(lk)} {obj(unc_k)} 먹지 않는다고 하지만 확인되지 않았다.", f"{topic(unc_k)} 안 먹는다고 함(미확인)."),
        "比": (f"{le} prefers {a_e} to {b_e}.", f"Prefers {a_e} to {b_e}.", f"{topic(lk)} {b_k}보다 {obj(a_k)} 더 좋아한다.", f"{b_k}보다 {a_k} 선호."),
        "禁": (f"{le} has been told by a doctor not to drink coffee at night.", "No coffee at night.",
               f"{topic(lk)} 의사에게 밤에 커피를 마시지 말라는 지시를 받았다.", "밤 커피 금지."),
        "過": (f"{le} previously used {d[0]}.", f"Previously {d[0]}.", f"{topic(lk)} 이전에 {obj(d[0])} 사용했다.", f"이전에는 {d[0]}."),
        "今": (f"{le} currently uses {d[1]}.", f"Now {d[1]}.", f"{topic(lk)} 현재 {obj(d[1])} 사용하고 있다.", f"현재는 {d[1]}."),
        "将": (f"{le} plans to use {d[2]} in the future.", f"Plans to use {d[2]}.", f"{topic(lk)} 앞으로 {obj(d[2])} 사용할 예정이다.", f"앞으로 {d[2]} 사용 예정."),
        "未": (f"{d[2]} has not been purchased yet.", f"{d[2]} not yet bought.", f"{topic(d[2])} 아직 구매하지 않았다.", f"{topic(d[2])} 미구매."),
        "若": (f"If {w[0]}, {le} does not {w[1]}.", f"If {w[0]}, doesn't {w[1]}.", f"{w[2]} {topic(lk)} {w[3]}.", f"{w[2]} {w[3]}."),
        "因": (c["en0"].format(L=le), c["en1"], c["ko0"].replace("{L}{topic}", topic(lk)), c["ko1"]),
    }
    q = [  # order must match gen.build_person's question list
        (f"In what year was {le} born?", f"{topic(lk)} 몇 년에 태어났습니까?", None),
        (f"What is {le}'s job?", f"{lk}의 직업은 무엇입니까?", f"{p['job']}|{job_e}|{job_k}"),
        (f"Where does {le} live?", f"{topic(lk)} 어디에 살고 있습니까?", f"{p['city']}|{city_e}|{city_k}"),
        (f"Does {le} like {food_e}?", f"{topic(lk)} {obj(food_k)} 좋아합니까?", None),
        (f"Does {le} drink alcohol?", f"{topic(lk)} 술을 마십니까?", None),
        (f"Is it confirmed that {le} does not eat {unc_e}?", f"{subj(lk)} {obj(unc_k)} 먹지 않는다는 것이 확인되었습니까?", None),
        (f"Which does {le} prefer, {a_e} or {b_e}?", f"{topic(lk)} {with_(a_k)} {b_k} 중 어느 쪽을 더 좋아합니까?", f"{p['cmp'][0]}|{a_e}|{a_k}"),
        (f"May {le} drink coffee at night?", f"{topic(lk)} 밤에 커피를 마셔도 됩니까?", None),
        (f"What device does {le} currently use?", f"{subj(lk)} 현재 사용하는 기기는 무엇입니까?", None),
        (f"What device does {le} plan to use in the future?", f"{subj(lk)} 앞으로 사용할 예정인 기기는 무엇입니까?", None),
        (f"What device did {le} use before?", f"{subj(lk)} 이전에 사용하던 기기는 무엇입니까?", None),
        (f"Has {le} already bought the {d[2]}?", f"{topic(lk)} {obj(d[2])} 이미 구매했습니까?", None),
        (f"When {w[0]}, does {le} {w[1]}?", f"{w[2]} {topic(lk)} {w[4]}?", None),
        (f"What is the cause of {le}'s {c['prob_en']}?", f"{lk}의 {c['prob_ko']} 원인은 무엇입니까?", c["ans"]),
        (f'What is the job of the person nicknamed "{alias_e}"?', f'"{alias_k}"라고 불리는 사람의 직업은 무엇입니까?', f"{p['job']}|{job_e}|{job_k}"),
    ]
    return T, q


def project_texts(L: Tuple[str, str, str], p: dict) -> Tuple[Dict[str, Row], List[Tuple[str, str, Optional[str]]]]:
    """p keys: due(y,m,d) owner size lang db vers non_db fa fb."""
    _, le, lk = L
    y, m, d = p["due"]
    owner_e, owner_k = SUR[p["owner"]]
    v, nd, fa, fb, lang, db, size = p["vers"], p["non_db"], p["fa"], p["fb"], p["lang"], p["db"], p["size"]
    sj = lambda w: subj(w)
    T: Dict[str, Row] = {
        "期": (f"{le} is due on {MONTHS[m - 1]} {d}, {y}.", f"Due {y}-{m:02d}-{d:02d}.", f"{lk}의 납기는 {y}년 {m}월 {d}일이다.", f"납기 {y}-{m:02d}-{d:02d}."),
        "担": (f"The person in charge of {le} is {owner_e}.", f"Owner: {owner_e}.", f"{lk}의 담당자는 {owner_k}씨이다.", f"담당: {owner_k}."),
        "人": (f"{le}'s team has {size} members.", f"{size}-person team.", f"{lk}의 팀은 {size}명으로 구성되어 있다.", f"{size}명 체제."),
        "言": (f"{le} is implemented in {lang}.", f"Language: {lang}.", f"{lk}의 구현 언어는 {lang}이다.", f"언어는 {lang}."),
        "D": (f"{le} uses {db} as its database.", f"DB: {db}.", f"{topic(lk)} 데이터베이스로 {obj(db)} 사용하고 있다.", f"DB는 {db}."),
        "過": (f"The previous production version of {le} was {v[0]}.", f"Previous {v[0]}.", f"{topic(lk)} 이전에 {v[0]} 버전으로 가동되었다.", f"이전 {v[0]}."),
        "今": (f"{le} currently runs {v[1]}.", f"Current {v[1]}.", f"{topic(lk)} 현재 {v[1]} 버전으로 가동 중이다.", f"현재 {v[1]}."),
        "将": (f"{le} will move to {v[2]} next.", f"Next {v[2]} planned.", f"{topic(lk)} 다음에 {v[2]} 버전으로 가동할 예정이다.", f"다음 {v[2]} 예정."),
        "未": (f"The load test for {le} has not been run yet.", "Load test not yet run.", f"{lk}의 부하 테스트는 아직 실시하지 않았다.", "부하 테스트 미실시."),
        "禁": (f"Directly changing production is prohibited on {le}.", "Direct production changes prohibited.",
               f"{lk}에서는 운영 환경을 직접 변경하는 것이 금지되어 있다.", "운영 직접 변경 금지."),
        "不": (f"Outsourcing is not allowed on {le}.", "No outsourcing.", f"{lk}에서는 외주가 허용되지 않는다.", "외주 불가."),
        "非": (f"The cause of the recent incident on {le} was not {nd}.", f"Incident cause not {nd}.",
               f"{lk}의 최근 장애 원인은 {sj(nd)} 아니었다.", f"장애 원인은 {nd} 아님."),
        "因": (f"{le} is delayed because of specification changes.", "Delayed by spec changes.",
               f"{topic(lk)} 사양 변경 때문에 개발이 지연되고 있다.", "사양 변경으로 지연."),
        "若": (f"On {le}, if the budget is exceeded, approval from the department head is required.",
               "Over budget: department head approval needed.", f"{topic(lk)} 예산을 초과하는 경우 부장의 승인이 필요하다.", "예산 초과 시 부장 승인 필요."),
        "比": (f"On {le}, {fa} is faster than {fb}.", f"{fa} faster than {fb}.", f"{lk}에서는 처리 속도가 {fb}보다 {sj(fa)} 더 빠르다.", f"{fb}보다 {sj(fa)} 빠름."),
        "疑": (f"{le} is considering a migration to AWS, but it is not final.", "AWS migration under consideration (unconfirmed).",
               f"{topic(lk)} AWS로의 이전을 검토 중이지만 확정은 아니다.", "AWS 이전 검토 중(미확정)."),
    }
    q = [  # order must match gen.build_project
        (f"What is the due date of {le}?", f"{lk}의 납기는 언제입니까?", None),
        (f"Who is in charge of {le}?", f"{lk}의 담당자는 누구입니까?", f"{p['owner']}|{owner_e}|{owner_k}"),
        (f"How many members are on the team of {le}?", f"{lk}의 팀은 몇 명입니까?", None),
        (f"What language is {le} implemented in?", f"{lk}의 구현 언어는 무엇입니까?", None),
        (f"Which database does {le} use?", f"{lk}에서 사용하는 데이터베이스는 무엇입니까?", None),
        (f"Which version of {le} is currently running?", f"{lk}의 현재 가동 중인 버전은 무엇입니까?", None),
        (f"Which version of {le} is planned next?", f"{lk}의 다음 가동 버전은 무엇입니까?", None),
        (f"Which version of {le} ran before?", f"{lk}의 이전에 가동되던 버전은 무엇입니까?", None),
        (f"Has the load test for {le} been run?", f"{lk}의 부하 테스트는 실시되었습니까?", None),
        (f"May production be changed directly on {le}?", f"{lk}에서 운영 환경을 직접 변경해도 됩니까?", None),
        (f"Can {le} be outsourced?", f"{lk}의 개발을 외주할 수 있습니까?", None),
        (f"Was the cause of the recent incident on {le} {nd}?", f"{lk}의 최근 장애 원인은 {nd}입니까?", None),
        (f"Why is {le} delayed?", f"{lk}의 개발이 지연되고 있는 이유는 무엇입니까?", "仕様変更|specification|spec|사양 변경"),
        (f"What is required on {le} if the budget is exceeded?", f"{lk}에서 예산을 초과하면 무엇이 필요합니까?", "部長|department head|부장"),
        (f"Which is faster on {le}, {fa} or {fb}?", f"{lk}에서 {with_(fa)} {fb} 중 처리 속도가 빠른 것은 어느 쪽입니까?", None),
        (f"Is the AWS migration of {le} confirmed?", f"{lk}의 AWS 이전은 확정되었습니까?", None),
    ]
    return T, q


def event_texts(L: Tuple[str, str, str], p: dict) -> Tuple[Dict[str, Row], List[Tuple[str, str, Optional[str]]]]:
    """p keys: month day place cap bud stalls parking."""
    _, le, lk = L
    mo, dd = p["month"], p["day"]
    pe, pk = PLACE[p["place"]]
    cap, bud, st = p["cap"], p["bud"], p["stalls"]
    yen = f"{bud * 10000:,}"
    T: Dict[str, Row] = {
        "日": (f"{le} takes place on {MONTHS[mo - 1]} {dd}.", f"Date: {MONTHS[mo - 1]} {dd}.", f"{topic(lk)} {mo}월 {dd}일에 열린다.", f"개최일 {mo}월 {dd}일."),
        "場": (f"{le} is held at {pe} Park.", f"Venue: {pe} Park.", f"{lk}의 장소는 {pk}공원이다.", f"장소: {pk}공원."),
        "定": (f"{le} has a capacity of {cap} visitors.", f"Capacity {cap}.", f"{lk}의 입장객 정원은 {cap}명이다.", f"정원 {cap}명."),
        "予": (f"The budget for {le} is {yen} yen.", f"Budget {yen} yen.", f"{lk}의 예산은 {bud}만 엔이다.", f"예산 {bud}만 엔."),
        "雨": (f"{le} is canceled if it rains.", "Canceled if rain.", f"{topic(lk)} 비가 오면 취소된다.", "우천 시 취소."),
        "風": (f"At {le}, food stalls are not set up if the wind is strong.", "No stalls if strong wind.",
               f"{lk}에서는 바람이 강하면 노점을 설치하지 않는다.", "강풍 시 노점 미설치."),
        "因": (f"{le} was canceled last time because of rain.", "Last time canceled due to rain.", f"{topic(lk)} 지난번에 비 때문에 취소되었다.", "지난번 우천 취소."),
        "禁": (f"The use of open flames is prohibited at {le}.", "Open flames prohibited.", f"{lk}의 행사장 내 화기 사용은 금지되어 있다.", "화기 금지."),
        "駐": ((f"{le} has a parking lot.", "Parking available.", f"{lk} 행사장에는 주차장이 있다.", "주차장 있음.") if p["parking"] else
               (f"{le} has no parking lot.", "No parking.", f"{lk} 행사장에는 주차장이 없다.", "주차장 없음.")),
        "未": (f"The police notification for {le} has not been filed yet.", "Police notification pending.", f"{lk}의 경찰 신고는 아직 하지 않았다.", "경찰 신고 미완료."),
        "過": (f"Last year, {le} had {st[0]} food stalls.", f"Last year: {st[0]} stalls.", f"작년 {lk}의 노점은 {st[0]}개였다.", f"작년 {st[0]}개."),
        "今": (f"Currently, {st[1]} food stalls have applied for {le}.", f"Applications now: {st[1]}.", f"현재 {lk}의 노점 신청은 {st[1]}개이다.", f"현재 신청 {st[1]}개."),
        "将": (f"{le} is expected to end up with {st[2]} food stalls.", f"Expected: {st[2]} stalls.", f"{lk}의 노점은 최종적으로 {st[2]}개가 될 예정이다.", f"최종 {st[2]}개 예정."),
        "疑": (f"{le} may have fireworks, but it has not been confirmed.", "Fireworks possible (unconfirmed).",
               f"{lk}에서는 불꽃놀이를 할 가능성이 있지만 아직 확정되지 않았다.", "불꽃놀이 가능(미확정)."),
    }
    q = [  # order must match gen.build_event
        (f"When does {le} take place?", f"{lk}의 개최일은 언제입니까?", f"{mo}月{dd}日|{MONTHS[mo - 1]} {dd}|{mo}월 {dd}일"),
        (f"Where is {le} held?", f"{lk}의 장소는 어디입니까?", f"{p['place']}|{pe}|{pk}"),
        (f"What is the visitor capacity of {le}?", f"{lk}의 입장객 정원은 몇 명입니까?", None),
        (f"Is {le} canceled if it rains?", f"{topic(lk)} 비가 오면 취소됩니까?", None),
        (f"Are food stalls set up at {le} when the wind is strong?", f"{lk}에서는 바람이 강할 때 노점이 설치됩니까?", None),
        (f"Why was {le} canceled last time?", f"{subj(lk)} 지난번에 취소된 이유는 무엇입니까?", "雨|rain|비"),
        (f"May open flames be used at {le}?", f"{lk}에서 화기를 사용해도 됩니까?", None),
        (f"Does {le} have a parking lot?", f"{lk}에는 주차장이 있습니까?", None),
        (f"Has the police notification for {le} been completed?", f"{lk}의 경찰 신고는 완료되었습니까?", None),
        (f"How many stalls have applied for {le} so far?", f"{lk}의 노점 신청은 현재 몇 개입니까?", None),
        (f"How many stalls did {le} have last year?", f"{lk}의 작년 노점은 몇 개였습니까?", None),
        (f"How many stalls are expected at {le} in the end?", f"{lk}의 최종 예정 노점은 몇 개입니까?", None),
        (f"Is the fireworks show at {le} confirmed?", f"{lk}의 불꽃놀이는 확정되었습니까?", None),
    ]
    return T, q
