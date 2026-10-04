"""Simplified-Chinese renderings of the canonical facts and questions (Issue #19 A).

Chinese is the control language for the kanji-label hypothesis: it uses hanzi natively, has little
inflection, and shares many hanzi with Japanese kanji (but not all: 職/职, 医療/医疗 ...). So in Chinese
the canonical kanji labels (IR-C) are *partly* native, and the localized labels (IR-L) are simplified-hanzi words.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import i18n


def _d(s: str) -> Dict[str, str]:
    return dict(x.split(":") for x in s.split())


SUR = _d("佐藤:佐藤 鈴木:铃木 高橋:高桥 田中:田中 伊藤:伊藤 渡辺:渡边 山本:山本 中村:中村 小林:小林 加藤:加藤 吉田:吉田 山田:山田 "
         "佐々木:佐佐木 山口:山口 松本:松本 井上:井上 木村:木村 林:林 斎藤:斋藤 清水:清水 山崎:山崎 森:森 池田:池田 橋本:桥本 阿部:阿部 "
         "石川:石川 山下:山下 中島:中岛 石井:石井 小川:小川 前田:前田 岡田:冈田 長谷川:长谷川 藤田:藤田 後藤:后藤 近藤:近藤 村上:村上 "
         "遠藤:远藤 青木:青木 坂本:坂本 斉藤:齐藤 福田:福田 太田:太田 西村:西村 藤井:藤井 金子:金子 岡本:冈本 藤原:藤原 三浦:三浦 "
         "中野:中野 原田:原田 松田:松田 竹内:竹内 小野:小野 田村:田村 中山:中山 和田:和田 石田:石田 上田:上田 森田:森田 柴田:柴田")
ALIAS = _d("ポン:阿胖 タロ:太郎 ミケ:三花 ハチ:阿八 ゴン:阿冈 コタ:小太 マメ:豆豆 ソラ:小空 ムギ:麦麦 クロ:小黑 シロ:小白 チビ:小不点 モモ:桃桃 "
           "ラム:阿拉 レオ:莱欧 ココ:可可 ナナ:娜娜 ユキ:小雪 ハル:小春 ケン:阿健 ジロ:二郎 サブ:三郎 ヨシ:阿义 トラ:小虎 ブン:阿文 ダイ:大大 "
           "ノリ:小则 リク:阿陆 ヒロ:阿宏 マル:小丸 ミツ:阿蜜")
CITY = _d("大阪:大阪 京都:京都 神戸:神户 名古屋:名古屋 福岡:福冈 札幌:札幌 仙台:仙台 広島:广岛 横浜:横滨 金沢:金泽 熊本:熊本 新潟:新泻")
PLACE = _d("桜ヶ丘:樱之丘 中央:中央 港町:港町 緑地:绿地 駅前:站前 河原:河滩 城址:城址 丘上:丘上 湖畔:湖畔 旧市街:旧城区 北口:北口 "
           "南公園:南公园 東通り:东大街 西浜:西滨 山手:山手 若葉:若叶")
SEASON = _d("春祭:春祭 夏祭:夏祭 秋祭:秋祭 冬祭:冬祭 花火大会:烟花大会 収穫祭:丰收节")
JOB = _d("IT:IT 医療:医疗 教育:教育 建設:建筑 金融:金融 農業:农业 運輸:运输 製造:制造 小売:零售 出版:出版 法務:法务 設計:设计")
JOB_ALT = {"IT": [], "医療": ["医药", "医学"], "教育": [], "建設": ["建设"], "金融": [], "農業": [], "運輸": ["物流", "交通"], "製造": ["生产"],
           "小売": ["零售业", "销售"], "出版": [], "法務": ["法律"], "設計": []}
FOOD = _d("十割蕎麦:纯荞麦面 カレー:咖喱 寿司:寿司 焼き鳥:烤鸡肉串 ラーメン:拉面 天ぷら:天妇罗 お好み焼き:大阪烧 餃子:饺子 うなぎ:鳗鱼 鍋:火锅")
UNC = _d("納豆:纳豆 ヨーグルト:酸奶 チーズ:奶酪 パクチー:香菜 生卵:生鸡蛋 レバー:肝脏")
PAIR = _d("蕎麦:荞麦面 うどん:乌冬面 珈琲:咖啡 紅茶:红茶 犬:狗 猫:猫 山:山 海:大海 朝:早晨 夜:夜晚 電車:电车 バス:公交车")
PAIR_ALT = {"蕎麦": ["荞麦"], "うどん": ["乌冬"], "犬": ["小狗"], "猫": ["小猫"], "海": ["海"], "朝": ["早上", "清晨"], "夜": ["晚上"], "電車": ["火车", "地铁"],
            "バス": ["巴士", "公共汽车"]}
# (cond clause, act as it appears in "去{act}" statements, act question)
WEATHER = {"雨": ("下雨", "去散步"), "雪": ("下雪", "去跑步"), "強風": ("刮大风", "去钓鱼")}
CAUSES = [
    dict(zh0="{L}因为睡眠不足，白天会犯困。", zh1="睡眠不足导致白天犯困。", prob="白天犯困", ans="睡眠|睡觉|犯困"),
    dict(zh0="{L}因为运动不足，体重增加了。", zh1="运动不足导致体重增加。", prob="体重增加", ans="运动|锻炼|缺乏运动"),
    dict(zh0="{L}因为加班过多，身体状况恶化了。", zh1="加班过多导致身体状况恶化。", prob="身体状况问题", ans="加班|工作过量|超时工作"),
]
CLS = {"人物": "人物", "案件": "项目", "催事": "活动"}


def labels(cls: str, name_ja: str, **kw) -> str:
    """Chinese display label, e.g. 人物石田 / 项目Atlas / 活动樱之丘春祭."""
    if cls == "人物":
        return f"人物{SUR[name_ja]}"
    if cls == "案件":
        return f"项目{name_ja}"
    return f"活动{PLACE[kw['place']]}{SEASON[kw['season']]}"


def job_ans(ja: str) -> str:
    return "|".join([i18n.job_ans(ja), JOB[ja], *JOB_ALT.get(ja, [])])


def pair_ans(ja: str) -> str:
    return "|".join([i18n.pair_ans(ja), PAIR[ja], *PAIR_ALT.get(ja, [])])


Row = Tuple[str, str]


def person_texts(L: str, p: dict):
    job, city, alias = JOB[p["job"]], CITY[p["city"]], ALIAS[p["alias"]]
    food, unc = FOOD[p["food"]], UNC[p["unc"]]
    a, b = PAIR[p["cmp"][0]], PAIR[p["cmp"][1]]
    d, w, c, y = p["dev"], WEATHER[p["w"]], CAUSES[p["ci"]], p["year"]
    T: Dict[str, Row] = {
        "生": (f"{L}出生于{y}年。", f"{y}年出生。"),
        "職": (f"{L}从事{job}行业。", f"职业：{job}。"),
        "住": (f"{L}住在{city}。", f"居住地：{city}。"),
        "愛": (f"周围的人称{L}为“{alias}”。", f"昵称：{alias}。"),
        "好": (f"{L}喜欢{food}。", f"喜欢{food}。"),
        "酒": (f"{L}喝酒。", "喝酒。") if p["drinks"] else (f"{L}不喝酒。", "不喝酒。"),
        "疑": (f"据说{L}不吃{unc}，但尚未得到确认。", f"据说不吃{unc}（未确认）。"),
        "比": (f"比起{b}，{L}更喜欢{a}。", f"喜欢{a}胜过{b}。"),
        "禁": (f"医生禁止{L}在夜间喝咖啡。", "夜间禁止喝咖啡。"),
        "過": (f"{L}以前使用{d[0]}。", f"以前用{d[0]}。"),
        "今": (f"{L}目前使用{d[1]}。", f"现在用{d[1]}。"),
        "将": (f"{L}今后打算使用{d[2]}。", f"今后打算用{d[2]}。"),
        "未": (f"{d[2]}还没有购买。", f"{d[2]}尚未购买。"),
        "若": (f"如果{w[0]}，{L}就不{w[1]}。", f"{w[0]}则不{w[1]}。"),
        "因": (c["zh0"].format(L=L), c["zh1"]),
    }
    q = [
        (f"{L}出生于哪一年？", None),
        (f"{L}从事什么行业？", job_ans(p["job"])),
        (f"{L}住在哪里？", None),
        (f"{L}喜欢{food}吗？", None),
        (f"{L}喝酒吗？", None),
        (f"{L}不吃{unc}这件事已经确认了吗？", None),
        (f"{L}更喜欢{a}还是{b}？", None),
        (f"{L}可以在夜间喝咖啡吗？", None),
        (f"{L}目前使用的设备是什么？", None),
        (f"{L}今后打算使用的设备是什么？", None),
        (f"{L}以前使用的设备是什么？", None),
        (f"{L}已经购买{d[2]}了吗？", None),
        (f"{w[0]}的时候，{L}会{w[1].replace('去', '去')}吗？", None),
        (f"{L}{c['prob']}的原因是什么？", c["ans"]),
        (f"被称为“{alias}”的人从事什么行业？", job_ans(p["job"])),
    ]
    # extra Chinese alternatives for questions whose answer_ml already exists (city / compare answers are names)
    extra = {2: "|".join([p["city"], city]), 6: pair_ans(p["cmp"][0])}
    return T, q, extra


def project_texts(L: str, p: dict):
    y, m, d = p["due"]
    owner = SUR[p["owner"]]
    v, nd, fa, fb, lang, db, size = p["vers"], p["non_db"], p["fa"], p["fb"], p["lang"], p["db"], p["size"]
    T: Dict[str, Row] = {
        "期": (f"{L}的交付期限是{y}年{m}月{d}日。", f"交付期限{y}-{m:02d}-{d:02d}。"),
        "担": (f"{L}的负责人是{owner}。", f"负责人：{owner}。"),
        "人": (f"{L}的团队由{size}人组成。", f"团队{size}人。"),
        "言": (f"{L}使用{lang}开发。", f"开发语言：{lang}。"),
        "D": (f"{L}使用{db}作为数据库。", f"数据库：{db}。"),
        "過": (f"{L}之前的运行版本是{v[0]}。", f"旧版{v[0]}。"),
        "今": (f"{L}目前运行的版本是{v[1]}。", f"现行{v[1]}。"),
        "将": (f"{L}下一个运行版本预计为{v[2]}。", f"下一版{v[2]}（计划）。"),
        "未": (f"{L}还没有进行压力测试。", "压力测试尚未进行。"),
        "禁": (f"{L}禁止直接修改生产环境。", "禁止直接修改生产环境。"),
        "不": (f"{L}不允许外包。", "不可外包。"),
        "非": (f"{L}最近一次故障的原因不是{nd}。", f"故障原因不是{nd}。"),
        "因": (f"{L}因为需求变更，开发出现了延迟。", "需求变更导致延迟。"),
        "若": (f"{L}如果超出预算，需要部长批准。", "超预算需部长批准。"),
        "比": (f"{L}中，{fa}的处理速度比{fb}更快。", f"{fa}比{fb}更快。"),
        "疑": (f"{L}正在考虑迁移到AWS，但尚未确定。", "AWS迁移在考虑中（未确定）。"),
    }
    q = [
        (f"{L}的交付期限是什么时候？", None),
        (f"{L}的负责人是谁？", None),
        (f"{L}的团队有多少人？", None),
        (f"{L}使用什么语言开发？", None),
        (f"{L}使用什么数据库？", None),
        (f"{L}目前运行的是哪个版本？", None),
        (f"{L}下一个运行版本是什么？", None),
        (f"{L}之前运行的是哪个版本？", None),
        (f"{L}的压力测试已经进行了吗？", None),
        (f"{L}可以直接修改生产环境吗？", None),
        (f"{L}可以外包吗？", None),
        (f"{L}最近一次故障的原因是{nd}吗？", None),
        (f"{L}开发延迟的原因是什么？", "需求变更|需求|变更"),
        (f"{L}超出预算时需要什么？", None),
        (f"{L}中，{fa}和{fb}哪个处理速度更快？", None),
        (f"{L}迁移到AWS已经确定了吗？", None),
    ]
    extra = {1: "|".join([p["owner"], owner]), 13: "部长"}
    return T, q, extra


def event_texts(L: str, p: dict):
    mo, dd = p["month"], p["day"]
    place = PLACE[p["place"]]
    cap, bud, st = p["cap"], p["bud"], p["stalls"]
    T: Dict[str, Row] = {
        "日": (f"{L}将于{mo}月{dd}日举行。", f"举办日期：{mo}月{dd}日。"),
        "場": (f"{L}的举办地点是{place}公园。", f"地点：{place}公园。"),
        "定": (f"{L}的入场人数上限为{cap}人。", f"容量{cap}人。"),
        "予": (f"{L}的预算是{bud}万日元。", f"预算{bud}万日元。"),
        "雨": (f"{L}遇到下雨就会取消。", "下雨则取消。"),
        "風": (f"{L}遇到大风就不设置摊位。", "大风则不设摊位。"),
        "因": (f"{L}上次因为下雨而取消了。", "上次因雨取消。"),
        "禁": (f"{L}会场内禁止使用明火。", "禁止明火。"),
        "駐": (f"{L}会场有停车场。", "有停车场。") if p["parking"] else (f"{L}会场没有停车场。", "没有停车场。"),
        "未": (f"{L}还没有向警方报备。", "警方报备尚未完成。"),
        "過": (f"去年{L}有{st[0]}个摊位。", f"去年{st[0]}个摊位。"),
        "今": (f"目前{L}已有{st[1]}个摊位申请。", f"目前申请{st[1]}个。"),
        "将": (f"{L}的摊位预计最终为{st[2]}个。", f"预计最终{st[2]}个。"),
        "疑": (f"{L}可能会有烟花表演，但尚未确定。", "可能有烟花（未确定）。"),
    }
    q = [
        (f"{L}在什么时候举行？", None),
        (f"{L}在哪里举行？", None),
        (f"{L}的入场人数上限是多少？", None),
        (f"下雨的话，{L}会取消吗？", None),
        (f"刮大风时，{L}会设置摊位吗？", None),
        (f"{L}上次取消的原因是什么？", "雨|下雨"),
        (f"{L}可以使用明火吗？", None),
        (f"{L}会场有停车场吗？", None),
        (f"{L}向警方报备完成了吗？", None),
        (f"{L}目前有多少个摊位申请？", None),
        (f"{L}去年有多少个摊位？", None),
        (f"{L}预计最终有多少个摊位？", None),
        (f"{L}的烟花表演确定了吗？", None),
    ]
    extra = {1: "|".join([p["place"], place])}
    return T, q, extra
