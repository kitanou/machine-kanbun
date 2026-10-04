"""Natural-sentence benchmark (Issue #19 C/D): 30 short passages x {JA, EN, KO, ZH}, 3 per semantic phenomenon.

The sentences are written independently of the canonical-fact generator (they are *not* produced from the
templates), so they test whether an LLM can turn unseen natural text into the IR. Every item has questions whose
answers depend on the phenomenon (negation, tense, modality, uncertainty, hearsay, causality, condition,
comparison, coreference, intention/cancellation). Span answers list alternatives for all four languages.
"""
from __future__ import annotations

from typing import Dict, List

LANGS = ("JA", "EN", "KO", "ZH")


def _item(id_, ph, ja, en, ko, zh, qs):
    return {"id": id_, "ph": ph, "t": {"JA": ja, "EN": en, "KO": ko, "ZH": zh}, "qs": qs}


def _yn(ja, en, ko, zh, gold):
    return {"q": {"JA": ja, "EN": en, "KO": ko, "ZH": zh}, "yn": gold, "ans": ""}


def _span(ja, en, ko, zh, ans):
    return {"q": {"JA": ja, "EN": en, "KO": ko, "ZH": zh}, "yn": None, "ans": ans}


ITEMS: List[dict] = [
    # ---- negation
    _item("n1", "否定", "田中さんは昨日の会議に出席しなかったが、資料は提出した。",
          "Mr. Tanaka did not attend yesterday's meeting, but he submitted the materials.",
          "다나카 씨는 어제 회의에 참석하지 않았지만 자료는 제출했다.", "田中昨天没有出席会议，但提交了资料。",
          [_yn("田中さんは昨日の会議に出席しましたか？", "Did Mr. Tanaka attend yesterday's meeting?", "다나카 씨는 어제 회의에 참석했습니까?", "田中昨天出席会议了吗？", False),
           _yn("田中さんは資料を提出しましたか？", "Did Mr. Tanaka submit the materials?", "다나카 씨는 자료를 제출했습니까?", "田中提交资料了吗？", True)]),
    _item("n2", "否定", "この店は現金を受け付けず、カードだけが使える。", "This shop does not accept cash; only cards can be used.",
          "이 가게는 현금을 받지 않고 카드만 사용할 수 있다.", "这家店不收现金，只能用卡。",
          [_yn("この店で現金は使えますか？", "Can cash be used at this shop?", "이 가게에서 현금을 사용할 수 있습니까?", "这家店可以用现金吗？", False),
           _yn("この店でカードは使えますか？", "Can cards be used at this shop?", "이 가게에서 카드를 사용할 수 있습니까?", "这家店可以用卡吗？", True)]),
    _item("n3", "否定", "彼は肉も魚も食べないが、卵は食べる。", "He eats neither meat nor fish, but he does eat eggs.",
          "그는 고기도 생선도 먹지 않지만 달걀은 먹는다.", "他不吃肉也不吃鱼，但吃鸡蛋。",
          [_yn("彼は魚を食べますか？", "Does he eat fish?", "그는 생선을 먹습니까?", "他吃鱼吗？", False),
           _yn("彼は卵を食べますか？", "Does he eat eggs?", "그는 달걀을 먹습니까?", "他吃鸡蛋吗？", True)]),
    # ---- tense / aspect
    _item("t1", "時制", "彼女は去年大阪に引っ越し、今は京都で働いている。来年は東京に移る予定だ。",
          "She moved to Osaka last year and now works in Kyoto. She plans to move to Tokyo next year.",
          "그녀는 작년에 오사카로 이사했고 지금은 교토에서 일하고 있다. 내년에는 도쿄로 옮길 예정이다.", "她去年搬到了大阪，现在在京都工作。明年打算搬到东京。",
          [_span("彼女は今どこで働いていますか？", "Where does she work now?", "그녀는 지금 어디에서 일하고 있습니까?", "她现在在哪里工作？", "京都|Kyoto|교토"),
           _span("彼女は来年どこに移る予定ですか？", "Where does she plan to move next year?", "그녀는 내년에 어디로 옮길 예정입니까?", "她明年打算搬到哪里？", "東京|Tokyo|도쿄|东京"),
           _span("彼女は去年どこに引っ越しましたか？", "Where did she move last year?", "그녀는 작년에 어디로 이사했습니까?", "她去年搬到了哪里？", "大阪|Osaka|오사카")]),
    _item("t2", "時制", "プロジェクトは先月完了したが、報告書はまだ提出されていない。",
          "The project was completed last month, but the report has not been submitted yet.",
          "프로젝트는 지난달에 완료되었지만 보고서는 아직 제출되지 않았다.", "项目上个月已经完成，但报告还没有提交。",
          [_yn("プロジェクトは完了していますか？", "Is the project completed?", "프로젝트는 완료되었습니까?", "项目完成了吗？", True),
           _yn("報告書は提出済みですか？", "Has the report been submitted?", "보고서는 제출되었습니까?", "报告已经提交了吗？", False)]),
    _item("t3", "時制", "彼は三年前からピアノを習っていて、来月初めての発表会に出る。",
          "He has been learning the piano for three years, and he will play in his first recital next month.",
          "그는 3년 전부터 피아노를 배우고 있으며 다음 달에 처음으로 발표회에 나간다.", "他三年前开始学钢琴，下个月将参加他的第一场演奏会。",
          [_yn("彼は今ピアノを習っていますか？", "Is he learning the piano now?", "그는 지금 피아노를 배우고 있습니까?", "他现在在学钢琴吗？", True),
           _yn("発表会はもう終わりましたか？", "Is the recital already over?", "발표회는 이미 끝났습니까?", "演奏会已经结束了吗？", False)]),
    # ---- modality
    _item("m1", "様相", "この資料は社外に持ち出してはならないが、社内で閲覧することはできる。",
          "This document must not be taken outside the company, but it can be viewed inside.",
          "이 자료는 사외로 반출해서는 안 되지만 사내에서 열람할 수는 있다.", "这份资料不得带出公司，但可以在公司内部查阅。",
          [_yn("この資料を社外に持ち出してもよいですか？", "May this document be taken outside the company?", "이 자료를 사외로 반출해도 됩니까?", "可以把这份资料带出公司吗？", False),
           _yn("この資料を社内で閲覧できますか？", "Can this document be viewed inside the company?", "이 자료를 사내에서 열람할 수 있습니까?", "这份资料可以在公司内部查阅吗？", True)]),
    _item("m2", "様相", "参加は任意で、希望する人だけ申し込めばよい。", "Participation is optional; only those who want to take part need to apply.",
          "참가는 자유이며 희망하는 사람만 신청하면 된다.", "参加是自愿的，只有希望参加的人才需要报名。",
          [_yn("参加は必須ですか？", "Is participation mandatory?", "참가는 필수입니까?", "参加是必须的吗？", False),
           _yn("参加を希望しない人は申し込む必要がありますか？", "Do people who do not want to take part need to apply?", "참가를 희망하지 않는 사람도 신청해야 합니까?", "不希望参加的人需要报名吗？", False)]),
    _item("m3", "様相", "申請には身分証明書が必要だが、写真は不要である。", "An ID is required for the application, but a photo is not needed.",
          "신청에는 신분증이 필요하지만 사진은 필요 없다.", "申请需要身份证件，但不需要照片。",
          [_yn("申請に身分証明書は必要ですか？", "Is an ID required for the application?", "신청에 신분증이 필요합니까?", "申请需要身份证件吗？", True),
           _yn("申請に写真は必要ですか？", "Is a photo required for the application?", "신청에 사진이 필요합니까?", "申请需要照片吗？", False)]),
    # ---- uncertainty
    _item("u1", "不確実性", "会議は来週の火曜日になるかもしれないが、まだ確定していない。",
          "The meeting may be held next Tuesday, but it has not been decided yet.",
          "회의는 다음 주 화요일이 될 수도 있지만 아직 확정되지 않았다.", "会议可能定在下周二，但还没有确定。",
          [_yn("会議の日程は確定していますか？", "Has the date of the meeting been decided?", "회의 일정은 확정되었습니까?", "会议的日期确定了吗？", False),
           _span("会議の候補日は何曜日ですか？", "Which day of the week is the candidate date of the meeting?", "회의 후보일은 무슨 요일입니까?", "会议的候选日期是星期几？", "火曜|Tuesday|화요일|周二|星期二")]),
    _item("u2", "不確実性", "新製品の発売は秋ごろと見られているが、正式な発表はない。",
          "The new product is expected to launch around autumn, but there has been no official announcement.",
          "신제품은 가을 무렵에 출시될 것으로 보이지만 공식 발표는 없다.", "新产品预计在秋季前后上市，但没有正式公布。",
          [_yn("発売時期は正式に発表されましたか？", "Has the launch time been officially announced?", "출시 시기는 공식 발표되었습니까?", "上市时间正式公布了吗？", False),
           _span("発売はいつごろと見られていますか？", "When is the launch expected?", "출시는 언제쯤으로 보입니까?", "预计什么时候上市？", "秋|autumn|fall|가을")]),
    _item("u3", "不確実性", "彼が転職するという話は本当かどうか分からない。", "It is not known whether the story that he is changing jobs is true.",
          "그가 이직한다는 이야기가 사실인지는 알 수 없다.", "不知道他要换工作的传闻是否属实。",
          [_yn("彼が転職することは確実ですか？", "Is it certain that he is changing jobs?", "그가 이직하는 것은 확실합니까?", "他要换工作是确定的吗？", False)]),
    # ---- evidentiality / hearsay
    _item("e1", "伝聞", "社長の話によると、来年度は予算が削減されるそうだ。", "According to the president, the budget will reportedly be cut next fiscal year.",
          "사장의 말에 따르면 내년도에는 예산이 삭감된다고 한다.", "据总经理说，下一财年的预算将被削减。",
          [_span("予算削減の情報は誰によるものですか？", "Who is the source of the information about the budget cut?", "예산 삭감 정보는 누구에게서 나온 것입니까?", "预算削减的消息来自谁？", "社長|president|사장|总经理|总裁"),
           _yn("予算削減は確定した事実として述べられていますか？", "Is the budget cut stated as a confirmed fact?", "예산 삭감은 확정된 사실로 서술되어 있습니까?", "预算削减是作为确定的事实陈述的吗？", False)]),
    _item("e2", "伝聞", "天気予報によると、明日は雪になるという。", "According to the weather forecast, it will snow tomorrow.",
          "일기예보에 따르면 내일은 눈이 온다고 한다.", "据天气预报，明天会下雪。",
          [_span("明日の天気の情報源は何ですか？", "What is the source of the information about tomorrow's weather?", "내일 날씨 정보의 출처는 무엇입니까?", "关于明天天气的信息来源是什么？", "天気予報|forecast|일기예보|天气预报")]),
    _item("e3", "伝聞", "山田さんが言うには、あの店は先月閉店したらしい。", "Mr. Yamada says that the shop seems to have closed last month.",
          "야마다 씨가 말하기를 그 가게는 지난달에 문을 닫은 것 같다.", "据山田说，那家店好像上个月已经关门了。",
          [_span("閉店の情報は誰から聞いたものですか？", "Who is the source of the information about the closure?", "폐점 정보는 누구에게서 들은 것입니까?", "关于关门的消息来自谁？", "山田|Yamada|야마다"),
           _yn("閉店は確認された事実ですか？", "Is the closure a confirmed fact?", "폐점은 확인된 사실입니까?", "关门是已确认的事实吗？", False)]),
    # ---- causality
    _item("c1", "因果", "電車が遅れたため、会議に間に合わなかった。", "Because the train was delayed, he did not make it to the meeting.",
          "전철이 늦어져서 회의에 제시간에 가지 못했다.", "因为电车晚点，没能赶上会议。",
          [_span("会議に間に合わなかった原因は何ですか？", "What was the cause of missing the meeting?", "회의에 제시간에 가지 못한 원인은 무엇입니까?", "没能赶上会议的原因是什么？", "電車|train|전철|电车"),
           _yn("彼は会議に間に合いましたか？", "Did he make it to the meeting?", "그는 회의에 제시간에 갔습니까?", "他赶上会议了吗？", False)]),
    _item("c2", "因果", "売上が伸びたのは、新しい広告を始めたからだ。", "Sales grew because a new advertising campaign was started.",
          "새로운 광고를 시작했기 때문에 매출이 늘었다.", "销售额增长是因为开始了新的广告。",
          [_span("売上が伸びた理由は何ですか？", "Why did sales grow?", "매출이 늘어난 이유는 무엇입니까?", "销售额为什么增长了？", "広告|advertis|광고|广告")]),
    _item("c3", "因果", "雨が降ったが、試合は予定どおり行われた。", "It rained, but the game was held as scheduled.",
          "비가 왔지만 경기는 예정대로 열렸다.", "虽然下雨了，但比赛照常举行。",
          [_yn("雨のために試合は中止になりましたか？", "Was the game canceled because of the rain?", "비 때문에 경기는 취소되었습니까?", "比赛因为下雨取消了吗？", False),
           _yn("試合は行われましたか？", "Was the game held?", "경기는 열렸습니까?", "比赛举行了吗？", True)]),
    # ---- condition
    _item("d1", "条件", "明日雨なら、遠足は中止になる。", "If it rains tomorrow, the excursion will be canceled.",
          "내일 비가 오면 소풍은 취소된다.", "如果明天下雨，郊游就取消。",
          [_span("遠足が中止になるのはどんな場合ですか？", "Under what condition will the excursion be canceled?", "소풍이 취소되는 것은 어떤 경우입니까?", "郊游在什么情况下会取消？", "雨|rain|비|下雨")]),
    _item("d2", "条件", "締め切りに間に合えば、追加料金はかからない。", "If you meet the deadline, no extra fee is charged.",
          "마감에 맞추면 추가 요금은 들지 않는다.", "如果赶上截止日期，就不收额外费用。",
          [_yn("締め切りに間に合った場合、追加料金はかかりますか？", "If you meet the deadline, is an extra fee charged?", "마감에 맞추면 추가 요금이 듭니까?", "如果赶上截止日期，要收额外费用吗？", False)]),
    _item("d3", "条件", "会員なら割引が受けられるが、非会員は定価である。", "Members get a discount, but non-members pay the regular price.",
          "회원은 할인을 받을 수 있지만 비회원은 정가를 내야 한다.", "会员可以享受折扣，非会员则按原价付费。",
          [_yn("非会員は割引を受けられますか？", "Can non-members get a discount?", "비회원은 할인을 받을 수 있습니까?", "非会员可以享受折扣吗？", False),
           _yn("会員は割引を受けられますか？", "Can members get a discount?", "회원은 할인을 받을 수 있습니까?", "会员可以享受折扣吗？", True)]),
    # ---- comparison
    _item("p1", "比較", "新型は旧型より軽く、バッテリーも長持ちする。", "The new model is lighter than the old model, and its battery lasts longer.",
          "신형은 구형보다 가볍고 배터리도 오래간다.", "新款比旧款更轻，电池也更耐用。",
          [_span("軽いのは新型と旧型のどちらですか？", "Which is lighter, the new model or the old model?", "신형과 구형 중 어느 쪽이 가볍습니까?", "新款和旧款哪个更轻？", "新型|new|신형|新款"),
           _span("バッテリーが長持ちするのはどちらですか？", "Which has the longer-lasting battery?", "배터리가 오래가는 것은 어느 쪽입니까?", "哪个的电池更耐用？", "新型|new|신형|新款")]),
    _item("p2", "比較", "東京は大阪より人口が多いが、物価は大阪の方が安い。", "Tokyo has a larger population than Osaka, but prices are lower in Osaka.",
          "도쿄는 오사카보다 인구가 많지만 물가는 오사카가 더 싸다.", "东京的人口比大阪多，但大阪的物价更便宜。",
          [_span("人口が多いのはどちらですか？", "Which has the larger population?", "인구가 많은 곳은 어디입니까?", "哪个城市人口更多？", "東京|Tokyo|도쿄|东京"),
           _span("物価が安いのはどちらですか？", "Where are prices lower?", "물가가 더 싼 곳은 어디입니까?", "哪个城市物价更便宜？", "大阪|Osaka|오사카")]),
    _item("p3", "比較", "太郎は次郎ほど速くないが、花子よりは速い。", "Taro is not as fast as Jiro, but he is faster than Hanako.",
          "타로는 지로만큼 빠르지 않지만 하나코보다는 빠르다.", "太郎不如次郎快，但比花子快。",
          [_span("三人の中で一番遅いのは誰ですか？", "Who is the slowest of the three?", "세 사람 중 가장 느린 사람은 누구입니까?", "三个人中谁最慢？", "花子|Hanako|하나코"),
           _span("三人の中で一番速いのは誰ですか？", "Who is the fastest of the three?", "세 사람 중 가장 빠른 사람은 누구입니까?", "三个人中谁最快？", "次郎|Jiro|지로")]),
    # ---- coreference
    _item("r1", "照応", "佐藤さんは鈴木さんに本を貸した。彼は翌日それを返した。", "Mr. Sato lent a book to Mr. Suzuki. He returned it the next day.",
          "사토 씨는 스즈키 씨에게 책을 빌려주었다. 그는 다음 날 그것을 돌려주었다.", "佐藤把一本书借给了铃木。他第二天就把它还了。",
          [_span("本を返したのは誰ですか？", "Who returned the book?", "책을 돌려준 사람은 누구입니까?", "是谁还了书？", "鈴木|Suzuki|스즈키|铃木")]),
    _item("r2", "照応", "山本課長は新人の中村さんを呼び、彼女に資料の作成を頼んだ。", "Manager Yamamoto called the new employee Ms. Nakamura and asked her to prepare the materials.",
          "야마모토 과장은 신입 나카무라 씨를 불러 그녀에게 자료 작성을 부탁했다.", "山本科长叫来新员工中村，请她准备资料。",
          [_span("資料を作成するのは誰ですか？", "Who will prepare the materials?", "자료를 작성하는 사람은 누구입니까?", "谁来准备资料？", "中村|Nakamura|나카무라")]),
    _item("r3", "照応", "犬は猫を追いかけたが、それは木に登って逃げた。", "The dog chased the cat, but it climbed a tree and escaped.",
          "개는 고양이를 쫓았지만 그것은 나무에 올라가 도망쳤다.", "狗追着猫，但它爬上树逃走了。",
          [_span("木に登ったのは何ですか？", "What climbed the tree?", "나무에 올라간 것은 무엇입니까?", "是什么爬上了树？", "猫|cat|고양이")]),
    # ---- intention / cancellation
    _item("i1", "意図・中止", "彼は大阪に出張するつもりだったが、台風のため取りやめた。", "He had intended to go on a business trip to Osaka, but canceled it because of a typhoon.",
          "그는 오사카로 출장 갈 생각이었지만 태풍 때문에 취소했다.", "他原本打算去大阪出差，但因为台风取消了。",
          [_yn("彼は大阪に出張しましたか？", "Did he go on a business trip to Osaka?", "그는 오사카로 출장을 갔습니까?", "他去大阪出差了吗？", False),
           _span("出張を取りやめた理由は何ですか？", "Why did he cancel the trip?", "출장을 취소한 이유는 무엇입니까?", "他取消出差的原因是什么？", "台風|typhoon|태풍|台风")]),
    _item("i2", "意図・中止", "来月の発表会は延期され、日程は未定である。", "Next month's presentation has been postponed, and the date is undecided.",
          "다음 달 발표회는 연기되었고 일정은 미정이다.", "下个月的发表会已被推迟，日期尚未确定。",
          [_yn("発表会は予定どおり行われますか？", "Will the presentation be held as planned?", "발표회는 예정대로 열립니까?", "发表会会按原计划举行吗？", False),
           _yn("発表会の日程は決まっていますか？", "Has the date of the presentation been decided?", "발표회 일정은 정해졌습니까?", "发表会的日期已经定了吗？", False)]),
    _item("i3", "意図・中止", "彼女は転職を考えているが、まだ決めていない。", "She is thinking about changing jobs, but she has not decided yet.",
          "그녀는 이직을 고민하고 있지만 아직 결정하지 않았다.", "她在考虑换工作，但还没有决定。",
          [_yn("彼女は転職を決めましたか？", "Has she decided to change jobs?", "그녀는 이직을 결정했습니까?", "她决定换工作了吗？", False),
           _yn("彼女は転職を考えていますか？", "Is she thinking about changing jobs?", "그녀는 이직을 고민하고 있습니까?", "她在考虑换工作吗？", True)]),
]

PHENOMENA = ["否定", "時制", "様相", "不確実性", "伝聞", "因果", "条件", "比較", "照応", "意図・中止"]
