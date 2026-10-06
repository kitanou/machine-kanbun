"""Issue #39 stage 1: tokenizer-only compactness on a parallel fact set (10 facts x 9 languages; no LLM).

Names are kept in Latin script in every language so name tokenisation is not a confound. Translations were written by the
assistant (not verified by native speakers); Greenlandic (kl) is left out because a reliable translation was not available.
Three compactness notions: linguistic (characters / UTF-8 bytes / whitespace words per fact), token (tokens per fact under a
tokenizer), semantic (always 1 fact per sentence here, so tokens per fact is the semantic cost).
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

from .jpbench import RES

OUT = RES.parent / "par"
FACTS = {
    "en": ["Yesterday I talked with Tanaka for the first time in a while.", "Tanaka is thinking of quitting the company but has not decided yet.", "Sato has already moved to Osaka.",
           "Suzuki did not go to the meeting.", "Yamada plans to start a new job next month.", "Ito wants to sell the house.",
           "I heard that Kato got married, but I have not confirmed it.", "Mori may change schools.", "It rained all day today and I feel down.", "The new cafe near the station opens next week."],
    "ja": ["昨日、Tanakaさんと久しぶりに話した。", "Tanakaさんは会社を辞めようか迷っているが、まだ決めていない。", "Satoさんはもう大阪に引っ越した。", "Suzukiさんは会議に行かなかった。",
           "Yamadaさんは来月から新しい仕事を始める予定だ。", "Itoさんは家を売りたがっている。", "Katoさんが結婚したと聞いたが、確認はしていない。", "Moriさんは転校するかもしれない。",
           "今日は一日中雨で、気分が沈んでいる。", "駅の近くの新しいカフェは来週オープンする。"],
    "zh": ["昨天和Tanaka久违地聊了聊。", "Tanaka在考虑辞职，但还没有决定。", "Sato已经搬到大阪了。", "Suzuki没有去开会。", "Yamada打算下个月开始新工作。", "Ito想卖掉房子。",
           "听说Kato结婚了，但我还没有确认。", "Mori可能会转学。", "今天下了一整天雨，我心情低落。", "车站附近的新咖啡馆下周开业。"],
    "ko": ["어제 Tanaka와 오랜만에 이야기했다.", "Tanaka는 회사를 그만둘까 고민 중이지만 아직 결정하지 않았다.", "Sato는 이미 오사카로 이사했다.", "Suzuki는 회의에 가지 않았다.",
           "Yamada는 다음 달부터 새 일을 시작할 예정이다.", "Ito는 집을 팔고 싶어 한다.", "Kato가 결혼했다고 들었지만 확인은 하지 않았다.", "Mori는 전학할지도 모른다.",
           "오늘은 하루 종일 비가 와서 기분이 가라앉는다.", "역 근처의 새 카페는 다음 주에 문을 연다."],
    "tr": ["Dün Tanaka ile uzun zamandan sonra konuştum.", "Tanaka işten ayrılmayı düşünüyor ama henüz karar vermedi.", "Sato çoktan Osaka'ya taşındı.", "Suzuki toplantıya gitmedi.",
           "Yamada gelecek ay yeni bir işe başlamayı planlıyor.", "Ito evini satmak istiyor.", "Kato'nun evlendiğini duydum ama doğrulamadım.", "Mori okul değiştirebilir.",
           "Bugün bütün gün yağmur yağdı ve moralim bozuk.", "İstasyonun yakınındaki yeni kafe gelecek hafta açılıyor."],
    "vi": ["Hôm qua tôi đã nói chuyện với Tanaka sau một thời gian dài.", "Tanaka đang cân nhắc nghỉ việc nhưng vẫn chưa quyết định.", "Sato đã chuyển đến Osaka rồi.", "Suzuki đã không đi họp.",
           "Yamada dự định bắt đầu công việc mới vào tháng sau.", "Ito muốn bán nhà.", "Tôi nghe nói Kato đã kết hôn nhưng chưa xác nhận.", "Mori có thể sẽ chuyển trường.",
           "Hôm nay trời mưa cả ngày nên tôi thấy buồn.", "Quán cà phê mới gần ga sẽ khai trương vào tuần sau."],
    "ru": ["Вчера я впервые за долгое время поговорил с Tanaka.", "Tanaka думает уволиться, но ещё не решил.", "Sato уже переехал в Осаку.", "Suzuki не пошёл на собрание.",
           "Yamada планирует начать новую работу в следующем месяце.", "Ito хочет продать дом.", "Я слышал, что Kato женился, но не проверял.", "Mori, возможно, сменит школу.",
           "Сегодня весь день шёл дождь, и у меня плохое настроение.", "Новое кафе возле станции откроется на следующей неделе."],
    "ar": ["تحدثت أمس مع Tanaka بعد غياب طويل.", "يفكر Tanaka في ترك الشركة لكنه لم يقرر بعد.", "انتقل Sato بالفعل إلى أوساكا.", "لم يذهب Suzuki إلى الاجتماع.", "يخطط Yamada لبدء عمل جديد الشهر القادم.",
           "يريد Ito بيع المنزل.", "سمعت أن Kato تزوج لكنني لم أتأكد من ذلك.", "قد ينتقل Mori إلى مدرسة أخرى.", "أمطرت اليوم طوال اليوم وأشعر بالإحباط.", "سيفتتح المقهى الجديد قرب المحطة الأسبوع القادم."],
    "eu": ["Atzo Tanaka-rekin hitz egin nuen aspaldiko partez.", "Tanaka lana uztea pentsatzen ari da, baina oraindik ez du erabaki.", "Sato dagoeneko Osakara joan da bizitzera.", "Suzuki ez zen bilerara joan.",
           "Yamada datorren hilean lan berri bati ekitea pentsatzen ari da.", "Ito-k etxea saldu nahi du.", "Kato ezkondu dela entzun dut, baina ez dut baieztatu.", "Mori ikastetxez alda daiteke.",
           "Gaur egun osoan euria egin du eta goibel nago.", "Geltokiaren ondoko kafetegi berria datorren astean irekiko da."],
}
TOK = {"o200k_base": None, "cl100k_base": None, "Qwen/Qwen3-8B": None, "NousResearch/Meta-Llama-3.1-8B-Instruct": None}


def counters():
    import tiktoken
    import glob
    from tokenizers import Tokenizer
    out = {}
    for n in ("o200k_base", "cl100k_base"):
        e = tiktoken.get_encoding(n)
        out[n] = lambda s, e=e: len(e.encode(s))
    for m in ("Qwen/Qwen3-8B", "NousResearch/Meta-Llama-3.1-8B-Instruct"):
        path = glob.glob(str(Path.home() / ".cache/huggingface/hub" / ("models--" + m.replace("/", "--")) / "snapshots/*/tokenizer.json"))[0]
        t = Tokenizer.from_file(path)
        out[m.split("/")[1]] = lambda s, t=t: len(t.encode(s, add_special_tokens=False).ids)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = counters()
    rows = []
    for lang, ss in FACTS.items():
        assert len(ss) == 10, lang
        r = dict(lang=lang, chars=sum(map(len, ss)) / 10, bytes=sum(len(s.encode()) for s in ss) / 10, words=sum(len(s.split()) for s in ss) / 10)
        for n, f in C.items():
            r[n] = sum(f(s) for s in ss) / 10
        rows.append(r)
    with (OUT / "compactness.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    en = next(r for r in rows if r["lang"] == "en")
    keys = [k for k in rows[0] if k != "lang"]
    L = ["# Issue #39 stage 1: 並列 10 文 × 9 言語のコンパクトさ(tokenizer のみ、LLM なし)\n",
         "1 文 = 1 事実あたりの平均。値は英語比(en=1.00)。chars/bytes/words が言語学的、トークン列が tokenizer 上の、(1 文 1 事実なので)tokens/fact が意味あたりのコスト。\n",
         "| 言語 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for r in rows:
        L.append(f"| {r['lang']} | " + " | ".join(f"{r[k]:.1f} ({r[k] / en[k]:.2f})" for k in keys) + " |")
    L.append("\n## 乖離(linguistic vs token compactness)\n")
    L.append("| 言語 | bytes比 | words比 | o200k比 | Qwen3比 | Llama3.1比 | 「chars比 → token比」の食い違い(o200k比/chars比) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['lang']} | {r['bytes'] / en['bytes']:.2f} | {r['words'] / en['words']:.2f} | {r['o200k_base'] / en['o200k_base']:.2f} | {r['Qwen3-8B'] / en['Qwen3-8B']:.2f} | "
                 f"{r['Meta-Llama-3.1-8B-Instruct'] / en['Meta-Llama-3.1-8B-Instruct']:.2f} | {(r['o200k_base'] / en['o200k_base']) / (r['chars'] / en['chars']):.2f} |")
    cv = {k: st.pstdev([r[k] for r in rows]) / st.mean([r[k] for r in rows]) for k in keys}
    L.append("\n変動係数(言語間): " + ", ".join(f"{k} {v:.2f}" for k, v in cv.items()) + "\n")
    L.append("注意: 翻訳は assistant が作成(母語話者の確認なし)。人名は全言語でラテン表記。グリーンランド語は信頼できる訳が用意できず対象外。")
    (OUT / "compactness.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
