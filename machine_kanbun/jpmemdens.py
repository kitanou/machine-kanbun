"""Issue #72: personal-AI long-term memory (conversation -> extracted NF memory -> SeCF-L1) and Memory Density.

Synthetic users (gold facts from templates, no real data); a multi-day conversation per user written by gemma from the gold facts (each gold fact is verified to appear
in its day's conversation); memory extraction by gemma (consolidated over all days, and incremental per day); SeCF-L1 of every memory line (jpchat.secf1_text);
QA over the full memory in each representation (keyword scoring, stale-value check), same-token-budget truncation, and a cache run.
Modes: gen | extract | convert | stats | qa | budget | report  (see scripts/run_dens72.sh).
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from . import jl1
from .gen import SURNAMES
from .jpchat import secf1_text
from .jpdata import GIVEN
from .lmstudio import chat

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "jp7"
MODEL = "google/gemma-4-12b"
EXTRA = {"reasoning_effort": "none"}
DAYS = 10
NUSERS = int(os.environ.get("MD_USERS", "10"))

FOODS = "味噌ラーメン 鯖の塩焼き 麻婆豆腐 カツカレー 親子丼 餃子 天ぷら そば 焼きうどん ハンバーグ 唐揚げ 肉じゃが".split()
DISLIKES = "ピーマン セロリ ナス きのこ パクチー ゴーヤ トマト 納豆 レバー ししゃも".split()
MUSIC = "ジャズ ボサノバ クラシック シティポップ ロック ヒップホップ アニソン フォーク".split()
DRINK = "ブラックコーヒー ほうじ茶 カフェオレ 緑茶 紅茶 ジンジャーエール".split()
HOBBY = "登山 写真 釣り 陶芸 将棋 ボルダリング ギター 読書 サイクリング 料理".split()
STUDY = "ポルトガル語 統計学 簿記 電子工作 天文学 書道 韓国語 ドイツ語".split()
PROJ = "家計簿アプリ 読書記録ツール 献立ジェネレーター 天気通知ボット 写真整理スクリプト 勉強タイマー".split()
LANG = "Rust Go Python TypeScript Kotlin Swift Ruby".split()
RULES = [("夜10時以降は仕事のメールを見ない", "10時"), ("週末は仕事をしない", "週末"), ("毎月1万円を寄付する", "寄付"), ("1日1回は外に出て歩く", "歩く"), ("買い物は必ず一晩考えてから決める", "一晩"),
         ("会議は30分以内で終わらせる", "30分")]
JOBS = "看護師 教師 建築士 エンジニア 美容師 翻訳家 栄養士 会計士".split()
CITIES = "福岡 仙台 札幌 金沢 広島 松本 那覇 高松".split()
PCS = ["M5 Pro 64GB", "M5 Max 64GB", "M5 Max 128GB", "ThinkPad X1", "Mac Studio", "Framework 16"]
PHONES = "Pixel iPhone Xperia AQUOS Galaxy".split()
EDITORS = "Neovim VSCode Emacs Zed Helix".split()
TRIPS = "沖縄 北海道 京都 長崎 屋久島 台湾 ベトナム ポルトガル".split()
EXAMS = [("基本情報技術者試験", "基本情報"), ("英検準1級", "準1級"), ("簿記2級", "簿記"), ("FP2級", "FP")]
ABROAD = "カナダ オーストラリア 台湾 イギリス ドイツ".split()
VIDEO = "Netflix Hulu U-NEXT Disney+".split()
SLEEP = ["11時", "12時", "2時", "10時半"]
RUN = ["20分", "30分", "45分"]

UNCERT = ["迷", "未定", "決まっていない", "決めていない", "検討", "考え中", "わからない", "分からない", "まだ", "未確定", "保留"]
STOPPED = ["やめ", "止め", "中止", "していない", "していません", "終了", "解約", "いません"]


def _name(r: random.Random) -> str:
    return r.choice(SURNAMES) + r.choice(GIVEN)


def make_user(uid: int, seed: int = 72) -> Dict:
    r = random.Random(seed * 1000 + uid)
    pick = lambda pool, k=1: r.sample(pool, k) if k > 1 else r.choice(pool)
    F: List[Dict] = []

    def add(cat, day, stmt, check, q, gold, stale=(), qtype="final", chain=None):
        F.append(dict(id=len(F), cat=cat, day=day, stmt=stmt, check=list(check), q=q, gold=list(gold), stale=list(stale), qtype=qtype, chain=chain))

    d = lambda lo=1, hi=DAYS: r.randint(lo, hi)
    v = pick(FOODS, 1); add("preference", d(), f"好きな食べ物は{v}", [v], "ユーザーが一番好きな食べ物は何ですか。", [v])
    v = pick(DISLIKES); add("preference", d(), f"{v}が苦手で食べられない", [v], "ユーザーが苦手な食べ物は何ですか。", [v])
    v = pick(MUSIC); add("preference", d(), f"最近よく聴く音楽は{v}", [v], "ユーザーが最近よく聴く音楽のジャンルは何ですか。", [v])
    v = pick(DRINK); add("preference", d(), f"毎朝{v}を飲む", [v], "ユーザーが毎朝飲む飲み物は何ですか。", [v])
    v = pick(HOBBY); add("interest", d(), f"趣味は{v}", [v], "ユーザーの趣味は何ですか。", [v])
    v = pick(STUDY); add("interest", d(), f"最近{v}を独学で勉強している", [v], "ユーザーが独学で勉強しているものは何ですか。", [v])
    pn = pick(PROJ); add("project", d(1, 3), f"個人プロジェクトとして「{pn}」を作っている", [pn], "ユーザーが作っている個人プロジェクトの名前は何ですか。", [pn])
    la, lb = pick(LANG, 2)
    d1 = d(2, 5); d2 = d(d1 + 2, DAYS)
    add("project", d1, f"「{pn}」は最初は{la}で書き始めた", [la], "「" + pn + "」の現在の実装言語は何ですか。", [lb], stale=[la], chain="lang")
    add("update", d2, f"「{pn}」の実装を{la}から{lb}に移行した(以後は{lb}で開発)", [lb], "「" + pn + "」は最初何の言語で書き始めましたか。", [la], qtype="history", chain="lang")
    ru, rk = pick(RULES); add("value", d(), f"仕事では「{ru}」と決めている", [rk], "ユーザーが仕事で決めているルールは何ですか。", [rk])
    pnm, pjob = _name(r), pick(JOBS); add("relation", d(), f"同居しているパートナーは{pnm}さんで、職業は{pjob}", [pnm, pjob], "ユーザーのパートナーの名前は何ですか。", [pnm])
    add("relation", d(), f"{pnm}さんの職業は{pjob}", [pjob], "ユーザーのパートナーの職業は何ですか。", [pjob])
    fn, fc = _name(r), pick(CITIES); add("relation", d(), f"大学時代の友人の{fn}さんは{fc}に住んでいる", [fn, fc], f"大学時代の友人の{fn}さんはどこに住んでいますか。", [fc])
    bn = _name(r); add("relation", d(), f"今の上司は{bn}さんで、1on1が毎週ある", [bn], "ユーザーの今の上司の名前は何ですか。", [bn])
    ph = pick(PHONES); add("possession", d(), f"スマホは{ph}を使っている", [ph], "ユーザーが使っているスマホは何ですか。", [ph])
    ed = pick(EDITORS); add("possession", d(), f"普段のエディタは{ed}", [ed], "ユーザーが普段使っているエディタは何ですか。", [ed])
    pa, pb = pick(PCS[:3], 2) if r.random() < 0.7 else pick(PCS[3:], 2)
    da = d(1, 3); db = d(da + 2, da + 4); dc = d(db + 2, DAYS)
    add("possession", da, f"新しいPCの候補として{pa}を検討中(まだ決めていない)", [pa], None, [pb], stale=[pa], chain="pc")
    add("update", db, f"新しいPCの候補に{pb}も追加した(まだ決めていない)", [pb], "ユーザーが最初に検討していたPCは何ですか。", [pa], qtype="history", chain="pc")
    add("update", dc, f"新しいPCは{pb}に決定した", [pb], "ユーザーの新しいPCは決まっていますか。決まっているなら何ですか。", [pb], stale=[pa], chain="pc")
    ta, tb = pick(TRIPS, 2); t1 = d(1, 4); t2 = d(t1 + 2, DAYS)
    add("plan", t1, f"夏休みの旅行は{ta}に行く予定", [ta], f"ユーザーの夏休みの旅行先はどこですか。", [tb], stale=[ta], chain="trip")
    add("update", t2, f"夏休みの旅行先を{ta}から{tb}に変更した", [tb], "ユーザーは最初、夏休みの旅行先をどこにする予定でしたか。", [ta], qtype="history", chain="trip")
    ex, ek = pick(EXAMS); m = r.randint(1, 12); add("plan", d(), f"来年{m}月に{ex}を受験する予定", [ex, f"{m}月"], "ユーザーが受験する予定の資格試験は何ですか。", [ek])
    add("plan", d(), "転職するかどうか迷っていて、まだ決めていない", ["転職"], "ユーザーは転職を決めていますか。", UNCERT, qtype="uncertain")
    ab = pick(ABROAD); add("past", d(), f"学生時代に{ab}へ短期留学した経験がある", [ab], "ユーザーが学生時代に短期留学した国はどこですか。", [ab])
    sl = pick(SLEEP); add("habit", d(), f"最近は夜{sl}に寝ている", [sl], "ユーザーの最近の就寝時刻は何時ですか。", [sl.replace("半", "")])
    rn = pick(RUN); add("habit", d(), f"毎朝{rn}ジョギングしている", [rn], "ユーザーは毎朝何分ジョギングしていますか。", [rn.replace("分", "")])
    d1 = d(1, 5); d2 = d(d1 + 2, DAYS)
    add("habit", d1, "ダイエットのため炭水化物を減らしている", ["ダイエット"], "ユーザーは現在ダイエットをしていますか。", STOPPED, chain="diet")
    add("update", d2, "ダイエットはやめて、普通に食べるようにした", ["ダイエット"], "ユーザーは現在ダイエットをしていますか。", STOPPED, chain="diet")
    vs = pick(VIDEO); d1 = d(1, 5); d2 = d(d1 + 2, DAYS)
    add("possession", d1, f"動画サービスは{vs}を契約している", [vs], f"ユーザーは今{vs}を契約していますか。", STOPPED, chain="video")
    add("update", d2, f"{vs}は解約した", [vs], f"ユーザーは今{vs}を契約していますか。", STOPPED, chain="video")
    # the diet / video chains share the question: keep only the later (update) question to avoid duplicates
    keep, seen = [], set()
    for f in F:
        if f["chain"] in ("diet", "video"):
            if f["cat"] != "update":
                f = dict(f, q=None)
        keep.append(f)
    for i, f in enumerate(keep):
        f["id"] = i
    return dict(uid=uid, facts=keep)


def users(n: int = None) -> List[Dict]:
    return [make_user(i) for i in range(n or NUSERS)]


def questions(u: Dict) -> List[Dict]:
    return [f for f in u["facts"] if f["q"]]


def keyword_ok(answer: str, f: Dict) -> bool:
    return any(g in answer for g in f["gold"])


def stale_hit(answer: str, f: Dict) -> bool:
    return bool(f["stale"]) and any(s in answer for s in f["stale"]) and not keyword_ok(answer, f)


CONV_SYSTEM = ("あなたは日本語の会話データを作ります。ユーザーと AI アシスタントの、その日の自然な雑談を書きます。\n"
               "- 形式は「ユーザー: …」「AI: …」を1行ずつ交互に、合計16〜24行。説明や見出しは書かない。\n"
               "- ユーザーは口語で話し、指定された事実を会話の流れの中で自然に(あちこちに分けて)話す。事実の固有名詞・数・状態(迷い・変更・やめた等)は指定どおりに書く。\n"
               "- 指定されていない個人的な事実は作らない。会話の半分以上は、事実と無関係な雑談(天気・仕事の愚痴・ニュース・今日の出来事・あいづち)にする。AI は共感・質問・短い助言をする。")


def gen_day(u: Dict, day: int, log=print) -> str:
    fs = [f for f in u["facts"] if f["day"] == day]
    if not fs:
        fs = []
    listing = "\n".join(f"- {f['stmt']}" for f in fs) or "(この日は個人的な事実は話さない。天気・仕事などの雑談だけ)"
    missing = []
    for attempt in range(3):
        extra = "" if not missing else "\n特に次の事実を必ず会話に含めること: " + " / ".join(missing)
        r = chat(MODEL, CONV_SYSTEM, f"その日にユーザーが話す事実:\n{listing}{extra}\n\n会話:", max_tokens=1100, extra=EXTRA, deadline=240, timeout=260)
        text = r.text.strip()
        missing = [f["stmt"] for f in fs if not any(c in text for c in f["check"]) or (f["chain"] and f["cat"] == "update" and not all(c in text for c in f["check"]))]
        if not missing:
            break
    return text, missing


def gen(log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "conversations.jsonl"
    done = {(r["uid"], r["day"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    for u in users():
        for day in range(1, DAYS + 1):
            if (u["uid"], day) in done:
                continue
            t0 = time.perf_counter()
            text, missing = gen_day(u, day)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(uid=u["uid"], day=day, text=text, missing=missing, seconds=round(time.perf_counter() - t0, 1)), ensure_ascii=False) + "\n")
            log(f"conv user {u['uid']} day {day}: {len(text)} chars, missing={len(missing)}")


def load_conv() -> Dict[int, Dict[int, Dict]]:
    out: Dict[int, Dict[int, Dict]] = {}
    for l in (OUT / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        out.setdefault(r["uid"], {})[r["day"]] = r
    return out


def conv_text(days: Dict[int, Dict]) -> str:
    return "\n\n".join(f"【Day {d}】\n{days[d]['text']}" for d in sorted(days))


EXT_SYSTEM = ("あなたはパーソナル AI の長期記憶を作ります。会話から、将来の会話で役に立つ『ユーザー自身についての情報』(嗜好・興味・プロジェクト・方針・人物関係・所有物・計画・経験・習慣)だけを抜き出し、"
              "自然な日本語の文で1行1項目、行頭を「- 」にして書きます。\n"
              "- 雑談・天気・その日だけのこと・AI の発言は書かない。固有名詞・数・時期は残す。\n"
              "- 確定していないこと(迷い・検討中)は未確定と明記し、確定したことにしない。\n"
              "- 情報が更新・撤回された場合は最新の状態を書く。変更前の状態が将来の会話で役立つ場合は「もともと〜、Day N に〜へ変更」のように簡潔に添える。\n"
              "- 前置きや説明は書かない。")
EXT_SYSTEM_DAY = EXT_SYSTEM.replace("情報が更新・撤回された場合は最新の状態を書く。変更前の状態が将来の会話で役立つ場合は「もともと〜、Day N に〜へ変更」のように簡潔に添える。",
                                    "この日の会話で出た情報だけを書く(過去の日の情報は知らない)。変更・撤回の発言があれば、その内容をそのまま書く。")


def bullets(text: str) -> List[str]:
    out = []
    for l in text.splitlines():
        l = l.strip().lstrip("-・* ").strip()
        if l:
            out.append(l)
    return out


def extract(log=print) -> None:
    path = OUT / "memory_nf.jsonl"
    done = {(r["uid"], r["kind"], r.get("day")) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    C = load_conv()
    for u in users():
        days = C[u["uid"]]
        if (u["uid"], "consolidated", None) not in done:
            t0 = time.perf_counter()
            r = chat(MODEL, EXT_SYSTEM, f"会話:\n{conv_text(days)}\n\n長期記憶メモ:", max_tokens=2400, extra=EXTRA, deadline=600, timeout=620)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(uid=u["uid"], kind="consolidated", day=None, lines=bullets(r.text), raw=r.text, prompt_tokens=r.prompt_tokens, seconds=round(time.perf_counter() - t0, 1)), ensure_ascii=False) + "\n")
            log(f"extract consolidated user {u['uid']}: {len(bullets(r.text))} lines")
        for day in sorted(days):
            if (u["uid"], "incremental", day) in done:
                continue
            r = chat(MODEL, EXT_SYSTEM_DAY, f"会話:\n{days[day]['text']}\n\n長期記憶メモ:", max_tokens=700, extra=EXTRA, deadline=240, timeout=260)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(uid=u["uid"], kind="incremental", day=day, lines=bullets(r.text), raw=r.text), ensure_ascii=False) + "\n")
        log(f"extract incremental user {u['uid']} done")


def load_mem() -> Dict[tuple, Dict]:
    out = {}
    for l in (OUT / "memory_nf.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        out[(r["uid"], r["kind"], r["day"])] = r
    return out


def memory_lines(uid: int, kind: str) -> List[Dict]:
    """[(day, line)] of one user's memory: consolidated has day=None per line (order produced), incremental carries the day."""
    M = load_mem()
    if kind == "consolidated":
        return [dict(day=None, nf=x) for x in M[(uid, "consolidated", None)]["lines"]]
    return [dict(day=d, nf=x) for d in range(1, DAYS + 1) if (uid, "incremental", d) in M for x in M[(uid, "incremental", d)]["lines"]]


def convert(log=print) -> None:
    conv = jl1.Converter("sudachi", "m")
    n = 0
    for u in users():
        for kind in ("consolidated", "incremental"):
            for L in memory_lines(u["uid"], kind):
                secf1_text(L["nf"])
                conv(L["nf"])
                n += 1
        log(f"secf1 converted user {u['uid']} ({n} lines)")


def o200k(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def rep_lines(uid: int, kind: str) -> Dict[str, List[Dict]]:
    conv = jl1.Converter("sudachi", "m")
    L = memory_lines(uid, kind)
    return {"nf": [dict(day=x["day"], text=x["nf"]) for x in L], "secf1": [dict(day=x["day"], text=secf1_text(x["nf"])) for x in L], "scf": [dict(day=x["day"], text=conv(x["nf"])) for x in L]}


def block(lines: List[Dict], with_day: bool) -> str:
    return "\n".join((f"[Day{x['day']}] " if with_day and x["day"] else "") + x["text"] for x in lines)


QA_SYS = "あなたは記憶メモだけを根拠に質問へ答えるアシスタントです。"
QA_INSTR = "記憶メモに書かれた最新の状態を根拠に、短い語句で一言で答えてください。メモにない場合は「不明」と答えてください。"


def lms(model: str, ctx: int = 0) -> None:
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    subprocess.run(["lms", "load", model] + (["-c", str(ctx), "--parallel", "1"] if ctx else []) + ["-y"], capture_output=True)


def jl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def contexts_for(uid: int) -> Dict[str, str]:
    """Full-memory contexts per condition: raw conversation, consolidated NF/SeCF-L1/SCF-L1, incremental NF/SeCF-L1."""
    C = load_conv()
    ctx = {"A_raw": "会話履歴:\n" + conv_text(C[uid])}
    for kind, tag in (("consolidated", "c"), ("incremental", "i")):
        R = rep_lines(uid, kind)
        for rep in ("nf", "secf1", "scf"):
            if kind == "incremental" and rep == "scf":
                continue
            ctx[f"{tag}_{rep}"] = "記憶メモ:\n" + block(R[rep], kind == "incremental")
    return ctx


def qa(n_users: int = NUSERS, log=print) -> None:
    path = OUT / "qa.jsonl"
    done = {(r["uid"], r["cond"], r["fid"]) for r in jl(path)}
    lms(MODEL, 19456)
    for u in users(n_users):
        ctx = contexts_for(u["uid"])
        for cond, c in ctx.items():
            rows = []
            for f in questions(u):
                if (u["uid"], cond, f["id"]) in done:
                    continue
                r = chat(MODEL, QA_SYS, f"{c}\n\n質問: {f['q']}\n{QA_INSTR}", max_tokens=60, extra=EXTRA, deadline=240, timeout=260)
                rows.append(dict(uid=u["uid"], cond=cond, fid=f["id"], cat=f["cat"], qtype=f["qtype"], answer=r.text, ok=keyword_ok(r.text, f), stale=stale_hit(r.text, f),
                                 prompt_tokens=r.prompt_tokens, ttft=r.ttft))
            with path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if rows:
                log(f"QA user {u['uid']} {cond}: acc={sum(r['ok'] for r in rows) / len(rows):.3f} prompt={rows[0]['prompt_tokens']}")


BUDGETS = (0.25, 0.4, 0.6)  # fraction of the o200k size of the user's full incremental NF memory


def budget_keep(R: Dict[str, List[Dict]], frac: float) -> Dict[str, List[Dict]]:
    """Per representation: keep the newest lines (oldest day dropped first) that fit frac * o200k(full incremental NF memory)."""
    cap = frac * o200k(block(R["nf"], True))
    out = {}
    for rep, L in R.items():
        keep, used = [], 0
        for x in reversed(L):
            t = o200k(("[Day%s] " % x["day"] if x["day"] else "") + x["text"] + "\n")
            if used + t > cap:
                break
            keep.append(x)
            used += t
        out[rep] = list(reversed(keep))
    return out


def coverage(uid: int, text: str) -> float:
    fs = [f for f in users(uid + 1)[uid]["facts"] if f["q"] and f["qtype"] == "final"]
    return sum(1 for f in fs if any(g in text for g in f["gold"])) / len(fs)


def budget(n_users: int = NUSERS, log=print) -> None:
    path = OUT / "budget.jsonl"
    done = {(r["uid"], r["frac"], r["rep"], r["fid"]) for r in jl(path)}
    lms(MODEL, 19456)
    for u in users(n_users):
        R = rep_lines(u["uid"], "incremental")
        for frac in BUDGETS:
            kept = budget_keep(R, frac)
            for rep in ("nf", "secf1", "scf"):
                text = block(kept[rep], True)
                rows = []
                for f in questions(u):
                    if (u["uid"], frac, rep, f["id"]) in done:
                        continue
                    r = chat(MODEL, QA_SYS, f"記憶メモ:\n{text}\n\n質問: {f['q']}\n{QA_INSTR}", max_tokens=60, extra=EXTRA, deadline=240, timeout=260)
                    rows.append(dict(uid=u["uid"], frac=frac, rep=rep, fid=f["id"], qtype=f["qtype"], answer=r.text, ok=keyword_ok(r.text, f), stale=stale_hit(r.text, f),
                                     lines=len(kept[rep]), lines_total=len(R[rep]), prompt_tokens=r.prompt_tokens))
                with path.open("a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                if rows:
                    log(f"budget user {u['uid']} frac={frac} {rep}: acc={sum(r['ok'] for r in rows) / len(rows):.3f} lines={len(kept[rep])}/{len(R[rep])}")


CACHE_USERS = (20, 24, 28, 32)


def cache(log=print) -> None:
    """Prefix cache with realistic personal memories: U users (the 10 incremental memories repeated under unique ids), K=4 slots, round-robin x 3 rounds; clean reload per representation.
    The cache hit is read from TTFT (round 1-2 TTFT vs the same user's round 0). #69 found the cache capacity ~25-29k total prompt tokens."""
    import uuid
    from . import jpcache
    path = OUT / "cache.jsonl"
    done = {(r["rep"], r["users"], r["round"], r["user"]) for r in jl(path)}
    base = {u["uid"]: rep_lines(u["uid"], "incremental") for u in users()}
    qs = {u["uid"]: [f["q"] for f in questions(u)] for u in users()}
    for U in CACHE_USERS:
        for rep in ("secf1", "nf"):
            if all((rep, U, rd, i) in done for rd in range(3) for i in range(U)):
                continue
            eff = jpcache.load(24576, parallel=4)
            jpcache.call("記憶メモ: 暖機", "ウォームアップ", nonce=f"[run:{uuid.uuid4().hex[:8]}]\n")
            nonces = {i: f"[run:{uuid.uuid4().hex[:8]}]\n" for i in range(U)}
            for rd in range(3):
                for i in range(U):
                    if (rep, U, rd, i) in done:
                        continue
                    jpcache.guard()
                    uid = i % len(base)
                    mem = block(base[uid][rep], True)
                    try:
                        r, err = jpcache.call(mem, qs[uid][(rd * U + i) % len(qs[uid])], nonce=nonces[i]), None
                    except Exception as e:
                        r, err = dict(ttft=None, total=None, prompt_tokens=None), str(e)[:100]
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(dict(rep=rep, users=U, round=rd, user=i, eff_ctx=eff, error=err, **r)) + "\n")
                    log(f"cache {rep} U={U} round={rd} user={i} tokens={r['prompt_tokens']} ttft={(r['ttft'] or 0):.2f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    nu = NUSERS
    if mode == "gen":
        gen(log=lambda s: print(s, flush=True))
    elif mode == "extract":
        extract(log=lambda s: print(s, flush=True))
    elif mode == "convert":
        convert(log=lambda s: print(s, flush=True))
    elif mode == "qa":
        qa(nu, log=lambda s: print(s, flush=True))
    elif mode == "cache":
        cache(log=lambda s: print(s, flush=True))
    elif mode == "budget":
        budget(nu, log=lambda s: print(s, flush=True))
