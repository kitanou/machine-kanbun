"""Issue #27: does the simple-L1 benefit survive natural, noisy conversation instead of template-generated memories?

No real chat corpus is available offline, so the conversations are written by an LLM (gemma-4-12b) from a *specification*
(person, event, status, 1-2 phenomena) that is independent of the template generator in jpdata.py: multi-turn LINE-style chats
with subject omission, demonstratives, self-corrections, fragments, slang, split facts across turns; typos are injected
programmatically. Gold labels come from the specification and are verified (the same model must read the original chat
back to the same status; disagreements are dropped, which favours L0 and is reported). Failure categories = phenomena.
"""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import jl1, jlvar, jpdata
from .jpbench import RES
from .jpllm import INSTR, SYSTEM, score
from .lmstudio import Stalled, chat

OUT = RES.parent / "jp2"
STATUS_DESC = {
    "DONE": "もう実際にそうした(完了している)", "NOT_DONE": "まだそうしていない(していないと明確に否定されている)", "PLANNED": "これからそうする予定・つもりである(まだ実行はしていない)",
    "WANTED": "そうしたいと望んでいる(まだ実行はしていない)", "UNDECIDED": "するかどうか迷っていて、まだ決まっていない", "HEARSAY": "人から聞いただけで、本当かどうかは確認できていない",
    "POSSIBLE": "そうする(した)可能性があるというだけで、確かではない(〜かもしれない)"}
PHENOM = {
    "subj_omit": "主語を省略する。人物のフルネームは最初の発言に1回だけ出し、以降は主語なしで話す",
    "anaphora": "『それ』『あれ』『その件』『前のやつ』などの指示語を使い、出来事そのものの名詞は最初の1回以外は繰り返さない",
    "selfcorrect": "途中で言い直しをする(『あ、違う、〜じゃなくて』など)。最終的に述べられた事実が正しい内容になる",
    "fragment": "文末を省略した断片的な短い発言や体言止めを多く使う",
    "slang": "俗語・若者言葉・くだけた口語(まじ、やばい、〜じゃん、〜っぽい)を多く使う",
    "multi_turn": "事実が複数の発言に分かれて語られる(1つの発言だけでは誰が何をしたか分からず、前後の発言と合わせて分かる)",
    "plain": "特別な癖のない自然なくだけた会話"}
TYPO_PROB = 0.35


def spec_list(n: int, seed: int = 27):
    rng = random.Random(seed)
    mems = jpdata.make_memories(n * 2, seed=seed)
    ev = {e.key: e for e in jpdata.EVENTS}
    out = []
    for m in mems[:n]:
        e = ev[m.event]
        f = jpdata.conj(e)
        ph = rng.sample([p for p in PHENOM if p != "plain"], rng.choice([1, 2]))
        out.append(dict(id=m.id, person=m.person, event=m.event, status=m.status, gold=jpdata.ANSWER[m.status], phrase=f"{e.np}{f['dict']}", q=f"{m.person}さんは{e.np}{f['ta']}？",
                        phenomena=ph, typo=rng.random() < TYPO_PROB))
    return out


def gen_prompt(s) -> str:
    ph = "\n".join(f"- {PHENOM[p]}" for p in s["phenomena"])
    return (f"友人同士のLINE風の雑談チャットを、日本語で4〜7発言(A:とB:の交互)書いてください。\n"
            f"話題: 共通の知り合いの「{s['person']}さん」が「{s['phrase']}」ことについて。\n"
            f"{s['person']}さんについての事実の状態: {STATUS_DESC[s['status']]}。この状態が会話から読み取れるようにしてください(ただし『完了』『未確定』などの単語は使わない)。\n"
            f"会話の特徴:\n{ph}\n"
            f"会話以外の説明や括弧書きは書かず、発言行だけを出力してください。")


def inject_typos(text: str, rng: random.Random) -> str:
    """1-2 mild noise edits that avoid the person's name: drop a kana, small-kana swap, or kanji -> hiragana of a common word."""
    swaps = [("です", "でs"), ("ちゃんと", "ちゃんと"), ("本当", "ほんと"), ("会社", "かいしゃ"), ("結婚", "けっこん"), ("仕事", "しごと"), ("やばい", "ヤバい"), ("そうだ", "そうだー")]
    lines = text.split("\n")
    for _ in range(rng.choice([1, 2])):
        k = rng.randrange(len(lines))
        for a, b in swaps:
            if a in lines[k] and a != b:
                lines[k] = lines[k].replace(a, b, 1)
                break
        else:
            kana = [i for i, c in enumerate(lines[k]) if "ぁ" <= c <= "ん" and i > 3]
            if kana:
                i = rng.choice(kana)
                lines[k] = lines[k][:i] + lines[k][i + 1:]
    return "\n".join(lines)


def generate(model: str, n: int = 90, seed: int = 27, base_url="http://localhost:1234/v1") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "natural_chats.jsonl"
    have = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    rng = random.Random(seed + 1)
    for s in spec_list(n, seed):
        if s["id"] in have:
            continue
        r = chat(model, f"[run:{uuid.uuid4().hex[:8]}] あなたは自然な日本語のチャットを書くライターです。", gen_prompt(s), max_tokens=500, base_url=base_url, extra=extra, deadline=240)
        text = "\n".join(l.strip() for l in r.text.splitlines() if re.match(r"\s*[AB][:：]", l))
        if len(text.splitlines()) < 3 or s["person"] not in text:
            rec = dict(s, ok_gen=False, text=r.text)
        else:
            raw = text
            if s["typo"]:
                text = inject_typos(text, rng)
            # verification: the model reads the ORIGINAL chat and must reach the gold status
            v = chat(model, f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}", f"記憶メモ:\n{raw}\n\n質問: {s['q']}\n{INSTR}", max_tokens=10, base_url=base_url, extra=extra, deadline=240)
            rec = dict(s, ok_gen=True, text=text, text_clean=raw, verified=score(v.text, s["gold"]), verify_answer=v.text)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_l1_chat(conv, text: str) -> str:
    """Convert line by line, keeping turn boundaries ('A ...' / 'B ...')."""
    out = []
    for ln in text.split("\n"):
        m = re.match(r"\s*([AB])[:：]\s*(.*)", ln)
        if not m:
            continue
        out.append(f"{m.group(1)} {conv(m.group(2))}")
    return "\n".join(out)


def load_chats(only_verified: bool = True) -> List[Dict]:
    path = OUT / "natural_chats.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r.get("ok_gen") and (r.get("verified") or not only_verified)]


class Refined:
    """Issue #27 refinement after the failure analysis (self-corrections, slang and negation scope were lost): keep conjugated surface
    forms and interjections / conjunctions / fillers (r1); additionally keep case particles and clause connectors (r2)."""

    def __init__(self, level: int):
        self.level = level
        self.an = jl1.make("sudachi")

    def __call__(self, text: str) -> str:
        return jl1.to_l1(self.an.analyze(text), surface=True, keep_filler=True, keep_case=self.level >= 2, keep_connectors=self.level >= 2, keep_quote=True)


REPS = {"L0": None, "sudachi-m": ("sudachi", "m"), "sudachi-g": ("sudachi", "g"), "naive": ("naive", "m"), "refined-r1": Refined(1), "refined-r2": Refined(2)}


def run_qa(model: str, base_url="http://localhost:1234/v1", log=print, max_chats: int = 0) -> None:
    """All verified chats form one memory store (a few thousand tokens); the question for every chat is asked against it."""
    chats = load_chats()
    path = OUT / f"nat_qa_{model.replace('/', '_')}.jsonl"
    done = {json.loads(l)["rep"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    rng = random.Random(9)
    order = list(range(len(chats)))
    rng.shuffle(order)
    chats = [chats[i] for i in order]
    if max_chats:
        chats = chats[:max_chats]
    for rep, spec in REPS.items():
        if rep in done:
            continue
        conv = (spec if isinstance(spec, Refined) else jl1.Converter(*spec)) if spec else None
        t0 = time.perf_counter()
        docs = [c["text"] if conv is None else to_l1_chat(conv, c["text"]) for c in chats]
        conv_ms = (time.perf_counter() - t0) / len(chats) * 1000
        ctx = "\n\n".join(f"[{i + 1}]\n{d}" for i, d in enumerate(docs))
        system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}"
        rows = []
        try:
            base = chat(model, f"[base:{uuid.uuid4().hex[:8]}] {SYSTEM}", f"記憶メモ:\n\n\n質問: x\n{INSTR}{nt}", max_tokens=12, base_url=base_url, extra=extra, deadline=240)
            for i, c in enumerate(chats):
                r = chat(model, system, f"記憶メモ:\n{ctx}\n\n質問: {c['q']}\n{INSTR}{nt}", max_tokens=8, base_url=base_url, extra=extra,
                         deadline=400 if i == 0 else 240, timeout=420 if i == 0 else 300)
                rows.append(dict(model=model, rep=rep, i=i, id=c["id"], status=c["status"], phenomena=c["phenomena"], typo=c["typo"], gold=c["gold"], answer=r.text,
                                 ok=score(r.text, c["gold"]), prompt_tokens=r.prompt_tokens, ctx_tokens=(r.prompt_tokens - base.prompt_tokens) if i == 0 else None,
                                 ttft=r.ttft, total=r.total, conv_ms=conv_ms, ctx_chars=len(ctx)))
        except Stalled:
            raise
        with path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"{model} natural {rep}: acc={sum(r['ok'] for r in rows) / len(rows):.1%} ctx={rows[0]['ctx_tokens']}tok cold_ttft={rows[0]['ttft']:.1f}s n={len(rows)}")
