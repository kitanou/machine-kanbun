"""Issue #67 (RESEARCH_MAP U13): why does the model read "まだ決めていない / 迷っている" as "いいえ"? Question format vs representation vs model prior.

Single-memory context (the "oracle" setting of #62, no retrieval), 15 memories per status from the #59 benchmark (jpdata.make_memories(1000, seed=28)),
three representations (NF / SCF-L1 / SeCF-L1 from the #59 cache), four question formats: V0 current 3-way (jpllm.INSTR), V1 V0 + explicit note on 迷い,
V2 7-way status choice mapped to the 3-way label, V3 V0 + few-shot examples. gemma-4-12b, temperature 0, deterministic scoring.
"""
from __future__ import annotations

import json
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from . import jl1, jpdata
from .jpbench import RES
from .jpchat import secf1_text
from .jpchatqa import option_text
from .jpllm import INSTR, SYSTEM, score
from .lmstudio import Stalled, chat

OUT = RES.parent / "jp4"
N_PER = 15
SEED = 28
LET = "ABCDEFG"
MAP3 = {"DONE": "はい", "NOT_DONE": "いいえ", "PLANNED": "いいえ", "WANTED": "いいえ", "UNDECIDED": "未確定", "HEARSAY": "未確定", "POSSIBLE": "未確定"}
V1_NOTE = ("注意: 「迷っている」「まだ決めていない」は、実行していないという意味ではなく、するかどうかが決まっていない状態なので「未確定」です。"
           "「しなかった」「しない」「予定」「したいと思っている」は「いいえ」です。")
FEWSHOT = ("例1 記憶メモ: ユウキさんは会社を辞めようか迷っているらしい。まだ決めたわけではないみたい。 質問: ユウキさんは会社を辞めた？ 答え: 未確定\n"
           "例2 記憶メモ: ハルカさんは来月家を売るつもりだ。 質問: ハルカさんは家を売った？ 答え: いいえ\n"
           "例3 記憶メモ: ソウタさんが結婚したって聞いたけど、本当かどうか分からない。 質問: ソウタさんは結婚した？ 答え: 未確定\n")
VARIANTS = ("V0", "V1", "V2", "V3")
REPS = ("nf", "scf", "secf1")
_FIRST = re.compile(r"\s*[「『]?(はい|いいえ|未確定|不明)")


def data():
    mems = jpdata.make_memories(1000, seed=SEED)
    chosen, cnt = [], Counter()
    for m in mems:
        if cnt[m.status] < N_PER:
            chosen.append(m)
            cnt[m.status] += 1
    return chosen


def rep_text(rep: str, text: str, conv) -> str:
    return text if rep == "nf" else conv(text) if rep == "scf" else secf1_text(text)


def prompt(variant: str, ctx: str, m, q) -> str:
    if variant == "V0":
        return f"記憶メモ:\n{ctx}\n\n質問: {q.text}\n{INSTR}"
    if variant == "V1":
        return f"記憶メモ:\n{ctx}\n\n質問: {q.text}\n{INSTR}{V1_NOTE}"
    if variant == "V3":
        return f"{FEWSHOT}\n記憶メモ:\n{ctx}\n\n質問: {q.text}\n{INSTR}"
    order = list(jpdata.STATUSES)
    random.Random(m.id).shuffle(order)
    opts = "\n".join(f"{LET[i]}. {option_text(dict(person=m.person, event=m.event), s)}" for i, s in enumerate(order))
    return f"記憶メモ:\n{ctx}\n\n{m.person}さんのことについて、記憶メモから最も正確なものはどれ?\n{opts}\n記号(A〜G)1文字だけで答えてください。", order


def run(model: str, log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"state_{model.replace('/', '_')}.jsonl"
    done = {(r["variant"], r["rep"], r["id"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    conv = jl1.Converter("sudachi", "m")
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    n = 0
    for m in data():
        q = jpdata.make_queries([m], 1, seed=m.id)[0]
        for rep in REPS:
            ctx = rep_text(rep, m.text, conv)
            for v in VARIANTS:
                if (v, rep, m.id) in done:
                    continue
                pr = prompt(v, ctx, m, q)
                order = None
                if v == "V2":
                    pr, order = pr
                r = chat(model, f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM}", pr, max_tokens=8, base_url="http://localhost:1234/v1", extra=extra, deadline=240)
                if v == "V2":
                    mm = re.search(r"[A-G]", r.text)
                    st7 = order[LET.index(mm.group(0))] if mm else None
                    pred = MAP3.get(st7)
                    rec = dict(variant=v, rep=rep, id=m.id, status=m.status, gold=q.answer, pred=pred, pred_status=st7, ok=pred == q.answer, exact7=st7 == m.status, raw=r.text[:8])
                else:
                    mm = _FIRST.match(r.text)
                    pred = mm.group(1) if mm else None
                    rec = dict(variant=v, rep=rep, id=m.id, status=m.status, gold=q.answer, pred=pred, ok=score(r.text, q.answer), raw=r.text[:8])
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 100 == 0:
                    log(f"state {n}")


def report() -> None:
    from .jpchatreport import boot_ci, perm_p
    rows = [json.loads(l) for l in (OUT / "state_google_gemma-4-12b.jsonl").read_text(encoding="utf-8").splitlines()]
    sts = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]
    cls = {"はい系(DONE)": ["DONE"], "いいえ系(NOT_DONE・PLANNED・WANTED)": ["NOT_DONE", "PLANNED", "WANTED"], "未確定系(UNDECIDED・HEARSAY・POSSIBLE)": ["UNDECIDED", "HEARSAY", "POSSIBLE"]}
    acc = lambda sel: float(np.mean([r["ok"] for r in sel])) if sel else float("nan")
    L = ["# Issue #67 「迷い」を「いいえ」と読む誤り: 質問形式 × 表現", f"\n単一の記憶(状態ごと {N_PER} 件 = {N_PER * 7} 件)、gemma-4-12b 温度 0。V0 現行 3 択、V1 迷いの明確化、V2 7 択を 3 択へ写像、V3 few-shot。\n",
         "## 1. 状態別の 3 択の正答率(表現 × 質問形式)\n", "| 表現 | 形式 | " + " | ".join(sts) + " | 全体 |", "|---|---|" + "---|" * (len(sts) + 1)]
    for rep in ("nf", "scf", "secf1"):
        for v in VARIANTS:
            sel = [r for r in rows if r["rep"] == rep and r["variant"] == v]
            if sel:
                L.append(f"| {rep} | {v} | " + " | ".join(f"{acc([r for r in sel if r['status'] == s]):.2f}" for s in sts) + f" | {acc(sel):.3f} |")
    L += ["\n## 2. 系統別の正答率(表現をまとめた平均)\n", "| 形式 | " + " | ".join(cls) + " |", "|---|---|---|---|"]
    for v in VARIANTS:
        L.append(f"| {v} | " + " | ".join(f"{acc([r for r in rows if r['variant'] == v and r['status'] in c]):.3f}" for c in cls.values()) + " |")
    L += ["\n## 3. UNDECIDED の変種間・表現間のペア比較(記憶単位、表現 3 通りの平均)\n", "| 比較 | 差[95%CI] | p |", "|---|---|---|"]

    def per_mem(v, reps=REPS):
        d = defaultdict(list)
        for r in rows:
            if r["variant"] == v and r["rep"] in reps and r["status"] == "UNDECIDED":
                d[r["id"]].append(float(r["ok"]))
        return {k: np.mean(x) for k, x in d.items()}
    base = per_mem("V0")
    for v in ("V1", "V2", "V3"):
        x = per_mem(v)
        dd = [x[k] - base[k] for k in base if k in x]
        lo, hi = boot_ci(dd)
        L.append(f"| {v} − V0(UNDECIDED) | {np.mean(dd) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}] | {perm_p(dd):.3f} |")
    for a, b in (("secf1", "nf"), ("scf", "nf"), ("secf1", "scf")):
        xa, xb = per_mem("V0", (a,)), per_mem("V0", (b,))
        dd = [xa[k] - xb[k] for k in xa if k in xb]
        lo, hi = boot_ci(dd)
        L.append(f"| V0: {a} − {b}(UNDECIDED) | {np.mean(dd) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}] | {perm_p(dd):.3f} |")
    L += ["\n## 4. UNDECIDED の誤答の行き先(形式別、表現まとめ)\n"]
    for v in VARIANTS:
        c = Counter(r["pred"] for r in rows if r["variant"] == v and r["status"] == "UNDECIDED" and not r["ok"])
        L.append(f"- {v}: " + ", ".join(f"{k}:{n}" for k, n in c.most_common()))
    ex7 = [r for r in rows if r["variant"] == "V2"]
    L += ["\n## 5. V2(7 択)の完全一致(状態そのものの正答率)\n", "| 状態 | 完全一致 |", "|---|---|"] + [f"| {s} | {np.mean([r['exact7'] for r in ex7 if r['status'] == s]):.2f} |" for s in sts]
    (OUT / "state_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    mode = sys.argv[1]
    try:
        if mode == "run":
            run(sys.argv[2], log=lambda s: print(s, flush=True))
        elif mode == "report":
            report()
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)
