"""Issue #57 (Meta #56 Priority 1): same token budget, NF-short (oldest exchanges cut) vs SCF-L1 / SeCF-L1 (also cut oldest-first).

Histories: 24 exchanges, target fact + detail placed at varying positions (0,4,...,20). Question: 8-way status choice (7 statuses + "not in the
conversation"), deterministic scoring; wrong answers are counted as hallucination / misattribution, the 8th option as abstention.
"""
from __future__ import annotations

import json
import random
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jpchat, jpdata
from .jpchat import OUT, PERSONA, USER_TEMPLATES, DETAIL, STATUS_NOTE, secf1_text, l1_text
from .jpchatqa import option_text, o200k
from .lmstudio import Stalled, chat

N_EX = 24
POS = (0, 4, 8, 12, 16, 20)
BUDGETS = (0.85, 0.70, 0.55, 0.40)
REPS = ("nf", "scf", "secf1")
SEEDS = (0, 1)
HFILE = "histories_longpos.jsonl"
LET8 = "ABCDEFGH"
NONE_OPT = "この人物についての話は、これまでの会話に出てこない"


def build(n: int = 35, seed: int = 5701) -> List[Dict]:
    rng = random.Random(seed)
    pool = jpdata.make_memories(n * 6, seed=seed)
    by = defaultdict(list)
    for m in pool:
        by[m.status].append(m)
    per = n // 7
    targets = [by[s][i] for i in range(per) for s in jpdata.STATUSES]
    used = {m.id for m in targets}
    others = [m for m in pool if m.id not in used]
    rng.shuffle(others)
    ev = {e.key: e for e in jpdata.EVENTS}
    out = []
    for k, t in enumerate(targets):
        pos = POS[k % len(POS)]
        turns = [others[(3 * k + j) % len(others)].text if j % 2 == 0 else rng.choice(USER_TEMPLATES) for j in range(N_EX)]
        turns[pos] = t.text
        turns[pos + 3] = f"{t.person}さんのことなんだけど、" + rng.choice(DETAIL)
        e = ev[t.event]
        f = jpdata.conj(e)
        out.append(dict(sid=k, person=t.person, event=t.event, status=t.status, pos=pos, user_turns=turns,
                        gold_note=STATUS_NOTE[t.status].format(p=t.person, np=e.np, **f)))
    return out


def gen_histories(model: str, log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / HFILE
    have = {json.loads(l)["sid"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    for sc in build():
        if sc["sid"] in have:
            continue
        msgs = [{"role": "system", "content": PERSONA + "返信は1〜2文で、相づち・共感・短い一言を中心にしてください。"}]
        ex = []
        for u in sc["user_turns"]:
            msgs.append({"role": "user", "content": u})
            r = chat(model, "", "", messages=[dict(m) for m in msgs], max_tokens=120, extra=extra, deadline=240)
            a = r.text.strip().replace("\n", " ")
            msgs.append({"role": "assistant", "content": a})
            ex.append([u, a])
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(sc, exchanges=ex), ensure_ascii=False) + "\n")
        log(f"history {sc['sid']} done")


def load() -> List[Dict]:
    return [json.loads(l) for l in (OUT / HFILE).read_text(encoding="utf-8").splitlines()]


def prep(log=print) -> None:
    n = 0
    for h in load():
        for u, a in h["exchanges"]:
            secf1_text(u)
            secf1_text(a)
        n += 1
        if n % 5 == 0:
            log(f"secf1 converted {n}/35")


def conv(rep: str, text: str) -> str:
    return text if rep == "nf" else l1_text(text) if rep == "scf" else secf1_text(text)


def make_question(sc: Dict, seed: int):
    order = list(jpdata.STATUSES) + ["NONE"]
    random.Random(sc["sid"] * 10 + seed).shuffle(order)
    opts = "\n".join(f"{LET8[i]}. {NONE_OPT if s == 'NONE' else option_text(sc, s)}" for i, s in enumerate(order))
    q = f"{sc['person']}さんのことについて、これまでの会話から最も正確なものはどれ?\n{opts}\n記号(A〜H)1文字だけで答えてください。"
    return q, LET8[order.index(sc["status"])], order



def kept_exchanges(sc: Dict, rep: str, frac: float) -> List[int]:
    """Indices (oldest first) of the exchanges that fit the budget = frac * o200k tokens of the full NF history; oldest are cut first."""
    full = o200k("\n".join(u + "\n" + a for u, a in sc["exchanges"]))
    budget = frac * full
    sizes = [o200k(conv(rep, u) + "\n" + conv(rep, a)) for u, a in sc["exchanges"]]
    kept, total = [], 0
    for i in range(len(sizes) - 1, -1, -1):  # newest first
        if total + sizes[i] > budget:
            break
        kept.insert(0, i)
        total += sizes[i]
    return kept


def run(model: str, log=print) -> None:
    path = OUT / f"budget_{model.replace('/', '_')}.jsonl"
    done = {(r["cond"], r["sid"], r["seed"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    n = 0
    for sc in load():
        for seed in SEEDS:
            q, gold, order = make_question(sc, seed)
            conds = [("nf_full", "nf", 1.0)] + [(f"{rep}@{b}", rep, b) for b in BUDGETS for rep in REPS]
            for cname, rep, b in conds:
                if (cname, sc["sid"], seed) in done:
                    continue
                kept = list(range(N_EX)) if b == 1.0 else kept_exchanges(sc, rep, b)
                msgs = [{"role": "system", "content": f"[run:{uuid.uuid4().hex[:8]}] {PERSONA}"}]
                for i in kept:
                    u, a = sc["exchanges"][i]
                    msgs += [{"role": "user", "content": conv(rep, u)}, {"role": "assistant", "content": conv(rep, a)}]
                msgs.append({"role": "user", "content": q})
                r = chat(model, "", "", messages=msgs, max_tokens=6, extra=extra, deadline=300, timeout=320)
                m = re.search(r"[A-H]", r.text)
                pred = order[LET8.index(m.group(0))] if m else None
                tgt = [i for i, (u, a) in enumerate(sc["exchanges"]) if sc["person"] in u or sc["person"] in a]
                rec = dict(cond=cname, rep=rep, budget=b, sid=sc["sid"], seed=seed, status=sc["status"], pos=sc["pos"], gold=gold, pred_letter=m.group(0) if m else None,
                           pred_status=pred, outcome="correct" if pred == sc["status"] else "abstain" if pred == "NONE" else "wrong", prompt_tokens=r.prompt_tokens,
                           n_kept=len(kept), fact_kept=sorted(set(tgt) & set(kept)) == tgt, first_kept=kept[0] if kept else None)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 50 == 0:
                    log(f"budget qa {n}")


if __name__ == "__main__":
    mode, model = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        if mode == "hist":
            gen_histories(model, log=lambda s: print(s, flush=True))
        elif mode == "prep":
            prep(log=lambda s: print(s, flush=True))
        elif mode == "run":
            run(model, log=lambda s: print(s, flush=True))
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)
