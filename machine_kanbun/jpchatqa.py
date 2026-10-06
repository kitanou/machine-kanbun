"""Issue #49: recall accuracy of old facts in a compressed chat history (multiple-choice status QA, deterministic scoring).

The last user turn asks which of 7 status statements about the target person is correct (letters A-G, option order permuted per
scenario/seed). Conditions: A (all NF), B (all SCF, message list), E (all SeCF, message list), N (NF with non-target exchanges dropped so
that its o200k size matches B), D / G (isolated SCF / SeCF block + instruction). Score = letter match; no LLM judge.
"""
from __future__ import annotations

import json
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jpchat, jpdata
from .jpchat import OUT, STATUS_NOTE, build_messages, load_histories, prep_secf, variant_files
from .lmstudio import Stalled, chat

SEEDS = (0, 1, 2)
LET = "ABCDEFG"
CONDS = {"A_base": dict(family="role", c=0), "B_scf": dict(family="role", c=-1), "E_secf": dict(family="role", c=-1, form="secf"),
         "N_nf_tokmatch": dict(family="role", c=0, trim=True), "D_scf_tag": dict(family="tag", c=-1, instr=True), "G_secf_tag": dict(family="tag", c=-1, instr=True, form="secf"),
         # Issue #51: markers made readable
         "B_legend": dict(family="role", c=-1, legend=True), "B_expl": dict(family="role", c=-1, form="scf_expl"), "B_expl_legend": dict(family="role", c=-1, form="scf_expl", legend=True)}
CONDS.update({"E1_secf1": dict(family="role", c=-1, form="secf1"), "G1_secf1_tag": dict(family="tag", c=-1, instr=True, form="secf1")})  # Issue #33 redo: SeCF-L1
CONDS.update({"T3_secf1_scf_nf": dict(family="tier", plan="T3"), "T2_secf1_nf": dict(family="tier", plan="T2"), "T3m_target_in_scf": dict(family="tier", plan="T3m")})  # Issue #54
BASE_CONDS = ("A_base", "B_scf", "E_secf", "N_nf_tokmatch", "D_scf_tag", "G_secf_tag")


def option_text(sc: Dict, status: str) -> str:
    e = jpdata.event_of(sc["event"])
    f = jpdata.conj(e)
    return STATUS_NOTE[status].format(p=sc["person"], np=e.np, **f).split("。")[0]


def make_question(sc: Dict, seed: int):
    order = list(jpdata.STATUSES)
    random.Random(sc["sid"] * 10 + seed).shuffle(order)
    opts = "\n".join(f"{LET[i]}. {option_text(sc, s)}" for i, s in enumerate(order))
    gold = LET[order.index(sc["status"])]
    q = f"{sc['person']}さんのことについて、これまでの会話から最も正確なものはどれ?\n{opts}\n記号(A〜G)1文字だけで答えてください。"
    return q, gold, order


def o200k(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def hist_tokens(exs: List[List[str]], conv) -> int:
    return o200k("\n".join(conv(u) + "\n" + conv(a) for u, a in exs))


def trim_to(sc: Dict, target: int) -> List[List[str]]:
    """Drop non-target exchanges (those not naming the target person), farthest first, to get closest to the target token count."""
    ex = [list(x) for x in sc["exchanges"]]
    keep = [i for i, (u, a) in enumerate(ex) if sc["person"] in u or sc["person"] in a]
    best, cur = list(range(len(ex))), list(range(len(ex)))
    err = lambda idx: abs(hist_tokens([ex[i] for i in idx], lambda x: x) - target)
    for i in [j for j in range(len(ex)) if j not in keep]:
        trial = [j for j in cur if j != i]
        if err(trial) < err(best):
            best = trial
        cur = trial
    return [ex[i] for i in best]


def tier_forms(sc: Dict, plan: str) -> List[str]:
    """Form per exchange (oldest first): T3 = 1/2 SeCF-L1, 1/4 SCF-L1, 1/4 NF; T2 = 3/4 SeCF-L1, 1/4 NF; T3m = T3 counts but the target-person exchanges are put in the SCF layer."""
    n = len(sc["exchanges"])
    q = n // 4
    if plan == "T2":
        return ["secf1"] * (n - q) + ["nf"] * q
    forms = ["secf1"] * (n - 2 * q) + ["scf"] * q + ["nf"] * q
    if plan == "T3m":
        tgt = [i for i, (u, a) in enumerate(sc["exchanges"]) if sc["person"] in u or sc["person"] in a]
        nf_idx = set(range(n - q, n))
        scf_idx = [i for i in tgt if i not in nf_idx]
        for i in range(n):  # fill the SCF layer with the oldest non-target exchanges up to q
            if len(scf_idx) >= q:
                break
            if i not in scf_idx and i not in nf_idx:
                scf_idx.append(i)
        scf_idx = scf_idx[:max(q, len(tgt))] if len(scf_idx) > q else scf_idx
        forms = ["nf" if i in nf_idx else "scf" if i in scf_idx else "secf1" for i in range(n)]
    return forms


def tier_messages(sc: Dict, plan: str, nonce: str, question: str):
    import time
    forms = tier_forms(sc, plan)
    msgs = [{"role": "system", "content": f"[run:{nonce}] {jpchat.PERSONA}"}]
    sync_ms = 0.0
    for (u, a), f in zip(sc["exchanges"], forms):
        if f == "scf":
            t0 = time.perf_counter()
            u, a = jpchat.l1_text(u), jpchat.l1_text(a)
            sync_ms += (time.perf_counter() - t0) * 1000
        elif f == "secf1":
            u, a = jpchat.secf1_text(u), jpchat.secf1_text(a)
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question})
    return msgs, sync_ms, forms


def spec_for(name: str, sc: Dict) -> Dict:
    s = dict(CONDS[name])
    if s.get("c") == -1:
        s["c"] = len(sc["exchanges"])
    return s


def run(model: str, variant: str, base_url="http://localhost:1234/v1", log=print, only=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"qa_{variant}_{model.replace('/', '_')}.jsonl"
    done = {(r["cond"], r["sid"], r["seed"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else set()
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    hv = "long" if variant == "long" else "short"
    n = 0
    for sc in load_histories(hv):
        for seed in SEEDS:
            q, gold, order = make_question(sc, seed)
            for cname in CONDS:
                if (only and cname not in only) or (cname, sc["sid"], seed) in done:
                    continue
                spec = spec_for(cname, sc)
                s2 = dict(sc, question=q)
                if spec.get("trim"):
                    target = hist_tokens(sc["exchanges"], jpchat.l1_text)
                    s2["exchanges"] = trim_to(sc, target)
                if spec["family"] == "tier":
                    msgs, conv_ms, forms = tier_messages(s2, spec["plan"], uuid.uuid4().hex[:8], q)
                else:
                    msgs, conv_ms = build_messages(s2, spec, uuid.uuid4().hex[:8])
                    forms = None
                r = chat(model, "", "", messages=msgs, max_tokens=6, base_url=base_url, extra=extra, deadline=300, timeout=320)
                m = re.search(r"[A-G]", r.text)
                rec = dict(cond=cname, sid=sc["sid"], seed=seed, status=sc["status"], gold=gold, pred=m.group(0) if m else None, pred_status=order[LET.index(m.group(0))] if m else None,
                           raw=r.text[:20], prompt_tokens=r.prompt_tokens, ttft=r.ttft, n_ex=len(s2["exchanges"]), conv_sync_ms=conv_ms, forms="".join(f[0] for f in forms) if forms else None)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 50 == 0:
                    log(f"{variant} qa {n}")


def prep(variant: str, log=print):
    prep_secf("long" if variant == "long" else "short", log=log)


if __name__ == "__main__":
    args = [a for a in sys.argv if a not in ("--51", "--secf1", "--tier")]
    mode, model, variant = args[1], args[2] if len(args) > 2 else "", args[-1]
    try:
        if mode == "prep":
            prep(variant, log=lambda s: print(s, flush=True))
        elif mode == "run":
            run(model, variant, log=lambda s: print(s, flush=True), only=("B_legend", "B_expl", "B_expl_legend") if "--51" in sys.argv else ("E1_secf1", "G1_secf1_tag") if "--secf1" in sys.argv else ("T3_secf1_scf_nf", "T2_secf1_nf", "T3m_target_in_scf") if "--tier" in sys.argv else None)
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)
