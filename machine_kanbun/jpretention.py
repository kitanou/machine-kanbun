"""Issue #20 experiment 4: what does the simple L1 keep of the important information?

Rule-based (no LLM): for every memory we check that the L1 still contains the person, the event (object + verb),
the time phrase / numbers, and the status marker (negation, planning, wish, undecided, hearsay, possibility); and
we classify the status from the L1 alone with the same kind of rules to see whether it can be *recovered* --
in particular whether an undecided / hearsay / possible item is mistaken for DONE ("退職済み" misread).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from . import jl1, jpdata
from .jpbench import CONFIGS, RES

TIME_WORDS = re.compile(r"先月|去年|3年前|5月|この前|先週の火曜日|1か月前|来月|来年|再来週|春ごろ|年内")


def norm(s: str) -> str:
    return re.sub(r"[\s/]", "", s)


def status_from_l0(t: str) -> str:
    """Rule classifier for the ORIGINAL text (calibration: should be ~100% on this benchmark)."""
    if "かもしれ" in t:
        return "POSSIBLE"
    if "って聞いた" in t or "と聞きました" in t:
        return "HEARSAY"
    if "迷って" in t:
        return "UNDECIDED"
    if "つもり" in t or "予定" in t:
        return "PLANNED"
    if "たい" in t or "たがって" in t:
        return "WANTED"
    if re.search(r"いない|ていません", t):
        return "NOT_DONE"
    return "DONE"


def status_from_l1(t: str) -> str:
    n = norm(t)
    if "かも" in n:
        return "POSSIBLE"
    if "聞く" in n and "ない" in n:
        return "HEARSAY"
    if "迷う" in n or ("まだ" in n and "決める" in n):
        return "UNDECIDED"
    if "つもり" in n or "予定" in n:
        return "PLANNED"
    if "たい" in n:
        return "WANTED"
    if "ない" in n:
        return "NOT_DONE"
    if "た" in n:
        return "DONE"
    return "UNKNOWN"


def event_present(l1: str, ev: jpdata.Event) -> bool:
    n = norm(l1)
    obj = re.sub(r"[をにがへ]$", "", ev.np.replace("新しい", ""))
    verb_stem = ev.noun or ev.verb[:-1]
    kan = re.findall(r"[一-鿿]+", verb_stem)
    needle = verb_stem if (ev.noun or verb_stem in n) else (kan[0] if kan else verb_stem)
    return (obj == "" or obj in n or (len(obj) > 1 and obj[:2] in n)) and (needle in n or (kan and kan[0][0] in n))


def run(n: int = 2000, seed: int = 7) -> Dict:
    mems = jpdata.make_memories(n, seed=seed)
    ev = {e.key: e for e in jpdata.EVENTS}
    out: Dict[str, Dict] = {}
    cal = Counter(status_from_l0(m.text) == m.status for m in mems)
    out["_calibration_l0_status_accuracy"] = cal[True] / n
    for an, var in CONFIGS:
        conv = jl1.Converter(an, var)
        stats = defaultdict(Counter)
        confusion: Dict[str, Counter] = defaultdict(Counter)
        for m in mems:
            l1 = conv(m.text)
            nl = norm(l1)
            stats["person"][m.person in nl] += 1
            stats["event"][event_present(l1, ev[m.event])] += 1
            times = set(TIME_WORDS.findall(m.text))
            if times:
                stats["time"][all(w in nl or w[:2] in nl for w in times)] += 1
            pred = status_from_l1(l1)
            confusion[m.status][pred] += 1
            stats["status"][pred == m.status] += 1
            if m.status in ("UNDECIDED", "HEARSAY", "POSSIBLE", "PLANNED", "WANTED"):
                stats["not_done_as_done"][pred == "DONE"] += 1
            stats["nums"][set(re.findall(r"\d+", m.text)) <= set(re.findall(r"\d+", l1))] += 1
        out[conv.name] = {k: v[True] / max(1, sum(v.values())) for k, v in stats.items()}
        out[conv.name]["by_status"] = {s: confusion[s][s] / max(1, sum(confusion[s].values())) for s in jpdata.STATUSES}
        out[conv.name]["confusion"] = {s: dict(confusion[s]) for s in jpdata.STATUSES}
        o = out[conv.name]
        print(f"{conv.name:12} person {o['person']:.0%} event {o['event']:.0%} time {o['time']:.0%} status-recoverable {o['status']:.0%} "
              f"misread-as-DONE {o['not_done_as_done']:.0%}", flush=True)
    return out


def main(n: int = 2000):
    RES.mkdir(parents=True, exist_ok=True)
    res = run(n)
    (RES / "retention.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
