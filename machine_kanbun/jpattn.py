"""Issue #29: how does a Transformer read L0 vs simple-L1 memories? Attention / hidden-state / logit analysis.

Model: Qwen3-1.7B (MLX 4-bit, local HF cache) -- the only small model with open weights available offline (LM Studio models are GGUF).
For every prompt the last prompt token's (the position that predicts the answer) attention over all context tokens is recomputed
per layer and head, and the next-token logits of the four answer words give the correct-answer logit margin.
Conditions: L0, sudachi-m, sudachi-g, and L0-token-matched (more memories of L0 so that its token count equals sudachi-m's),
with the target memory inserted at 5 controlled positions among the same distractors for every condition.
Span annotation is approximate: person = the name string, event = object + verb/noun of the event, state = tokens Sudachi maps
to meaning markers or the status cue words (迷う/聞く/予定/つもり/決める).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3
from tokenizers import Tokenizer

from . import jl1, jlvar, jpdata
from .jpllm import INSTR, SYSTEM

OUT = Path(__file__).parent.parent / "results" / "jp2"
REPO = "mlx-community/Qwen3-1.7B-4bit"
CUE_WORDS = {"迷う", "聞く", "予定", "つもり", "決める", "かも", "らしい", "みたい", "たい", "ない", "た", "う"}
ANSWERS = ["はい", "いいえ", "未確定", "不明"]
REC: Dict = {"on": False}


def _patch():
    if getattr(qwen3.Attention, "_patched", False):
        return
    orig_attn, orig_blk = qwen3.Attention.__call__, qwen3.TransformerBlock.__call__

    def attn(self, x, mask=None, cache=None):
        out = orig_attn(self, x, mask, cache)
        if REC["on"]:
            B, L, _ = x.shape
            q = self.q_norm(self.q_proj(x[:, -1:, :]).reshape(B, 1, self.n_heads, -1)).transpose(0, 2, 1, 3)
            k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            q, k = self.rope(q, offset=L - 1), self.rope(k)
            k = mx.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
            w = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * self.scale, axis=-1)[0, :, 0, :]
            REC["attn"].append(np.array(w.astype(mx.float32)))
        return out

    def blk(self, x, mask=None, cache=None):
        out = orig_blk(self, x, mask, cache)
        if REC["on"]:
            idx = REC.get("idx") or []
            last = np.array(out[0, -1, :].astype(mx.float32))
            span = np.array(out[0, mx.array(idx), :].astype(mx.float32).mean(axis=0)) if idx else last * 0
            REC["hid_last"].append(last)
            REC["hid_span"].append(span)
        return out

    qwen3.Attention.__call__, qwen3.TransformerBlock.__call__ = attn, blk
    qwen3.Attention._patched = True


class Probe:
    def __init__(self):
        _patch()
        self.model, _ = load(REPO)
        import glob
        snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots/*/tokenizer.json"))[0]
        self.tok = Tokenizer.from_file(snap)
        self.first = [self.tok.encode(a, add_special_tokens=False).ids[0] for a in ANSWERS]

    def run(self, prompt: str, spans: Dict[str, List[Tuple[int, int]]]) -> Dict:
        """prompt = full chat-formatted text; spans = char ranges per category."""
        enc = self.tok.encode(prompt, add_special_tokens=False)
        ids, offs = enc.ids, enc.offsets
        T = len(ids)
        tokidx = {k: [i for i, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in v)] for k, v in spans.items()}
        REC.update(on=True, attn=[], hid_last=[], hid_span=[], idx=tokidx.get("target") or [])
        h = self.model.model(mx.array([ids]))
        logits = self.model.model.embed_tokens.as_linear(h[:, -1:, :])[0, 0].astype(mx.float32)
        mx.eval(logits)
        REC["on"] = False
        lg = np.array(logits)
        cand = lg[self.first]
        attn = np.stack(REC["attn"])  # L,H,T
        res = dict(T=T, logits=cand.tolist(), spans={k: len(v) for k, v in tokidx.items()})
        w = attn.mean(axis=1)  # L,T head-averaged
        for k, idx in tokidx.items():
            if idx:
                res[f"mass_{k}"] = w[:, idx].sum(axis=1).tolist()
                res[f"maxhead_{k}"] = attn[:, :, idx].sum(axis=2).max(axis=1).tolist()  # strongest single head per layer
        res["entropy"] = (-(attn * np.log(attn + 1e-12)).sum(axis=2).mean(axis=1)).tolist()
        res["hid_last"] = np.stack(REC["hid_last"]).astype(np.float32)
        res["hid_span"] = np.stack(REC["hid_span"]).astype(np.float32)
        return res


def chat_prompt(ctx: str, q: str) -> Tuple[str, int, int]:
    pre = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n記憶メモ:\n"
    post = f"\n\n質問: {q}\n{INSTR}\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return pre + ctx + post, len(pre), len(pre) + len(ctx)


def state_spans(text: str) -> List[Tuple[int, int]]:
    out, pos = [], 0
    toks = jl1.merge_kamo(jlvar.sud().analyze(text))
    for i, t in enumerate(toks):
        j = text.find(t.surface, pos)
        if j < 0:
            continue
        pos = j + len(t.surface)
        if jl1._marker(t, toks[i + 1:i + 4]) or t.surface == "かも" or t.lemma in CUE_WORDS:
            out.append((j, j + len(t.surface)))
    return out


def find_all(text: str, needle: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)] if needle else []


def event_needles(ev: jpdata.Event) -> List[str]:
    obj = re.sub(r"[をにがへ]$", "", ev.np.replace("新しい", ""))
    verb = ev.noun or (re.findall(r"[一-鿿]+", ev.verb) or [ev.verb[:-1]])[0]
    return [x for x in (obj, verb) if x]


POS = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_context(texts: List[str], target_idx_in_list: int):
    ctx, starts, pos = [], [], 0
    for t in texts:
        starts.append(pos)
        ctx.append(t)
        pos += len(t) + 1
    return "\n".join(ctx), starts


def run(n: int = 100, per_pos: int = 8, seed: int = 29, out: str = "attn") -> None:
    rng = random.Random(seed)
    mems = jpdata.make_memories(500, seed=seed)
    ev = {e.key: e for e in jpdata.EVENTS}
    reps = {"L0": jlvar.VARIANTS["L0"], "sudachi-m": jlvar.VARIANTS["sudachi-m"], "sudachi-g": jlvar.VARIANTS["sudachi-g"]}
    cache = {k: [f(m.text) for m in mems] for k, f in reps.items()}
    pr = Probe()
    count = lambda s: len(pr.tok.encode(s, add_special_tokens=False).ids)
    tok_per = {k: sum(count(t) for t in v[:200]) / 200 for k, v in cache.items()}
    n_tm_m = int(round(n * tok_per["sudachi-m"] / tok_per["L0"]))
    n_tm_g = int(round(n * tok_per["sudachi-g"] / tok_per["L0"]))
    conds = {"L0": ("L0", n), "sudachi-m": ("sudachi-m", n), "sudachi-g": ("sudachi-g", n), f"L0-tm(m)@{n_tm_m}": ("L0", n_tm_m), f"L0-tm(g)@{n_tm_g}": ("L0", n_tm_g)}
    prompts = []
    targets = rng.sample(range(len(mems)), per_pos * len(POS))
    for k, t in enumerate(targets):
        p = POS[k % len(POS)]
        dis = [i for i in rng.sample(range(len(mems)), n_tm_m + 40) if i != t]
        prompts.append((t, p, dis))
    results = []
    hid = {}
    for cname, (rep, nm) in conds.items():
        for (t, p, dis) in prompts:
            ids = dis[:nm - 1]
            slot = int(round(p * (len(ids))))
            order = ids[:slot] + [t] + ids[slot:]
            texts = [cache[rep][i] for i in order]
            ctx, starts = build_context(texts, slot)
            m = mems[t]
            e = ev[m.event]
            q = f"{m.person}さんは{e.np}{jpdata.conj(e)['ta']}？"
            prompt, a, b = chat_prompt(ctx, q)
            ts = a + starts[slot]
            te = ts + len(texts[slot])
            tgt = texts[slot]
            spans = {"target": [(ts, te)], "question": [(b, len(prompt))], "prefix": [(0, a)],
                     "person": [(ts + s, ts + e_) for s, e_ in find_all(tgt, m.person)],
                     "event": [(ts + s, ts + e_) for nd in event_needles(e) for s, e_ in find_all(tgt, nd)],
                     "state": [(ts + s, ts + e_) for s, e_ in state_spans(tgt)]}
            r = pr.run(prompt, spans)
            gold = ANSWERS.index(jpdata.ANSWER[m.status])
            lg = np.array(r["logits"])
            margin = float(lg[gold] - np.max(np.delete(lg, gold)))
            rec = dict(cond=cname, rep=rep, n_mem=nm, target=t, pos=p, status=m.status, T=r["T"], correct=bool(lg.argmax() == gold), margin=margin,
                       span_tokens=r["spans"], entropy=r["entropy"], **{k: v for k, v in r.items() if k.startswith(("mass_", "maxhead_"))})
            results.append(rec)
            hid[f"{cname}|{t}|{p}|last"] = r["hid_last"].astype(np.float16)
            hid[f"{cname}|{t}|{p}|span"] = r["hid_span"].astype(np.float16)
        done = [x for x in results if x["cond"] == cname]
        print(f"{cname}: acc={np.mean([x['correct'] for x in done]):.2f} margin={np.mean([x['margin'] for x in done]):.2f} T={np.mean([x['T'] for x in done]):.0f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{out}.json").write_text(json.dumps(dict(model=REPO, n=n, per_pos=per_pos, tokens_per_memory=tok_per, conds=list(conds), results=results), ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / f"{out}_hidden.npz", **hid)


if __name__ == "__main__":
    kw = {}
    if len(sys.argv) > 1:
        kw["n"] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kw["per_pos"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kw["out"] = sys.argv[3]
    run(**kw)
