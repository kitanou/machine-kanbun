"""Issue #31 experiments: tokens per fact (L0 vs parser-based L1) across languages and tokenizers, plus LLM QA / state retention."""
from __future__ import annotations

import json
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jpml, tokcross
from .jpbench import RES
from .lmstudio import Stalled, chat
from .jpllm import SYSTEM

OUT = RES.parent / "jp2"
INSTR = {
    "ja": "上の記憶メモだけを根拠に、質問の出来事が確定しているかを答えてください。既に完了していると確定している場合は「はい」、していない・予定や希望の段階・否定されている場合は「いいえ」、迷っている・聞いた話で未確認・可能性にとどまる場合は「未確定」と、最初の語だけで答えてください。該当する記憶メモがない場合は「不明」と答えてください。",
    "en": "Based only on the memo notes above, answer whether the event in the question is confirmed. If it is confirmed to have happened, answer \"Yes\". If it has not happened, is only planned or wished for, or is denied, answer \"No\". If the person is undecided, it is only unconfirmed hearsay, or only a possibility, answer \"Undetermined\". If there is no relevant memo, answer \"Unknown\". Answer with the first word only.",
    "ko": "위의 메모만을 근거로 질문의 일이 확정되었는지 답하세요. 이미 일어났다고 확정된 경우는 \"예\", 아직 하지 않았거나 예정·희망 단계이거나 부정된 경우는 \"아니요\", 고민 중이거나 들은 이야기라 미확인이거나 가능성에 그치는 경우는 \"미확정\"이라고 첫 단어만으로 답하세요. 해당하는 메모가 없으면 \"모름\"이라고 답하세요.",
    "zh": "只根据上面的备忘记录，回答问题中的事情是否已确定。如果确定已经发生，回答“是”；如果没有发生、只是计划或愿望、或被否定，回答“否”；如果还在犹豫、只是未经证实的传闻、或只是可能性，回答“未确定”。只回答第一个词。如果没有相关记录，回答“不知道”。",
}
HEAD = {"ja": ("記憶メモ", "質問"), "en": ("Memo notes", "Question"), "ko": ("메모", "질문"), "zh": ("备忘记录", "问题")}


def score(lang: str, answer: str, gold_idx: int) -> bool:
    a = answer.strip().strip("「」\"'“”『』").lower()
    word = jpml.ANS[lang][gold_idx].lower()
    return a.startswith(word) and not (lang == "ko" and gold_idx == 1 and a.startswith("예")) and not (lang == "en" and gold_idx == 2 and False)


def token_stats(n: int = 600, boots: int = 2000, seed: int = 1) -> Dict:
    facts = jpml.make_facts(n)
    cnt = tokcross.counters()
    texts = {lg: [jpml.render(f, lg) for f in facts] for lg in jpml.LANGS}
    t0 = {}
    l1 = {}
    parse_ms = {}
    for lg in jpml.LANGS:
        t = time.perf_counter()
        l1[lg] = [jpml.L1[lg](x) for x in texts[lg]]
        parse_ms[lg] = (time.perf_counter() - t) / n * 1000
    out = {"n_facts": n, "parse_ms_per_fact": parse_ms, "tokenizers": {}}
    rng = np.random.default_rng(seed)
    for name, c in cnt.items():
        a0 = np.array([[c(x) for x in texts[lg]] for lg in jpml.LANGS], dtype=float)  # lang x fact
        a1 = np.array([[c(x) for x in l1[lg]] for lg in jpml.LANGS], dtype=float)
        cv = lambda a: float(a.mean(axis=1).std() / a.mean(axis=1).mean())
        var_fact = lambda a: (a.std(axis=0) / a.mean(axis=0))  # per-fact CV across the 4 languages
        d_boot = []
        for _ in range(boots):
            idx = rng.integers(0, n, n)
            d_boot.append(cv(a1[:, idx]) - cv(a0[:, idx]))
        pf0, pf1 = var_fact(a0), var_fact(a1)
        out["tokenizers"][name] = dict(
            tok_per_fact_L0={lg: float(a0[i].mean()) for i, lg in enumerate(jpml.LANGS)}, tok_per_fact_L1={lg: float(a1[i].mean()) for i, lg in enumerate(jpml.LANGS)},
            ratio={lg: float(a1[i].mean() / a0[i].mean()) for i, lg in enumerate(jpml.LANGS)}, cv_L0=cv(a0), cv_L1=cv(a1), cv_diff_ci=[float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5))],
            p_cv_L1_lt_L0=float(np.mean(np.array(d_boot) < 0)), var_between_langs_L0=float(a0.mean(axis=1).var()), var_between_langs_L1=float(a1.mean(axis=1).var()),
            per_fact_cv_L0=float(pf0.mean()), per_fact_cv_L1=float(pf1.mean()), per_fact_L1_lower=float((pf1 < pf0).mean()))
    # state retention (rule-based, native cues)
    ret = {}
    for lg in jpml.LANGS:
        ok = fd = 0
        for f, x in zip(facts, l1[lg]):
            p = jpml.status_from_l1(x, lg)
            ok += p == f.status
            fd += p == "DONE" and f.status != "DONE"
        ret[lg] = dict(status_recovered=ok / n, false_done=fd, chars_L0=float(np.mean([len(x) for x in texts[lg]])), chars_L1=float(np.mean([len(x) for x in l1[lg]])))
    out["retention"] = ret
    # sensitivity: is the convergence just an artefact of the marker design? (a) the same ASCII markers in all four languages, (b) no markers at all
    ja_map = {"ない": "NOT", "た": "PST", "らしい": "HEAR", "みたい": "HEAR", "たい": "WANT", "かも": "MAY"}
    ascii_all = {lg: ([" ".join(ja_map.get(w, w) for w in x.replace(" / ", " ").split() if w != "う") for x in l1[lg]] if lg == "ja" else l1[lg]) for lg in jpml.LANGS}
    markers = {"NOT", "PST", "HEAR", "WANT", "MAY", "ない", "た", "らしい", "みたい", "たい", "かも", "う"}
    content = {lg: [" ".join(w for w in x.split() if w not in markers) for x in l1[lg]] for lg in jpml.LANGS}
    out["sensitivity"] = {}
    for vname, docs in (("L1_ascii_markers_all_languages", ascii_all), ("L1_content_words_only", content)):
        out["sensitivity"][vname] = {}
        for name, c in cnt.items():
            a0 = np.array([[c(x) for x in texts[lg]] for lg in jpml.LANGS], dtype=float)
            a1 = np.array([[c(x) for x in docs[lg]] for lg in jpml.LANGS], dtype=float)
            cv = lambda a: float(a.mean(axis=1).std() / a.mean(axis=1).mean())
            db = []
            for _ in range(boots):
                idx = rng.integers(0, n, n)
                db.append(cv(a1[:, idx]) - cv(a0[:, idx]))
            out["sensitivity"][vname][name] = dict(cv_L0=cv(a0), cv_L1=cv(a1), p_cv_L1_lt_L0=float(np.mean(np.array(db) < 0)), tok_L1={lg: float(a1[i].mean()) for i, lg in enumerate(jpml.LANGS)})
    out["examples"] = {lg: [(texts[lg][i], l1[lg][i]) for i in range(7)] for lg in jpml.LANGS}
    (OUT / "ml_tokens.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def run_qa(model: str, n: int = 300, nq: int = 60, base_url: str = "http://localhost:1234/v1", log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ml_qa_{model.replace('/', '_')}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["lang"], r["rep"]))
    facts = jpml.make_facts(600)[:n]
    qsel = random.Random(5).sample(range(n), nq)
    extra = {"reasoning_effort": "none"} if "gemma" in model else None
    nt = "\n/no_think" if ("qwen3" in model and "qwen3." not in model) else ""
    for lg in jpml.LANGS:
        texts = [jpml.render(f, lg) for f in facts]
        l1 = [jpml.L1[lg](x) for x in texts]
        l1n = [jpml.l1n(x, lg) for x in texts]
        for rep, docs in (("L0", texts), ("L1", l1), ("L1n", l1n)):
            if rep == "L1n" and lg == "ja":
                continue
            if (lg, rep) in done:
                continue
            ctx = "\n".join(docs)
            h, qh = HEAD[lg]
            system = f"[run:{uuid.uuid4().hex[:8]}] {SYSTEM if lg == 'ja' else 'You answer questions using only the memo notes.'}"
            rows = []
            try:
                base = chat(model, f"[base:{uuid.uuid4().hex[:8]}] {SYSTEM if lg == 'ja' else 'You answer questions using only the memo notes.'}", f"{h}:\n\n\n{qh}: x\n{INSTR[lg]}{nt}", max_tokens=12, base_url=base_url, extra=extra, deadline=240)
                for i, k in enumerate(qsel):
                    f = facts[k]
                    r = chat(model, system, f"{h}:\n{ctx}\n\n{qh}: {jpml.question(f, lg)}\n{INSTR[lg]}{nt}", max_tokens=8, base_url=base_url, extra=extra,
                             deadline=1800 if i == 0 else 240, timeout=1800 if i == 0 else 300)
                    rows.append(dict(model=model, lang=lg, rep=rep, i=i, fact=f.id, status=f.status, answer=r.text, ok=score(lg, r.text, jpml.GOLD[f.status]), prompt_tokens=r.prompt_tokens,
                                     ctx_tokens=(r.prompt_tokens - base.prompt_tokens) if i == 0 else None, ttft=r.ttft, total=r.total))
            except Stalled:
                raise
            except Exception as e:
                rows = [dict(lang=lg, rep=rep, error=f"{type(e).__name__}: {e}"[:300])]
            with path.open("a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if "error" not in rows[0]:
                log(f"{model} {lg} {rep}: acc={sum(r['ok'] for r in rows) / len(rows):.1%} ctx={rows[0]['ctx_tokens']}tok cold_ttft={rows[0]['ttft']:.1f}s")


if __name__ == "__main__":
    if sys.argv[1] == "tokens":
        o = token_stats()
        for k, v in o["tokenizers"].items():
            print(k, {lg: round(x, 1) for lg, x in v["tok_per_fact_L0"].items()}, "->", {lg: round(x, 1) for lg, x in v["tok_per_fact_L1"].items()},
                  f"CV {v['cv_L0']:.3f}->{v['cv_L1']:.3f}  P(L1<L0)={v['p_cv_L1_lt_L0']:.2f}")
        print(o["retention"])
    else:
        try:
            run_qa(sys.argv[2], log=lambda s: print(s, flush=True))
        except Stalled as e:
            print(f"STALLED: {e}", flush=True)
            sys.exit(75)
