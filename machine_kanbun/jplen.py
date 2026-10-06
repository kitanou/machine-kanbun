"""Issue #60: SeCF-L1 compression ratio, TTFT reduction and conversion-cost break-even as a function of text length (Japanese only).

Samples (NF length buckets in o200k tokens, per text type): chat (concatenated exchanges of the #33 histories), news/explanation (consecutive FLORES-200
Japanese sentences), bullets (FLORES sentences as a bullet list; already dense). For each sample: SeCF-L1 conversion (gemma, same rewrite style as #33 re-run, extended
to multi-sentence input; latency and output tokens recorded), SCF-L1 (sudachi-m, token count only), and TTFT of the NF and the SeCF-L1 text as a prompt (unique prefix to
avoid prompt-cache reuse). Quality is only a proxy here (retention of numbers, katakana terms and "…さん" names), not QA.
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import jl1
from .jpbench import RES
from .jpchat import OUT as JP3, SECF1_SYSTEM
from .lmstudio import Stalled, chat

ROOT = RES.parent.parent
OUT = RES.parent / "jp5"
FL = ROOT / "data_ext/flores200_dataset"
TYPES = ("chat", "news", "bullets")
EXTRA_RULE = "\n- 入力が複数の文・複数行でも、内容を落とさず同じ順序で最後まで書き直す(箇条書きは箇条書きのまま、行を保つ)。"
SYSTEM = SECF1_SYSTEM + EXTRA_RULE
SEED = 60
_enc = None


def o200k(text: str) -> int:
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("o200k_base")
    return len(_enc.encode(text))


_units_cache: Dict[str, List[List[tuple]]] = {}


def units(kind: str, short: bool = False, merged: bool = False) -> List[List[tuple]]:
    """Ordered (unit text, o200k tokens) grouped into documents (a unit is a sentence or one chat exchange); consecutive units are coherent."""
    key = kind + ("-short" if short else "") + ("-merged" if merged else "")
    if key in _units_cache:
        return _units_cache[key]
    if kind == "chat":
        docs = []
        for l in (JP3 / "histories_long.jsonl").read_text(encoding="utf-8").splitlines():
            h = json.loads(l)
            docs.append([f"ユーザー: {u}" for u, a in h["exchanges"]] if short else [f"ユーザー: {u}\nアシスタント: {a}" for u, a in h["exchanges"]])
    else:
        sents = []
        for f in ("dev/jpn_Jpan.dev", "devtest/jpn_Jpan.devtest"):
            sents += [x.strip() for x in (FL / f).read_text(encoding="utf-8").splitlines() if x.strip()]
        docs = [sents]
    if merged and kind == "chat":  # one history is only ~1,100 tokens: join consecutive histories so that 2,000-token windows exist
        docs = [docs[i] + docs[(i + 1) % len(docs)] for i in range(len(docs))]
    _units_cache[key] = [[(u, o200k(u)) for u in d] for d in docs]
    return _units_cache[key]


def make_sample(kind: str, target: int, rng: random.Random):
    docs = units(kind, short=(kind == "chat" and target <= 50), merged=(kind == "chat" and target > 900))
    hi = max(1.3 * target, target + 25)  # short targets: a single sentence / utterance is already longer than 1.3x
    for _ in range(3000):
        doc = rng.choice(docs)
        i = rng.randrange(len(doc))
        buf, n = [], 0
        for u, t in doc[i:]:
            buf.append(u)
            n += t + 1
            if n >= 0.8 * target:
                break
        if 0.8 * target <= n <= hi:
            text = "\n".join("- " + u for u in buf) if kind == "bullets" else ("\n".join(buf) if kind == "chat" else "".join(buf))
            return text, o200k(text)
    return None, 0


def build(lengths, n: int) -> List[Dict]:
    rng = random.Random(SEED)
    out = []
    for kind in TYPES:
        for T in lengths:
            seen, fails = set(), 0
            while len([o for o in out if o["kind"] == kind and o["target"] == T]) < n and fails < 50:
                text, tok = make_sample(kind, T, rng)
                if text is None or text in seen:
                    fails += 1
                    continue
                seen.add(text)
                out.append(dict(id=f"{kind}-{T}-{len(seen)}", kind=kind, target=T, text=text, nf_o200k=tok))
    return out


def samples(pilot: bool) -> List[Dict]:
    p = OUT / ("samples_pilot.json" if pilot else "samples.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    s = build((25, 100, 500) if pilot else (25, 100, 500, 2000), 6 if pilot else 20)
    p.write_text(json.dumps(s, ensure_ascii=False))
    return s


def add_extras() -> List[Dict]:
    """Chat 2,000-token bucket (needs merged histories). Appended to samples.json; the ids of existing samples are not changed."""
    p = OUT / "samples.json"
    cur = json.loads(p.read_text(encoding="utf-8"))
    if any(x["kind"] == "chat" and x["target"] == 2000 for x in cur):
        return cur
    rng = random.Random(SEED + 1)
    seen, fails, extra = set(), 0, []
    while len(extra) < 20 and fails < 100:
        text, tok = make_sample("chat", 2000, rng)
        if text is None or text in seen:
            fails += 1
            continue
        seen.add(text)
        extra.append(dict(id=f"chat-2000-{len(extra) + 1}", kind="chat", target=2000, text=text, nf_o200k=tok))
    cur += extra
    p.write_text(json.dumps(cur, ensure_ascii=False))
    return cur


def jl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def convert(model: str, pilot: bool, log=print) -> None:
    path = OUT / ("convert_pilot.jsonl" if pilot else "convert.jsonl")
    done = {r["id"] for r in jl(path)}
    scf = jl1.Converter("sudachi", "m")
    for s in samples(pilot):
        if s["id"] in done:
            continue
        t0 = time.perf_counter()
        scf_text = scf(s["text"].replace("\n", " "))
        scf_ms = (time.perf_counter() - t0) * 1000
        r = None
        for mt in (int(s["nf_o200k"] * 1.6) + 200, int(s["nf_o200k"] * 2.4) + 400):
            try:
                r = chat(model, SYSTEM, f"入力: {s['text']}\n出力:", max_tokens=mt, extra={"reasoning_effort": "none"}, deadline=900, timeout=920)
                break
            except ValueError:
                continue
        secf = re.sub(r"^出力[:：]\s*", "", r.text.strip()) if r else ""
        rec = dict(id=s["id"], kind=s["kind"], target=s["target"], nf_o200k=s["nf_o200k"], secf_text=secf, secf_o200k=o200k(secf) if secf else None, scf_text=scf_text, scf_o200k=o200k(scf_text),
                   secf_ms=r.total * 1000 if r else None, secf_out_tokens=r.completion_tokens if r else None, scf_ms=scf_ms, fallback=not secf)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"convert {s['id']} ratio={(rec['secf_o200k'] or 0) / s['nf_o200k']:.2f} ms={rec['secf_ms'] or 0:.0f}")


def ttft(model: str, pilot: bool, log=print) -> None:
    path = OUT / ("ttft_pilot.jsonl" if pilot else "ttft.jsonl")
    done = {r["id"] for r in jl(path)}
    S = {s["id"]: s for s in samples(pilot)}
    for c in jl(OUT / ("convert_pilot.jsonl" if pilot else "convert.jsonl")):
        if c["id"] in done or not c["secf_text"]:
            continue
        rec = dict(id=c["id"], kind=c["kind"], target=c["target"])
        order = ["nf", "secf"] if hash(c["id"]) % 2 == 0 else ["secf", "nf"]  # alternate which goes first to cancel drift
        for which in order:
            text = S[c["id"]]["text"] if which == "nf" else c["secf_text"]
            r = chat(model, "", f"[run:{uuid.uuid4().hex[:10]}]\n{text}\n\n上の文章を読みました。「了解」とだけ答えてください。", max_tokens=4, extra={"reasoning_effort": "none"}, deadline=300, timeout=320)
            rec[f"{which}_ttft"], rec[f"{which}_prompt_tokens"] = r.ttft, r.prompt_tokens
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"ttft {c['id']} nf={rec['nf_ttft']:.2f}s secf={rec['secf_ttft']:.2f}s")


def sim(pilot: bool, log=print) -> None:
    """Embedding cosine similarity of the NF text with its SeCF-L1 / SCF-L1 text (qwen3-embedding-0.6b via LM Studio)."""
    import subprocess
    from .jprag import embed
    emb = "text-embedding-qwen3-embedding-0.6b"
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    subprocess.run(["lms", "load", emb, "-y"], capture_output=True)
    S = {x["id"]: x for x in samples(pilot)}
    C = [c for c in jl(OUT / ("convert_pilot.jsonl" if pilot else "convert.jsonl")) if c["secf_text"]]
    path = OUT / ("sim_pilot.jsonl" if pilot else "sim.jsonl")
    done = {r["id"] for r in jl(path)}
    for c in C:
        if c["id"] in done:
            continue
        v = embed(emb, [S[c["id"]]["text"][:6000], c["secf_text"][:6000], c["scf_text"][:6000]])
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(id=c["id"], kind=c["kind"], target=c["target"], cos_secf=float(v[0] @ v[1]), cos_scf=float(v[0] @ v[2]))) + "\n")
    log("sim done")


_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_KATA = re.compile(r"[ァ-ヶー]{3,}")
_NAME = re.compile(r"[一-龥ぁ-んァ-ヶ]{2,6}さん")


def retention(orig: str, conv: str) -> Dict[str, float]:
    out = {}
    for k, rx in (("number", _NUM), ("katakana", _KATA), ("name", _NAME)):
        items = set(rx.findall(orig))
        if items:
            out[k] = sum(1 for x in items if x in conv or (k == "name" and x.replace("さん", "") in conv)) / len(items)
    return out


def report(pilot: bool) -> None:
    sfx = "_pilot" if pilot else ""
    C = {r["id"]: r for r in jl(OUT / f"convert{sfx}.jsonl")}
    T = {r["id"]: r for r in jl(OUT / f"ttft{sfx}.jsonl")}
    S = {s["id"]: s for s in samples(pilot)}
    kinds = ("chat", "news", "bullets")
    lengths = sorted({c["target"] for c in C.values()})
    L = ["# Issue #60 SeCF-L1 の文章長別の圧縮率・TTFT・変換コストの損益分岐点" + ("(パイロット)" if pilot else "") + "\n",
         "日本語のみ。NF の長さは o200k トークンの目標値で、実際の NF トークン数は表の通り。TTFT・プロンプトトークンは gemma-4-12b(LM Studio、温度 0、先頭に一意の識別子を入れてプロンプトキャッシュを避ける)。"
         "変換は SeCF-L1(gemma の書き直し)で、SCF-L1(sudachi-m)はトークン数のみ参考。意味保持は代理指標(数・カタカナ語・「〜さん」の保持率)で、QA ではない。\n",
         "## 1. 圧縮率(SeCF-L1 / NF、o200k)と変換時間\n", "| 文種 | 目標長 | n | NF tok | SeCF-L1 tok | SeCF/NF | SCF/NF | 変換時間 s(中央) | 変換速度 NF tok/s | 数の保持 | カタカナ語の保持 | 名前の保持 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for k in kinds:
        for t in lengths:
            rs = [c for c in C.values() if c["kind"] == k and c["target"] == t and c["secf_text"]]
            if not rs:
                continue
            ret = [retention(S[c["id"]]["text"], c["secf_text"]) for c in rs]
            rmean = lambda key: (f"{np.mean([r[key] for r in ret if key in r]):.2f}" if any(key in r for r in ret) else "-")
            nf = st.mean(c["nf_o200k"] for c in rs)
            table[(k, t)] = dict(nf=nf, ratio=st.mean(c["secf_o200k"] / c["nf_o200k"] for c in rs), ms=st.median(c["secf_ms"] for c in rs))
            L.append(f"| {k} | {t} | {len(rs)} | {nf:.0f} | {st.mean(c['secf_o200k'] for c in rs):.0f} | {table[(k, t)]['ratio']:.2f} | {st.mean(c['scf_o200k'] / c['nf_o200k'] for c in rs):.2f} | "
                     f"{table[(k, t)]['ms'] / 1000:.1f} | {nf / (table[(k, t)]['ms'] / 1000):.0f} | {rmean('number')} | {rmean('katakana')} | {rmean('name')} |")
    L += ["\n## 2. TTFT と損益分岐(gemma の実トークン数)\n", "| 文種 | 目標長 | n | NF プロンプト tok | SeCF プロンプト tok | tok 削減 | NF TTFT s | SeCF TTFT s | TTFT 短縮 s(%) | 同期の総差 s(変換+SeCF TTFT − NF TTFT) | 事前変換の総差 s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    series, bsr = {}, {}
    for k in kinds:
        for t in lengths:
            rs = [(T[i], C[i]) for i in T if C[i]["kind"] == k and C[i]["target"] == t]
            if not rs:
                continue
            nf_t, sf_t = [a["nf_ttft"] for a, _ in rs], [a["secf_ttft"] for a, _ in rs]
            red = [a["nf_ttft"] - a["secf_ttft"] for a, _ in rs]
            sync = [c["secf_ms"] / 1000 + a["secf_ttft"] - a["nf_ttft"] for a, c in rs]
            pre = [a["secf_ttft"] - a["nf_ttft"] for a, _ in rs]
            nfp, sfp = st.mean(a["nf_prompt_tokens"] for a, _ in rs), st.mean(a["secf_prompt_tokens"] for a, _ in rs)
            L.append(f"| {k} | {t} | {len(rs)} | {nfp:.0f} | {sfp:.0f} | {1 - sfp / nfp:+.1%} | {st.mean(nf_t):.2f} | {st.mean(sf_t):.2f} | {st.mean(red):+.2f} ({st.mean(red) / st.mean(nf_t):+.0%}) | {st.mean(sync):+.2f} | {st.mean(pre):+.2f} |")
            series.setdefault(f"{k} SeCF/NF", []).append((table[(k, t)]["nf"], table[(k, t)]["ratio"]))
            bsr.setdefault(k, []).append((table[(k, t)]["nf"], st.mean(sync)))
    SM = jl(OUT / f"sim{sfx}.jsonl")
    if SM:
        L += ["\n## 3. 意味の類似度(埋め込み cos(NF, 変換後)。qwen3-embedding-0.6b)\n", "| 文種 | 目標長 | n | SeCF-L1 | SCF-L1 |", "|---|---|---|---|---|"]
        for k in kinds:
            for t in lengths:
                rs = [x for x in SM if x["kind"] == k and x["target"] == t]
                if rs:
                    L.append(f"| {k} | {t} | {len(rs)} | {np.mean([x['cos_secf'] for x in rs]):.3f} | {np.mean([x['cos_scf'] for x in rs]):.3f} |")
    L += ["\n同期変換の総差が 0 を下回る(SeCF のほうが速い)文章長が損益分岐点。事前変換は変換コストを除いた差(負 = SeCF のほうが速い)。\n"]
    (OUT / f"len_report{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    try:
        from .jpreport import svg_lines
        if series:
            svg_lines(series, "NF 長 vs SeCF-L1/NF トークン比", "NF のトークン数(o200k)", "SeCF-L1 / NF", OUT / f"chart_len_ratio{sfx}.svg", logx=True)
        if bsr:
            svg_lines({f"{k} 同期の総差(s)": v for k, v in bsr.items()}, "NF 長 vs 同期変換の総レイテンシ差(正 = NF が速い)", "NF のトークン数(o200k)", "変換+SeCF TTFT − NF TTFT(秒)", OUT / f"chart_len_breakeven{sfx}.svg", logx=True)
    except Exception as e:  # charts are optional
        print("chart skipped:", e)
    print("\n".join(L))


if __name__ == "__main__":
    mode, pilot = sys.argv[1], "--pilot" in sys.argv
    model = next((a for a in sys.argv[2:] if not a.startswith("--")), "")
    try:
        if mode == "build":
            samples(pilot)
        elif mode == "extras":
            print(len(add_extras()), "samples")
        elif mode == "convert":
            convert(model, pilot, log=lambda s: print(s, flush=True))
        elif mode == "ttft":
            ttft(model, pilot, log=lambda s: print(s, flush=True))
        elif mode == "sim":
            sim(pilot, log=lambda s: print(s, flush=True))
        elif mode == "report":
            report(pilot)
    except Stalled as e:
        print(f"STALLED: {e}", flush=True)
        sys.exit(75)
