"""Issue #19 (F): token counts of the same documents under many tokenizers.

Separates representation effects from tokenizer effects: for each tokenizer we report, per
language, the size of L0 / L1 / IR-L / IR-C / IR-ID (and CV across languages), and whether the
Korean advantage of the kanji-label IR (IR-C shorter than IR-L) reproduces.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path
from typing import Callable, Dict, List

from . import irlabel, mlenc
from .gen import generate
from .model import Profile

RES = Path(__file__).parent.parent / "results"
HF = {"qwen3": "Qwen/Qwen3-8B", "gemma2": "unsloth/gemma-2-9b-it", "llama3.1": "NousResearch/Meta-Llama-3.1-8B-Instruct",
      "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407", "deepseek-v3": "deepseek-ai/DeepSeek-V3"}
LANGS = ("JA", "EN", "KO")


def counters() -> Dict[str, Callable[[str], int]]:
    import tiktoken
    from tokenizers import Tokenizer

    out: Dict[str, Callable[[str], int]] = {}
    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        out[name] = lambda s, enc=enc: len(enc.encode(s))
    for k, repo in HF.items():
        try:
            tok = Tokenizer.from_pretrained(repo)
            out[k] = lambda s, tok=tok: len(tok.encode(s, add_special_tokens=False).ids)
        except Exception as e:  # gated / unavailable
            print(f"skip {k}: {str(e)[:80]}")
    return out


def render(ents: List[Profile], lang: str, rep: str) -> str:
    if rep in ("L0", "L1"):
        return mlenc.encode_ml(ents, lang, rep)
    return irlabel.render(ents, lang, rep)


def main(extra_langs=(), reps=("L0", "L1", "IRL", "IRC", "IRID"), lengths=(2000, 8000)):
    langs = LANGS + tuple(extra_langs)
    cnt = counters()
    base = cnt["o200k_base"]
    L: List[str] = []
    P = L.append
    P("# Issue #19 (F) トークナイザ横断\n")
    P("同じ文書(seed 1)を各トークナイザで数える。CV は言語間(" + "/".join(langs) + ")。"
      "「IR-C vs IR-L」は共通漢字ラベルが各言語ラベルより短いか(負 = 漢字が有利)。\n")
    for length in lengths:
        ents, _ = generate(length, base, seed=1, n_questions=8)
        docs = {(lg, r): render(ents, lg, r) for lg in langs for r in reps}
        P(f"\n## 文脈長 {length}\n")
        P("| トークナイザ | " + " | ".join(f"CV {r}" for r in reps) + " | L1 vs L0 (JA/EN/KO) | IR-C vs L1 (JA/EN/KO) | IR-C vs IR-L (EN/KO) |")
        P("|---|" + "---|" * (len(reps) + 3))
        for name, c in cnt.items():
            t = {k: c(v) for k, v in docs.items()}
            cvs = [st.pstdev([t[(lg, r)] for lg in langs]) / st.mean([t[(lg, r)] for lg in langs]) for r in reps]
            l1 = "/".join(f"{t[(lg, 'L1')] / t[(lg, 'L0')] - 1:+.0%}" for lg in LANGS)
            c1 = "/".join(f"{t[(lg, 'IRC')] / t[(lg, 'L1')] - 1:+.0%}" for lg in LANGS)
            cl = "/".join(f"{t[(lg, 'IRC')] / t[(lg, 'IRL')] - 1:+.0%}" for lg in ("EN", "KO"))
            P(f"| {name} | " + " | ".join(f"{x:.1%}" for x in cvs) + f" | {l1} | {c1} | {cl} |")
    text = "\n".join(L)
    (RES / "synthesis" / "tokenizer_cross.md").write_text(text + "\n", encoding="utf-8")
    print(text)
