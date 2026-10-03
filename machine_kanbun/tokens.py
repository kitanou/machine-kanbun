"""Tokenizer comparison. tiktoken is always available; HF tokenizers are optional."""
from __future__ import annotations

from typing import Callable, Dict, List


def get_counters(hf_models: List[str] = ()) -> Dict[str, Callable[[str], int]]:
    counters: Dict[str, Callable[[str], int]] = {}
    import tiktoken

    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        counters[name] = lambda s, enc=enc: len(enc.encode(s))
    for m in hf_models:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(m)
        counters[m] = lambda s, tok=tok: len(tok.encode(s, add_special_tokens=False))
    return counters
