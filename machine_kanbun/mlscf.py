"""Issue #39 stage 2: language-specific SCF on FLORES-200 (parallel news sentences) and cross-lingual convergence.

SCF per language = segment (Sudachi for ja, jieba for zh, regex words otherwise) -> delete function words (spaCy stop-word list of that
language, no models needed) except a small hand-written negation/modality list that is protected (Issue #30: negation must stay)
-> drop punctuation -> join (no space for ja/zh, space otherwise). Deterministic, no LLM.
Metrics (no generation): word deletion rate, token ratio (o200k / Qwen3), embedding retention cos(NF_L, SCF_L), and convergence =
cross-lingual retrieval Recall@1 (does SCF_L sentence i find SCF_en sentence i among N) for NF vs SCF, with qwen3-embedding-0.6b.
FLORES-200 (CC BY-SA 4.0, Meta) is downloaded to data_ext/ and not committed (scripts/fetch_flores.sh). Greenlandic is not in FLORES-200.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np

from .jpbench import RES

ROOT = RES.parent.parent
FL = ROOT / "data_ext/flores200_dataset/dev"
OUT = ROOT / "results/par"
CODES = {"en": "eng_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans", "ko": "kor_Hang", "tr": "tur_Latn", "vi": "vie_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab", "eu": "eus_Latn"}
NEG = {"en": "not no never n't nor without cannot", "tr": "değil yok hayır hiç asla", "vi": "không chưa chẳng đừng", "ru": "не нет ни никогда без", "ar": "لا لم لن ليس ما غير",
       "eu": "ez ezin inoiz gabe", "zh": "不 没 没有 无 未 别 非", "ja": "ない なかっ ぬ ず ません", "ko": "아니 않 못 없 안"}
EMB_MODEL = "text-embedding-qwen3-embedding-0.6b"
_W = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.U)
_sud = None


def stop(lang: str) -> set:
    mod = importlib.import_module(f"spacy.lang.{lang}.stop_words")
    s = {w.lower() for w in mod.STOP_WORDS}
    return s - {w for n in NEG[lang].split() for w in (n, n.lower())}


def segment(lang: str, s: str) -> List[str]:
    global _sud
    if lang == "ja":
        if _sud is None:
            from . import jlvar
            _sud = jlvar.sud()
        return [t.surface for t in _sud.analyze(s) if t.pos not in ("補助記号", "空白")]
    if lang == "zh":
        import jieba
        return [w for w in jieba.lcut(s) if _W.fullmatch(w) or re.fullmatch(r"[一-鿿]+", w)]
    return _W.findall(s)


def scf(lang: str, s: str, sw: set) -> str:
    keep = [w for w in segment(lang, s) if w.lower() not in sw]
    return ("" if lang in ("ja", "zh") else " ").join(keep)


def load(n: int) -> Dict[str, List[str]]:
    return {l: (FL / f"{c}.dev").read_text(encoding="utf-8").splitlines()[:n] for l, c in CODES.items()}


def embed(texts: List[str], cache: Dict[str, List[float]], batch: int = 32) -> np.ndarray:
    todo = [t for t in dict.fromkeys(texts) if hashlib.md5(t.encode()).hexdigest() not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        req = urllib.request.Request("http://localhost:1234/v1/embeddings", json.dumps({"model": EMB_MODEL, "input": [c or "。" for c in chunk]}).encode(), {"Content-Type": "application/json"})
        for t, d in zip(chunk, json.load(urllib.request.urlopen(req, timeout=300))["data"]):
            cache[hashlib.md5(t.encode()).hexdigest()] = d["embedding"]
    m = np.array([cache[hashlib.md5(t.encode()).hexdigest()] for t in texts], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-9)


def main(n: int = 200):
    from .mlpar import counters
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p = OUT / "emb_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    data = load(n)
    C = counters()
    nf, sc, rows = {}, {}, []
    for l, ss in data.items():
        sw = stop(l)
        nf[l] = ss
        sc[l] = [scf(l, s, sw) for s in ss]
        wd = 1 - sum(len(segment(l, x)) for x in sc[l]) / max(sum(len(segment(l, x)) for x in ss), 1)
        r = dict(lang=l, n=len(ss), chars_nf=np.mean([len(s) for s in ss]), chars_scf=np.mean([len(s) for s in sc[l]]), word_deletion=wd)
        for k in ("o200k_base", "Qwen3-8B"):
            r[f"tok_{k}"] = sum(C[k](x) for x in sc[l]) / sum(C[k](x) for x in ss)
        rows.append(r)
    E_nf = {l: embed(nf[l], cache) for l in data}
    E_sc = {l: embed(sc[l], cache) for l in data}
    cache_p.write_text(json.dumps(cache))
    ids = np.arange(n)

    def recall(A, B):  # Recall@1 of retrieving the parallel sentence of A in B
        return float(((A @ B.T).argmax(axis=1) == ids).mean())

    for r in rows:
        l = r["lang"]
        r["retention_cos"] = float((E_nf[l] * E_sc[l]).sum(1).mean())
        r["R1_nf_to_en"] = recall(E_nf[l], E_nf["en"])
        r["R1_scf_to_scf_en"] = recall(E_sc[l], E_sc["en"])
        r["R1_scf_to_nf_en"] = recall(E_sc[l], E_nf["en"])
        r["cos_nf_en"] = float((E_nf[l] * E_nf["en"]).sum(1).mean())
        r["cos_scf_scf_en"] = float((E_sc[l] * E_sc["en"]).sum(1).mean())
    langs = list(data)
    pair = lambda E: float(np.mean([recall(E[a], E[b]) for a in langs for b in langs if a != b]))
    pcos = lambda E: float(np.mean([(E[a] * E[b]).sum(1).mean() for a in langs for b in langs if a < b]))
    summ = dict(pair_R1_nf=pair(E_nf), pair_R1_scf=pair(E_sc), pair_cos_nf=pcos(E_nf), pair_cos_scf=pcos(E_sc))
    (OUT / "scf_flores.json").write_text(json.dumps(dict(rows=rows, summary=summ, n=n), ensure_ascii=False, indent=1, default=float))
    L = [f"# Issue #39 段階 2: FLORES-200 dev 先頭 {n} 文 × 9 言語の言語別 SCF と言語間の収束(LLM 生成なし)\n",
         "SCF = 分かち書き → spaCy ストップワードの削除(否定語は保護) → 句読点除去。埋め込みは qwen3-embedding-0.6b。\n",
         "| 言語 | 文字数 NF→SCF | 語の削除率 | トークン比 o200k | トークン比 Qwen3 | 保持 cos(NF,SCF) | R@1 NF→en | R@1 SCF→SCF(en) | R@1 SCF→NF(en) | cos NF–en | cos SCF–SCF(en) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['lang']} | {r['chars_nf']:.0f}→{r['chars_scf']:.0f} | {r['word_deletion']:.2f} | {r['tok_o200k_base']:.2f} | {r['tok_Qwen3-8B']:.2f} | {r['retention_cos']:.3f} | "
                 f"{r['R1_nf_to_en']:.2f} | {r['R1_scf_to_scf_en']:.2f} | {r['R1_scf_to_nf_en']:.2f} | {r['cos_nf_en']:.3f} | {r['cos_scf_scf_en']:.3f} |")
    L.append(f"\n全言語対の平均: R@1 NF {summ['pair_R1_nf']:.3f} → SCF {summ['pair_R1_scf']:.3f}、cos NF {summ['pair_cos_nf']:.3f} → SCF {summ['pair_cos_scf']:.3f}")
    (OUT / "scf_flores.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
