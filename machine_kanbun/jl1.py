"""Simple L1 for Japanese chat text built on EXISTING analyzers + lightweight rules (Issue #20).

No LLM and no new parser: morphological analyzers (SudachiPy, MeCab via fugashi/unidic-lite, Janome) and the
GiNZA dependency parser are used as-is; their output is mapped to a telegraphic L1 by a few rules.

    L1 = content words (nouns, verb/adjective lemmas, selected adverbs) + the few functional morphemes that carry
         meaning (negation, past, hearsay/inference, wish, "かも") ; particles, polite/copula endings, fillers dropped.

Variants:  m  morphology only          p  keep case particles (role information)
           c  keep clause connectors   d  GiNZA dependency clauses + subject completion (zero-anaphora heuristic)
           naive  analyzer-free baseline: drop hiragana runs
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

FILLER_ADV = {"なんか", "ちょっと", "まあ", "えっと", "どう", "そう", "こう", "ああ", "やっぱり", "やはり", "ほんと", "本当に", "とても", "すごく", "ふと"}
NEG = {"ない", "ぬ", "ず", "ん", "なかっ", "なく"}
PAST = {"た", "だっ"}
HEARSAY = {"らしい"}
INFER = {"みたい", "よう", "ようだ", "ようです"}
WANT = {"たい", "たがる", "たがっ"}
DROP_FORMAL_NOUN = {"こと", "もの", "ん", "の", "わけ", "ところ"}
DROP_VERB_AUX = {"いる", "ある", "おく", "しまう", "くる", "いく", "みる", "あげる", "くれる", "もらう", "する", "為る", "居る", "有る"}
HONORIFIC = {"さん", "君", "くん", "ちゃん", "様", "さま"}
CASE = {"が", "を", "に", "で", "と", "から", "へ", "まで", "より", "は", "も"}
CONNECT = {"けど": "が", "けれど": "が", "けれども": "が", "が": "が", "から": "ため", "ので": "ため", "のに": "のに", "ば": "なら", "たら": "なら", "なら": "なら"}
PUNCT_END = {"。", "！", "？", "!", "?", "．"}


@dataclass
class Tok:
    surface: str
    lemma: str
    pos: str        # major POS in UniDic-style Japanese ("名詞", "動詞", "助詞", ...)
    sub: str = ""   # second-level POS
    cform: str = ""
    i: int = 0
    head: int = -1  # dependency head index (GiNZA only)
    dep: str = ""
    ctype: str = ""  # conjugation type (Sudachi only), e.g. "五段-カ行"


class Analyzer:
    name = "base"

    def analyze(self, text: str) -> List[Tok]:
        raise NotImplementedError


class Sudachi(Analyzer):
    name = "sudachi"

    def __init__(self, mode: str = "C"):
        from sudachipy import dictionary, tokenizer
        self._t = dictionary.Dictionary().create()
        self._m = getattr(tokenizer.Tokenizer.SplitMode, mode)

    def analyze(self, text):
        out = []
        for i, m in enumerate(self._t.tokenize(text, self._m)):
            p = m.part_of_speech()
            out.append(Tok(m.surface(), m.dictionary_form(), p[0], p[1], p[5] if len(p) > 5 else "", i, ctype=p[4] if len(p) > 4 else ""))
        return out


class MeCab(Analyzer):
    name = "mecab"

    def __init__(self):
        from fugashi import Tagger
        self._t = Tagger()

    def analyze(self, text):
        out = []
        for i, w in enumerate(self._t(text)):
            f = w.feature
            lemma = getattr(f, "orthBase", None) or w.surface
            out.append(Tok(w.surface, lemma if lemma != "*" else w.surface, f.pos1, f.pos2, getattr(f, "cForm", "") or "", i))
        return out


class Janome(Analyzer):
    name = "janome"

    def __init__(self):
        from janome.tokenizer import Tokenizer
        self._t = Tokenizer()

    def analyze(self, text):
        out = []
        for i, t in enumerate(self._t.tokenize(text)):
            p = t.part_of_speech.split(",")
            lemma = t.base_form if t.base_form != "*" else t.surface
            pos = {"接頭詞": "接頭辞", "フィラー": "感動詞", "記号": "補助記号"}.get(p[0], p[0])
            if p[0] == "名詞" and p[1] == "代名詞":
                pos = "代名詞"
            if p[0] == "名詞" and p[1] == "接尾":
                pos = "接尾辞"
            out.append(Tok(t.surface, lemma, pos, p[1], t.infl_form if hasattr(t, "infl_form") else "", i))
        return out


class Ginza(Analyzer):
    name = "ginza"

    def __init__(self):
        import spacy
        self._nlp = spacy.load("ja_ginza")

    def analyze(self, text):
        out = []
        for t in self._nlp(text):
            tag = t.tag_.split("-")
            out.append(Tok(t.text, t.lemma_, tag[0], tag[1] if len(tag) > 1 else "", "", t.i, t.head.i, t.dep_))
        return out


def make(name: str) -> Analyzer:
    return {"sudachi": Sudachi, "mecab": MeCab, "janome": Janome, "ginza": Ginza}[name]()


def _marker(t: Tok, nxt: List[Tok]) -> Optional[str]:
    """Semantic functional morpheme -> short canonical word, or None."""
    s, l = t.surface, t.lemma
    if (l in NEG or s in NEG) and t.pos in ("助動詞", "AUX", "形容詞", "ADJ") and s != "ん" or (s == "ん" and t.pos in ("助動詞", "AUX")):
        return "ない"
    if l in PAST and t.pos in ("助動詞", "AUX"):
        return "た"
    if l in HEARSAY or s in HEARSAY:
        return "らしい"
    if l in WANT or s in WANT:
        return "たい"
    if s == "かも" or (s == "か" and nxt and nxt[0].surface == "も" and len(nxt) > 1 and nxt[1].surface.startswith("しれ")):
        return "かも"
    if s.startswith("かもしれ"):
        return "かも"
    if (l in INFER and t.pos in ("助動詞", "形状詞")) or (s == "よう" and t.sub.startswith("非自立")):
        return "みたい"
    if s == "って" and nxt and nxt[0].lemma in ("聞く", "言う"):
        return None
    return None


def merge_kamo(toks: List[Tok]) -> List[Tok]:
    """か|も|しれ|(ませ|ん|ない|ます) -> one token "かも" (analyzers split "かもしれない" differently)."""
    out: List[Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.surface == "なん" and i + 1 < len(toks) and toks[i + 1].surface == "か" and (i == 0 or toks[i - 1].surface in ("、", "。")):
            out.append(Tok("なんか", "なんか", "副詞", "", "", t.i))  # sentence-initial filler (Sudachi splits it into なん|か)
            i += 2
            continue
        if t.surface.startswith("かもしれ") or (t.surface == "かも" and i + 1 < len(toks) and toks[i + 1].surface.startswith("しれ")) or (t.surface == "か" and i + 2 < len(toks) and toks[i + 1].surface == "も" and toks[i + 2].surface.startswith("しれ")):
            j = i + (1 if t.surface.startswith("かもしれ") else 2 if t.surface == "かも" else 3)
            while j < len(toks) and (toks[j].pos in ("助動詞", "AUX") or toks[j].surface in ("ない", "ん", "ませ", "ます", "です")):
                j += 1
            out.append(Tok("かも", "かも", "助詞", "かも", "", t.i, t.head, t.dep))
            i = j
            continue
        out.append(t)
        i += 1
    return out


def to_l1(toks: List[Tok], keep_case: bool = False, keep_connectors: bool = False, sep: str = " ", *, drop: frozenset = frozenset(),
          surface: bool = False, keep_filler: bool = False, keep_polite: bool = False, no_slash: bool = False, drop_person: bool = False) -> str:
    """Ablation switches (Issues #25/#30; defaults reproduce the Issue #20 variants):
    drop = marker words to delete (ない/た/らしい/たい/かも/みたい/う); surface = keep conjugated surface instead of lemma;
    keep_filler / keep_polite = keep fillers+interjections+conjunctions / polite+copula auxiliaries; no_slash = no sentence marker;
    drop_person = drop proper nouns and honorifics (control: the question's subject disappears)."""
    toks = merge_kamo(toks)
    out: List[str] = []
    last_noun = False
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        nxt = toks[i + 1:i + 4]
        s = t.surface
        if s in PUNCT_END:
            if out and out[-1] != "/" and not no_slash:
                out.append("/")
            i += 1
            continue
        if t.pos in ("補助記号", "空白", "記号", "PUNCT") or (t.pos in ("接続詞", "感動詞") and not keep_filler):
            i += 1
            continue
        if t.pos in ("接続詞", "感動詞"):
            out.append(s)
            i += 1
            continue
        mk = _marker(t, nxt)
        if t.surface == "かも":
            if "かも" not in drop:
                out.append("かも")
            last_noun = False
            i += 1
            continue
        if mk in drop or (t.surface in ("う", "よう") and t.pos in ("助動詞", "AUX") and "う" in drop and False):
            i += 1
            continue
        if t.surface == "かも":
            out.append("かも")
            last_noun = False
            i += 1
            continue
        if mk:
            if not (out and out[-1] == mk):
                out.append(mk)
            last_noun = False
            i += 1
            continue
        if t.pos in ("助詞", "ADP", "SCONJ", "PART"):
            if keep_case and s in CASE and out and last_noun:
                out[-1] += s
            elif keep_connectors and s in CONNECT:
                out.append(CONNECT[s])
            i += 1
            continue
        if t.pos in ("助動詞", "AUX"):  # polite, copula, volitional ... dropped
            if keep_polite:
                out.append(s)
            i += 1
            continue
        if t.pos == "接尾辞" and (s in HONORIFIC or t.lemma in HONORIFIC):
            i += 1
            continue
        if drop_person and t.pos == "名詞" and t.sub.startswith("固有名詞"):
            i += 1
            continue
        if t.pos in ("名詞", "代名詞", "NOUN", "PROPN", "PRON", "接頭辞"):
            if s in DROP_FORMAL_NOUN or t.lemma in DROP_FORMAL_NOUN:
                i += 1
                continue
            out.append(s)
            last_noun = True
            i += 1
            continue
        if t.pos in ("動詞", "VERB"):
            if t.sub.startswith("非自立") and t.lemma in DROP_VERB_AUX or t.lemma in ("する", "為る") and out:
                i += 1  # light verb after a nominal ("結婚した") or progressive "いる"
                continue
            out.append(s if surface else t.lemma)
            last_noun = False
            if "意志" in t.cform and "う" not in drop:
                out.append("う")
            i += 1
            continue
        if t.pos in ("形容詞", "形状詞", "ADJ"):
            if t.lemma in NEG:
                if "ない" not in drop:
                    out.append("ない")
            else:
                out.append(t.lemma if (t.pos != "形状詞" and not surface) else s)
            i += 1
            continue
        if t.pos in ("副詞", "ADV", "連体詞"):
            if s not in FILLER_ADV or keep_filler:
                out.append(s)
            i += 1
            continue
        i += 1
    while out and out[-1] == "/":
        out.pop()
    return sep.join(out) if sep == " " else sep.join(w for w in out)


def naive_l1(text: str) -> str:
    """Analyzer-free baseline: keep kanji / katakana / latin / digit runs, drop everything else."""
    return " ".join(re.findall(r"[一-鿿々〆ヶ]+|[ァ-ヴー]+|[A-Za-z0-9]+", text))


def clause_l1(toks: List[Tok]) -> str:
    """GiNZA dependency variant: one chunk per predicate clause (arguments before the predicate), the subject carried
    over from the previous clause / sentence when a clause has none (zero-anaphora heuristic)."""
    toks = merge_kamo(toks)
    byi = {t.i: t for t in toks}
    is_pred = lambda t: t.pos in ("動詞", "形容詞", "形状詞") and not t.sub.startswith("非自立") and t.dep in ("ROOT", "advcl", "conj", "ccomp", "acl", "parataxis", "csubj")
    preds = [t.i for t in toks if is_pred(t)]
    sentences: List[List[Tok]] = [[]]
    for t in toks:
        sentences[-1].append(t)
        if t.surface in PUNCT_END:
            sentences.append([])
    parts: List[str] = []
    last_person: Optional[str] = None
    for sent in sentences:
        if not sent:
            continue
        idx = {t.i for t in sent}
        heads = [p for p in preds if p in idx]
        groups = {h: [] for h in heads} if heads else {sent[0].i: []}
        cur = next(iter(groups))
        for t in sent:
            j, seen = t.i, 0
            while j in byi and j not in groups and byi[j].head != j and byi[j].head in idx and seen < 50:
                j, seen = byi[j].head, seen + 1
            cur = j if j in groups else cur
            groups[cur].append(t)
        for h in sorted(groups):
            ts = groups[h]
            text = to_l1(ts)
            if not text:
                continue
            names = [t.surface for t in ts if t.pos == "名詞" and t.sub.startswith("固有名詞")]
            if names:
                last_person = " ".join(names)
            elif last_person and any(t.pos in ("動詞", "形容詞") for t in ts):
                text = f"{last_person} {text}"
            parts.append(text)
    return " / ".join(parts)


class Converter:
    """analyzer + variant -> L1 text."""

    def __init__(self, analyzer: str, variant: str = "m"):
        self.analyzer_name, self.variant = analyzer, variant
        self.an = None if analyzer == "naive" else make(analyzer)

    @property
    def name(self) -> str:
        return f"{self.analyzer_name}-{self.variant}" if self.an else "naive"

    def __call__(self, text: str) -> str:
        if self.an is None:
            return naive_l1(text)
        toks = self.an.analyze(text)
        v = self.variant
        if v == "d":
            return clause_l1(toks)
        # "g": glued (no spaces) -- fewer tokens with Japanese tokenizers, less readable
        return to_l1(toks, keep_case=(v == "p"), keep_connectors=(v == "c"), sep="" if v == "g" else " ")
