# Machine Kanbun — Transformer 向けの最小十分な言語表現の研究

> 自然言語の表層的な冗長性を削ったとき、事前学習済みの Transformer が意味を保てる**最小十分な言語表現**を探る研究。

**Core Question**: Transformer / LLM にとって必要な最小十分な言語表現とは何か。

当初は「漢字 + 漢文的省略 + 構造記号」の機械漢文(MKW)の評価だったが、研究が進み、**圧縮の効果の源泉・何を削ってよく何を残すべきか・実際の Chat 履歴と長期記憶に使えるか**という一般的な問いに移った。

## 表現の軸
```text
NF   Natural Form(人間向けの自然言語)
 │
SCF  Structural Compact Form(解析器 + 決定的規則で冗長性を削る。LLM 推論なし)
 │
SeCF Semantic Compact Form(意味を保ち、意味単位で書き直す。生成コストを許容)
```
圧縮の強さを表す L0〜L5 は別の軸(圧縮プロファイル)として残している。詳しくは [NF / SCF / SeCF](NF-SCF-SeCF)。

## 現在の研究方針(Meta [#56](https://github.com/kitanou/machine-kanbun/issues/56))
日本語の **Chat / Memory** を主軸に置き、次の役割分担を検証している。

| 役割 | 表現 |
|---|---|
| ユーザーとの入出力、直近の会話 | **NF** |
| Chat Context(会話中の高速・決定的な圧縮) | **SCF-L1** |
| Long-term Memory(非同期・意味中心の書き直し) | **SeCF-L1** |

SeCF-L5 / 機械漢文(L5)の研究は終了。多言語・Transformer 内部機構は、再開条件が整うまで保留。

## ページ一覧
- [5 分で分かる現在までの成果](Five-Minute-Summary)
- [NF / SCF / SeCF](NF-SCF-SeCF)
- [Research Map](Research-Map)
- [Confirmed Findings](Confirmed-Findings)
- [Experiments](Experiments)
- [Chat / Memory](Chat-Memory)
- [Long Context](Long-Context)
- [Tokenizer](Tokenizer)
- [Transformer / Models](Transformer-Models)
- [Languages](Languages)
- [Compiler / Applications](Compiler-Applications)
- [Glossary](Glossary)

## 原簿との関係
このWiki は読みやすさのための入口で、**事実と状態の原簿ではない**。確認済みの結果は [RESULTS.md](https://github.com/kitanou/machine-kanbun/blob/main/RESULTS.md)、研究の体系と未解決問題は [RESEARCH_MAP.md](https://github.com/kitanou/machine-kanbun/blob/main/RESEARCH_MAP.md)、実験の詳細は [docs/](https://github.com/kitanou/machine-kanbun/blob/main/docs)、生データは [results/](https://github.com/kitanou/machine-kanbun/blob/main/results)、作業履歴は [Issues](https://github.com/kitanou/machine-kanbun/issues)。Wiki の内容が原簿とずれたら、原簿を正とする。

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
