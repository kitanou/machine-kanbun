# NF / SCF / SeCF

表現を「圧縮の強さ」ではなく**生成の仕方と保持するもの**で名付ける(Issue [#35](https://github.com/kitanou/machine-kanbun/issues/35))。正式な定義と境界条件は [terminology.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/terminology.md)。

## NF — Natural Form
元の自然言語。文体・助詞・語順・ニュアンスを保持し、圧縮処理をしない。ユーザーとの入出力に使う。

## SCF — Structural Compact Form
既存の言語解析器(形態素解析・係り受け解析)と決定的な規則で、冗長性を削る表現。追加の LLM 推論を必要とせず、高速で、同じ入力に同じ出力になる。日本語では #20 の `sudachi-m`(内容語を保持、助動詞を状態標識に写像、助詞・丁寧語・フィラーを削除、空白区切り)が主な実装。用途は **Chat Context**。

## SeCF — Semantic Compact Form
意味保持を最優先に、意味単位で書き直して冗長性を除く表現。LLM などの生成コストを許容する。用途は **Long-term Memory**。
- **SeCF-L1**: #19 の L1 = 助詞・活用を残した**簡潔な自然言語**への書き直し(例: 「なんか、森田さんは手術を受けようか迷っているらしい。まだ決めたわけではないみたい。」→「森田さんは手術を受けるか迷っており、未決定。」)。
- **SeCF-L5**: 構造化された機械漢文 / Semantic IR(`{}`・field 名・演算子)。**研究は終了**。

## 旧用語との対応(過去の Issue を読むための対応表)
| 旧称 | 新称 | 備考 |
|---|---|---|
| L0 | NF | |
| 軽量 L1 / Parser L1 / 簡易 L1(#20 以降の `sudachi-m` など) | **SCF-L1** | 裸の「L1」は #20 以降では SCF |
| L1(#19 まで、構造化 Fact から作る簡潔な日本語要約) | **SeCF-L1** | 裸の「L1」は #19 までは SeCF |
| L2〜L4(漢字圧縮・擬似漢文・構造漢文) | SeCF-L2〜L4 | |
| L5 / 機械漢文 / MKW / IR | SeCF-L5 | 研究終了 |
| 文言文(C1) | NF の一種(別の自然言語 baseline) | |

表現(NF / SCF / SeCF)と圧縮プロファイル(L0 / L1 / L5)は別の軸で、`SCF-L1`、`SeCF-L1`、`SeCF-L5` のように組み合わせて書く。

## SCF と SeCF の境界(要約)
1. 生成に LLM 推論・学習済みの意味モデルが必要か → 必要なら SeCF(形態素解析器・係り受け解析器は含めない)。
2. 情報の編成を変えるか(言い換え、事実の統合、状態の正規化など) → 変えるなら SeCF。
3. 語の削除、品詞・活用に基づく写像、表記の書き換えだけ → SCF。

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
