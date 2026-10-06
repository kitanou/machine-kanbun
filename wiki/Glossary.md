# Glossary

## 表現
- **NF (Natural Form)**: 元の自然言語。人間向け。
- **SCF (Structural Compact Form)**: 既存の解析器と決定的規則で冗長性を削る表現。LLM 推論なし。**SCF-L1** は日本語の `sudachi-m`(助詞・丁寧語・フィラーを削除、助動詞を状態標識に写像、空白区切り)。
- **SeCF (Semantic Compact Form)**: 意味保持を優先して意味単位で書き直す表現。生成コストを許容。**SeCF-L1** は助詞・活用を残した簡潔な自然言語への書き直し、**SeCF-L5** は構造化 IR(研究終了)。
- **圧縮プロファイル (L0〜L5)**: 圧縮の強さ。表現(NF / SCF / SeCF)とは別の軸。`SCF-L1`、`SeCF-L5` のように組み合わせる。

## 評価
- **Semantic Preservation(意味保持)**: 圧縮後も QA や状態の想起ができること。
- **Compression Tolerance(圧縮耐性)**: モデルが圧縮表現をどこまで読めるか。
- **Tokens / Fact**: 1 つの事実を表すのに要するトークン数。
- **Fragmentation**: 1 語が複数のサブワードに割れること。
- **状態 7 種**: DONE(実行済み)、NOT_DONE(していない)、PLANNED(予定)、WANTED(願望)、UNDECIDED(迷い)、HEARSAY(伝聞)、POSSIBLE(可能性)。想起 QA の対象。
- **NF-short**: 同じ token budget に収まるまで古い往復から切り捨てた NF。

## 概念
- **Minimum Sufficient Representation(最小十分表現)**: 意味を保てる最小の表現。現時点の日本語では「内容語 + 状態標識 + 語順 + 主語」が必須(R-C3)。
- **Semantic Density(意味密度)**: 単位トークンあたりの意味情報の量。長文脈で SCF が高精度に見えた効果を、トークン削減・文脈長と分けて検証した(#25)。
- **Human–Machine Shared Language**: 人間にも LLM にも読み書きできる共有の言語表現という仮説。SCF が候補だが未検証(#39)。
- **Semantic IR / Semantic Unit**: 意味を構造化した中間表現とその単位。L5 方向で扱ったが研究は終了。
- **consolidation(統合)**: 同じ(人物, 出来事)の更新列を 1 つの経緯メモにまとめること(#62)。
- **Dual-Index**: 語彙検索(BM25 on SCF)と意味検索(ベクトル on NF)の融合(#28、#59)。

## 旧用語
- **L0 / L1 / L2〜L5**: 旧圧縮レベル。#20 以降の L1 は SCF-L1、#19 までの L1 は SeCF-L1、L5 は SeCF-L5。対応表は [NF / SCF / SeCF](NF-SCF-SeCF)。
- **MKW(機械漢文, Machine Kanbun)**: 漢字 + 漢文的省略 + 構造記号の擬似漢文 IR。SeCF-L5 に相当し、研究は終了。

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
