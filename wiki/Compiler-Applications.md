# Compiler / Applications

## 研究としては終了した方向
機械漢文コンパイラ(`mkwc`)、Semantic IR、SeCF-L5、LLMLingua 系との比較は、結果を保存したうえで**研究対象から外した**(Meta [#56](https://github.com/kitanou/machine-kanbun/issues/56)。#2、#6、#8、#13、#14)。理由: 圧縮の大半は文法的冗長性の除去で、構造化・漢字化に上乗せがなく(R-A3、R-A4)、L5 方向は SeCF-L1 と区別できなかった。

## 応用の構想(研究上の Finding とは分けて記述する)
Finding ではなく、現時点の結果から**設計として考えられる使い方**。未検証の部分を明記する。

| 用途 | 構想 | 根拠・状態 |
|---|---|---|
| Chat Context | 直近は NF、会話中の圧縮は SCF-L1(同期)、書き直し済みの古い履歴は SeCF-L1(非同期) | 三層履歴は SeCF-L1 並みの想起を出せた(R-D14)。実運用の二段構成は未検証 |
| Memory / RAG | 検索は BM25(SCF-L1)+ ベクトル(NF)の Dual、LLM への入力は NF か SeCF-L1 | R-C6、R-M1、R-M2。変換コストは約 40 回の検索で回収(R-M3) |
| 書き直しの優先順位 | 状態を述べた往復から先に SeCF-L1 化する | 未検証(U10)。SCF 層に重要な事実が残ると想起が落ちる(R-D15) |
| 文章長に応じた切り替え | 短文は NF、長文は SeCF-L1 | **同期変換では長さに依らず成り立たない**(R-L3)。長文を事前に SeCF-L1 化しておく運用なら、再利用回数が十分多い場合のみ得(R-L4) |
| モデル別の最適化 | トークナイザ別の書き方 | 効果は強く依存(R-C2)。精度への影響は検出できず |

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
