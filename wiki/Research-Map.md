# Research Map

一次資料は [RESEARCH_MAP.md](https://github.com/kitanou/machine-kanbun/blob/main/RESEARCH_MAP.md)(未解決問題 U1〜U13 と依存関係を含む)。ここは地図の要約。

```text
Core Question: Transformer / LLM にとって必要な最小十分な言語表現とは
│
├─ Representation(圧縮の源泉)
├─ Minimum Sufficient(何を削ってよいか)
├─ Tokenizer
├─ Transformer / Models
├─ Languages
├─ Long Context
├─ Chat / Memory(現在の主軸)
└─ Conversion Cost / Latency
```

| 領域 | 現在の答え | 不確実性 | 主な Issue |
|---|---|---|---|
| Representation | 圧縮の大半は文法的冗長性の除去。漢字化・構造化に上乗せなし | 低 | #1 #4 #10 #19 |
| Minimum Sufficient | 助詞・丁寧語・フィラーは削除可。否定・不確実性・語順・主語は必須。願望・完了の区別も助詞・活用に依存 | 中 | #30 #49 #51 |
| Long Context | 効果の大半はトークン削減・文脈長で説明可能。意味密度の優位は示せず | 高(検出力不足) | #25 |
| Tokenizer | 効果は強く依存し、逆転もある。トークナイザ別に最適化できる | 低 | #26 |
| Retrieval | 検索は SCF-L1(語彙)、LLM 入力は NF の Dual。SeCF-L1 に検索上の利点なし | 中 | #28 #59 |
| Chat | 文体は汚染されない。SeCF-L1 は想起が NF 並み、SCF-L1 は低下 | 中 | #33 #49 #51 #54 #57 |
| Memory | 更新・統合の利得は示せず。難しいのは「迷い」に戻る遷移 | 中 | #59 #62 |
| Languages | トークン数は設計上収束するが QA は日本語以外で保てない。粗い SCF は収束しない | 高 | #11 #31 #39 |
| Transformer 内部 | 未確認(動かせる小型モデルが課題を解けず) | 高 | #29 |
| 変換コスト・遅延 | SCF は無視できる。SeCF-L1 は同期変換では 25〜2000 トークンのどの長さでも遅く(変換は節約の約 22 倍)、事前変換でも 24〜56 回の再利用で回収 | 低〜中 | #20 #33 #60 |

## 未解決問題(要約)
- **進行中・候補**: #20・#59・#62 の 3 択 QA を改善した指示で再測定(U15)、重要度に基づく書き直しの優先順位(U10)、表現効果を検出できる難しい課題(U11)、SCF-L1 が状態を取り違える原因(U9)、SeCF-L1 の変換を同期で成立させる条件(U14。変換速度が約 22 倍必要)。
- **保留(再開条件あり)**: 多言語 SCF(U1、U2。形態素解析つき SCF、母語話者評価、信頼できる並列コーパスが必要)、Transformer 内部機構(U8)、人間の可読性(U7、被験者が必要)、モデル依存性(U6)、実会話での再現(U4)。
- **解決済み**: 長文脈での「事実に言及されやすい」(U3、取り下げ)、二段構成(U5、成立)、状態が変わる記憶の統合(U12、利得なし)。

方針の全体は Meta [#56](https://github.com/kitanou/machine-kanbun/issues/56)。各領域の詳細は [Chat / Memory](Chat-Memory)・[Long Context](Long-Context)・[Tokenizer](Tokenizer)・[Transformer / Models](Transformer-Models)・[Languages](Languages)。

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
