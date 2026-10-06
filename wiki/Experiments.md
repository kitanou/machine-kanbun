# Experiments

Issue 番号順ではなく目的別。詳細は各 docs([docs/](https://github.com/kitanou/machine-kanbun/blob/main/docs))、生データは [results/](https://github.com/kitanou/machine-kanbun/blob/main/results)。

| 目的 | Question | 結果 | 状態 | Issue |
|---|---|---|---|---|
| **Compression** | 7 形式(L0〜L5、JSON)の精度とトークンは | 文脈 -49%、L1 も -46%。漢文化の上乗せなし | 完了 | #1 |
| **Semantic Density** | なぜ長文脈で SCF が高精度に見えたか | トークン削減・文脈長で説明可能。#20 の優位は再現せず | 完了 | #25 |
| **Compression** | 日本語 NF と解析器ベースの SCF | 変換は無視できるコスト。検索は SCF が有利 | 完了 | #20 |
| **Ablation** | L1 から 1 要素ずつ変えると | 助詞削除だけ -9%。L1 は既に最小に近い | 完了 | #10 |
| **Ablation** | 圧縮の限界(最小十分表現) | 否定・不確実性・語順・主語が必須 | 完了 | #30 |
| **Tokenizer** | トークナイザ別の最適な書き方 | 最小スタイル 0.59〜0.66、精度差は検出できず | 完了 | #26 |
| **Tokenizer** | NF / SCF / SeCF のトークン圧縮特性の横断評価 | 一部は #20・#26 で測定済み。残りは未実施 | 保留 | #41 |
| **Transformer** | NF と SCF の内部での読まれ方 | 小型モデルが課題を解けず検出不能 | 保留 | #29 |
| **Transformer** | Transformer 横断評価、決定要因の分解 | 未実施 | 保留 | #42 #43 |
| **Multilingual** | 漢字ラベルは何に効くか | トークン安定性に効くが精度には効かない。意味の透明性が精度を決める | 完了 | #17 |
| **Multilingual** | EN / KO / JA を共通 MKW に | CV 1% は設計の帰結。MKW は L1 に勝たない | 完了 | #11 |
| **Multilingual** | 多言語パーサベース L1 | トークンは収束、QA は日本語以外で保てない | 完了 | #31 |
| **Multilingual** | 9 言語のコンパクトさと言語別 SCF の収束 | 文字数とトークンは乖離。粗い SCF は収束しない | 完了(一旦停止) | #39 |
| **Long Context** | 2k〜32k で | MKW は L1 と区別できない。TTFT はトークンに比例 | 完了 | #4 |
| **Retrieval** | NF/SCF の Dual-Index RAG | 強い埋め込みで最良(Recall@5 0.97〜0.99) | 完了 | #28 |
| **Chat** | SCF / SeCF-L1 履歴は回答の文体を汚染するか | 汚染は検出されず | 完了 | #33 |
| **Chat** | 圧縮履歴から古い事実を取り出せるか | SCF-L1 は -10〜-17pt、SeCF-L1 は NF 並み | 完了 | #49 |
| **Chat** | SCF の標識の凡例・自明語化 | 回復を検出できず | 完了 | #51 |
| **Chat** | 三層履歴 | SeCF-L1 並みの想起を同期変換 14〜86 ms で | 完了 | #54 |
| **Chat** | 同一 token budget で NF-short と比較 | SeCF-L1 が +14〜20pt(保持量で説明)。SCF-L1 は勝てない | 完了 | #57 |
| **Memory** | SeCF-L1 を Long-term Memory 表現にすると | 検索上の独自の利点なし。LLM 入力は NF 並み〜SCF-L1 より正確 | 完了 | #59 |
| **Memory** | 状態が変わる記憶の統合 | 利得なし。「迷い」に戻る遷移が難しい | 完了(停止) | #62 |
| **Question format** | 「迷い」を「いいえ」と読む誤りの原因(質問形式・表現・モデル) | 原因の大半は 3 択の指示の不足(+20pt)と SCF-L1 の断片。現行の 3 択 QA は未確定系を過小評価 | 完了 | #67 |
| **Cache** | SeCF-L1 長期記憶 × Prefix / KV キャッシュ | ヒットで TTFT 1.3〜4 秒(最大約 87 倍短縮)、NF と SeCF-L1 の差はほぼ消える。ミスでは SeCF-L1 が約 20% 短い。限界付近の保持は一貫しない | 完了(停止) | #69 |
| **Cost / Latency** | 文章長ごとの SeCF-L1 の圧縮率・TTFT・損益分岐 | 圧縮率は文体で決まる。TTFT 短縮は長さに比例(2000 トークンで 2〜2.5 秒)。同期変換はどの長さでも遅く損益分岐点なし | 完了 | #60 |
| **Compiler** | 機械漢文コンパイラ・L5・LLMLingua 系との比較 | 方針により終了 | 終了 | #2 #6 #8 #13 #14 |
| **Synthesis** | 三層分解と次期実験 | 圧縮の大半は層 1(文法的冗長性) | 完了 | #19 |

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
