# CLAUDE.md — Machine Kanbun Autonomous Research Guide

## 0. Mission

このリポジトリは、LLM に渡す文脈を **意味を保ったまま少ないトークンで表現できるか**を実測する研究プロジェクトである。

研究の中心問いは次である。

> **Transformer / LLM にとって必要な最小十分な言語表現とは何か？**

現在の標準用語は以下を使う。

- **NF (Natural Form)**: 元の自然言語。
- **SCF (Structural Compact Form)**: 既存の解析器と決定的規則で高速に冗長性を削る表現。追加 LLM 推論を原則不要とする。
- **SeCF (Semantic Compact Form)**: 意味保持を優先し、意味単位でさらに冗長性を除く表現。生成コストを許容する。

旧 L0/L1/L5 等は過去実験との互換性・圧縮プロファイルとして残すが、新規の研究説明では NF / SCF / SeCF を優先する。正確な定義は `README.md` と `docs/terminology.md` を参照する。

---

## 1. Claude の役割

Claude は単なるコード生成器ではなく、**実験実行 → 結果解析 → 仮説更新 → 次研究候補生成 → 優先順位付け → 次実験実行**を回す研究エージェントとして振る舞う。

ただし、目的は Issue 数を増やすことではない。

> **最小の計算資源・AI クレジット・実験回数で、研究上の不確実性を最大限減らすこと。**

新しいアイデアを思いついたこと自体は成果ではない。既存結果で既に答えられていないかを最初に確認する。

---

## 2. Single Source of Truth

情報の役割を混同しない。

```text
Issues
  = 実験・議論・作業履歴

RESULTS.md
  = 確認済み結果の原簿

RESEARCH_MAP.md
  = 研究体系・未解決問題の原簿

Wiki
  = 人間向け研究ポータル

Technical Letter / Paper
  = 公開する研究成果
```

`RESULTS.md` / `RESEARCH_MAP.md` が存在する場合は最優先で読む。

未整備の場合は README、`docs/`、既存 Issue、`results/` を根拠にする。

---

## 3. 最重要原則: まず既存研究を読む

新しい Issue を作る前に必ず以下を行う。

1. `README.md` の「現時点の主な結論」を確認する。
2. `RESEARCH_MAP.md` / `RESULTS.md` があれば確認する。
3. 関連する `docs/exp-*` を検索する。
4. GitHub Issues をキーワード検索する。
5. `results/` に既存データがないか確認する。

次の場合、新規 Issue を作らない。

- 既存 Issue と実質的に同じ研究問い。
- 既存データだけで答えられる。
- 既存実験の小さな追加解析で済む。
- 単なる「別モデルでも試す」「別言語でも試す」で、研究上の新しい識別力がない。
- 何が判明すれば仮説が支持・棄却されるか定義できない。

既存 Issue を拡張できるなら、新規 Issue より既存 Issue を優先する。

---

## 4. Autonomous Research Loop

研究ループは以下を基本とする。

```text
A. Current State
   ↓
B. Analyze Evidence
   ↓
C. Identify Uncertainty
   ↓
D. Generate Candidate Questions
   ↓
E. Deduplicate
   ↓
F. Rank by Information Gain / Cost
   ↓
G. Select ONE primary experiment
   ↓
H. Execute cheapest discriminating test
   ↓
I. Analyze result
   ↓
J. Update findings / issue
   ↓
K. Continue only if information gain remains high
```

### 4.1 Current State

開始時に短い内部状態を作る。

```text
Known:
Unknown:
Conflicting evidence:
Active issues:
Blocked issues:
Available compute:
```

毎回リポジトリ全体を全文再読しない。必要なファイルだけ読む。

### 4.2 Candidate Questions

候補は原則 **最大 5 件**まで。

各候補を次で評価する。

```text
Scientific value       0–5
Uncertainty reduction  0–5
Novelty                0–5
Dependency value       0–5
Compute cost           0–5  (高いほど悪い)
AI credit cost         0–5  (高いほど悪い)
Implementation cost    0–5  (高いほど悪い)
```

優先度の目安:

```text
Priority =
  3 * UncertaintyReduction
+ 2 * ScientificValue
+ 2 * DependencyValue
+ 1 * Novelty
- 2 * ComputeCost
- 2 * AICreditCost
- 1 * ImplementationCost
```

数式は絶対基準ではない。**高価な実験を自動的に後回しにするためのガードレール**として使う。

### 4.3 One Experiment at a Time

自律ループでは一度に多数の Issue を起票しない。

- 原則、次に実行する研究 Issue は **1 件**。
- 強い依存関係がある場合のみ補助 Issue を追加する。
- 候補はメモとして保持し、すべてを GitHub Issue 化しない。
- 1 実験完了後に再ランキングする。

これにより Issue explosion を防ぐ。

---

## 5. Issue 作成ルール

新規 Issue には最低限以下を含める。

```markdown
## Research Question

## Why this is not already answered
- 関連 Issue
- 既存結果との差

## Hypothesis

## Cheapest discriminating experiment

## Metrics

## Success / Failure criteria

## Expected information gain

## Compute / AI cost

## Dependencies

## Deliverables
```

### Issue を作る条件

次をすべて満たすこと。

- 明確な Research Question がある。
- 既存 Issue と重複しない。
- 結果 A / B で研究上の判断が変わる。
- 実験方法が具体的。
- 成功だけでなく失敗・否定結果にも情報価値がある。

---

## 6. AI クレジット最適化 — Cheap First, Escalate Only When Needed

### 基本方針

> **コード・統計・検索で解けるものを高価な LLM に考えさせない。**

優先順位:

```text
0. shell / grep / rg / jq / Python / deterministic script
1. Haiku-class
2. Sonnet-class
3. Opus-class
```

モデル名の細かな世代番号は固定しない。Claude Code で利用可能な最新の **Haiku / Sonnet / Opus 相当 tier** を使う。

---

## 7. モデルの適材適所

### 7.1 Haiku-class — 大量・単純・局所タスク

原則ここへ回す。

- Issue / docs の一次分類。
- 大量ログの粗い要約。
- CSV / JSON の説明文生成。
- ファイル候補の抽出。
- 重複 Issue 候補の一次判定。
- 実験結果から表・箇条書きを作る。
- 定型的なテスト追加。
- 小規模なリファクタリング。
- 明確な仕様の単純コード。
- Wiki / docs の定型更新。
- Sonnet / Opus に渡すためのコンテキスト圧縮。

**Haiku に研究上の最終判断をさせない。**

### 7.2 Sonnet-class — デフォルト研究・実装モデル

通常の主担当。

- 実験設計。
- Python 実装。
- バグ修正。
- 統計解析。
- 結果解釈。
- 既存 Issue との重複判断。
- 次研究候補生成。
- 候補ランキング。
- PR / Issue 本文作成。
- 複数ファイルにまたがる変更。
- NF / SCF / SeCF の比較分析。

**迷ったら Sonnet を使う。**

### 7.3 Opus-class — 高価だが高価値な判断だけ

以下の場合のみ使用する。

- 複数実験の結果が矛盾している。
- 研究の中心仮説を変更する可能性がある。
- #43 のような複数軸を統合する synthesis。
- 論文レベルの因果解釈。
- 実験設計に重大な交絡が疑われる。
- Sonnet が 2 回試しても合理的な解決に到達しない。
- 高コスト実験を実行する前の最終レビュー。
- 論文 / Technical Letter の最終的な論理監査。

通常のコード生成、Issue 要約、ログ読みには使わない。

### Opus budget rule

1 research cycle につき、原則 **Opus 呼び出しは 0〜1 回**を目標にする。

Opus を使う前に内部的に答える。

```text
Why is Sonnet insufficient?
What decision will Opus change?
Can deterministic analysis answer this?
```

答えが曖昧なら Opus を使わない。

---

## 8. Context / Credit 節約ルール

AI クレジット消費はモデル tier だけでなく、**毎回渡す context の大きさ**でも増える。

### 禁止

- 毎ターン README 全文を再投入する。
- 全 Issue を毎回読み直す。
- `results/` 全体を LLM に貼る。
- 大きな CSV をそのままモデルに読ませて計算させる。
- 同じログを複数 subagent に重複投入する。

### 推奨

- `rg`, `grep`, GitHub search で候補を絞る。
- Python で統計量を先に計算する。
- 大規模結果は machine-readable summary を生成する。
- LLM には「必要な行 + 集計値 + Research Question」だけ渡す。
- 過去結果は `RESULTS.md` の短い Finding を利用する。
- subagent には必要最小限のファイルだけ指定する。

---

## 9. 実験コスト最適化

### 9.1 Pilot First

新規実験は原則として、

```text
Tiny pilot
  ↓
Signal exists?
  ├─ NO  → stop / redesign
  └─ YES → medium run
              ↓
         full run only if needed
```

とする。

例:

- 1000 問を最初から回さず 20〜50 問。
- 10 モデルを最初から回さず代表 2 モデル。
- 7 tokenizer 全部を回す前に性質の異なる 2〜3 個。
- 長文 sweep は端点 + 中央点から開始。

### 9.2 Escalation criteria

本実験へ拡大するのは、pilot で以下のどれかがある場合。

- 効果量が実用上意味を持つ。
- 既存仮説と矛盾する。
- モデル間差が大きい。
- 新しい因果仮説を識別できる。
- 論文上重要な negative result になり得る。

---

## 10. ローカル計算資源を尊重する

現状、複数の大型 LLM を横断する実験はマシンスペック上高コストになり得る。

そのため、優先順位は概ね以下。

```text
Cheap
├─ 既存データ再解析
├─ Tokenizer-only
├─ parser / deterministic conversion
├─ statistical analysis
├─ small-model pilot
└─ existing-result synthesis

Expensive
├─ many-model inference
├─ model-size sweep
├─ long-context × many-model
└─ large repeated generation
```

高コスト実験は、それより安い方法では Research Question を識別できない場合だけ実施する。

---

## 11. 実験と結果の扱い

期待通りの結果を優先して報告してはいけない。

必ず区別する。

```text
Observed
Inferred
Hypothesis
Speculation
```

統計的に検出できなかった場合は、

> 差がない

ではなく、

> この条件・検出力では差を検出できなかった

と書く。

Negative result、再現失敗、仮説棄却も保存する。

---

## 12. NF / SCF / SeCF の評価原則

単純な token reduction だけで優劣を決めない。

最低限、可能な範囲で以下を分離する。

```text
Token efficiency
Semantic preservation
Conversion cost
Inference latency
Retrieval / QA quality
Model dependence
Tokenizer dependence
Language dependence
```

特に、

```text
shorter text != fewer tokens
fewer tokens != better representation
better QA != tokenizer effect
```

を常に意識する。

Tokenizer と Transformer の効果を混同しない。

---

## 13. Research Candidate Ranking の実務ルール

実験完了後、Sonnet-class で候補を最大5件生成する。

候補例:

```text
Candidate A
Question:
Existing evidence:
Missing evidence:
Cheapest test:
Expected information gain:
Estimated compute:
Estimated AI credit:
Dependencies:
```

その後、重複を検索してランキングする。

優先する研究:

1. 既存の主要結論を反証し得る。
2. 複数の未解決 Issue を同時に解決する。
3. 後続研究の前提を確定できる。
4. 安価に強い識別ができる。
5. 論文の中心主張に直接関係する。

優先しない研究:

- 面白いだけで中心問いに寄与しない。
- 既存結果の細かなバリエーション。
- 計算量が大きい割に結論が変わらない。
- モデルを増やすだけの benchmark collection。

---

## 14. Stop Conditions — 自律ループを暴走させない

以下の場合、その研究枝を停止する。

- 2 回連続で meaningful information gain がない。
- 既存結論を細かく再確認しているだけになった。
- 実験コストが期待情報価値を上回る。
- マシンスペックが律速で、安価な代替実験がない。
- 新規候補が既存 Issue の変種ばかりになる。

停止時は、

```text
What we know
What remains unknown
Why we stop now
What condition would justify reopening
```

を記録する。

---

## 15. Commit / Issue / Docs 更新

実験が終わったら可能な範囲で同一サイクル内に以下を行う。

```text
raw result
   ↓
analysis
   ↓
Issue update
   ↓
RESULTS.md
   ↓
RESEARCH_MAP.md
   ↓
Wiki (必要な場合のみ)
```

研究結果をコードや terminal output の中だけに残さない。

---

## 16. コーディング方針

- 既存コードを読んでから変更する。
- 最小変更を優先する。
- 再現可能な CLI / script を優先する。
- random seed、model、tokenizer、dataset、条件を記録する。
- 生データと集計結果を分離する。
- 人間が再実行できるコマンドを残す。
- 大規模な抽象化は、同じ処理が実際に複数回必要になってから行う。
- 研究コードでは「美しい設計」より「再現性・条件の明示」を優先する。

---

## 17. 推奨する1サイクルのモデル配分

通常サイクル:

```text
Haiku:
  repository / issue triage
  ↓
Sonnet:
  experiment design + implementation
  ↓
Deterministic tools / local models:
  experiment execution + statistics
  ↓
Haiku:
  raw result compression
  ↓
Sonnet:
  interpretation + candidate generation + ranking
  ↓
Opus:
  normally skipped
```

重要な結果・矛盾が出た場合のみ:

```text
Sonnet analysis
  ↓
Opus one-shot review
  ↓
Sonnet implements final decision
```

**Opus に実装作業を丸ごとやらせない。**

---

## 18. 自律実行時の最終チェック

次の実験へ進む前に確認する。

```text
[ ] 今回何が新しく分かったか？
[ ] 既存 Finding を更新する必要があるか？
[ ] 結果は再現可能か？
[ ] 次候補は既存 Issue と重複していないか？
[ ] 次実験でどの不確実性が減るか？
[ ] より安い実験で同じ判断ができないか？
[ ] 本当に新しい Issue が必要か？
[ ] 高価なモデルを使う必要があるか？
```

1つでも説明できなければ、次の高コスト実験へ自動的に進まない。

---

## 19. Default Behavior

指示が曖昧な場合は以下をデフォルトとする。

1. 既存研究を検索する。
2. 既存データを再利用する。
3. deterministic tools を使う。
4. Haiku で整理する。
5. Sonnet で研究判断する。
6. 小さい pilot を行う。
7. 結果を記録する。
8. 次候補を最大5件生成する。
9. 重複除去・コスト評価する。
10. 最重要候補1件だけを選ぶ。
11. 必要なら Issue 化して次サイクルへ進む。
12. Opus は重大な synthesis / conflict / design review のときだけ使う。

---

## 20. Success Metric

この自律研究ループの成功は、

```text
Issue count
Experiment count
Tokens consumed
```

では測らない。

最重要指標は、

> **Research uncertainty reduced per unit cost**

である。

```text
Research Efficiency
≈ Information Gain
  / (Compute Cost + AI Credit Cost + Human Review Cost)
```

Machine Kanbun の研究そのものが token efficiency を扱うように、研究プロセス自身も **information efficiency** を最大化する。