# 表現の用語: NF / SCF / SeCF(Issue #35)

圧縮の強さ(レベル)ではなく、**表現がどのように生成され、何を保持するか**で名前を付ける。

| 新名称 | 略称 | 旧称 | 定義 |
|---|---|---|---|
| Natural Form | **NF** | L0 | 元の自然言語。文体・助詞・語順・ニュアンスを保持し、圧縮処理なし |
| Structural Compact Form | **SCF** | 軽量L1 / Parser L1 | 形態素解析・構文解析など既存の言語解析器と、品詞・係り受け・語彙に基づく規則で、高速かつ決定的に冗長性を削減した表現。追加の LLM 推論を必要としない |
| Semantic Compact Form | **SeCF** | L1(Issue #19 までの意味)・L5/機械漢文(IR) | 意味保持を最優先に、意味単位で整理しながらさらに冗長性を除去した表現。意味解析・LLM などの生成コストを許容し、オフライン処理・長期記憶・コンテキスト圧縮に向く |

変換モデル: `NF → SCF → SeCF`(構造的変換 → 意味的変換)。逐次である必要はなく、`NF → SeCF` を直接生成してもよい。

## 2 つの軸: 表現と圧縮プロファイル

`L0 / L1 / L5 …` は廃止せず、**Compression Profile(圧縮プロファイル)**として表現形式とは別軸にする。

| 軸 | 値 | 意味 |
|---|---|---|
| Representation | NF / SCF / SeCF | 表現の性質(生成方法と保持するもの) |
| Compression Profile | L0 / L1 / L5 … | 圧縮の強さ(L0=なし、L1=表層文法の削除、L5=強い意味再符号化) |

組み合わせは `SCF-L1`、`SeCF-L1`、`SeCF-L5` のように書く。本リポジトリの Issue #20 以降の日本語実験の簡易 L1(`sudachi-m` など)はすべて **SCF-L1**、Issue #4〜#19 の L1(構造化 Fact からの電報体)は **SeCF-L1**、L5 / 機械漢文 / IR(IR-C・IR-L・IR-ID)は **SeCF-L5** にあたる。文言文(C1)は別の自然言語 baseline で、NF の一種として扱う。

## SCF と SeCF の境界

次の順に判定する。

1. **生成に LLM 推論・学習済みの意味モデルが必要か**(形態素解析器・係り受け解析器・品詞タガーは含めない)。必要 → SeCF。
2. **情報の編成を変えるか**: 言い換え、複数文の事実の統合・再配列、状態や関係の正規化、照応解決を明示的な意味表現に置き換えるなど。変える → SeCF。
3. **語・形態素の削除、品詞・活用に基づく写像(助詞・丁寧語・フィラーの削除、標識語への置換)、表記の書き換え(空白・区切り・文字種)だけか** → SCF。

境界の例:

| 変換 | 分類 | 理由 |
|---|---|---|
| sudachi-m / sudachi-g / mecab-m / janome-m / refined-r1・r2 | SCF | 解析器の出力を規則で写像・削除するだけ |
| ginza-d(係り受けで節に分け、省略主語を補完) | SCF | 決定的な規則。ただし主語補完は照応解決のヒューリスティックで、境界に近い |
| naive(ひらがな連の削除) | SCF | 解析器を使わない最も単純な規則。意味保持は不十分 |
| tokenizer-aware L1(`taw:*`) | SCF | 書き方だけを探索。意味は変えない |
| 多言語パーサベース L1(ja/en/ko/zh) | SCF | 各言語の既存解析器と規則 |
| LLM による要約・電報体への書き換え | SeCF | LLM 推論が必要 |
| 構造化 Fact から作る電報体(旧 L1)、機械漢文 L5、IR-C/IR-L/IR-ID | SeCF | 意味構造(Fact)を経由して再符号化 |

**生成要件**: SCF は(a) 追加の LLM 推論なし、(b) 入力長に対してほぼ線形で、リアルタイム処理に耐える、(c) 同じ入力に同じ出力(決定的)。SeCF は意味保持を QA・状態復元で検証することを要件にし、生成コストは許容する。

## 使い方(解析・表・論文)

- 以後のスクリプト出力・表・グラフ凡例・Issue・論文では `NF` / `SCF` / `SeCF` を用いる。`machine_kanbun/terms.py` の `display()` / `form_of()` が、データキー(`L0`、`sudachi-m` …)を標準名に変換する。
- 過去実験との対応が必要なときだけ旧称を併記する: `NF (formerly L0)`、`SCF (formerly lightweight/Parser L1)`、`SeCF (formerly L1)`。
- 裸の `L1` は文脈で意味が違う。**Issue #20 以降の日本語実験では SCF**、Issue #19 までは SeCF を指す(`form_of(name, era="jp" | "legacy")`)。

## フィールド名・データキーの方針

- 既存の JSON / JSONL / CSV のキー(`L0`、`sudachi-m`、`rep`、`cond` …)は**変更しない**(再現性・再開処理のため)。
- 新しい出力では、表現を示す列に標準名(`form`: NF / SCF / SeCF)と、必要なら圧縮プロファイル(`profile`: L0 / L1 / L5)を**追加の列**として持たせる。
- 表・グラフの凡例は `terms.display()` を通して標準名で出力する。

## 論文ドラフトの Terminology 節(原稿)

> We distinguish three representations of the same content. **Natural Form (NF)** is the original natural-language text. **Structural Compact Form (SCF)** is obtained by deterministic, analyzer-based transformations (morphological and dependency analysis with lightweight rules) that remove structural redundancy without any additional language-model inference. **Semantic Compact Form (SeCF)** removes redundancy at the level of meaning units and may require semantic analysis or language-model generation. Representation (NF/SCF/SeCF) is independent of the compression profile (L0, L1, L5, …), which indicates compression strength; e.g., SCF-L1 and SeCF-L5 are distinct configurations.

(論文本体のドラフトはこのリポジトリにはまだない。上の原稿を Terminology 節の正式文とする。)
