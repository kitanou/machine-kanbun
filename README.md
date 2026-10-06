# Machine Kanbun（機械漢文）— LLM の文脈を小さくする言語表現の研究

LLM に渡す文脈(会話履歴・記憶・文書)を、**意味を保ったまま少ないトークンで表す言語表現**にできるかを実測する研究リポジトリ。
当初は「漢字 + 漢文的省略 + 構造記号」の擬似漢文 IR(機械漢文, MKW)の評価だったが、研究が進むにつれて
**圧縮の効果はどこから来るのか、何を削ってよく何を残すべきか、実際の検索・チャット履歴に使えるか**という一般的な問いに移った。

- 評価は LM Studio 上のローカルモデル(主に `google/gemma-4-12b`、`qwen/qwen3-8b`)と、トークナイザ 7 種の実測で行う。
- 結果・失敗・再現しなかった結果は、そのまま記録する(§2.5)。各節の詳細は `docs/`、生データと全表は `results/`。

**現在の研究方針(Meta [#56](https://github.com/kitanou/machine-kanbun/issues/56))**: 日本語の Chat / Memory を主軸にする。**SCF-L1 = Chat Context**(会話中の高速・決定的な圧縮)、**SeCF-L1 = Long-term Memory**(非同期で意味中心に書き直す)、**NF = ユーザーとの入出力と直近の会話**という役割分担を検証している。SeCF-L5 / 機械漢文(L5)の方向の研究は終了し、多言語・Transformer 内部機構は再開条件が整うまで保留している(§7)。

**どこに何があるか**

| 知りたいこと | 場所 |
|---|---|
| 確認済みの結果(1 行 1 結果、状態つき) | [RESULTS.md](RESULTS.md) |
| 研究の体系・未解決問題・依存関係 | [RESEARCH_MAP.md](RESEARCH_MAP.md) |
| 初めて読む人向けの解説・用語集・ナビゲーション | [Wiki](https://github.com/kitanou/machine-kanbun/wiki)(原稿は [wiki/](wiki/)) |
| 各実験の詳細と生データ | [docs/](docs/)、[results/](results/) |
| 実験・議論・作業履歴 | [Issues](https://github.com/kitanou/machine-kanbun/issues) |
| 自律研究ループの運用ルール | [CLAUDE.md](CLAUDE.md) |

## 1. 用語: NF / SCF / SeCF(Issue #35)

圧縮の強さ(L0, L1, L5)ではなく**表現の性質**で名前を付ける。

| 新名称 | 略称 | 旧称 | 定義 |
|---|---|---|---|
| Natural Form | **NF** | L0 | 元の自然言語。文体・助詞・語順・ニュアンスを保持し、圧縮処理なし |
| Structural Compact Form | **SCF** | 軽量L1 / Parser L1 | 既存の言語解析器と規則で、高速・決定的に冗長性を削る表現。追加の LLM 推論は不要 |
| Semantic Compact Form | **SeCF** | L1(#19 まで)・L5/機械漢文 | 意味保持を最優先に、意味単位でさらに冗長性を除いた表現。生成コストを許容 |

- `L0 / L1 / … / L5` は**圧縮プロファイル**として別軸に残し、`SCF-L1`、`SeCF-L1`、`SeCF-L5` のように組み合わせる(L0〜L5 の定義は次節)。
- 本リポジトリの #20 以降の簡易 L1(sudachi-m など)は **SCF-L1**、#4〜#19 の L1〜L5・IR は **SeCF-L1〜L5**。裸の「L1」は文脈で意味が違う(#20 以降は SCF、#19 まで SeCF)。
- 過去の文書は旧称のまま記述し、冒頭に読み替えを付けてある。生成するレポートは `machine_kanbun/terms.py` で標準名に読み替える。JSON/CSV の既存キー(`L0`、`sudachi-m` …)は変更せず、CSV に `form` 列を追加した。
- 境界条件・生成要件・論文 Terminology 節の原稿は [docs/terminology.md](docs/terminology.md)。

### 機械漢文の圧縮レベル(L0〜L5, Issue #1〜#19)

Issue #1〜#19 の「L0〜L5」は、**同じ構造化データ(Fact)から決定的に描画した 6 段階の表現**で、1 段階ごとに「何を削り、何を足すか」が決まっている。L1 から L5 へは一気に飛ぶのではなく、次の梯子になっている(Issue #10 はこの梯子の間を 1 要素ずつ分解した)。

| レベル | 名称 | 新名称 | その段階で起きること | トークン(o200k) | 例(同じ人物の先頭部分) |
|---|---|---|---|---|---|
| L0 | 自然日本語 | NF | 元の文章。主語や文末を毎回書く | 233 | `人物Aは1976年に生まれた。人物Aの職業はIT関係である。人物Aは大阪に住んでいる。…` |
| L1 | 日本語要約 | SeCF-L1 | 主語の反復・冗長な文末を除いた**普通の短い日本語**。助詞・活用は残り、そのまま読める | 127(-45%) | `1976年生まれ。職業はIT。大阪在住。十割蕎麦が好き。…酒は飲まない。納豆は食べないらしい(未確認)。` |
| L2 | 漢字圧縮 | SeCF-L2 | **助詞・活用を捨てて漢字語の列**にする。意味演算子を導入(下表)。群の接頭(食・機・習…)は各句に反復する | 111(-52%) | `人物A。生1976。職IT。住大阪。食十割蕎麦好。食酒不飲。食疑納豆不食。食好蕎麦>うどん。機将iPad。習若雨→散歩不行。` |
| L3 | 擬似漢文 | SeCF-L3 | 群ごとに**まとめて**群名を 1 回だけ書き、`:` でキーと値を区切り、`、` でつなぐ(漢文風の句読) | 112(-52%) | `人物A：生:1976、職:IT、住:大阪。食・十割蕎麦好、酒不飲、疑:納豆不食、好:蕎麦>うどん。機・過:ThinkPad、今:MacBook、将:iPad。` |
| L4 | 構造漢文 | SeCF-L4 | 群を `群{ … }` の**ブロック**にして、改行とインデントで構造を明示する。読みやすいが改行・括弧でトークンが増える(L1 比 +11%) | 141(-39%) | `人物A{`<br>` 食{`<br>`  酒不飲`<br>`  疑:納豆不食`<br>` }`<br>` 機{ 過:ThinkPad … }`<br>`}` |
| L5 | 機械漢文 | SeCF-L5 | L4 の構造を**1 行に詰める**(`;` 区切り)。条件の `若A→B` は `A?B` に | 110(-53%) | `人物A{生:1976;職:IT;住:大阪;食{十割蕎麦好;酒不飲;疑:納豆不食;好:蕎麦>うどん};機{過:ThinkPad;今:MacBook;将:iPad};習{雨?散歩不行}}` |
| — | JSON | — | 比較用の標準的な JSON | 362(+55%) | `{"subject":"人物A","facts":[{"type":"attribute","key":"生","value":"1976"},…` |

(数値は `machine_kanbun/data/profiles.json` の 1 人目、o200k。長い生成文書では L5 は L1 より約 2〜3% 多く、L0 比で約 -54%。)

**意味演算子**(L2 以降。L3 以降は `:` を伴う): `不` 否定 / `無` 不存在 / `未` 未了 / `非` 〜でない / `禁` 禁止 / `過` `今` `将` 時制 / `若` 条件 / `故` 因果 / `疑` 不確実 / `>` 比較。

**読み方**: L1 は「普通の日本語の簡潔化」、L2 は「漢字語への圧縮」、L3 は「群ごとの整理」、L4 は「構造化」、L5 は「1 行への詰め込み」と、段階ごとに別の操作が足されている。
このため #19 では、圧縮の大半が **L0 → L1(文法的冗長性の除去)**で得られ、**L1 → L5(漢字化・構造化)は追加の利得をほとんど生まない**ことを三層に分けて確認している([docs/exp-19](docs/exp-19-synthesis.md))。
各段階の違いの精度への効果(助詞削除・漢字置換・構造化などを 1 つずつ)は [docs/exp-10](docs/exp-10-l1-ablation.md)、全形式の精度とトークンは [docs/exp-01](docs/exp-01-basic-formats.md) を参照。

## 2. 現時点の主な結論

数字は gemma-4-12b を中心にした値で、特記がない限り各条件 48〜120 問(誤差は ±5〜10pt)。検出できなかった差は「差がない」ではなく「検出できない」と読む。

### 2.1 圧縮の効果はどこから来るのか(Issue #1, #4, #10, #19)
- **大半は自然言語の文法的冗長性の除去**から来る。NF → 簡潔な日本語(SeCF-L1)でトークンは -48〜-61%(全トークナイザで頑健)、精度は変わらない。
- **機械漢文(SeCF-L5)は SeCF-L1 と統計的に区別できない**(p≥0.07)。構造化(`{}` 記法)はトークンを減らさず、精度を有意に下げる(gemma -5.5pt、qwen -9.4pt)。漢字置換そのものの効果も小さい。
- **演算子の意味を読めるかが精度を決める**: 意味のない ID にすると約 -30pt。凡例(数十〜250 トークン)で一部回復するが戻りきらない。不確実性(疑)が最も弱い。
- **TTFT は文脈トークンにほぼ比例**する(gemma の prefill ≈100 tok/s)。-55% のトークンで TTFT も約 -55% だが、これは SeCF-L1 でも同じで機械漢文固有の効果ではない。
- → [Issue #1](docs/exp-01-basic-formats.md) / [#4](docs/exp-04-long-context.md) / [#10](docs/exp-10-l1-ablation.md) / [#19](docs/exp-19-synthesis.md)

### 2.2 漢字・ラベルと多言語(Issue #11, #17, #19)
- MKW の**言語間トークン数のばらつき(CV)は約 1%**(NF 12〜18%、L1 8〜13%)。ただし設計上の帰結で、ラベルを各言語の語に戻すと約 10 倍(約 10%)に悪化する。**効いているのは漢字ではなく「言語共通の構造語彙」**で、内容語まで共通化するとさらに 0.8% まで下がる(#17 の解釈を #19 で修正)。
- 精度はほぼ変わらず(ラベルの局所化で ±2pt)、**MKW が SeCF-L1 に勝つ言語はない**(英語は不確実性の読み落としで -5.7pt)。韓国語では Hangul のトークン効率でトークン優位が最大(-10%)、中国語では出ない。
- 自然文から直接 IR を作る(日本語を経由しない)と、凡例なしで 78.8%、**演算子の凡例を付けると 93.9%**(原文は 99.5%)。往復(IR → 自然文)は IR+凡例より低い。
- → [#11](docs/exp-11-multilingual.md) / [#17](docs/exp-17-label-localization.md) / [#19](docs/exp-19-synthesis.md)

### 2.3 日本語の SCF(Issue #20, #25〜#31, #33)
LLM を使わず、既存の形態素解析器(Sudachi / MeCab / Janome)と係り受け解析器(GiNZA)と規則だけで作る SCF を、会話体の記憶・チャットで評価した。
- **生成コストは無視できる**: 1 通 0.05 ms(約 2 万通/秒)、GiNZA でも約 20 ms。変換コストは最初の使用で回収できる(損益分岐 R≈0.002〜0.4)。トークナイザによって効果が逆転する: Qwen3 では空白区切りの SCF-L1 が **+22%増**、空白なし連結は全トークナイザで -28〜-35%。トークナイザ別に書き方を探索すると 0.59〜0.66 まで下がる(#26、精度への有意な影響は検出できず)。
- **何を削ってよいか(#30)**: 助詞・丁寧語・フィラー・文区切りは削除しても精度は落ちない(むしろ 85〜88%)。**状態標識(否定・不確実性の「かも」)、語順、主語を消すと壊れる**(content-only 55%、語順の逆転 27%、naive 35%)。最小十分表現は「内容語 + 状態標識 + 語順 + 主語」。
- **長文脈での精度(#25)**: #20 で見えた「N=600 で SCF が NF より +20pt(p=0.022)」は**再現しなかった**。質問を倍にすると +7.5pt(p=0.22)に縮み、同トークン数にそろえた NF と差がない。**効果の大半はトークン削減と文脈長で説明でき、意味密度の優位を示す証拠はない**(検出力は低い)。
- **検索(#20, #28)**: BM25 では SCF が大きく改善(Recall@5 0.72 → 0.94〜0.99、10,000 件)。埋め込みでは強いモデル(qwen3-embedding)なら NF のほうが強い。**検索は SCF(語彙)、LLM への入力は NF** の Dual-Index が最良(Recall@5 0.97〜0.99)。弱い埋め込みでは SCF 単独が最良。
- **実会話に近いデータ(#27)**: LLM 生成の自然会話では汎用 SCF が NF より有意に悪化(71.1% 対 82.9%)。否定・言い直し・俗語・伝聞の「って」が落ちるためで、精製ルールで NF と同等(82.9%、トークン -18%)に回復した(同じデータからの精製で過学習の可能性)。
- **チャット履歴・メモリ(#33 再実行、#49〜#62)**: SCF-L1 / SeCF-L1 を Chat 履歴・長期記憶に使う検証の結果は §2.4 にまとめた。
- **多言語・内部表現(#29, #31)**: 各言語の既存パーサで L1 化するとトークン数の言語間ばらつきは 7 トークナイザすべてで収束するが、QA は日本語以外で保てない(中国語 90% → 40%、標識を各言語の語で書くと 72%)。注意・隠れ状態の解析は、動かせる小型モデルが課題を解けず検出不能。
- → [#20](docs/exp-20-japanese-scf.md) / [#25〜#31](docs/exp-25-31-followups.md) / [#33](docs/exp-33-chat-style.md)

### 2.4 Chat 履歴と Long-term Memory(Issue #33 再実行, #49, #51, #54, #57, #59, #60, #62)
SCF-L1 = Chat Context、SeCF-L1 = Long-term Memory という役割分担([#56](https://github.com/kitanou/machine-kanbun/issues/56))を検証した。**SeCF-L1 は #19 の L1(助詞・活用を残した簡潔な自然言語への書き直し。ここでは gemma-4-12b が各発話を書き直す)**。初回 #33 の `主体{…}` 型の構造化条件は SeCF-L5 相当で、SeCF-L1 の結果としては扱わない。状態 7 択 QA(はい / いいえ / 未確定などの状態を問う、決定的採点)で想起を測った。
- **文体(#33)**: 古い履歴を SCF-L1 / SeCF-L1 にしても、NF の回答の自然さ・会話らしさ・Persona の汚染は検出されない。直近 NF の窓 0 往復でも問題なし。
- **想起(#49, #33 再実行)**: **SCF-L1 は古い事実の想起が NF より -10〜-17pt**(トークン数をそろえた NF は NF 並み)。特に願望(WANTED)・完了(DONE)を取り違える。**SeCF-L1 は NF 並み**(短 0.962 / 長 0.968、NF 0.971 / 1.000)で、圧縮率も 0.85〜0.88(SCF-L1 は 0.90〜0.92)。構造化 `{}` 型(SeCF-L5 相当)は 0.70〜0.82 に落ちトークンも増える。標識の凡例や自明語化では SCF-L1 の低下は回復を検出できない(#51)。かつての「圧縮履歴のほうが事実に言及されやすい」は語の有無の指標による見かけで、取り下げた。
- **三層履歴(#54)**: 最古 SeCF-L1 / 中間 SCF-L1 / 直近 NF の三層は、SeCF-L1 一本と同等の想起(短 0.990 / 長 0.952)を、同期変換 14〜86 ms で出せる。ただし目標の事実が SCF 層に残ると想起が落ちる(長 0.794)ので、SCF-L1 は書き直し前のつなぎにとどめる。
- **同一 token budget(#57)**: 古い往復から切り捨てる NF に対し、**SCF-L1 は勝てない(+0〜6pt、有意でない。事実が残っていても別の状態を答える誤答が出る。budget 0.85 で NF の約 8 倍)**。**SeCF-L1 は +14〜20pt 上回る**が、理由は「同じ budget により多くの履歴が残る」ことで、表現そのものの効果ではない(事実が残った場合の正答率は NF と同等)。
- **Long-term Memory の検索(#59)**: SeCF-L1 に検索上の独自の利点はない(ベクトルは NF 0.947 > SeCF-L1 0.900、BM25 は SCF-L1 0.893 > SeCF-L1 0.807)。最良は #28 と同じ Dual(BM25 = SCF-L1、ベクトル = NF、R@1 0.980)。LLM に渡す記憶としては SeCF-L1 ≒ NF > SCF-L1 だが、節約は 7〜8%、変換は 1 件約 1.2 秒で、約 40 回検索されないと回収できない。
- **状態が変わる記憶(#62)**: 日付つきの 2〜3 回更新の記憶は、生の記憶列のまま最新状態を約 0.87〜0.90 で答えられ、SeCF-L1 で統合しても利得はない(0.867、トークン +6%)。難しいのは「迷い」に戻る遷移(上限でも 0.67)。
- **文章長と変換コスト(#60)**: SeCF-L1 の圧縮率は文章長ではなく文体で決まる(chat 0.74〜0.76 で平坦、news 0.88 → 0.81)。TTFT の短縮は長さに比例するが小さく(2000 トークンで 2〜2.5 秒、16〜20%)、**同期変換は 25〜2000 トークンのどの長さでも SeCF-L1 のほうが遅い**(変換は節約の約 22 倍)。事前変換した記憶でも、変換コストの回収に 1 件あたり約 24〜56 回の再利用が必要。
- → [#33](docs/exp-33-chat-style.md) / [#59](docs/exp-mem-secf1.md) / [#62](docs/exp-mem-update.md) / [#60](docs/exp-60-length-breakeven.md)

### 2.5 再現しなかった・修正された結果
- **#20 の N=600 の優位**(+20pt, p=0.022)は、質問サンプルを足すと有意でなくなった(#25)。#20 の p 値の一部は偶然だった可能性が高い。
- **#17 の「漢字が共有語彙」**は、#19 で「共通の構造語彙(文字種は問わない)」に修正した。
- **#20 の変換器の不具合**(文頭の「なんか」が「なん」と残る)を #25 で修正した。#20 のレポートは再生成済み(トークナイザ表が 0.01〜0.02 変わる。例: Qwen3 の m 1.24 → 1.22)。
- **#19 の自然文実験**では、「IR のみ」と「往復」の比較に凡例の有無の混同があり、凡例つきの条件を足して切り分けた。
- **#33 の「圧縮履歴のほうが事実に言及されやすい」**は、語の有無の指標による見かけで、状態 7 択 QA(#49)では逆転したため取り下げた。
- **#33 の初回の「SeCF」**(`主体{…}` 型の構造化)は SeCF-L5 相当で、SeCF-L1 ではなかった。Issue #33 の訂正に従い、SeCF-L1 で再実行した(§2.4)。

## 3. 実験一覧

| Issue | 問い | 主な結果 | 詳細 |
|---|---|---|---|
| #1 | 7 形式(L0〜L5, JSON)で精度・トークンは? | 文脈 -49%、L1 も -46% で漢文化の上乗せは見えない。弱点は不確実性 | [exp-01](docs/exp-01-basic-formats.md) |
| #4 | 2k〜32k の長文脈では? | MKW は L1 と区別できない。TTFT はトークンに比例 | [exp-04](docs/exp-04-long-context.md) |
| #10 | L1 から 1 要素ずつ変えると? | 助詞削除だけ -9%。L1 は既に最小に近い | [exp-10](docs/exp-10-l1-ablation.md) |
| #11 | EN/KO/JA を共通 MKW に | CV 1% だが設計の帰結。MKW は L1 に勝たない | [exp-11](docs/exp-11-multilingual.md) |
| #17 | 漢字ラベルは何に効く? | トークン安定性に効くが精度には効かない。意味の透明性が精度を決める | [exp-17](docs/exp-17-label-localization.md) |
| #19 | 三層分解と次期実験 | 圧縮の大半は層 1(文法的冗長性)。構造化は精度を下げる | [exp-19](docs/exp-19-synthesis.md) |
| #20 | 日本語 NF と解析器ベースの SCF | 変換は無視できるコスト。検索は SCF が有利 | [exp-20](docs/exp-20-japanese-scf.md) |
| #25 | なぜ長文脈で SCF が高精度に見えたか | トークン削減・文脈長で説明可能。#20 の優位は再現せず | [exp-25-31](docs/exp-25-31-followups.md) |
| #26 | トークナイザ別の最適な書き方 | 最小スタイルは 0.59〜0.66(連結形)。精度差は検出できず | 同上 |
| #27 | 実会話(に近いデータ)で再現するか | 汎用 SCF は悪化、精製ルールで回復 | 同上 |
| #28 | NF/SCF の Dual-Index RAG | 強い埋め込みで最良(Recall@5 0.97〜0.99) | 同上 |
| #29 | Transformer は NF と SCF をどう読むか | 小型モデルが課題を解けず検出不能 | 同上 |
| #30 | 圧縮の限界(最小十分表現) | 否定・不確実性・語順・主語が必須 | 同上 |
| #31 | 多言語パーサベース L1 の収束 | トークンは収束、QA は日本語以外で保てない | 同上 |
| #33 | SCF 履歴は回答の文体を汚染するか | SCF-L1・SeCF-L1 とも汚染は検出されず。想起は SeCF-L1 が NF 並み、SCF-L1 は -10〜-17pt。変換コストは SeCF-L1 が 3 桁大 | [exp-33](docs/exp-33-chat-style.md) |
| #39 段階 1-2 | 多言語(9 言語)の compactness は一致するか、言語別 SCF は収束するか | 乖離し、粗い SCF では収束しない。文字数は ja 0.50・zh 0.33 だが token は ja 1.14〜1.89。eu・tr は語数が少なくても token が 1.1〜1.7 倍 | [exp-39](docs/exp-39-multilingual-scf.md) |
| #49 | 圧縮履歴から古い事実を取り出せるか(状態 7 択、決定的採点) | SCF-L1 は NF より -10〜-17pt、SeCF-L1 は NF 並み。「事実に言及されやすい」を取り下げ | [exp-33](docs/exp-33-chat-style.md) |
| #51 | SCF の標識の凡例・自明語化で想起は回復するか | 回復を検出できず(凡例 +5〜7pt、有意でない) | 同上 |
| #54 | 三層履歴(SeCF-L1 / SCF-L1 / NF)で SeCF-L1 並みの想起を低遅延で出せるか | 出せる(同期変換 14〜86 ms)。SCF 層に目標の事実が残ると落ちる | 同上 |
| #57 | 同一 token budget で NF-short と SCF-L1 / SeCF-L1 | SCF-L1 は勝てない。SeCF-L1 は +14〜20pt(保持量で説明) | 同上 |
| #59 | SeCF-L1 を Long-term Memory 表現にすると | 検索上の独自の利点なし。LLM 入力としては NF 並み〜SCF-L1 より正確 | [exp-mem-secf1](docs/exp-mem-secf1.md) |
| #62 | 状態が変わる記憶で SeCF-L1 統合は有効か | 利得なし(最新状態 0.867 対 生の NF 0.883)。「迷い」に戻る遷移が難しい | [exp-mem-update](docs/exp-mem-update.md) |
| #60 | SeCF-L1 の文章長別の圧縮率・TTFT 短縮・変換コストの損益分岐点 | 圧縮率は文体で決まる(chat 0.74、news 0.81〜0.88)。TTFT 短縮は長さに比例(2000 トークンで 2〜2.5 秒)。同期変換はどの長さでも遅く損益分岐点なし(変換は節約の約 22 倍)。事前変換は 24〜56 回の再利用で回収 | [exp-60](docs/exp-60-length-breakeven.md) |
| #35 | 名称を NF / SCF / SeCF に統一 | — | [terminology](docs/terminology.md) |

未着手・未達の Issue は §7。

## 4. リポジトリ構成

| 領域 | 場所 |
|---|---|
| **共通** | `machine_kanbun/lmstudio.py`(LM Studio ストリーミングクライアント。TTFT・実測 prompt_tokens)、`tokens.py` `tokcross.py`(トークナイザ比較)、`qa.py`(プロンプトと採点)、`sysmem.py`、`terms.py`(NF/SCF/SeCF の表示名) |
| **機械漢文 IR(#1〜#19)** | `model.py` `encoder.py`(KCR エンコーダ L0〜L5)、`data/profiles.json`(ベンチマーク)、`gen.py` `legend.py` `longrun.py` `longreport.py`(#4)、`ablate.py` `ablreport.py` `convcost.py`(#10)、`i18n.py` `mlenc.py` `mlconv.py` `mlreport.py`(#11)、`irlabel.py` `irreport.py`(#17)、`layers.py` `labelfam.py` `i18n_zh.py` `zhreport.py` `core.py` `natural*.py`(#19) |
| **日本語 SCF(#20)** | `jl1.py`(SCF 変換器: Sudachi/MeCab/Janome/GiNZA + 規則)、`jpdata.py`(会話体メモリ生成)、`jpbench.py` `jpretention.py` `jprag.py` `jpllm.py` `jprun.py` `jpbreak.py` `jpretime.py` `jpstats.py` `jpreport.py` |
| **要因分解・一般化(#25〜#31)** | `jlvar.py`(表現バリアント)、`jpctx.py` `jpctxrun.py`(制御付き長文脈 QA)、`tokaware.py` `jptokaware.py`(#26)、`jpnat.py` `jpnatrun.py`(#27)、`jpdual.py`(#28)、`jpattn.py`(#29, MLX)、`jpml.py` `jpmlrun.py`(#31)、`jp2report.py` |
| **チャット履歴(#33, #49, #51, #54, #57)** | `jpchat.py` `jpchatreport.py` `jpchatsecf.py`(初回の構造化 = SeCF-L5 相当)`jpchatsecf1.py`(SeCF-L1)`jpchatqa.py` `jpchatqareport.py`(想起 QA、三層履歴)`jpchatbudget.py` `jpchatbudgetreport.py`(同一 budget) |
| **Long-term Memory(#59, #62)** | `jpmem.py`(記憶の検索・回答)、`jpmemupd.py`(状態が変わる記憶・統合) |
| **文章長と変換コスト(#60)** | `jplen.py`(長さ別の圧縮率・TTFT・変換時間・損益分岐) |
| **多言語の続き(#39)** | `mlpar.py`(9 言語の並列文のコンパクトさ)、`mlscf.py`(FLORES-200 での言語別 SCF と収束) |
| **実行・監視** | `scripts/run_*.sh`(いずれも再開可能。LM Studio が固まると再ロードして再開)、`scripts/status_*.sh` |
| **テスト** | `tests/`(`python -m pytest tests -q`) |
| **結果** | `results/`(JSON/JSONL、CSV、SVG、`*_report.md`)。生データと全表はここ |
| **文書** | `docs/`(用語と、各 Issue の詳細)、`RESULTS.md` `RESEARCH_MAP.md`(原簿)、`wiki/`(Wiki の原稿)、`CLAUDE.md`(自律研究ループのルール) |

## 5. 再現手順

**環境**
- Python 3.9 以上。基本: `pip install -r requirements.txt`(`tiktoken`、`pytest`)。日本語/多言語解析器: `pip install -r requirements-ja.txt`(Python 3.9 の `.venv-ja` で実行)。#29 のみ Apple silicon + Python 3.11 の別環境: `pip install -r requirements-mlx.txt`(`.venv-mlx`)。
- LM Studio のローカルサーバ(`http://localhost:1234/v1`)に `qwen/qwen3-8b` と `google/gemma-4-12b` を `lms load <model> -c 20000 --parallel 1` でロードする。gemma は `reasoning_effort: none`、qwen3 は `/no_think` を付ける(コード側で自動)。埋め込みは `text-embedding-nomic-embed-text-v1.5` と `text-embedding-qwen3-embedding-0.6b`。
- HF トークナイザ(Qwen3、gemma2、Llama-3.1、Mistral-Nemo、DeepSeek-V3)はローカルキャッシュを使う。

**主なコマンド**
```bash
python3 -m pytest tests -q                              # テスト
python3 -m machine_kanbun tokens                        # #1 オフラインのトークン比較
scripts/run_long.sh                                     # #4 長文脈(再開可能)
python3 -m machine_kanbun ablreport                     # #10 レポート(実行は scripts/run_ablation*.sh)
scripts/run_ml.sh; scripts/run_ir.sh                    # #11, #17
PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpreport        # #20 レポート再生成
scripts/run_jp2.sh; scripts/run_jp2b.sh                 # #25〜#31(LM Studio、約 10 時間)
PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jp2report       # #25〜#31 のレポートと CSV・グラフ
scripts/run_jp3.sh; scripts/run_jp3b.sh                 # #33 初回(SCF、構造化 SeCF)
PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpchatreport    # #33 のレポート
scripts/run_jp3_secf1.sh; PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpchatsecf1   # #33 再実行(SeCF-L1)
scripts/run_qa49.sh; scripts/run_qa51.sh; scripts/run_qa54.sh      # #49 想起 QA、#51 標識、#54 三層履歴
PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpchatqareport  # 想起 QA のレポート
scripts/run_budget57.sh; PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpchatbudgetreport   # #57 同一 budget
scripts/run_mem59.sh; PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpmem report            # #59 Long-term Memory
scripts/run_upd62.sh; PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jpmemupd report         # #62 状態が変わる記憶
scripts/run_len60.sh; PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.jplen report   # #60 文章長(--pilot で小規模)
scripts/fetch_flores.sh                                            # #39 段階 2 のコーパス(FLORES-200、約 24 MB、リポジトリには含めない)
PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.mlpar; PYTHONPATH=. .venv-ja/bin/python -m machine_kanbun.mlscf   # #39 段階 1・2
```
各 `scripts/run_*.sh` は途中経過を `results/<dir>/timing.log` と `progress.txt` に残し、完了済みの段階を飛ばして再開する。

**LM Studio で分かっていること**: キャッシュ命中リクエストで「prompt processing 98.8%」のまま固まることがある(`lmstudio.py` はウォールクロックの期限で検出し、スクリプトがモデルを再ロードして再開)。16 GB 機では qwen3-8b が 4,000 トークン超の文脈で停止・文字化け・Metal のメモリ不足を起こしたため、qwen の長文脈条件は評価できていない(#4, #25〜#27)。同名モデルの二重ロードがあると、要求が詰まった古いインスタンスに流れる。

## 6. 評価の方法と共通の限界
- **検定**: 同じ質問・同じ文脈でのペア符号検定(必要に応じて置換検定とブートストラップ)。1 問は 1〜2pt で、多重比較の補正はしていない。
- **データ**: ほとんどの実験は生成器(テンプレート)由来の合成データ。自然会話(#27)・チャット履歴(#33)は LLM 生成で、実会話ではない。採点は文字列一致と LLM 採点(#33、条件を伏せた別モデル)で、人手評価は未実施。
- **モデル**: 長文脈の主評価は gemma-4-12b 1 モデルで、qwen3-8b は短い条件のみ。内部表現(#29)は動く小型モデル 1 種のみ。
- **seed**: 多くは 1 seed・1 文書。seed 間のばらつきは未評価。

## 7. 未完・今後
研究の方向は Meta Issue [#56](https://github.com/kitanou/machine-kanbun/issues/56) と [RESEARCH_MAP.md](RESEARCH_MAP.md) の未解決問題(U1〜U13)で管理している。
- **終了した方向**: SeCF-L5 / 機械漢文 / Semantic IR(#2、#6、#8、#14 など。結果は保存済み)、LLMLingua 系との比較(#13)。
- **次の候補**: 「まだ決めていない」を LLM が「いいえ」と読む問題(U13)、重要度に基づく SeCF-L1 への書き直しの優先順位(U10)、表現効果を検出できる難しい課題(U11)。
- **保留(再開条件つき)**: 多言語 SCF(#39。形態素解析つき SCF、母語話者評価、信頼できる並列コーパスが必要。グリーンランド語は FLORES-200 にも含まれない)、Transformer 内部機構(#29。動かせるモデルが律速)、Tokenizer / Transformer 横断評価(#41〜#43)。
- **データ・評価の限界**: 実会話コーパスでの再現、人手の匿名評価(`results/jp3/blind_sheet.csv` を用意済み)、より大きいモデル、qwen の長文脈の再評価。
- **Wiki**: 研究ポータルとして整備中(#45。原稿は `wiki/`)。

## 8. 文書の索引
[用語](docs/terminology.md) ・ [#1 基本比較](docs/exp-01-basic-formats.md) ・ [#4 長文脈](docs/exp-04-long-context.md) ・ [#10 L1 アブレーション](docs/exp-10-l1-ablation.md) ・ [#11 多言語](docs/exp-11-multilingual.md) ・ [#17 ラベル局所化](docs/exp-17-label-localization.md) ・ [#19 研究整理](docs/exp-19-synthesis.md) ・ [#20 日本語 SCF](docs/exp-20-japanese-scf.md) ・ [#25〜#31](docs/exp-25-31-followups.md) ・ [#33 チャット文体・想起・三層・budget](docs/exp-33-chat-style.md) ・ [#39 多言語 SCF](docs/exp-39-multilingual-scf.md) ・ [#59 Long-term Memory](docs/exp-mem-secf1.md) ・ [#62 状態が変わる記憶](docs/exp-mem-update.md) ・ [#60 文章長と変換コスト](docs/exp-60-length-breakeven.md) ・ [RESULTS](RESULTS.md) ・ [RESEARCH_MAP](RESEARCH_MAP.md) ・ [Wiki](https://github.com/kitanou/machine-kanbun/wiki)
