# Machine Kanbun（機械漢文）— Issue #1 検証基盤

LLM に渡す文脈を「漢字 + 漢文的省略 + 構造記号」の擬似漢文 IR に変換し、
トークン数・意味保持・TTFT がどう変わるかを測る。

## 構成
- `machine_kanbun/model.py` — 構造化 Fact / Question / Profile
- `machine_kanbun/encoder.py` — KCR エンコーダ。L0〜L5 と JSON を生成
- `machine_kanbun/data/profiles.json` — ベンチマーク(3 プロファイル / 49 facts / 47 問)。否定(不無未非禁)・時制・条件・因果・比較・数値・不確実性を含む
- `machine_kanbun/tokens.py` — トークナイザ比較(tiktoken。HF は任意)
- `machine_kanbun/lmstudio.py` — LM Studio(OpenAI 互換, ストリーミング)クライアント。TTFT とモデル実測の prompt_tokens を取得
- `machine_kanbun/qa.py` — プロンプト構築と採点(yes/no は先頭語、span は必須語/禁止語)
- `machine_kanbun/cli.py` — `tokens` / `qa` / `report`(Pareto frontier 付き)

```bash
python3 -m machine_kanbun tokens                       # オフライン・トークン比較
python3 -m machine_kanbun qa --model qwen/qwen3-8b     # 凡例なし(ゼロショット)
python3 -m machine_kanbun qa --model qwen/qwen3-8b --legend   # 演算子の凡例つき
python3 -m machine_kanbun qa --model google/gemma-4-12b
python3 -m machine_kanbun report
```

## 現時点の結果(qwen/qwen3-8b, temperature 0, 47 問 × 7 形式)
文脈のみのトークン(`results/tokens.txt`, o200k): L0 649 → L1 -46% / L2 -49% / L3 -49% / L4 -36% / L5 -49% / JSON +58%。

QA 精度(凡例なし → 凡例あり):

| 形式 | 精度 | 精度(凡例) |
|---|---|---|
| L0 自然日本語 | 100% | 100% |
| JSON | 93.6% | 93.6% |
| L1 日本語要約 | 91.5% | 91.5% |
| L2 漢字圧縮 | 93.6% | 93.6% |
| L3 擬似漢文 | 91.5% | 89.4% |
| L4 構造漢文 | 89.4% | 93.6% |
| L5 機械漢文 | 87.2% | 91.5% |

所見:
1. 文脈トークンは L2/L3/L5 で約 -49%(目標 -30% 達成)。JSON は逆に +58%。
2. ただし **L1(普通の日本語要約)も -46%** で、トークン効率は漢文系とほぼ同じ。漢文化の上乗せ効果はこのデータでは見えない。
3. 否定(不・無・未・非・禁)は全形式でほぼ満点。懸念されていた否定の区別は問題にならなかった。
4. 弱点は**不確実性(疑)**: L1〜L5 は 0〜33%。モデルが「疑」を答えとしてそのまま出力してしまう。L0 は 100%。時制も凡例なしの L4/L5 で劣化。
5. 精度 95% 維持の条件は、L1〜L5 のいずれでも未達(L0 との差 6〜13pt)。ただし 47 問・1 モデルなので 1 問 = 2.1pt。誤差が大きく、確定的な結論ではない。
6. TTFT は全形式で約 1.0 秒でほぼ同じ。1 リクエスト約 240〜470 トークン中、固定のシステム/指示部が約 175 を占め、短い文脈では差が出ない。TTFT 改善の検証には長い文脈(数千トークン以上)が必要。

## 未完・制約
- **gemma-4-12b は未評価**(ダウンロード完了待ち)。完了後に上記 `qa --model google/gemma-4-12b` を実行すれば `report` に並ぶ。
- 文脈は Fact から決定的に生成している(手書き構造化)。自然文 → 擬似漢文を LLM で行う Pseudo-Kanbun Generator、「要約 LLM との比較」は未実装(L1 は手書き要約で代用)。
- 評価データが小さい。問題数・プロファイル数・文脈長の拡張が必要。
- HF トークナイザ比較は transformers が古く Qwen3 を読めなかった。モデル別の実トークン数は LM Studio の `prompt_tokens` で代用。
