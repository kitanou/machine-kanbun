# Long Context

詳細: [docs/exp-04-long-context.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/exp-04-long-context.md)、[docs/exp-25-31-followups.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/exp-25-31-followups.md)。

## 現在までの結論
- **TTFT は文脈トークンにほぼ比例**する(gemma の prefill ≈ 100 tok/s)。-55% のトークンで TTFT も約 -55% だが、これは SeCF-L1 でも同じで機械漢文固有ではない(R-A6)。
- **#20 で見えた「N=600 で SCF が NF より +20pt」は再現しなかった**。質問を倍にすると +7.5pt(p=0.22)に縮み、同トークン数にそろえた NF と差がない。効果の大半はトークン削減と文脈長で説明でき、意味密度の優位の証拠はない(検出力は低い)(R-C4、R-C5)。
- 同一 token budget で古い履歴を切り捨てる運用では、SeCF-L1 が NF-short を上回るが、理由は保持量(#57、R-D16、R-D17)。

## 未検証
Lost-in-the-middle・distractor density・モデル依存の分離は、小規模・1 モデルでの確認にとどまる。qwen3-8b の 4,000 トークン超は 16 GB 機で停止・文字化け・Metal OOM で評価できていない(R-E2)。

## Evidence
- #4 — 2k〜32k の長文脈
- #25 — Token 量と Semantic Density の切り分け
- #57 — 同一 token budget での比較

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
