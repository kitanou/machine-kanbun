# Languages

多言語の研究は、**各言語固有の表層差を除いたとき意味構造が収束するか**を問う。現在は**低優先で保留**(Meta [#56](https://github.com/kitanou/machine-kanbun/issues/56))。母語話者でない言語の自然さ・ニュアンスの評価が難しく、LLM 採点への依存を増やしたくないため。

## 測定済み
- **機械漢文(MKW)**: トークン数の言語間 CV は約 1% だが、設計上の帰結(ラベルを各言語の語に戻すと約 10%)。効いているのは漢字ではなく言語共通の構造語彙(#11、#17、#19)。MKW が SeCF-L1 に勝つ言語はない。
- **言語パーサベースの SCF**(ja / en / ko / zh): トークン数は 7 トークナイザで収束するが、QA は日本語以外で保てない(中国語 90% → 40%)(#31、R-B5)。
- **並列 9 言語の NF**(ja / en / zh / ko / tr / vi / ru / ar / eu): 文字数・語数・トークンが言語間で乖離する(R-B6)。
- **FLORES-200 の言語別ストップワード SCF**: 言語間で収束しない。削除率は ko 0.04、tr・eu 0.15、en 0.44、ja 0.53。ただしストップワード表の大きさとの交絡がある(R-B7)。

## 未実施・対象外
- グリーンランド語: 信頼できる並列訳が未入手(FLORES-200 にも含まれない)(R-E4)。
- 形態素解析つきの SCF、QA での意味保持、人間の可読性、Chat への適用: 未実施。

## 再開条件(#39)
母語話者評価、信頼できる並列コーパス、十分な自動意味評価、日本語で確立した SCF 評価方法を他言語へ移植できる状態。

Evidence: #11 #17 #19 #31 #39 / [docs/exp-39-multilingual-scf.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/exp-39-multilingual-scf.md)

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
