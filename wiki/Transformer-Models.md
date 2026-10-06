# Transformer / Models

## 確定していること
- 評価は主に **gemma-4-12b**(長文脈・Chat・Memory)と **qwen3-8b**(短い条件)の 2 モデル。同じ表現でもモデルによって精度が大きく違う(例: qwen で `疑` の理解が弱い)。
- 演算子の意味が読めないと精度が約 -30pt。凡例で一部回復する(R-A5)。

## 検出できなかった・未検証
- **内部機構(注意・隠れ状態)**: 動かせる小型モデルが課題を解けず、NF と SCF の読まれ方の違いは検出不能(#29、R-E1)。保留。
- **モデルサイズと圧縮耐性**、複数モデル族での結論の保持(#42、#43): 未実施。単なる「別モデルでも試す」は新規性が低いため、主要仮説が固まってから行う(#56)。
- 16 GB 機では qwen3-8b が 4,000 トークン超で停止するなど、計算資源が律速。

確定した事項と未検証の仮説を混同しないこと。確定事項は [Confirmed Findings](Confirmed-Findings)。

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
