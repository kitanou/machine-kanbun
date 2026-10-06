# Chat / Memory(現在の主軸)

**SCF-L1 = Chat Context、SeCF-L1 = Long-term Memory、NF = ユーザーとの入出力と直近の会話**という役割分担を検証している(Meta [#56](https://github.com/kitanou/machine-kanbun/issues/56))。詳細は [docs/exp-33-chat-style.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/exp-33-chat-style.md)、[docs/exp-mem-secf1.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/exp-mem-secf1.md)、[docs/exp-mem-update.md](https://github.com/kitanou/machine-kanbun/blob/main/docs/exp-mem-update.md)。

## 現在までの結論
**Chat 履歴**(#33、#49、#51、#54、#57)
- 古い履歴を SCF-L1 / SeCF-L1 にしても、NF の回答の自然さ・会話らしさ・Persona の汚染は検出されない。
- 想起(状態 7 択 QA、決定的採点)は **SeCF-L1 が NF 並み**(0.962 / 0.968)、**SCF-L1 は -10〜-17pt**(願望・完了を取り違える)。トークン数をそろえた NF は NF 並み。
- 三層履歴(最古 SeCF-L1 / 中間 SCF-L1 / 直近 NF)は SeCF-L1 一本と同等の想起を、同期変換 14〜86 ms で出せる。ただし目標の事実が SCF 層に残ると落ちる(0.794)。
- 同一 token budget では、SeCF-L1 が古い往復を切り捨てる NF を +14〜20pt 上回るが、理由は**より多くの履歴が残る**こと。表現そのものの効果は未検出。SCF-L1 は勝てず、事実が残っていても別の状態を答える誤答が出る。

**Long-term Memory**(#59、#62)
- 検索の最良は BM25(SCF-L1)+ ベクトル(NF)の Dual(R@1 0.980)。SeCF-L1 に検索上の独自の利点はない。
- LLM に渡す記憶としては SeCF-L1 ≒ NF > SCF-L1。節約は 7〜8% で、変換(1 件約 1.2 秒)は約 40 回検索されないと回収できない。
- 状態が変わる記憶は、日付つきの生の記憶列でも最新状態を約 0.87〜0.90 で答えられ、SeCF-L1 の統合に利得はない。

**文章長と変換コスト**(#60)
- SeCF-L1 の圧縮率は文章長ではなく文体で決まる(chat 0.74〜0.76 で平坦、news 0.88 → 0.81)。
- TTFT の短縮は長さに比例するが小さい(2000 トークンで 2〜2.5 秒、16〜20%)。**同期変換は 25〜2000 トークンのどの長さでも SeCF-L1 のほうが遅い**(変換は節約の約 22 倍)。事前変換でも 1 件あたり約 24〜56 回の再利用で回収。

**Prefix / KV キャッシュ**(#69)
- キャッシュにヒットすると、記憶量に依らず TTFT は 1.3〜4 秒で、NF と SeCF-L1 の差はほぼ消える。SeCF-L1 の利点は、キャッシュが外れたときの約 20% の短縮と、限界付近で保持できる数が増える場合がある点に限られる。

## 取り下げ・訂正
- 「圧縮した履歴のほうが事実に言及されやすい」は語の有無の指標による見かけで、取り下げ(R-D4)。
- 初回 #33 の「SeCF」は `{}` 型の構造化(SeCF-L5 相当)で、SeCF-L1 ではなかった。訂正して SeCF-L1 で再実行した。

## 限界
合成の会話・記憶、gemma-4-12b 1 モデル(書き直しと回答が同じモデル)、選択式 QA、1 回生成、多重比較の補正なし。SeCF-L1 は #19 の決定的テンプレートではなく LLM による書き直し。人手評価と実会話での再現は未実施。

## Open Questions
重要度に基づく SeCF-L1 への書き直し順(U10)、表現効果を検出できる難しい課題(U11)、「まだ決めていない」を「いいえ」と読む問題(U13)。SeCF-L1 の変換を同期で成立させる条件(U14)。

---
[Home](Home) ・ [Research Map](Research-Map) ・ [Findings](Confirmed-Findings) ・ [Glossary](Glossary)
