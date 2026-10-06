"""Issue #57 report: per budget, NF-short vs SCF-L1 / SeCF-L1 (all cut oldest-first): correct / abstain / wrong rates, by target position."""
from __future__ import annotations

import statistics as st
from collections import defaultdict

import numpy as np

from .jpchat import OUT
from .jpchatbudget import BUDGETS, POS
from .jpchatreport import boot_ci, jl, perm_p

REPS = (("nf", "NF-short"), ("scf", "SCF-L1"), ("secf1", "SeCF-L1"))


def main():
    R = jl(OUT / "budget_google_gemma-4-12b.jsonl")
    if not R:
        print("no data")
        return
    by = defaultdict(lambda: defaultdict(list))   # cond -> sid -> list of rows
    for r in R:
        by[r["cond"]][r["sid"]].append(r)
    sids = sorted(set.intersection(*[set(d) for d in by.values()]))
    rate = lambda c, f: {s: float(np.mean([f(r) for r in by[c][s]])) for s in sids}
    corr = lambda r: r["outcome"] == "correct"
    L = ["# Issue #57 同一 token budget での NF-short と SCF-L1 / SeCF-L1(古い往復から切り捨て)\n",
         f"35 シナリオ(完了 {len(sids)})× 2 並び。履歴 24 往復、目標の事実の位置 {list(POS)} 往復目。budget = 全 NF の o200k token 数の倍率。どの表現も、収まるまで古い往復から切り捨てる。8 択(状態 7 + 「会話にない」)。誤答 = 別の状態を選んだ(誤帰属)、棄権 = 「会話にない」。\n"]
    ref = rate("nf_full", corr)
    L += ["## 1. 正答率・棄権率・誤答率\n", f"参照: 全 NF(切り捨てなし)の正答率 {np.mean(list(ref.values())):.3f}\n",
          "| budget | 表現 | 正答率 | 棄権率 | 誤答率 | 保持往復数 | 目標の事実を保持 | prompt tok(全 NF 比) | vs NF-short 差[95%CI], p |", "|---|---|---|---|---|---|---|---|---|"]
    full_tok = st.mean(r["prompt_tokens"] for r in R if r["cond"] == "nf_full")
    for b in BUDGETS:
        base = rate(f"nf@{b}", corr)
        for rep, nm in REPS:
            c = f"{rep}@{b}"
            if c not in by:
                continue
            rows = [r for s in sids for r in by[c][s]]
            a = rate(c, corr)
            d = [a[s] - base[s] for s in sids]
            lo, hi = boot_ci(d)
            cell = "-" if rep == "nf" else f"{np.mean(d) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}], p={perm_p(d):.3f}"
            L.append(f"| {b} | {nm} | {np.mean(list(a.values())):.3f} | {np.mean([r['outcome'] == 'abstain' for r in rows]):.3f} | {np.mean([r['outcome'] == 'wrong' for r in rows]):.3f} | "
                     f"{st.mean(r['n_kept'] for r in rows):.1f} | {np.mean([r['fact_kept'] for r in rows]):.2f} | {st.mean(r['prompt_tokens'] for r in rows):.0f} ({st.mean(r['prompt_tokens'] for r in rows) / full_tok:.2f}) | {cell} |")
    L += ["\n## 2. 目標の事実の位置別の正答率(古い = 位置 0〜8 往復目、新しい = 12〜20 往復目)\n", "| budget | 表現 | 古い位置 | 新しい位置 |", "|---|---|---|---|"]
    for b in BUDGETS:
        for rep, nm in REPS:
            c = f"{rep}@{b}"
            if c not in by:
                continue
            old = [r["outcome"] == "correct" for s in sids for r in by[c][s] if r["pos"] <= 8]
            new = [r["outcome"] == "correct" for s in sids for r in by[c][s] if r["pos"] >= 12]
            L.append(f"| {b} | {nm} | {np.mean(old):.3f} (n={len(old)}) | {np.mean(new):.3f} (n={len(new)}) |")
    L += ["\n## 3. 状態別の正答率(budget 0.70)\n", "| 表現 | DONE | NOT_DONE | PLANNED | WANTED | UNDECIDED | HEARSAY | POSSIBLE |", "|---|---|---|---|---|---|---|---|"]
    for rep, nm in REPS:
        c = f"{rep}@0.7"
        if c in by:
            L.append(f"| {nm} | " + " | ".join(f"{np.mean([r['outcome'] == 'correct' for s in sids for r in by[c][s] if r['status'] == st_]):.2f}" for st_ in ("DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE")) + " |")
    L += ["\n## 4. 保持と表現の分離(目標の事実が履歴に残っている場合の正答率)\n", "| budget | 表現 | 事実が残った件数 | 残った場合の正答率 | 残った場合の誤答率 | 残らなかった場合の棄権率 |", "|---|---|---|---|---|---|"]
    for b in BUDGETS:
        for rep, nm in REPS:
            rows = [r for r in R if r["cond"] == f"{rep}@{b}"]
            k = [r for r in rows if r["fact_kept"]]
            nk = [r for r in rows if not r["fact_kept"]]
            f = lambda X, o: f"{np.mean([r['outcome'] == o for r in X]):.3f}" if X else "-"
            L.append(f"| {b} | {nm} | {len(k)}/{len(rows)} | {f(k, 'correct')} | {f(k, 'wrong')} | {f(nk, 'abstain')} |")
    (OUT / "budget_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
