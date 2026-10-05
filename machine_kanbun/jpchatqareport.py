"""Issue #49 report: status-recall accuracy per condition (per-scenario mean over 3 option orders), paired tests, per-status, confusion."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import jpdata
from .jpchat import OUT
from .jpchatqa import CONDS
from .jpchatreport import boot_ci, perm_p

MODAL = ["NOT_DONE", "UNDECIDED", "HEARSAY", "POSSIBLE"]


def rows(variant):
    p = OUT / f"qa_{variant}_google_gemma-4-12b.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def main():
    L = ["# Issue #49 Chat 履歴の想起精度(状態の 7 択 QA、決定的採点)\n",
         "回答 gemma-4-12b(温度 0)。正答 = 選択肢の記号の一致(ランダム 14%)。シナリオごとに 3 通りの選択肢の並びを回して平均。差はシナリオ間のペア比較(置換検定、ブートストラップ 95%CI)。\n"]
    csv_rows = []
    for variant, label in (("short", "短い履歴(8 往復、35 シナリオ)"), ("long", "長い履歴(24 往復、21 シナリオ)")):
        R = rows(variant)
        if not R:
            continue
        by = defaultdict(lambda: defaultdict(list))
        tok, nex = defaultdict(list), defaultdict(list)
        for r in R:
            by[r["cond"]][r["sid"]].append(r["pred"] == r["gold"])
            tok[r["cond"]].append(r["prompt_tokens"])
            nex[r["cond"]].append(r["n_ex"])
        acc = {c: {s: float(np.mean(v)) for s, v in d.items()} for c, d in by.items()}
        sids = sorted(set.intersection(*[set(d) for d in acc.values()]))
        pt0 = st.mean(tok["A_base"])
        L += [f"## {label}\n", "| 条件 | n | 正答率 | prompt tok(圧縮率) | 残した往復数 | vs A 差[95%CI], p | vs N(トークン一致 NF) 差[95%CI], p |", "|---|---|---|---|---|---|---|"]
        for c in CONDS:
            if c not in acc:
                continue
            a = np.array([acc[c][s] for s in sids])
            cells = []
            for ref in ("A_base", "N_nf_tokmatch"):
                if c == ref or ref not in acc:
                    cells.append("-")
                    continue
                d = list(a - np.array([acc[ref][s] for s in sids]))
                lo, hi = boot_ci(d)
                cells.append(f"{np.mean(d) * 100:+.1f}pt [{lo * 100:+.1f},{hi * 100:+.1f}], p={perm_p(d):.3f}")
            L.append(f"| {c} | {len(sids)} | {a.mean():.3f} | {st.mean(tok[c]):.0f} ({st.mean(tok[c]) / pt0:.2f}) | {st.mean(nex[c]):.1f} | {cells[0]} | {cells[1]} |")
            csv_rows.append(dict(variant=variant, cond=c, acc=round(float(a.mean()), 4), tokens=round(st.mean(tok[c]), 1), ratio=round(st.mean(tok[c]) / pt0, 3)))
        L += ["\n### 状態別の正答率\n", "| 条件 | " + " | ".join(jpdata.STATUSES) + " | モダリティ系4状態 |", "|---|" + "---|" * (len(jpdata.STATUSES) + 1)]
        for c in CONDS:
            if c not in by:
                continue
            per = {s: [x["pred"] == x["gold"] for x in R if x["cond"] == c and x["status"] == s] for s in jpdata.STATUSES}
            modal = [v for s in MODAL for v in per[s]]
            L.append(f"| {c} | " + " | ".join(f"{np.mean(per[s]):.2f}" if per[s] else "-" for s in jpdata.STATUSES) + f" | {np.mean(modal):.2f} |")
        L += ["\n### 誤答の行き先(E_secf と A_base、正解 → 選んだ状態の上位 5 件)\n"]
        for c in ("A_base", "E_secf", "B_scf"):
            conf = defaultdict(int)
            for x in R:
                if x["cond"] == c and x["pred"] != x["gold"]:
                    conf[(x["status"], x["pred_status"])] += 1
            top = sorted(conf.items(), key=lambda kv: -kv[1])[:5]
            L.append(f"- **{c}**: " + ", ".join(f"{a}→{b}:{n}" for (a, b), n in top))
        L.append("")
    (OUT / "qa_report.md").write_text("\n".join(L), encoding="utf-8")
    if csv_rows:
        with (OUT / "qa_results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
