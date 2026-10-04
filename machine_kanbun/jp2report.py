"""Report for Issues #25 / #30 (context ablations), plus helpers shared by the other Issue #26-#31 reports."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List

OUT = Path(__file__).parent.parent / "results" / "jp2"


def rows_of(pattern: str) -> List[Dict]:
    out = []
    for f in sorted(OUT.glob(pattern)):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if "error" not in r:
                out.append(r)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def svg_scatter(points, title: str, xlabel: str, ylabel: str, path: Path, hull_label=True):
    """points: [(name, x, y, kind)]; kind picks the colour; the Pareto frontier (max y for min x) is drawn."""
    W, H, M = 860, 520, 66
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x0, x1, y0, y1 = min(xs) - 0.03, max(xs) + 0.03, max(0, min(ys) - 0.05), min(1.0, max(ys) + 0.05)
    sx = lambda v: M + (v - x0) / ((x1 - x0) or 1) * (W - 2 * M - 120)
    sy = lambda v: H - M - (v - y0) / ((y1 - y0) or 1) * (H - 2 * M)
    cols = {"L0": "#c0392b", "surgical": "#e67e22", "l1": "#2471a3", "ablation": "#1e8449", "control": "#7f8c8d"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">', f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
         f'<line x1="{M}" y1="{H - M}" x2="{W - M - 120}" y2="{H - M}" stroke="#444"/><line x1="{M}" y1="{M}" x2="{M}" y2="{H - M}" stroke="#444"/>',
         f'<text x="{(W - 120) / 2}" y="{H - 14}" text-anchor="middle">{xlabel}</text>', f'<text x="16" y="{H / 2}" transform="rotate(-90 16 {H / 2})" text-anchor="middle">{ylabel}</text>']
    for t in range(5):
        o.append(f'<text x="{M + (W - 2 * M - 120) * t / 4:.0f}" y="{H - M + 14}" text-anchor="middle">{x0 + (x1 - x0) * t / 4:.2f}</text>')
        o.append(f'<text x="{M - 6}" y="{H - M - (H - 2 * M) * t / 4 + 4:.0f}" text-anchor="end">{y0 + (y1 - y0) * t / 4:.2f}</text>')
    front, best = [], -1
    for n_, x, y, k in sorted(points, key=lambda p: (p[1], -p[2])):
        if y > best:
            front.append((x, y))
            best = y
    o.append('<polyline fill="none" stroke="#999" stroke-dasharray="5 3" points="' + " ".join(f"{sx(x):.0f},{sy(y):.0f}" for x, y in front) + '"/>')
    for n_, x, y, k in points:
        o.append(f'<circle cx="{sx(x):.0f}" cy="{sy(y):.0f}" r="4" fill="{cols.get(k, "#444")}"/><text x="{sx(x) + 6:.0f}" y="{sy(y) - 5:.0f}" font-size="9">{n_}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def cond_table(rows: List[Dict], group: str):
    by = defaultdict(list)
    for r in rows:
        if r["group"] == group:
            by[r["cond"]].append(r)
    return by


def summarize(rs: List[Dict]) -> Dict:
    st = [r for r in rs if r["kind"] == "status"]
    tm = [r for r in rs if r["kind"] == "time"]
    cold = next(r for r in rs if r["cold"])
    return dict(n_status=len(st), acc=sum(r["ok"] for r in st) / len(st), ci=wilson(sum(r["ok"] for r in st), len(st)), time_acc=sum(r["ok"] for r in tm) / max(1, len(tm)),
                ctx=cold["ctx_tokens"], ttft=cold["ttft"], warm=sum(r["total"] for r in rs if not r["cold"]) / max(1, len(rs) - 1))


def paired(by: Dict[str, List[Dict]], a: str, b: str):
    """questions answered correctly only under a / only under b (status questions, same question set)."""
    ra = {r["qi"]: r["ok"] for r in by[a] if r["kind"] == "status"}
    rb = {r["qi"]: r["ok"] for r in by[b] if r["kind"] == "status"}
    x = sum(1 for i in ra if ra[i] and not rb.get(i, False))
    y = sum(1 for i in ra if rb.get(i, False) and not ra[i])
    return x, y, sign_p(x, y)


def section_ctx(model_key: str, P):
    rows = rows_of(f"ctx_{model_key}.jsonl")
    if not rows:
        return
    groups = sorted({r["group"] for r in rows})
    csvrows = []
    for g in groups:
        by = cond_table(rows, g)
        base = "L0" if "L0" in by else next((c for c in by if c.startswith("L0@")), None)
        P(f"\n### グループ {g}({model_key})\n")
        P("| 条件 | 文脈tok | トークン比 | 状態QA | 95%CI | 時期QA | コールドTTFT(s) | ウォーム(s/問) | 精度/1k tok | L0に対する符号検定(のみ正解 L1:L0, p) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        s0 = summarize(by[base]) if base else None
        for c, rs in by.items():
            s = summarize(rs)
            sg = ""
            if base and c != base:
                x, y, p = paired(by, c, base)
                sg = f"{x}:{y}, p={p:.3f}"
            ratio = s["ctx"] / s0["ctx"] if s0 else float("nan")
            P(f"| {c} | {s['ctx']} | {ratio:.2f} | {s['acc']:.1%} | {s['ci'][0]:.2f}-{s['ci'][1]:.2f} | {s['time_acc']:.0%} | {s['ttft']:.0f} | {s['warm']:.1f} | {s['acc'] / (s['ctx'] / 1000):.3f} | {sg} |")
            csvrows.append(dict(model=model_key, group=g, cond=c, ctx_tokens=s["ctx"], token_ratio=round(ratio, 3), status_acc=round(s["acc"], 4), time_acc=round(s["time_acc"], 4),
                                cold_ttft=round(s["ttft"], 1), warm_s=round(s["warm"], 2)))
        # by status category and position
        P("\n状態別の正答率(状態QA):\n")
        sts = ["DONE", "NOT_DONE", "PLANNED", "WANTED", "UNDECIDED", "HEARSAY", "POSSIBLE"]
        P("| 条件 | " + " | ".join(sts) + " |")
        P("|---|" + "---|" * len(sts))
        for c, rs in by.items():
            cells = []
            for s_ in sts:
                q = [r for r in rs if r["kind"] == "status" and r["status"] == s_]
                cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f}" if q else "-")
            P(f"| {c} | " + " | ".join(cells) + " |")
        P("\n正解メモリの位置別の正答率(状態QA, 位置ビン0=先頭〜4=末尾):\n")
        P("| 条件 | " + " | ".join(f"bin{b}" for b in range(5)) + " |")
        P("|---|---|---|---|---|---|")
        for c, rs in by.items():
            cells = []
            for b in range(5):
                q = [r for r in rs if r["kind"] == "status" and r["pos_bin"] == b]
                cells.append(f"{sum(r['ok'] for r in q) / len(q):.2f}" if q else "-")
            P(f"| {c} | " + " | ".join(cells) + " |")
    with (OUT / "ctx_summary.csv").open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csvrows[0]))
        if fh.tell() == 0:
            w.writeheader()
        w.writerows(csvrows)


def pareto(model_key: str):
    rows = [r for r in rows_of(f"ctx_{model_key}.jsonl") if r["group"].startswith("ablation")]
    if not rows:
        return
    by = cond_table(rows, rows[0]["group"])
    base = summarize(by["L0"])
    kind = lambda c: "L0" if c == "L0" else "control" if "control" in c or c == "naive" else "surgical" if c in ("no-filler", "no-polite", "no-particle", "state-normalized", "no-chitchat(oracle)", "surface-stripped", "surface-stripped+state") else "l1" if c in ("sudachi-m", "sudachi-g", "content-only") else "ablation"
    pts = []
    for c, rs in by.items():
        s = summarize(rs)
        pts.append((c, s["ctx"] / base["ctx"], s["acc"], kind(c)))
    svg_scatter(pts, f"トークン比 vs 状態QA正答率({model_key}, N=300)", "トークン比(L0=1.0)", "状態QA正答率", OUT / f"chart_pareto_{model_key}.svg")
