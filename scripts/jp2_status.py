"""Summarise the Issue #25-#31 result files for the 30-minute progress line."""
import json
import glob
import os
import collections

base = os.path.join(os.path.dirname(__file__), "..", "results", "jp2")
parts = []
for f in sorted(glob.glob(os.path.join(base, "ctx_*.jsonl"))):
    conds = collections.OrderedDict()
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if "cond" in r:
            conds[(r["group"], r["cond"])] = 1
    last = list(conds)[-1] if conds else None
    parts.append(f"ctx実験 {os.path.basename(f)[4:-6]}: {len(conds)}条件完了(最新 {last[0]}/{last[1] if last else ''})")
for name in ("dual_index.json", "tokaware.json", "natural_chats.jsonl", "ml_qa.jsonl", "attn.json"):
    p = os.path.join(base, name)
    if os.path.exists(p):
        parts.append(f"{name} あり")
print("; ".join(parts))
