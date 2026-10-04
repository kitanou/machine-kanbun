"""Issue #20: paired sign tests, L1 variant vs L0, on the LLM end-to-end answers (warm, same questions)."""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb

from .jpbench import RES

REPS = ("sudachi-m", "sudachi-g", "sudachi-p", "ginza-d", "naive")


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def main():
    out = []
    for path in sorted(RES.glob("llm_*.jsonl")):
        d = defaultdict(dict)
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r["cold"]:
                d[(r["n"], r["rep"])][r["i"]] = r["ok"]
        for n in sorted({k[0] for k in d}):
            base = d.get((n, "L0"))
            for rep in REPS:
                cur = d.get((n, rep))
                if not base or not cur:
                    continue
                qs = [i for i in base if i in cur]
                a = sum(1 for i in qs if cur[i] and not base[i])
                b = sum(1 for i in qs if base[i] and not cur[i])
                out.append(dict(model=path.stem[4:], n=n, rep=rep, questions=len(qs), l1_only_correct=a,
                                l0_only_correct=b, p=round(sign_p(a, b), 4)))
    (RES / "signtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
