"""Best-effort memory sampling on macOS (LM Studio exposes no KV-cache / memory API).

`lmstudio_rss_mb`: summed RSS of processes belonging to LM Studio.
`used_mb`: system-wide (active + wired + compressed) pages; captures unified/GPU memory
that RSS misses. Both are coarse: use deltas between a baseline and the request peak.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Dict


def _vm_used_mb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))
    return (g("Pages active") + g("Pages wired down") + g("Pages occupied by compressor")) * page / 1048576


def _lms_rss_mb() -> float:
    out = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True).stdout
    tot = 0
    for line in out.splitlines():
        if "LM Studio" in line or ".lmstudio" in line:
            tot += int(line.split(None, 1)[0])
    return tot / 1024


def snapshot() -> Dict[str, float]:
    return {"rss_mb": _lms_rss_mb(), "used_mb": _vm_used_mb()}


class Peak:
    """Context manager sampling memory every 0.5 s; `.base` before, `.peak` during."""

    def __enter__(self):
        self.base = snapshot()
        self.peak = dict(self.base)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.5):
            s = snapshot()
            for k in self.peak:
                self.peak[k] = max(self.peak[k], s[k])

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        s = snapshot()
        for k in self.peak:
            self.peak[k] = max(self.peak[k], s[k])
