"""Minimal streaming client for LM Studio's OpenAI-compatible server (stdlib only)."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

BASE_URL = "http://localhost:1234/v1"


class Stalled(RuntimeError):
    """The server stopped responding (wall-clock deadline exceeded)."""


@dataclass
class Reply:
    text: str
    prompt_tokens: Optional[int]  # as counted by the model's own tokenizer
    completion_tokens: Optional[int]
    ttft: float  # seconds to first token of any kind
    total: float  # seconds to completion


def chat(model: str, system: str, user: str, base_url: str = BASE_URL,
         max_tokens: int = 300, timeout: float = 300, extra: Optional[dict] = None,
         deadline: float = 600) -> Reply:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **(extra or {}),
    }).encode()
    req = urllib.request.Request(base_url + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = 0.0
    parts, usage = [], {}
    stalled = threading.Event()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # LM Studio sends keep-alive bytes while stuck, which defeat the socket timeout,
        # so a wall-clock deadline closes the connection from a watchdog thread.
        def _abort():
            stalled.set()
            r.close()

        timer = threading.Timer(deadline, _abort)
        timer.daemon = True
        timer.start()
        try:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                if ev.get("error"):  # e.g. context overflow arrives as an SSE error event
                    raise ValueError(f"server error: {ev['error']}")
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    delta = ch.get("delta") or {}
                    # TTFT = first token of any kind (reasoning or content) = prompt-processing latency
                    if not ttft and (delta.get("content") or delta.get("reasoning_content")):
                        ttft = time.perf_counter() - t0
                    if delta.get("content"):
                        parts.append(delta["content"])
        except (OSError, ValueError, AttributeError):
            if not stalled.is_set():
                raise
        finally:
            timer.cancel()
    if stalled.is_set():
        raise Stalled(f"no completion within {deadline}s")
    total = time.perf_counter() - t0
    if not parts or usage.get("prompt_tokens") is None:
        raise ValueError(f"empty response (usage={usage}, chunks={len(parts)})")
    text = re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.S).strip()
    return Reply(text, usage.get("prompt_tokens"), usage.get("completion_tokens"), ttft, total)
