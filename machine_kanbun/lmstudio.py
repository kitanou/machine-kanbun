"""Minimal streaming client for LM Studio's OpenAI-compatible server (stdlib only)."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

BASE_URL = "http://localhost:1234/v1"


@dataclass
class Reply:
    text: str
    prompt_tokens: Optional[int]  # as counted by the model's own tokenizer
    completion_tokens: Optional[int]
    ttft: float  # seconds to first content token
    total: float  # seconds to completion


def chat(model: str, system: str, user: str, base_url: str = BASE_URL,
         max_tokens: int = 200, timeout: float = 300) -> Reply:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base_url + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = 0.0
    parts, usage = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if not ttft:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    total = time.perf_counter() - t0
    text = re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.S).strip()
    return Reply(text, usage.get("prompt_tokens"), usage.get("completion_tokens"), ttft, total)
