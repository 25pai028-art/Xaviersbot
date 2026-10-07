"""Runtime guardrail: how many questions one visitor may send (per minute and per day).

Kept in memory only, keyed by the visitor's IP address and never written anywhere (no chat logs, see the
privacy decision). A restart forgets it, which is fine for its purpose: stopping floods and scripts that
would block everyone else, since the AI answers one question at a time.
"""
from __future__ import annotations

import threading
import time
from collections import deque

MINUTE, DAY = 60.0, 86_400.0
_PRUNE_AT = 50_000  # visitors remembered before old entries are swept


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, per_minute: int, per_day: int, now: float | None = None) -> str | None:
        """Record a question from `key`. Returns None if allowed, else "minute" or "day" (the limit hit).
        A limit of 0 means no limit."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= DAY:
                hits.popleft()
            if per_day and len(hits) >= per_day:
                return "day"
            if per_minute and sum(1 for t in reversed(hits) if now - t < MINUTE) >= per_minute:
                return "minute"
            hits.append(now)
            if len(self._hits) > _PRUNE_AT:
                self._sweep(now)
            return None

    def _sweep(self, now: float) -> None:
        for k in [k for k, h in self._hits.items() if not h or now - h[-1] >= DAY]:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


chat_limiter = RateLimiter()
