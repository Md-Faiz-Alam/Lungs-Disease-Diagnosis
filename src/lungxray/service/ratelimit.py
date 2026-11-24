"""In-memory sliding-window rate limiting for the expensive (model inference) endpoints.

Per replica, which is enough to stop scripted abuse of a public demo; Container Apps caps replicas
at 3, so the global ceiling is bounded. A shared store (e.g. Redis) would be the next step at scale.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from starlette.requests import Request

MAX_TRACKED_CLIENTS = 10_000


def parse_rate(spec: str) -> tuple[int, float]:
    """'30/600' -> 30 requests per 600 seconds."""
    count, seconds = spec.split("/", 1)
    limit, window = int(count), float(seconds)
    if limit <= 0 or window <= 0:
        raise ValueError(f"Invalid rate limit {spec!r}")
    return limit, window


def client_key(request: Request) -> str:
    # Behind Azure Container Apps ingress the original client is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """Record a hit. Returns 0 if allowed, otherwise seconds until the next slot frees up."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            if len(self._hits) > MAX_TRACKED_CLIENTS:  # drop idle clients to bound memory
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return 0.0
