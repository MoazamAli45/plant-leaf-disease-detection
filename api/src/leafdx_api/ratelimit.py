"""A small in-memory sliding-window rate limiter (per client address).

Good enough for a single-instance demo deployment. Behind a proxy, the client address comes
from the first ``X-Forwarded-For`` entry.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, per_minute: int, window_s: float = 60.0) -> None:
        self.per_minute = per_minute
        self.window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """(allowed, seconds until the next request would be allowed)."""
        if self.per_minute <= 0:
            return True, 0.0
        t = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and t - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, self.window - (t - q[0])
            q.append(t)
            return True, 0.0
