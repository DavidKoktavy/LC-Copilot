"""Tool-calling that doesn't break: retries, timeouts, circuit breaking, rate limiting, idempotency.

Everything here is dependency-free and async-first so it can wrap LLM calls, vector search,
database queries and MCP tool invocations with the same, predictable semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class TransientError(Exception):
    """Errors worth retrying (HTTP 429/5xx, timeouts, connection resets)."""


class PermanentError(Exception):
    """Errors that must not be retried (validation, auth, policy violations)."""


class CircuitOpenError(PermanentError):
    """Raised immediately while a dependency's circuit is open."""


class RateLimitedError(TransientError):
    pass


# ---------------------------------------------------------------- retry + timeout
@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 4.0
    timeout_s: float | None = 30.0
    retry_on: tuple[type[BaseException], ...] = (TransientError, asyncio.TimeoutError, ConnectionError)

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with *full jitter* (AWS architecture blog pattern)."""
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return random.uniform(0, cap)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy = RetryPolicy(),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Run ``fn`` with a per-attempt timeout and jittered exponential backoff.

    Only exceptions in ``policy.retry_on`` are retried; everything else surfaces at once,
    so a validation error never burns three attempts and three model calls.
    """
    last: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            if policy.timeout_s is None:
                return await fn()
            return await asyncio.wait_for(fn(), timeout=policy.timeout_s)
        except policy.retry_on as exc:  # type: ignore[misc]
            last = exc
            if attempt == policy.attempts:
                break
            if on_retry:
                on_retry(attempt, exc)
            await asyncio.sleep(policy.backoff(attempt))
    assert last is not None
    raise last


# ---------------------------------------------------------------- circuit breaker
@dataclass
class CircuitBreaker:
    """Closed → Open after ``failure_threshold`` consecutive failures → Half-open after ``reset_after_s``."""

    name: str
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    _failures: int = 0
    _opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            return "half_open"
        return "open"

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if self.state == "open":
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            result = await fn()
        except PermanentError:
            raise  # caller bugs should not trip the breaker
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold or self.state == "half_open":
                self._opened_at = time.monotonic()
            raise
        self._failures = 0
        self._opened_at = None
        return result


# ---------------------------------------------------------------- rate limiting
@dataclass
class TokenBucket:
    """Async token bucket. ``acquire(wait=False)`` fails fast (API edge); ``wait=True`` queues (outbound calls)."""

    rate_per_s: float
    capacity: float
    _tokens: float = field(init=False)
    _updated: float = field(init=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate_per_s)
        self._updated = now

    async def acquire(self, cost: float = 1.0, wait: bool = True) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                if not wait:
                    raise RateLimitedError("rate limit exceeded")
                await asyncio.sleep((cost - self._tokens) / self.rate_per_s)


class KeyedRateLimiter:
    """One bucket per key (client id, tool name, ...)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._buckets: dict[str, TokenBucket] = {}

    async def check(self, key: str) -> None:
        bucket = self._buckets.setdefault(
            key, TokenBucket(rate_per_s=self.per_minute / 60.0, capacity=float(self.per_minute))
        )
        await bucket.acquire(wait=False)


# ---------------------------------------------------------------- idempotency
def fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class IdempotencyConflict(PermanentError):
    """Same idempotency key re-used with a different request body."""


class IdempotencyStore:
    """LRU+TTL store for Idempotency-Key replays. Swap for Redis in a multi-replica deployment.

    Concurrent duplicates are serialised with a per-key lock, so the work runs exactly once.
    """

    def __init__(self, max_items: int = 10_000, ttl_s: float = 24 * 3600) -> None:
        self._data: OrderedDict[str, tuple[float, str, Any]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self.max_items = max_items
        self.ttl_s = ttl_s

    async def run_once(self, key: str, payload: Any, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Returns ``(result, replayed)``."""
        fp = fingerprint(payload)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            hit = self._data.get(key)
            if hit and time.time() - hit[0] < self.ttl_s:
                if hit[1] != fp:
                    raise IdempotencyConflict(f"idempotency key '{key}' reused with a different payload")
                self._data.move_to_end(key)
                return hit[2], True
            result = await fn()
            self._data[key] = (time.time(), fp, result)
            while len(self._data) > self.max_items:
                old, _ = self._data.popitem(last=False)
                self._locks.pop(old, None)
            return result, False
