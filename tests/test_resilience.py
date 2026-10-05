import asyncio

import pytest

from lc_copilot.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    IdempotencyConflict,
    IdempotencyStore,
    PermanentError,
    RateLimitedError,
    RetryPolicy,
    TokenBucket,
    TransientError,
    call_with_retry,
)
from tests.conftest import run

FAST = RetryPolicy(attempts=3, base_delay_s=0.001, max_delay_s=0.002, timeout_s=0.5)


def test_retries_transient_then_succeeds():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("503")
        return "ok"

    assert run(call_with_retry(flaky, FAST)) == "ok"
    assert len(calls) == 3


def test_never_retries_permanent_errors():
    calls = []

    async def bad():
        calls.append(1)
        raise PermanentError("400 invalid request")

    with pytest.raises(PermanentError):
        run(call_with_retry(bad, FAST))
    assert len(calls) == 1


def test_timeout_is_retried_then_raised():
    async def slow():
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        run(call_with_retry(slow, RetryPolicy(attempts=2, base_delay_s=0.001, timeout_s=0.02)))


def test_circuit_opens_after_threshold_and_fails_fast():
    cb = CircuitBreaker("llm", failure_threshold=2, reset_after_s=60)

    async def boom():
        raise TransientError("down")

    async def scenario():
        for _ in range(2):
            with pytest.raises(TransientError):
                await cb.call(boom)
        assert cb.state == "open"
        with pytest.raises(CircuitOpenError):
            await cb.call(boom)

    run(scenario())


def test_circuit_half_open_recovers():
    cb = CircuitBreaker("llm", failure_threshold=1, reset_after_s=0.01)

    async def scenario():
        with pytest.raises(TransientError):
            await cb.call(lambda: _raise(TransientError("x")))
        await asyncio.sleep(0.02)
        assert cb.state == "half_open"
        assert await cb.call(_ok) == "ok"
        assert cb.state == "closed"

    run(scenario())


async def _ok():
    return "ok"


async def _raise(e):
    raise e


def test_token_bucket_fail_fast():
    async def scenario():
        b = TokenBucket(rate_per_s=0.001, capacity=2)
        await b.acquire(wait=False)
        await b.acquire(wait=False)
        with pytest.raises(RateLimitedError):
            await b.acquire(wait=False)

    run(scenario())


def test_idempotency_replays_and_detects_conflicts():
    store = IdempotencyStore()
    calls = []

    async def work():
        calls.append(1)
        await asyncio.sleep(0.01)
        return {"id": len(calls)}

    async def scenario():
        # concurrent duplicates execute the side effect exactly once
        r = await asyncio.gather(*[store.run_once("k1", {"a": 1}, work) for _ in range(5)])
        assert len(calls) == 1
        assert [x[1] for x in r].count(False) == 1  # one original, four replays
        with pytest.raises(IdempotencyConflict):
            await store.run_once("k1", {"a": 2}, work)

    run(scenario())
