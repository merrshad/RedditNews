"""Tests for the shared retry engine (`app/retry.py`) — NFR-2 / Invariant 8.

Every external call in the project goes through this module (RSS, LLM, Telegram), so the
attempt semantics, the backoff growth and the "do not retry a permanent failure" rule are
tested here once instead of being implied three times over in the call-site tests.

Phase 6 added the awaitable twin of the engine (`call_with_retries_async`) and taught the
decorator to recognise coroutine functions, so both halves are pinned here: a wait that
blocked the event loop would stall every button press handled at the same time.
"""

from __future__ import annotations

import pytest

from app.retry import (
    BACKOFF_FACTOR,
    BASE_DELAY_SECONDS,
    PermanentError,
    call_with_retries,
    call_with_retries_async,
    resolve_attempts,
    retryable,
)
from tests.conftest import run


def _recorder() -> tuple[list[int], list[float]]:
    """A pair of lists: attempts made, and the delays the engine asked for."""
    return [], []


def _operation(calls: list[int], *, fail_times: int, error: Exception | None = None):
    """A callable that raises ``error`` (default: RuntimeError) for its first calls."""
    error = error or RuntimeError("boom")

    def operation() -> str:
        calls.append(1)
        if len(calls) <= fail_times:
            raise error
        return "ok"

    return operation


# --- attempt semantics ------------------------------------------------------------


def test_attempts_is_the_total_number_of_calls_including_the_first() -> None:
    """The documented meaning of `HTTP_MAX_RETRIES` (AGENTS.md section 11)."""
    calls, slept = _recorder()

    with pytest.raises(RuntimeError):
        call_with_retries(
            _operation(calls, fail_times=99), attempts=3, description="t", sleep=slept.append
        )

    assert len(calls) == 3


def test_a_call_that_succeeds_on_the_last_attempt_returns_its_value() -> None:
    calls, slept = _recorder()

    result = call_with_retries(
        _operation(calls, fail_times=2), attempts=3, description="t", sleep=slept.append
    )

    assert result == "ok"
    assert len(calls) == 3


def test_a_happy_call_is_not_retried_and_does_not_sleep() -> None:
    calls, slept = _recorder()

    assert call_with_retries(
        _operation(calls, fail_times=0), attempts=3, description="t", sleep=slept.append
    ) == "ok"

    assert (len(calls), slept) == (1, [])


def test_a_budget_below_one_is_a_programming_error() -> None:
    calls, slept = _recorder()

    with pytest.raises(ValueError, match="attempts must be >= 1"):
        call_with_retries(
            _operation(calls, fail_times=0), attempts=0, description="t", sleep=slept.append
        )

    assert calls == []  # nothing was attempted


# --- backoff ----------------------------------------------------------------------


def test_the_delay_grows_exponentially_between_attempts() -> None:
    calls, slept = _recorder()

    with pytest.raises(RuntimeError):
        call_with_retries(
            _operation(calls, fail_times=99), attempts=4, description="t", sleep=slept.append
        )

    assert slept == [
        BASE_DELAY_SECONDS,
        BASE_DELAY_SECONDS * BACKOFF_FACTOR,
        BASE_DELAY_SECONDS * BACKOFF_FACTOR**2,
    ]  # no sleep after the final attempt


def test_a_custom_base_delay_is_used_as_given() -> None:
    calls, slept = _recorder()

    with pytest.raises(RuntimeError):
        call_with_retries(
            _operation(calls, fail_times=99),
            attempts=2,
            description="t",
            base_delay=0.25,
            sleep=slept.append,
        )

    assert slept == [0.25]


# --- permanent failures (the phase-4 rule) ---------------------------------------


def test_a_permanent_error_is_not_retried() -> None:
    """A rejected token or key answers identically forever, so the budget is not spent."""
    calls, slept = _recorder()

    with pytest.raises(PermanentError, match="rejected"):
        call_with_retries(
            _operation(calls, fail_times=99, error=PermanentError("rejected")),
            attempts=5,
            description="t",
            sleep=slept.append,
        )

    assert (len(calls), slept) == (1, [])


def test_the_original_exception_type_survives_exhaustion() -> None:
    """Callers match on their own error types, so the engine must not wrap them."""
    calls, slept = _recorder()

    class CustomError(RuntimeError):
        pass

    with pytest.raises(CustomError):
        call_with_retries(
            _operation(calls, fail_times=99, error=CustomError("nope")),
            attempts=1,
            description="t",
            sleep=slept.append,
        )


# --- configuration and the decorator ---------------------------------------------


def test_the_default_budget_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR-7: `HTTP_MAX_RETRIES` is read at call time, never baked in at import."""
    monkeypatch.setenv("HTTP_MAX_RETRIES", "4")

    assert resolve_attempts(None) == 4
    assert resolve_attempts(2) == 2  # an explicit number always wins


def test_the_decorator_retries_and_labels_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "2")

    @retryable(description=lambda name: f"fetch {name}")
    def flaky(name: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first one is always bad")
        return f"got {name}"

    assert flaky("feed-a") == "got feed-a"
    assert len(calls) == 2


# --- the awaitable twin (phase 6) --------------------------------------------------


def _async_recorder() -> tuple[list[int], list[float]]:
    return [], []


def _async_operation(calls: list[int], *, fail_times: int, error: Exception | None = None):
    """A coroutine function that raises ``error`` for its first calls."""
    error = error or RuntimeError("boom")

    async def operation() -> str:
        calls.append(1)
        if len(calls) <= fail_times:
            raise error
        return "ok"

    return operation


def test_the_async_engine_retries_until_the_budget_is_spent() -> None:
    calls, slept = _async_recorder()

    with pytest.raises(RuntimeError):
        run(
            call_with_retries_async(
                _async_operation(calls, fail_times=99),
                attempts=3,
                description="t",
                sleep=_record_async_sleep(slept),
            )
        )

    assert len(calls) == 3


def test_the_async_engine_returns_a_late_success_and_grows_the_delay() -> None:
    """The backoff schedule must be the same as the synchronous engine's."""
    calls, slept = _async_recorder()

    result = run(
        call_with_retries_async(
            _async_operation(calls, fail_times=2),
            attempts=3,
            description="t",
            sleep=_record_async_sleep(slept),
        )
    )

    assert result == "ok"
    assert slept == [BASE_DELAY_SECONDS, BASE_DELAY_SECONDS * BACKOFF_FACTOR]


def test_a_permanent_error_ends_the_async_engine_at_once() -> None:
    calls, slept = _async_recorder()

    with pytest.raises(PermanentError, match="rejected"):
        run(
            call_with_retries_async(
                _async_operation(calls, fail_times=99, error=PermanentError("rejected")),
                attempts=5,
                description="t",
                sleep=_record_async_sleep(slept),
            )
        )

    assert (len(calls), slept) == (1, [])


def test_the_async_engine_validates_the_attempt_budget() -> None:
    calls, slept = _async_recorder()

    with pytest.raises(ValueError, match="attempts must be >= 1"):
        run(
            call_with_retries_async(
                _async_operation(calls, fail_times=0),
                attempts=0,
                description="t",
                sleep=_record_async_sleep(slept),
            )
        )

    assert calls == []


def test_the_decorator_retries_a_coroutine_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """One decorator covers both worlds: a coroutine function gets the async engine."""
    calls: list[int] = []
    monkeypatch.setattr("app.retry.async_sleep", _anonymous_async_sleep)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "2")

    @retryable(description=lambda name: f"fetch {name}")
    async def flaky(name: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first one is always bad")
        return f"got {name}"

    assert run(flaky("feed-a")) == "got feed-a"
    assert len(calls) == 2


def test_the_async_engine_waits_without_blocking_the_event_loop() -> None:
    """The default wait is ``asyncio.sleep``, never ``time.sleep``.

    A blocking wait would freeze every other task in the worker — including the Telegram
    update loop — for the whole backoff schedule.
    """
    waited: list[float] = []

    async def _operation() -> str:
        if len(waited) < 1:
            raise RuntimeError("flaky")
        return "ok"

    async def _sleep(seconds: float) -> None:
        waited.append(seconds)

    assert run(
        call_with_retries_async(
            _operation, attempts=2, description="t", sleep=_sleep
        )
    ) == "ok"

    assert waited == [BASE_DELAY_SECONDS]


def _record_async_sleep(slept: list[float]):
    """A capture-only wait for the async engine."""

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    return _sleep


async def _anonymous_async_sleep(_seconds: float) -> None:
    return None
