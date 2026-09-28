"""Tests for the shared retry engine (`app/retry.py`) — NFR-2 / Invariant 8.

Every external call in the project goes through this module (RSS, LLM, Telegram), so the
attempt semantics, the backoff growth and the "do not retry a permanent failure" rule are
tested here once instead of being implied three times over in the call-site tests.
"""

from __future__ import annotations

import pytest

from app.retry import (
    BACKOFF_FACTOR,
    BASE_DELAY_SECONDS,
    PermanentError,
    call_with_retries,
    resolve_attempts,
    retryable,
)


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
