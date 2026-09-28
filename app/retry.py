"""Shared retry/backoff helper for every external call (NFR-2, Invariant 8, DRY).

Two entry points, one implementation:

- :func:`retryable` — the decorator every external call site uses (RSS, LLM, Telegram),
- :func:`call_with_retries` — the engine underneath, for calls that are not a plain
  function (a bound SDK method, a closure).
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

from app.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

BASE_DELAY_SECONDS = 1.0
BACKOFF_FACTOR = 2.0

# Either a literal label or a callable that derives one from the wrapped call's own
# arguments (e.g. the feed URL). It must never return secrets (Invariant 6).
Description = str | Callable[..., str]


def resolve_attempts(attempts: int | None) -> int:
    """`HTTP_MAX_RETRIES` when the caller did not pin a number (NFR-7)."""
    if attempts is not None:
        return attempts
    return get_settings().http_max_retries


def call_with_retries(
    operation: Callable[[], T],
    *,
    attempts: int,
    description: str,
    base_delay: float = BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``operation`` until it succeeds or the attempt budget is exhausted.

    ``attempts`` is the total number of attempts (the first call included), so
    ``HTTP_MAX_RETRIES=3`` means at most 3 calls. The delay between attempts grows
    exponentially (``base_delay * 2 ** (attempt - 1)``).

    The final failure is re-raised unchanged so callers can decide what to do with
    the post/feed (Invariant 8). ``description`` must never contain secrets.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts:
                logger.error(
                    "%s failed after %d attempt(s): %s", description, attempts, exc
                )
                raise
            delay = base_delay * (BACKOFF_FACTOR ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")  # pragma: no cover


def _describe(description: Description | None, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Label for the log lines: explicit text, a call-derived label, or the name."""
    if callable(description):
        return description(*args, **kwargs)
    return description or func.__qualname__


def retryable(
    *,
    attempts: int | None = None,
    description: Description | None = None,
    base_delay: float = BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a call so it runs with bounded retries and exponential backoff.

    ``attempts=None`` (the default) reads ``HTTP_MAX_RETRIES`` from settings at call
    time, so the limit stays configurable without threading it through every
    signature (NFR-7, AGENTS.md section 11). ``sleep`` defaults to ``time.sleep``,
    resolved at call time so tests can replace it.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retries(
                lambda: func(*args, **kwargs),
                attempts=resolve_attempts(attempts),
                description=_describe(description, func, args, kwargs),
                base_delay=base_delay,
                sleep=time.sleep if sleep is None else sleep,
            )

        return wrapper

    return decorator
