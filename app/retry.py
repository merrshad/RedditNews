"""Shared retry/backoff helper for every external call (NFR-2, Invariant 8, DRY).

The whole worker runs inside one asyncio event loop (phase 6), so the module offers the
same engine twice: :func:`call_with_retries` for a synchronous callable and
:func:`call_with_retries_async` for a coroutine function. Both keep identical semantics —
``attempts`` is the total number of attempts, the delay grows exponentially, and a
:class:`PermanentError` ends the loop immediately.

- :func:`retryable` — the decorator every external call site uses (RSS, LLM, Telegram).
  It detects whether the callable it wraps is a coroutine function and applies the
  matching engine, so one decorator covers both worlds.
- :class:`PermanentError` — the marker a call site raises to opt out of retrying at all
  (a rejected token or API key never becomes valid on the second attempt).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

from app.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

BASE_DELAY_SECONDS = 1.0
BACKOFF_FACTOR = 2.0


class PermanentError(RuntimeError):
    """A failure retrying cannot fix, so the attempt budget must not be spent on it.

    A call site raises this when the remote end already said the *request* is wrong
    rather than the attempt unlucky — a rejected bot token, an unknown chat, a 404 feed,
    an invalid API key. Every attempt would fail identically, so
    :func:`call_with_retries` re-raises it at once instead of sleeping through the
    remaining attempts. Measured in a real run (phase 4): an invalid key cost ~6s per
    post, ~3 minutes per 25-post cycle, forever.
    """

# Either a literal label or a callable that derives one from the wrapped call's own
# arguments (e.g. the feed URL). It must never return secrets (Invariant 6).
Description = str | Callable[..., str]


def resolve_attempts(attempts: int | None) -> int:
    """`HTTP_MAX_RETRIES` when the caller did not pin a number (NFR-7)."""
    if attempts is not None:
        return attempts
    return get_settings().http_max_retries


def async_sleep(seconds: float) -> Awaitable[None]:
    """Awaitable wait, in one place so tests can replace it without touching asyncio.

    ``retryable`` resolves this at call time (never as a default argument value), which is
    what lets a test monkeypatch ``app.retry.async_sleep`` and run the whole backoff
    schedule instantly (NFR-8).
    """
    return asyncio.sleep(seconds)


def _backoff_delay(attempt: int, base_delay: float) -> float:
    """Delay before the attempt that follows ``attempt`` — exponential, no jitter."""
    return base_delay * (BACKOFF_FACTOR ** (attempt - 1))


def _validate_attempts(attempts: int) -> None:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")


def call_with_retries(
    operation: Callable[[], T],
    *,
    attempts: int,
    description: str,
    base_delay: float = BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Run ``operation`` until it succeeds or the attempt budget is exhausted.

    ``attempts`` is the total number of attempts (the first call included), so
    ``HTTP_MAX_RETRIES=3`` means at most 3 calls. The delay between attempts grows
    exponentially (``base_delay * 2 ** (attempt - 1)``).

    ``sleep`` defaults to ``time.sleep``, resolved here rather than as a default
    argument, so a test can replace ``time.sleep`` and never really wait.

    The final failure is re-raised unchanged so callers can decide what to do with
    the post/feed (Invariant 8). ``description`` must never contain secrets.
    """
    _validate_attempts(attempts)
    sleep = time.sleep if sleep is None else sleep

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except PermanentError as exc:
            # Retrying cannot change the answer; fail now and leave the backoff unused.
            logger.error("%s failed permanently: %s", description, exc)
            raise
        except Exception as exc:
            if attempt >= attempts:
                logger.error(
                    "%s failed after %d attempt(s): %s", description, attempts, exc
                )
                raise
            delay = _backoff_delay(attempt, base_delay)
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


async def call_with_retries_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    description: str,
    base_delay: float = BASE_DELAY_SECONDS,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """:func:`call_with_retries` for a coroutine function — same semantics, `await`ed.

    The wait is ``asyncio.sleep`` by default, so a retry never blocks the event loop and
    therefore never delays a button press that is being handled concurrently (phase 6).
    """
    _validate_attempts(attempts)
    sleep = async_sleep if sleep is None else sleep

    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except PermanentError as exc:
            logger.error("%s failed permanently: %s", description, exc)
            raise
        except Exception as exc:
            if attempt >= attempts:
                logger.error(
                    "%s failed after %d attempt(s): %s", description, attempts, exc
                )
                raise
            delay = _backoff_delay(attempt, base_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            await sleep(delay)

    raise AssertionError("unreachable")  # pragma: no cover


def _describe(
    description: Description | None,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Label for the log lines: explicit text, a call-derived label, or the name."""
    if callable(description):
        return description(*args, **kwargs)
    return description or func.__qualname__


def retryable(
    *,
    attempts: int | None = None,
    description: Description | None = None,
    base_delay: float = BASE_DELAY_SECONDS,
    sleep: Callable[[float], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a call so it runs with bounded retries and exponential backoff.

    Works on plain functions and on ``async def`` functions: the wrapper checks
    :func:`inspect.iscoroutinefunction` and picks the matching engine, so a call site
    does not have to know which one it needs.

    ``attempts=None`` (the default) reads ``HTTP_MAX_RETRIES`` from settings at call
    time, so the limit stays configurable without threading it through every
    signature (NFR-7, AGENTS.md section 11). ``sleep`` is resolved at call time too.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        label = description
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await call_with_retries_async(
                    lambda: func(*args, **kwargs),
                    attempts=resolve_attempts(attempts),
                    description=_describe(label, func, args, kwargs),
                    base_delay=base_delay,
                    sleep=async_sleep if sleep is None else sleep,
                )

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return call_with_retries(
                lambda: func(*args, **kwargs),
                attempts=resolve_attempts(attempts),
                description=_describe(label, func, args, kwargs),
                base_delay=base_delay,
                sleep=time.sleep if sleep is None else sleep,
            )

        return wrapper

    return decorator
