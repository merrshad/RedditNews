"""Shared retry/backoff helper for every external call (NFR-2, Invariant 8, DRY)."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

BASE_DELAY_SECONDS = 1.0
BACKOFF_FACTOR = 2.0


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
