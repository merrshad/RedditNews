"""Entry point: one asyncio event loop running the worker's two jobs side by side.

The worker has always had two jobs; phase 6 stops them taking turns. ``run_forever`` opens an
``asyncio.TaskGroup`` with two long-lived tasks:

- ``_updates_loop`` long-polls Telegram (``app.telegram_updates``) so a ✅/❌ press or a panel
  button is handled within seconds, and
- ``_pipeline_loop`` runs ``pipeline.run_once()`` every ``POLL_INTERVAL_SECONDS`` and the much
  cheaper ``process_approved_posts()`` as soon as a decision really changed a row.

Before this, the single ``while True`` did one thing at a time: a cycle that spent a minute on
RSS fetches, model calls and review messages made every button press wait for it, which is
exactly the sluggishness the admin noticed. Two tasks in one loop is still one process, one
set of credentials and no queue — the old constraint (only one consumer may call ``getUpdates``)
is kept: ``_updates_loop`` is the only place that polls.

Deliberately still no scheduler dependency and no message queue (YAGNI, AGENTS.md section 8).
Two guarantees matter here: a failing cycle must never kill the worker (NFR-2), and
``docker stop`` (SIGTERM) must end the process cleanly instead of waiting for a poll or a
timeout.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from collections.abc import Awaitable, Callable

import psycopg

from app import pipeline, repository, telegram_updates
from app.settings import get_settings
from app.telegram_notifier import LONG_POLL_SECONDS

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# A failed getUpdates answers immediately instead of waiting for the long poll, so the
# loop would spin; this is the floor between two attempts.
MIN_POLL_INTERVAL_SECONDS = 5.0

# Transport libraries log every request, and their messages carry the credentials: httpx
# prints the full URL, which for Telegram is `.../bot<TOKEN>/sendMessage`, and at DEBUG it
# prints the headers, which hold the LLM bearer key. A real phase-4 run showed the bot
# token in `docker logs` at the default INFO level, so these loggers stay at WARNING no
# matter what LOG_LEVEL asks for (Invariant 6, NFR-4).
SECRET_BEARING_LOGGERS = ("httpx", "httpcore", "openai")


def setup_logging(level: str) -> None:
    """Structured-enough logging to stdout, which Docker collects (NFR-3)."""
    name = level.upper()
    logging.basicConfig(level=name, format=LOG_FORMAT, stream=sys.stdout, force=True)

    # `LOG_LEVEL=DEBUG` still must not print a token, so the transport loggers are capped.
    requested = logging.getLevelNamesMapping().get(name, logging.INFO)
    for logger_name in SECRET_BEARING_LOGGERS:
        logging.getLogger(logger_name).setLevel(max(requested, logging.WARNING))


def _request_shutdown(_signum: int, _frame: object) -> None:
    """Translate SIGTERM/SIGINT into the same clean exit path as Ctrl+C.

    Raising ``KeyboardInterrupt`` also interrupts whatever the event loop is waiting on
    (the long poll, a backoff sleep), so the container stops immediately.
    """
    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


async def _sleep(seconds: float) -> None:
    """The loop's only wait, in one place so a test can drive the schedule (NFR-8)."""
    await asyncio.sleep(seconds)


async def _run_safely(step: Callable[[], Awaitable[None]]) -> None:
    """Run one worker step; a failure is logged and never ends the loop (NFR-2)."""
    try:
        await step()
    except Exception:
        logger.exception("%s failed; continuing", getattr(step, "__name__", step))


async def _updates_loop(decision: asyncio.Event) -> None:
    """Consume Telegram updates forever, signalling every real review decision.

    The long poll is the task's own wait, so it costs nothing while a cycle runs: the
    scheduler task is free to work, and this one wakes up the moment Telegram answers.
    """
    offset: int | None = None

    while True:
        started = time.monotonic()
        result: telegram_updates.PollResult | None
        try:
            result = await telegram_updates.poll_once(
                offset, long_poll_seconds=LONG_POLL_SECONDS
            )
        except Exception:
            # `poll_once` contains its own per-update guard; anything reaching here is a bug,
            # and it must not take the worker down (NFR-2).
            logger.exception("The Telegram update loop failed; retrying shortly")
            result = None

        if result is not None:
            offset = result.offset
            if result.decisions:
                # An admin pressed a button and is waiting: analyse/publish right away.
                decision.set()

        elapsed = time.monotonic() - started
        if elapsed < MIN_POLL_INTERVAL_SECONDS:
            await _sleep(MIN_POLL_INTERVAL_SECONDS - elapsed)


async def _wait_for_decision(decision: asyncio.Event, timeout: float) -> bool:
    """Wait up to ``timeout`` for a review decision; ``True`` when one arrived first."""
    try:
        await asyncio.wait_for(decision.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return False
    decision.clear()
    return True


async def _pipeline_loop(decision: asyncio.Event) -> None:
    """Run a full cycle on its own schedule, and the fast path after every decision."""
    settings = get_settings()

    while True:
        await _run_safely(pipeline.run_once)
        if await _wait_for_decision(decision, settings.poll_interval_seconds):
            # The cycle above already drained the queues, so only the newly approved posts
            # are left; the RSS fetch keeps its own schedule.
            await _run_safely(pipeline.process_approved_posts)


async def run_forever() -> None:
    """Poll for button presses and run the pipeline, concurrently, until shutdown."""
    decision = asyncio.Event()
    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(_updates_loop(decision), name="telegram-updates")
            group.create_task(_pipeline_loop(decision), name="pipeline")
    except* KeyboardInterrupt:
        # SIGTERM/SIGINT is delivered to whichever task is parked in an await, so it can
        # arrive as a child-task failure inside the group. Both spellings must mean the
        # same thing, or `docker stop` would exit with a traceback instead of cleanly.
        raise KeyboardInterrupt from None


def preflight() -> None:
    """Fail fast (and loudly) when the database cannot serve this code (Invariant 9).

    `db/schema.sql` is applied by the `db` container only on a *first* boot, so a volume from
    an earlier phase keeps the old shape and every cycle would die with ``UndefinedTable``
    while the worker still looked alive. Exiting with the fix in the message turns that
    silent loop into one actionable line (NFR-3).
    """
    try:
        unusable = repository.find_unusable_tables()
    except psycopg.OperationalError as exc:
        raise SystemExit(f"cannot reach the database ({exc}); is the `db` service up?") from exc

    if unusable:
        raise SystemExit(
            f"database schema is not ready: {', '.join(unusable)} cannot be queried. "
            "There are no migrations (the schema is applied once, on a fresh database), so "
            "recreate the volume: `docker compose down -v && docker compose up -d --build`"
        )


def main() -> None:
    """Log, then poll and cycle forever; the pipeline reads its own settings."""
    settings = get_settings()
    setup_logging(settings.log_level)
    preflight()
    _install_signal_handlers()
    logger.info(
        "Starting reddit-telegram-digest: model=%s, interval=%ds, review_channel=%s",
        settings.openai_model,
        settings.poll_interval_seconds,
        settings.telegram_review_channel_id,
    )

    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting")


if __name__ == "__main__":
    main()
