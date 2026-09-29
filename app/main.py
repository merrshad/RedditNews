"""Entry point: long-poll the bot for admin decisions and run the pipeline on schedule.

Phase 5 made the worker two interleaved jobs in one process:

- it long-polls Telegram (``app.telegram_updates``), so a ✅/❌ press is acted on within
  seconds, and
- it runs ``pipeline.run_once()`` every ``POLL_INTERVAL_SECONDS``.

Deliberately still no scheduler dependency, no queue and no threads (YAGNI, AGENTS.md
section 8). Two guarantees matter here: a failing cycle must never kill the worker
(NFR-2), and ``docker stop`` (SIGTERM) must end the process cleanly instead of waiting for
a poll or a timeout.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from collections.abc import Callable

from app import pipeline, telegram_updates
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

    Raising ``KeyboardInterrupt`` also interrupts the long poll and the short sleep
    between polls, so the container stops immediately.
    """
    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


def _run_safely(step: Callable[[], None]) -> None:
    """Run one worker step; a failure is logged and never ends the loop (NFR-2)."""
    try:
        step()
    except Exception:
        logger.exception("%s failed; continuing", getattr(step, "__name__", step))


def run_forever() -> None:
    """Poll for decisions forever, running a full pipeline cycle on its own schedule."""
    settings = get_settings()
    offset: int | None = None
    next_cycle_at = 0.0

    while True:
        due = time.monotonic() >= next_cycle_at
        started = time.monotonic()

        # When a cycle is due, do not sit in a long poll first: ask for updates and go.
        result = telegram_updates.poll_once(
            offset, long_poll_seconds=0 if due else LONG_POLL_SECONDS
        )
        offset = result.offset

        if due:
            _run_safely(pipeline.run_once)
            next_cycle_at = time.monotonic() + settings.poll_interval_seconds
        elif result.decisions:
            # An admin pressed a button and is waiting: analyse/publish right away,
            # without dragging a full RSS cycle in (the fetch keeps its own schedule).
            _run_safely(pipeline.process_approved_posts)

        elapsed = time.monotonic() - started
        if elapsed < MIN_POLL_INTERVAL_SECONDS:
            time.sleep(MIN_POLL_INTERVAL_SECONDS - elapsed)


def main() -> None:
    """Log, then poll and cycle forever; the pipeline reads its own settings."""
    settings = get_settings()
    setup_logging(settings.log_level)
    _install_signal_handlers()
    logger.info(
        "Starting reddit-telegram-digest: model=%s, interval=%ds, review_channel=%s",
        settings.openai_model,
        settings.poll_interval_seconds,
        settings.telegram_review_channel_id,
    )

    try:
        run_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting")


if __name__ == "__main__":
    main()
