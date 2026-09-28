"""Entry point: a simple periodic loop around one pipeline run (AGENTS.md section 8).

No scheduler dependency on purpose (YAGNI): ``while True: run_once(); sleep(...)``. Two
guarantees matter here: a failing cycle must never kill the worker (NFR-2), and a
``docker stop`` (SIGTERM) must end the process cleanly instead of waiting for the timeout.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from app import pipeline
from app.settings import get_settings

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str) -> None:
    """Structured-enough logging to stdout, which Docker collects (NFR-3)."""
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def _request_shutdown(_signum: int, _frame: object) -> None:
    """Translate SIGTERM/SIGINT into the same clean exit path as Ctrl+C.

    Raising ``KeyboardInterrupt`` also interrupts the ``time.sleep`` between cycles, so
    the container stops immediately rather than after up to ``POLL_INTERVAL_SECONDS``.
    """
    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


def main() -> None:
    """Log, then poll forever; the pipeline reads its own settings and topics config."""
    settings = get_settings()
    setup_logging(settings.log_level)
    _install_signal_handlers()
    logger.info(
        "Starting reddit-telegram-digest: model=%s, interval=%ds",
        settings.openai_model,
        settings.poll_interval_seconds,
    )

    try:
        while True:
            try:
                pipeline.run_once()
            except Exception:
                # The loop is the last safety net: nothing but a shutdown may end it.
                logger.exception("Pipeline cycle failed; continuing after the poll interval")

            time.sleep(settings.poll_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting")


if __name__ == "__main__":
    main()
