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
