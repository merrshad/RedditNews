"""Entry point: a simple periodic loop around one pipeline run (AGENTS.md section 8).

No scheduler dependency on purpose (YAGNI): ``while True: run(); sleep(...)``.

``SIGTERM`` is turned into the same ``KeyboardInterrupt`` path as Ctrl-C so that
``docker stop`` leaves the worker cleanly instead of killing the process mid-cycle
(NFR-2). Nothing here re-implements signal handling beyond that one seam.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from app.pipeline import Pipeline
from app.reddit_source import TopicsConfig, load_topics_config
from app.settings import Settings, get_settings

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


def _exit_on_sigterm(signum: int, frame: object) -> None:
    """Raise ``KeyboardInterrupt`` so SIGTERM follows the existing shutdown path.

    The repository writes are autocommit and one cycle is idempotent (Invariant 7),
    so aborting a cycle mid-flight is safe: the next run simply picks the post up.
    """
    raise KeyboardInterrupt


def install_signal_handlers() -> None:
    """Handle SIGTERM like Ctrl-C so ``docker stop`` exits the loop (NFR-2).

    SIGINT already raises ``KeyboardInterrupt`` by default. Installing a handler is
    only possible from the main thread, so a non-main-thread caller (some test
    runners) is tolerated rather than fatal.
    """
    try:
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
    except ValueError:  # pragma: no cover - only reachable off the main thread
        logger.warning("SIGTERM handler not installed: not running on the main thread")


def run_once(settings: Settings, *, topics_config: TopicsConfig) -> None:
    """Execute a single pipeline cycle (the repository opens its own connections)."""
    Pipeline(
        settings=settings,
        topics_config=topics_config,
    ).run_once()


def run_forever(settings: Settings, *, topics_config: TopicsConfig) -> None:
    """Poll forever; a failed cycle must never kill the worker (NFR-2)."""
    while True:
        try:
            run_once(settings, topics_config=topics_config)
        except Exception:
            logger.exception("Pipeline cycle failed; retrying after the poll interval")

        logger.info("Sleeping %d second(s) until the next cycle", settings.poll_interval_seconds)
        time.sleep(settings.poll_interval_seconds)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    topics_config = load_topics_config()
    logger.info(
        "Starting reddit-telegram-digest: %d topic(s), %d feed(s), model=%s, interval=%ds",
        len(topics_config.topics),
        sum(len(topic.feeds) for topic in topics_config.topics),
        settings.openai_model,
        settings.poll_interval_seconds,
    )

    install_signal_handlers()
    try:
        run_forever(settings, topics_config=topics_config)
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting")


if __name__ == "__main__":
    main()
