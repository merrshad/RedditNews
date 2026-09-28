"""Entry point: a simple periodic loop around one pipeline run (AGENTS.md section 8).

No scheduler dependency on purpose (YAGNI): ``while True: run(); sleep(...)``.
"""

from __future__ import annotations

import logging
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

    try:
        run_forever(settings, topics_config=topics_config)
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting")


if __name__ == "__main__":
    main()
