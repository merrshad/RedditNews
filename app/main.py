"""Entry point: a simple periodic loop around one pipeline run (AGENTS.md section 8).

No scheduler dependency on purpose (YAGNI): ``while True: run(); sleep(...)``.
"""

from __future__ import annotations

import logging
import sys
import time

from app.llm_client import LlmClient
from app.pipeline import Pipeline
from app.repository import PostRepository, connect
from app.settings import Settings, TopicConfig, get_settings, load_topics
from app.telegram_notifier import TelegramNotifier

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


def build_llm_client(settings: Settings) -> LlmClient:
    return LlmClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        max_retries=settings.http_max_retries,
    )


def build_notifier(settings: Settings) -> TelegramNotifier:
    return TelegramNotifier(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        max_retries=settings.http_max_retries,
    )


def run_once(
    settings: Settings,
    *,
    topics: list[TopicConfig],
    llm_client: LlmClient,
    notifier: TelegramNotifier,
) -> None:
    """Open a fresh connection and execute a single pipeline cycle."""
    with connect(settings.database_url) as connection:
        Pipeline(
            settings=settings,
            repository=PostRepository(connection),
            llm_client=llm_client,
            notifier=notifier,
            topics=topics,
        ).run_once()


def run_forever(
    settings: Settings,
    *,
    topics: list[TopicConfig],
    llm_client: LlmClient,
    notifier: TelegramNotifier,
) -> None:
    """Poll forever; a failed cycle must never kill the worker (NFR-2)."""
    while True:
        try:
            run_once(settings, topics=topics, llm_client=llm_client, notifier=notifier)
        except Exception:
            logger.exception("Pipeline cycle failed; retrying after the poll interval")

        logger.info("Sleeping %d second(s) until the next cycle", settings.poll_interval_seconds)
        time.sleep(settings.poll_interval_seconds)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    topics = load_topics()
    logger.info(
        "Starting reddit-telegram-digest: %d topic(s), %d feed(s), model=%s, interval=%ds",
        len(topics),
        sum(len(topic.feeds) for topic in topics),
        settings.openai_model,
        settings.poll_interval_seconds,
    )

    llm_client = build_llm_client(settings)
    notifier = build_notifier(settings)

    try:
        run_forever(settings, topics=topics, llm_client=llm_client, notifier=notifier)
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting")


if __name__ == "__main__":
    main()
