"""Shared fixtures. Unit tests never touch the network or a real database (NFR-8)."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from app.settings import Settings, TopicConfig


class FakeCursor:
    """Minimal psycopg cursor stand-in that records SQL and replays canned rows."""

    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self._rows: list[Any] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self._connection.executed.append((sql, params))
        self._rows = list(self._connection.queue.pop(0)) if self._connection.queue else []

    def fetchone(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class FakeConnection:
    """Records executed statements and serves queued result sets in order."""

    def __init__(self, responses: Sequence[Sequence[Any]] | None = None) -> None:
        self.executed: list[tuple[str, Sequence[Any] | None]] = []
        self.queue: list[Sequence[Any]] = [list(response) for response in responses or []]

    def cursor(self, *args: Any, **kwargs: Any) -> FakeCursor:
        return FakeCursor(self)

    @property
    def statements(self) -> list[str]:
        return [sql for sql, _ in self.executed]


class FakeChatCompletion:
    """Stand-in for ``app.analyzer.chat_completion``: records prompts, replays an answer."""

    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


@pytest.fixture
def fake_connection() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def settings() -> Settings:
    """Explicit settings so tests never depend on the developer's env/.env."""
    return Settings(
        database_url="postgresql://postgres:postgres@localhost:5432/reddit_digest",
        openai_api_key="test-key",
        openai_base_url="https://llm.example/v1",
        openai_model="test-model",
        telegram_bot_token="000000:test-token",
        telegram_chat_id="-100123",
        poll_interval_seconds=1,
        similarity_lookback_limit=50,
        similarity_lookback_hours=72,
        min_importance_to_send="low",
        http_max_retries=2,
        log_level="INFO",
    )


@pytest.fixture
def topics() -> list[TopicConfig]:
    return [
        TopicConfig(
            key="ai",
            name="هوش مصنوعی",
            feeds=["https://www.reddit.com/r/MachineLearning/new/.rss"],
        ),
        TopicConfig(
            key="startup",
            name="استارتاپ",
            feeds=["https://www.reddit.com/r/startups/new/.rss"],
        ),
    ]
