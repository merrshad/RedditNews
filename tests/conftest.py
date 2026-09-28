"""Shared fixtures and test doubles.

Unit tests never touch the network (RSS/LLM/Telegram are faked, NFR-8). The database
tests are integration tests against a real Postgres (`docker compose up -d db`); when
no database is reachable they skip themselves with a clear message, so a plain
`pytest` still works on a machine without Docker.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from app.models import PostRecord
from app.reddit_source import TopicConfig, TopicsConfig
from app.settings import Settings, get_settings

# Defaults for the env vars `app/settings.py` requires. `setdefault` never overrides
# what the developer configured (e.g. a DATABASE_URL pointing at another database).
DATABASE_URL = os.environ.setdefault(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/reddit_digest"
)
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://llm.example/v1")
os.environ.setdefault("OPENAI_MODEL", "test-model")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "000000:test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "-100123")
os.environ.setdefault("HTTP_MAX_RETRIES", "2")
os.environ.setdefault("LOG_LEVEL", "INFO")

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

# Rows created by the database tests carry this prefix so they can be cleaned up
# without touching anything else in the database.
TEST_REDDIT_ID_PREFIX = "t3_test_"

_REPOSITORY_API = (
    "exists",
    "save",
    "update_status",
    "fetch_recent_candidates",
    "fetch_pending_to_send",
)


class StubLlmClient:
    """Records the prompts it receives and replays a canned answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


class FakeRepository:
    """In-memory stand-in for the ``app.repository`` functions (NFR-8).

    Mirrors that module's public API exactly, so a test can swap in the whole storage
    layer with :func:`patch_repository` and still assert on the order of the calls.
    """

    def __init__(
        self,
        *,
        existing: tuple[str, ...] = (),
        candidates: tuple[PostRecord, ...] = (),
        pending: tuple[PostRecord, ...] = (),
    ) -> None:
        self.existing: dict[str, int] = {
            reddit_id: 50 + index for index, reddit_id in enumerate(existing)
        }
        self.candidates = list(candidates)
        self.pending = list(pending)
        self.saved: list[PostRecord] = []
        self.updates: list[tuple[int, str, datetime | None]] = []
        self.calls: list[str] = []
        self._next_id = 100

    def exists(self, reddit_id: str) -> bool:
        self.calls.append("exists")
        return reddit_id in self.existing

    def save(self, post: PostRecord) -> int:
        self.calls.append("save")
        if post.reddit_id in self.existing:
            return self.existing[post.reddit_id]
        self._next_id += 1
        self.existing[post.reddit_id] = self._next_id
        self.saved.append(post)
        return self._next_id

    def update_status(self, post_id: int, status: str, sent_at: datetime | None = None) -> None:
        self.calls.append("update_status")
        self.updates.append((post_id, status, sent_at))

    def fetch_recent_candidates(self, limit: int, hours: int) -> list[PostRecord]:
        self.calls.append(f"fetch_recent_candidates(limit={limit},hours={hours})")
        return list(self.candidates)

    def fetch_pending_to_send(self) -> list[PostRecord]:
        self.calls.append("fetch_pending_to_send")
        return list(self.pending)

    @property
    def sent_ids(self) -> list[int]:
        """Ids that were marked ``sent``, in order."""
        return [post_id for post_id, status, _ in self.updates if status == "sent"]

    @property
    def last_id(self) -> int:
        return self._next_id


def patch_repository(monkeypatch: pytest.MonkeyPatch, repository: FakeRepository) -> FakeRepository:
    """Point every ``app.repository`` function at an in-memory fake."""
    for name in _REPOSITORY_API:
        monkeypatch.setattr(f"app.repository.{name}", getattr(repository, name))
    return repository


@pytest.fixture(autouse=True)
def _fresh_settings_cache() -> Iterator[None]:
    """`get_settings()` is process-cached; tests that patch env need a clean cache."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Explicit settings so tests never depend on the developer's env/.env."""
    return Settings(
        database_url=DATABASE_URL,
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
def topics_config() -> TopicsConfig:
    return TopicsConfig(
        topics=[
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
    )


def apply_schema() -> None:
    """Apply ``db/schema.sql`` — the single source of truth for the schema (Invariant 9).

    The container applies it on first boot too; running it again is idempotent and
    keeps the tests working against any fresh Postgres.
    """
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def delete_test_rows(connection: psycopg.Connection) -> None:
    """Remove only the rows these tests created (matched by reddit_id prefix)."""
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE posts SET duplicate_of_id = NULL WHERE reddit_id LIKE %s",
            (f"{TEST_REDDIT_ID_PREFIX}%",),
        )
        cursor.execute(
            "DELETE FROM posts WHERE reddit_id LIKE %s", (f"{TEST_REDDIT_ID_PREFIX}%",)
        )


@pytest.fixture(scope="session")
def postgres_database() -> str:
    """URL of the Postgres under test, or a skip when none is reachable."""
    try:
        apply_schema()
    except psycopg.OperationalError as exc:
        pytest.skip(f"no Postgres at {DATABASE_URL} ({exc}); start it with `docker compose up -d db`")
    return DATABASE_URL


@pytest.fixture
def db_connection(postgres_database: str) -> Iterator[psycopg.Connection]:
    """A dict-row connection with this test's rows removed before and after."""
    with psycopg.connect(postgres_database, row_factory=dict_row, autocommit=True) as connection:
        delete_test_rows(connection)
        try:
            yield connection
        finally:
            delete_test_rows(connection)
