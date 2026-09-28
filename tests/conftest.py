"""Shared fixtures and test doubles.

Unit tests never touch the network (RSS/LLM/Telegram are faked, NFR-8). The database
tests are integration tests against a real Postgres (`docker compose up -d db`); when
no database is reachable they skip themselves with a clear message, so a plain
`pytest` still works on a machine without Docker.

The database tests own a *separate* database (`<DATABASE_URL>_test`, created on demand).
They used to run against the configured one, which meant a real `docker compose up` run —
which is exactly what phase 4 asks for — wrote analysed posts into the same `posts` table
and broke five candidate/duplicate tests (observed). Isolating the two is what keeps
`pytest` green while a real worker is running against the development database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

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

# Name suffix of the database the integration tests use, and its URL.
TEST_DATABASE_SUFFIX = "_test"


def _with_test_suffix(url: str) -> str:
    """`postgresql://.../reddit_digest` -> `postgresql://.../reddit_digest_test`.

    Idempotent on purpose: pytest imports this file once as ``conftest`` and once as
    ``tests.conftest`` (the test modules import helpers from it by name), so the suffix
    must not be appended twice.
    """
    parts = urlsplit(url)
    if parts.path.endswith(TEST_DATABASE_SUFFIX):
        return url
    return urlunsplit(parts._replace(path=f"{parts.path}{TEST_DATABASE_SUFFIX}"))


TEST_DATABASE_URL = _with_test_suffix(DATABASE_URL)

# `app.repository` and `app.settings` read this env var directly, so pointing the process
# at the test database is what makes every DB test — and every real `repository` call the
# pipeline makes — land there instead of in the development database. Set (not
# `setdefault`): the tests must never run against the database a real worker writes to.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

_REPOSITORY_API = (
    "exists",
    "save",
    "update_status",
    "fetch_recent_candidates",
    "fetch_pending_to_send",
)


class FakeChatCompletion:
    """Stand-in for ``app.analyzer.chat_completion``: records prompts, replays an answer."""

    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


class FakeSender:
    """Stand-in for ``app.telegram_notifier.send_message`` (NFR-8: no network).

    It mirrors the real contract — text in, ``True`` only when Telegram accepted the
    message — so a rejected send comes back as ``False`` and the pipeline keeps the record
    as ``to_send`` for the next run (FR-10/FR-11).
    """

    def __init__(self, *, accept: bool = True, fail_first: bool = False) -> None:
        self.accept = accept
        self.fail_first = fail_first
        self.messages: list[str] = []

    def __call__(self, text: str) -> bool:
        if self.fail_first:
            self.fail_first = False
            return False
        if not self.accept:
            return False
        self.messages.append(text)
        return True


class FailingSender(FakeSender):
    """A notifier that always reports a rejected send (Telegram is down)."""

    def __init__(self) -> None:
        super().__init__(accept=False)


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
        database_url=TEST_DATABASE_URL,
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
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as connection:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def create_test_database() -> None:
    """Create ``<DATABASE_URL>_test`` if it does not exist yet.

    Connects to the configured (development) database first: that one is guaranteed to
    exist, and ``CREATE DATABASE`` cannot run inside a transaction, hence autocommit. The
    name comes from configuration, never from user input, and is quoted defensively.
    """
    name = urlsplit(TEST_DATABASE_URL).path.lstrip("/")
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if exists is None:
            connection.execute(f'CREATE DATABASE "{name}"')


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
    """URL of the isolated Postgres under test, or a skip when none is reachable."""
    try:
        create_test_database()
        apply_schema()
    except psycopg.OperationalError as exc:
        pytest.skip(
            f"no Postgres at {DATABASE_URL} ({exc}); start it with `docker compose up -d db`"
        )
    return TEST_DATABASE_URL


@pytest.fixture
def db_connection(postgres_database: str) -> Iterator[psycopg.Connection]:
    """A dict-row connection with this test's rows removed before and after."""
    with psycopg.connect(postgres_database, row_factory=dict_row, autocommit=True) as connection:
        delete_test_rows(connection)
        try:
            yield connection
        finally:
            delete_test_rows(connection)
