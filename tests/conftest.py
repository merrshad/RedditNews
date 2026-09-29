"""Shared fixtures and test doubles.

Unit tests never touch the network (RSS/LLM/Telegram are faked, NFR-8). Storage is *not*
faked any more (phase 5): the review/publication flow is a state machine spread over
conditional SQL, so the pipeline and repository tests run against a real Postgres through
``db/schema.sql``. When no database is reachable they skip themselves with a clear
message, so a plain ``pytest`` still works on a machine without Docker.

Those database tests own a *separate* database (`<DATABASE_URL>_test`, created on demand).
They used to run against the configured one, which meant a real `docker compose up` run —
which is exactly what phase 4 asks for — wrote analysed posts into the same `posts` table
and broke five candidate/duplicate tests (observed). Isolating the two is what keeps
`pytest` green while a real worker is running against the development database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg.rows import dict_row

from app.models import TopicRecord
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
os.environ.setdefault("TELEGRAM_REVIEW_CHANNEL_ID", "-100999")
os.environ.setdefault("TELEGRAM_ADMIN_IDS", "777,888")
os.environ.setdefault("RSS_FETCH_LIMIT", "25")
os.environ.setdefault("HTTP_MAX_RETRIES", "2")
os.environ.setdefault("LOG_LEVEL", "INFO")

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

# Everything the tests create carries one of these prefixes, so they can be cleaned up
# without touching the seeded topics/sources or anything else in the database.
TEST_REDDIT_ID_PREFIX = "t3_test_"
TEST_TOPIC_KEY_PREFIX = "t_test_"
TEST_SOURCE_URL_PREFIX = "https://test.example/"

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


class FakeChatCompletion:
    """Stand-in for ``app.analyzer.chat_completion``: records prompts, replays an answer."""

    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


class FakeTelegram:
    """Stand-in for the whole ``app.telegram_notifier`` surface the flow uses (NFR-8).

    It mirrors the real contract — a send answers with Telegram's ``message_id`` or ``None``
    — so a rejected send keeps the post for a later run (FR-10/FR-11). ``fail_next_sends``
    simulates a Telegram outage for the next N sends, which is how the retry paths are
    exercised.
    """

    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.fail_next_sends = 0
        # What the real `send_message` falls back to when no chat is given: the public
        # channel. `patch_telegram` fills it from the settings under test.
        self.default_chat_id: str = ""
        #: (text, chat_id, reply_markup)
        self.sent: list[tuple[str, str | None, dict[str, Any] | None]] = []
        #: (text, chat_id, message_id)
        self.edits: list[tuple[str, str, int]] = []
        #: (callback_query_id, answer text)
        self.answers: list[tuple[str, str | None]] = []
        self._message_id = 1000

    def send_message(
        self,
        text: str,
        *,
        chat_id: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> int | None:
        if self.fail_next_sends > 0:
            self.fail_next_sends -= 1
            return None
        if not self.accept:
            return None
        self._message_id += 1
        self.sent.append((text, chat_id or self.default_chat_id, reply_markup))
        return self._message_id

    def edit_message_text(
        self,
        text: str,
        *,
        chat_id: str,
        message_id: int,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        self.edits.append((text, chat_id, message_id))
        return True

    def answer_callback_query(self, callback_query_id: str, *, text: str | None = None) -> bool:
        self.answers.append((callback_query_id, text))
        return True

    def messages_to(self, chat_id: str | None) -> list[str]:
        """Texts delivered to one chat, in order."""
        return [text for text, target, _ in self.sent if target == chat_id]

    def buttons_to(self, chat_id: str | None) -> list[dict[str, Any] | None]:
        return [markup for _, target, markup in self.sent if target == chat_id]


def patch_telegram(
    monkeypatch: pytest.MonkeyPatch, fake: FakeTelegram, *, default_chat_id: str = ""
) -> FakeTelegram:
    """Point the Telegram boundary at a fake (one seam for notifications and reviews)."""
    if default_chat_id:
        fake.default_chat_id = default_chat_id
    monkeypatch.setattr("app.telegram_notifier.send_message", fake.send_message)
    monkeypatch.setattr("app.telegram_notifier.edit_message_text", fake.edit_message_text)
    monkeypatch.setattr(
        "app.telegram_notifier.answer_callback_query", fake.answer_callback_query
    )
    return fake


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
        telegram_review_channel_id="-100999",
        telegram_admin_ids="777,888",
        poll_interval_seconds=1,
        rss_fetch_limit=25,
        similarity_lookback_limit=50,
        similarity_lookback_hours=72,
        min_importance_to_send="low",
        http_max_retries=2,
        log_level="INFO",
    )


def apply_schema() -> None:
    """Rebuild the *test* database from ``db/schema.sql`` (Invariant 9).

    ``db/schema.sql`` is the single source of truth for the schema, but it is not a
    migration: ``CREATE TABLE IF NOT EXISTS`` cannot add a column to a table that already
    exists, so a test database left over from an earlier phase would silently keep the old
    columns. Dropping the three tables first is safe here — this is the dedicated
    ``<DATABASE_URL>_test`` database that ``conftest`` itself creates, never the one a
    running worker uses — and it makes the tests fail loudly instead of mysteriously when
    the schema moves.
    """
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS posts, sources, topics CASCADE")
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
    """Remove only the rows these tests created (matched by their test prefixes)."""
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE posts SET duplicate_of_id = NULL WHERE reddit_id LIKE %s",
            (f"{TEST_REDDIT_ID_PREFIX}%",),
        )
        cursor.execute(
            "DELETE FROM posts WHERE reddit_id LIKE %s", (f"{TEST_REDDIT_ID_PREFIX}%",)
        )
        # Deleting a topic cascades to its sources (db/schema.sql).
        cursor.execute(
            "DELETE FROM topics WHERE key LIKE %s", (f"{TEST_TOPIC_KEY_PREFIX}%",)
        )
        cursor.execute(
            "DELETE FROM sources WHERE rss_url LIKE %s", (f"{TEST_SOURCE_URL_PREFIX}%",)
        )


def insert_topic(
    connection: psycopg.Connection,
    key: str,
    name: str = "موضوع تست",
    *,
    is_active: bool = True,
) -> int:
    """Create a test topic directly in the database and return its id."""
    row = connection.execute(
        "INSERT INTO topics (key, name, is_active) VALUES (%s, %s, %s) RETURNING id",
        (f"{TEST_TOPIC_KEY_PREFIX}{key}", name, is_active),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def insert_source(
    connection: psycopg.Connection,
    topic_key: str,
    rss_url: str,
    *,
    fetch_limit: int | None = None,
    is_active: bool = True,
) -> int:
    """Create a test source under a test topic and return its id."""
    row = connection.execute(
        """
        INSERT INTO sources (topic_id, rss_url, fetch_limit, is_active)
        SELECT id, %s, %s, %s FROM topics WHERE key = %s
        RETURNING id
        """,
        (f"{TEST_SOURCE_URL_PREFIX}{rss_url}", fetch_limit, is_active, f"{TEST_TOPIC_KEY_PREFIX}{topic_key}"),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def temp_topic_key(key: str) -> str:
    """The prefixed key a test topic is stored under."""
    return f"{TEST_TOPIC_KEY_PREFIX}{key}"


def temp_source_url(path: str) -> str:
    """The prefixed URL a test source is stored under."""
    return f"{TEST_SOURCE_URL_PREFIX}{path}"


def prepare_test_database() -> None:
    """Create ``<DATABASE_URL>_test`` and load ``db/schema.sql`` into it (Invariant 9)."""
    create_test_database()
    apply_schema()


@pytest.fixture(scope="session", autouse=True)
def _test_database_ready() -> Iterator[None]:
    """Make the isolated test database ready before any test can touch it.

    This used to happen only inside `postgres_database`, so a test that talks to
    `repository` *without* requesting that fixture — the admin-command tests — died as soon
    as the database was not there yet. Measured after `docker compose down -v`, which is
    exactly what a first run on CI or a new clone looks like: 16 failures, and whether the
    suite passed at all depended on test order.

    Best effort on purpose: with no Postgres reachable the unit tests still run untouched,
    and the tests that genuinely need a database skip themselves (see `postgres_database`).
    """
    try:
        prepare_test_database()
    except psycopg.OperationalError:
        pass
    yield


@pytest.fixture(scope="session")
def postgres_database() -> str:
    """URL of the isolated Postgres under test, or a skip when none is reachable."""
    try:
        prepare_test_database()
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


@pytest.fixture
def topic_records() -> list[TopicRecord]:
    """A small in-memory taxonomy for the tests that never touch the database."""
    return [
        TopicRecord(id=1, key="ai", name="هوش مصنوعی"),
        TopicRecord(id=2, key="startup", name="استارتاپ"),
    ]
