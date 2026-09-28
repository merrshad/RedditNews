"""FR-2 / FR-4 / FR-9 / FR-11 against a real Postgres (`docker compose up -d db`).

Deliberately integration tests rather than cursor mocks: the parametrized SQL, the
JSONB round-trip, the UNIQUE(reddit_id) idempotency and the ``to_send`` filtering are
exactly what this phase must prove, and none of them can be proven by a fake cursor.
They skip themselves with a clear message when no Postgres is reachable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import pytest

from app import repository
from app.models import PostRecord
from tests.conftest import TEST_REDDIT_ID_PREFIX


def _record(reddit_id: str, **overrides: Any) -> PostRecord:
    """A fully analysed post; ``overrides`` tweak one field per test."""
    fields: dict[str, Any] = {
        "reddit_id": reddit_id,
        "subreddit": "MachineLearning",
        "source_topic_key": "ai",
        "title": "A new open model was released",
        "url": f"https://www.reddit.com/r/MachineLearning/comments/{reddit_id}/x/",
        "author": "somebody",
        "raw_content": "Body text",
        "published_at": datetime.now(timezone.utc),
        "is_relevant": True,
        "duplicate_of_id": None,
        "topic": "ai",
        "importance": "high",
        "summary_fa": "خلاصه فارسی پست.",
        "key_points": ["نکته اول", "نکته دوم"],
        "llm_raw_response": {"is_relevant": True},
        "status": "to_send",
    }
    fields.update(overrides)
    return PostRecord(**fields)


def _row(connection: psycopg.Connection, post_id: int) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        row = cursor.fetchone()
    assert row is not None
    return row


def _count(connection: psycopg.Connection, reddit_id: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) AS total FROM posts WHERE reddit_id = %s", (reddit_id,))
        return int(cursor.fetchone()["total"])


# --- exists (FR-2) ---------------------------------------------------------------


def test_exists_is_false_before_save_and_true_after(db_connection: psycopg.Connection) -> None:
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}exists"

    assert repository.exists(reddit_id) is False

    post_id = repository.save(_record(reddit_id))

    assert post_id > 0
    assert repository.exists(reddit_id) is True


def test_exists_treats_its_argument_as_data_not_sql(db_connection: psycopg.Connection) -> None:
    """Every query is parametrized (AGENTS.md section 12)."""
    assert repository.exists("t3_x'; DROP TABLE posts; --") is False
    assert repository.exists("t3_1abcde") is False


# --- save (FR-9) -----------------------------------------------------------------


def test_save_persists_every_field(db_connection: psycopg.Connection) -> None:
    record = _record(f"{TEST_REDDIT_ID_PREFIX}fields")

    post_id = repository.save(record)

    row = _row(db_connection, post_id)
    assert row["reddit_id"] == record.reddit_id
    assert row["subreddit"] == "MachineLearning"
    assert row["source_topic_key"] == "ai"
    assert row["title"] == record.title
    assert row["url"] == record.url
    assert row["author"] == "somebody"
    assert row["raw_content"] == "Body text"
    assert row["published_at"] == record.published_at
    assert row["is_relevant"] is True
    assert row["duplicate_of_id"] is None
    assert row["topic"] == "ai"
    assert row["importance"] == "high"
    assert row["summary_fa"] == "خلاصه فارسی پست."
    assert row["key_points"] == ["نکته اول", "نکته دوم"]
    assert row["llm_raw_response"] == {"is_relevant": True}
    assert row["status"] == "to_send"
    assert row["sent_at"] is None


def test_save_stores_an_unanalysed_post_as_failed(db_connection: psycopg.Connection) -> None:
    """Invariant 3 — a rejected LLM answer is still recorded, with empty analysis."""
    post_id = repository.save(
        _record(
            f"{TEST_REDDIT_ID_PREFIX}failed",
            status="failed",
            is_relevant=None,
            topic=None,
            importance=None,
            summary_fa=None,
            key_points=[],
            llm_raw_response=None,
        )
    )

    row = _row(db_connection, post_id)
    assert row["status"] == "failed"
    assert row["is_relevant"] is None
    assert row["key_points"] == []
    assert row["llm_raw_response"] is None


def test_save_is_idempotent_for_the_same_reddit_id(db_connection: psycopg.Connection) -> None:
    """NFR-1 / Invariant 1 — a repeated run never writes a second row."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}dupe"

    first = repository.save(_record(reddit_id, summary_fa="خلاصه اول"))
    second = repository.save(_record(reddit_id, summary_fa="خلاصه دوم"))

    assert first == second
    assert _count(db_connection, reddit_id) == 1
    # The conflicting save must not overwrite what is already stored.
    assert _row(db_connection, first)["summary_fa"] == "خلاصه اول"


def test_save_never_overwrites_the_analysis_of_an_existing_row(
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 1 / NFR-1, the part a re-run could realistically get wrong.

    `save` is an `INSERT ... ON CONFLICT DO NOTHING`, so a second analysis of the same
    ``reddit_id`` cannot rewrite the stored verdict or move the row backwards: a post that
    is already ``sent`` stays ``sent`` with its original ``sent_at`` and summary. The
    pipeline avoids the call entirely via ``exists()`` (FR-2); this is the database-level
    guarantee behind that shortcut.
    """
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}no_overwrite"
    sent_at = datetime.now(timezone.utc).replace(microsecond=0)
    post_id = repository.save(
        _record(reddit_id, status="to_send", importance="high", summary_fa="خلاصه اول")
    )
    repository.update_status(post_id, "sent", sent_at)

    again = repository.save(
        _record(
            reddit_id,
            status="to_send",
            importance="low",
            summary_fa="خلاصه دوم",
            is_relevant=False,
        )
    )

    assert again == post_id
    assert _count(db_connection, reddit_id) == 1
    row = _row(db_connection, post_id)
    assert row["status"] == "sent"
    assert row["sent_at"] == sent_at
    assert row["importance"] == "high"
    assert row["is_relevant"] is True
    assert row["summary_fa"] == "خلاصه اول"


def test_save_keeps_duplicate_of_id_pointing_at_another_row(db_connection: psycopg.Connection) -> None:
    original_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}original"))

    duplicate_id = repository.save(
        _record(
            f"{TEST_REDDIT_ID_PREFIX}duplicate",
            duplicate_of_id=original_id,
            status="skipped_duplicate",
        )
    )

    assert _row(db_connection, duplicate_id)["duplicate_of_id"] == original_id


# --- fetch_recent_candidates (FR-4 / Invariant 10) --------------------------------


def test_fetch_recent_candidates_returns_at_most_limit_newest_first(
    db_connection: psycopg.Connection,
) -> None:
    now = datetime.now(timezone.utc)
    for index in range(3):
        repository.save(
            _record(
                f"{TEST_REDDIT_ID_PREFIX}recent{index}",
                published_at=now - timedelta(hours=index),
            )
        )

    candidates = repository.fetch_recent_candidates(limit=2, hours=72)

    assert [candidate.reddit_id for candidate in candidates] == [
        f"{TEST_REDDIT_ID_PREFIX}recent0",
        f"{TEST_REDDIT_ID_PREFIX}recent1",
    ]
    assert isinstance(candidates[0].id, int)
    assert candidates[0].importance == "high"
    assert candidates[0].key_points == ["نکته اول", "نکته دوم"]


def test_fetch_recent_candidates_excludes_unanalysed_and_old_posts(
    db_connection: psycopg.Connection,
) -> None:
    now = datetime.now(timezone.utc)
    repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}keep", published_at=now - timedelta(hours=1)))
    repository.save(
        _record(
            f"{TEST_REDDIT_ID_PREFIX}no_summary",
            summary_fa=None,
            status="failed",
            published_at=now,
        )
    )
    repository.save(
        _record(f"{TEST_REDDIT_ID_PREFIX}old", published_at=now - timedelta(hours=100))
    )

    candidates = repository.fetch_recent_candidates(limit=50, hours=72)

    assert [candidate.reddit_id for candidate in candidates] == [f"{TEST_REDDIT_ID_PREFIX}keep"]


def test_fetch_recent_candidates_honours_the_limit_without_rows_to_spare(
    db_connection: psycopg.Connection,
) -> None:
    repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}only"))

    assert len(repository.fetch_recent_candidates(limit=5, hours=72)) == 1


# --- fetch_pending_to_send (FR-11) ------------------------------------------------


def test_fetch_pending_to_send_returns_only_to_send_rows(db_connection: psycopg.Connection) -> None:
    pending_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}pending"))
    repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}sent", status="sent"))
    repository.save(
        _record(
            f"{TEST_REDDIT_ID_PREFIX}irrelevant",
            status="skipped_irrelevant",
            is_relevant=False,
            summary_fa=None,
        )
    )

    records = repository.fetch_pending_to_send()

    assert [record.id for record in records] == [pending_id]
    assert records[0].status == "to_send"
    assert records[0].summary_fa == "خلاصه فارسی پست."
    assert records[0].key_points == ["نکته اول", "نکته دوم"]


def test_fetch_pending_to_send_returns_oldest_first(db_connection: psycopg.Connection) -> None:
    first = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}pending_a"))
    second = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}pending_b"))

    assert [record.id for record in repository.fetch_pending_to_send()] == [first, second]


# --- update_status (FR-10 / FR-11) ------------------------------------------------


def test_update_status_marks_sent_with_the_given_timestamp(db_connection: psycopg.Connection) -> None:
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}status_update"))
    sent_at = datetime.now(timezone.utc).replace(microsecond=0)

    repository.update_status(post_id, "sent", sent_at)

    row = _row(db_connection, post_id)
    assert row["status"] == "sent"
    assert row["sent_at"] == sent_at
    assert repository.fetch_pending_to_send() == []


def test_update_status_keeps_the_previous_timestamp_when_none_is_passed(
    db_connection: psycopg.Connection,
) -> None:
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}keep_sent_at"))
    sent_at = datetime.now(timezone.utc).replace(microsecond=0)
    repository.update_status(post_id, "sent", sent_at)

    repository.update_status(post_id, "failed")

    row = _row(db_connection, post_id)
    assert row["status"] == "failed"
    assert row["sent_at"] == sent_at


def test_update_status_rejects_a_status_outside_the_schema(db_connection: psycopg.Connection) -> None:
    """The CHECK constraint in db/schema.sql is the last line of defence (section 10)."""
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}bad_status"))

    with pytest.raises(psycopg.errors.CheckViolation):
        repository.update_status(post_id, "teleported")
