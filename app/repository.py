"""All Postgres access for the ``posts`` table (FR-2, FR-4, FR-9, FR-11).

Raw parametrized SQL through psycopg 3 — no ORM and no migration tool (AGENTS.md
section 8); schema changes only ever happen in ``db/schema.sql`` (Invariant 9).

Every function opens one short-lived autocommit connection and closes it again: the
pipeline is single-threaded and runs once per ``POLL_INTERVAL_SECONDS``, so a
connection pool would only add moving parts. Autocommit keeps the write durable the
moment it returns, which is what Invariant 2(d) needs — a post is stored *before*
any attempt to send it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.models import PostRecord
from app.settings import get_settings

logger = logging.getLogger(__name__)

# One column list for every read, in the order PostRecord expects them.
_POST_COLUMNS = """
id, reddit_id, subreddit, source_topic_key, title, url, author, raw_content,
published_at, is_relevant, duplicate_of_id, topic, importance, summary_fa,
key_points, status
"""

_SAVE_SQL = f"""
INSERT INTO posts (
    reddit_id, subreddit, source_topic_key, title, url, author, raw_content, published_at,
    is_relevant, duplicate_of_id, topic, importance, summary_fa, key_points,
    llm_raw_response, status
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s
)
ON CONFLICT (reddit_id) DO NOTHING
RETURNING id
"""

_ID_BY_REDDIT_ID_SQL = "SELECT id FROM posts WHERE reddit_id = %s"

_CANDIDATES_SQL = f"""
SELECT {_POST_COLUMNS}
FROM posts
WHERE summary_fa IS NOT NULL
  AND COALESCE(published_at, fetched_at) >= now() - make_interval(hours => %s)
ORDER BY COALESCE(published_at, fetched_at) DESC
LIMIT %s
"""

_PENDING_TO_SEND_SQL = f"""
SELECT {_POST_COLUMNS}
FROM posts
WHERE status = 'to_send'
ORDER BY id
"""

_UPDATE_STATUS_SQL = "UPDATE posts SET status = %s, sent_at = COALESCE(%s, sent_at) WHERE id = %s"


@contextmanager
def _connection() -> Iterator[psycopg.Connection]:
    """One short-lived autocommit connection per repository call (no pool)."""
    with psycopg.connect(
        get_settings().database_url, row_factory=dict_row, autocommit=True
    ) as connection:
        yield connection


def exists(reddit_id: str) -> bool:
    """FR-2 — cheap exact duplicate check before spending an LLM call."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM posts WHERE reddit_id = %s", (reddit_id,))
        return cursor.fetchone() is not None


def save(post: PostRecord) -> int:
    """FR-9 — persist one analysed post and return its real row id.

    Idempotent by design (NFR-1 / Invariant 1): saving the same ``reddit_id`` again
    inserts nothing and returns the id already stored, so a repeated or overlapping
    run can never write a second row.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            _SAVE_SQL,
            (
                post.reddit_id,
                post.subreddit,
                post.source_topic_key,
                post.title,
                post.url,
                post.author,
                post.raw_content,
                post.published_at,
                post.is_relevant,
                post.duplicate_of_id,
                post.topic,
                post.importance,
                post.summary_fa,
                Jsonb(post.key_points),
                Jsonb(post.llm_raw_response) if post.llm_raw_response else None,
                post.status,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            post_id = int(row["id"])
            logger.info(
                "Stored reddit_id=%s as id=%s status=%s", post.reddit_id, post_id, post.status
            )
            return post_id

        cursor.execute(_ID_BY_REDDIT_ID_SQL, (post.reddit_id,))
        existing = cursor.fetchone()

    if existing is None:  # pragma: no cover - the row cannot vanish between the two statements
        raise RuntimeError(f"reddit_id={post.reddit_id} was not visible after a conflicting insert")

    post_id = int(existing["id"])
    logger.info(
        "reddit_id=%s already stored as id=%s; save was a no-op", post.reddit_id, post_id
    )
    return post_id


def update_status(post_id: int, status: str, sent_at: datetime | None = None) -> None:
    """Move a stored post to another status (FR-10, FR-11).

    ``sent_at`` is written only when the caller passes a timestamp; passing ``None``
    leaves an already recorded timestamp untouched.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_UPDATE_STATUS_SQL, (status, sent_at, post_id))
    logger.info("id=%s status -> %s", post_id, status)


def fetch_recent_candidates(limit: int, hours: int) -> list[PostRecord]:
    """FR-4 / Invariant 10 — at most ``limit`` analysed posts of the last ``hours``.

    Newest first, so the indexes handed to the LLM stay stable and bounded.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_CANDIDATES_SQL, (hours, limit))
        rows = cursor.fetchall()

    candidates = [PostRecord.model_validate(row) for row in rows]
    logger.debug("Loaded %d similarity candidate(s)", len(candidates))
    return candidates


def fetch_pending_to_send() -> list[PostRecord]:
    """FR-11 — posts analysed earlier whose send never completed."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_PENDING_TO_SEND_SQL)
        rows = cursor.fetchall()
    return [PostRecord.model_validate(row) for row in rows]
