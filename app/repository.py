"""All Postgres access for the ``posts`` table (FR-2, FR-9, FR-10 status, FR-11).

Schema changes only ever happen in ``db/schema.sql`` (Invariant 9).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.models import AnalyzedPost, PostRecord

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO posts (
    reddit_id, subreddit, source_topic_key, title, url, author, raw_content, published_at,
    is_relevant, duplicate_of_id, topic, importance, summary_fa, key_points,
    llm_raw_response, status
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s,
    %s, %s
)
ON CONFLICT (reddit_id) DO NOTHING
RETURNING id
"""

_CANDIDATES_SQL = """
SELECT id, reddit_id, subreddit, source_topic_key, title, url, author,
       topic, importance, summary_fa, key_points, published_at, status
FROM posts
WHERE summary_fa IS NOT NULL
  AND COALESCE(published_at, fetched_at) >= now() - make_interval(hours => %s)
ORDER BY COALESCE(published_at, fetched_at) DESC
LIMIT %s
"""

_PENDING_SEND_SQL = """
SELECT id, reddit_id, subreddit, source_topic_key, title, url, author, topic,
       importance, summary_fa, key_points, published_at, status
FROM posts
WHERE status = 'to_send'
ORDER BY id
"""


@contextmanager
def connect(database_url: str) -> Iterator[psycopg.Connection]:
    """Open a connection for one pipeline run and always close it.

    The connection runs in autocommit mode: every write is durably stored before the
    next pipeline step continues, which is what Invariant 2(d) requires (a post is
    persisted *before* any attempt to send it).
    """
    connection = psycopg.connect(database_url, row_factory=dict_row, autocommit=True)
    try:
        yield connection
    finally:
        connection.close()


class PostRepository:
    """Every query the pipeline needs; nothing else talks to the database."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def exists(self, reddit_id: str) -> bool:
        """FR-2 — cheap exact duplicate check before spending an LLM call."""
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM posts WHERE reddit_id = %s", (reddit_id,))
            return cursor.fetchone() is not None

    def get_similarity_candidates(self, *, limit: int, hours: int) -> list[PostRecord]:
        """FR-4 / Invariant 10 — at most ``limit`` recently analysed posts.

        Only the raw rows are returned; the presenter (``analyzer``) is responsible for
        exposing just their 1-based position/title/summary to the LLM (Invariant 4).
        """
        with self._connection.cursor() as cursor:
            cursor.execute(_CANDIDATES_SQL, (hours, limit))
            rows = cursor.fetchall()

        candidates = [PostRecord.model_validate(row) for row in rows]
        logger.debug("Loaded %d similarity candidate(s)", len(candidates))
        return candidates

    def insert_post(self, record: AnalyzedPost) -> int | None:
        """FR-9 — persist the full record; the returned id is the real row id.

        Returns ``None`` when the ``reddit_id`` was already stored (the UNIQUE
        constraint makes a concurrent second run a no-op — NFR-1/Invariant 1).
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                _INSERT_SQL,
                (
                    record.reddit_id,
                    record.subreddit,
                    record.source_topic_key,
                    record.title,
                    record.url,
                    record.author,
                    record.raw_content,
                    record.published_at,
                    record.is_relevant,
                    record.duplicate_of_id,
                    record.topic,
                    record.importance,
                    record.summary_fa,
                    Jsonb(record.key_points),
                    Jsonb(record.llm_raw_response) if record.llm_raw_response else None,
                    record.status,
                ),
            )
            row = cursor.fetchone()

        if row is None:
            logger.info("reddit_id=%s already stored, insert skipped", record.reddit_id)
            return None

        post_id = int(row["id"])
        logger.info("Stored reddit_id=%s as id=%s status=%s", record.reddit_id, post_id, record.status)
        return post_id

    def mark_sent(self, post_id: int) -> None:
        """FR-10 — final state after a successful Telegram send."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                "UPDATE posts SET status = 'sent', sent_at = now() WHERE id = %s",
                (post_id,),
            )
        logger.info("Marked id=%s as sent", post_id)

    def fetch_pending_send(self) -> list[PostRecord]:
        """FR-11 — posts analysed earlier whose send never completed."""
        with self._connection.cursor() as cursor:
            cursor.execute(_PENDING_SEND_SQL)
            rows = cursor.fetchall()
        return [PostRecord.model_validate(row) for row in rows]
