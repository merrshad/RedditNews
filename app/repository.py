"""All Postgres access for ``topics``, ``sources`` and ``posts`` (FR-2..FR-14).

Raw parametrized SQL through psycopg 3 — no ORM and no migration tool (AGENTS.md
section 8); schema changes only ever happen in ``db/schema.sql`` (Invariant 9).

Every function opens one short-lived autocommit connection and closes it again: the
pipeline is single-threaded and runs once per ``POLL_INTERVAL_SECONDS``, so a
connection pool would only add moving parts. Autocommit keeps the write durable the
moment it returns, which is what Invariant 2 needs — a post is stored *before* any
attempt to send it anywhere.

The review/publication transitions are **conditional single statements** on purpose
(Invariant 12): ``UPDATE ... WHERE id = %s AND status = <expected>`` succeeds for exactly
one caller, so two admins clicking at the same time, or a retried publication, can never
run the transition twice. The boolean these functions return is "I won the race".
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.models import (
    LlmAnalysis,
    PostRecord,
    ReviewDecision,
    SourceRecord,
    TopicRecord,
)
from app.settings import get_settings

logger = logging.getLogger(__name__)

# One column list for every read, in the order PostRecord expects them.
_POST_COLUMNS = """
id, reddit_id, subreddit, source_topic_key, title, url, author, raw_content,
posted_at, status, review_status, reviewed_by, reviewed_by_name, reviewed_at, approved_at,
rejected_at, private_channel_id, private_message_id, is_relevant, duplicate_of_id, topic,
importance, summary_fa, key_points, llm_raw_response, ai_processed_at, ai_error,
public_channel_id, public_message_id, published_at
"""

_TOPIC_COLUMNS = "id, key, name, is_active"

# `topic_key` is the joined topic key, which is what every caller wants to see.
_SOURCE_COLUMNS = "s.id, t.key AS topic_key, s.rss_url, s.fetch_limit, s.is_active"

_SAVE_SQL = f"""
INSERT INTO posts (
    reddit_id, subreddit, source_topic_key, title, url, author, raw_content, posted_at,
    status, review_status
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s
)
ON CONFLICT (reddit_id) DO NOTHING
RETURNING id
"""

_ID_BY_REDDIT_ID_SQL = "SELECT id FROM posts WHERE reddit_id = %s"

_CANDIDATES_SQL = f"""
SELECT {_POST_COLUMNS}
FROM posts
WHERE summary_fa IS NOT NULL
  AND COALESCE(posted_at, fetched_at) >= now() - make_interval(hours => %s)
ORDER BY COALESCE(posted_at, fetched_at) DESC
LIMIT %s
"""

# --- the three queues the worker drains, each in its own status (FR-11, FR-12, FR-14) --

_NEW_REVIEWS_SQL = f"SELECT {_POST_COLUMNS} FROM posts WHERE status = 'new' ORDER BY id"

_APPROVED_SQL = f"SELECT {_POST_COLUMNS} FROM posts WHERE status = 'approved' ORDER BY id"

_PENDING_TO_SEND_SQL = f"SELECT {_POST_COLUMNS} FROM posts WHERE status = 'to_send' ORDER BY id"

# Claiming and finishing a publication (Invariant 12: never published twice).
_CLAIM_PUBLISH_SQL = """
UPDATE posts SET status = 'publishing'
WHERE id = %s AND status = 'to_send'
RETURNING id
"""

_MARK_PUBLISHED_SQL = """
UPDATE posts SET status = 'sent', public_channel_id = %s, public_message_id = %s,
                 published_at = now()
WHERE id = %s AND status = 'publishing'
RETURNING id
"""

_RELEASE_PUBLISH_SQL = """
UPDATE posts SET status = 'to_send'
WHERE id = %s AND status = 'publishing'
RETURNING id
"""

# Marking a review message as delivered (new -> awaiting_review).
_MARK_DISPATCHED_SQL = """
UPDATE posts SET status = 'awaiting_review', private_channel_id = %s, private_message_id = %s
WHERE id = %s AND status = 'new'
RETURNING id
"""

# The two admin decisions. Both are single conditional statements, so exactly one of two
# simultaneous clicks wins (Invariant 12); the loser gets False and a "already reviewed".
_APPROVE_SQL = """
UPDATE posts SET review_status = 'approved', status = 'approved', reviewed_by = %s,
                 reviewed_by_name = %s, reviewed_at = now(), approved_at = now()
WHERE id = %s AND status = 'awaiting_review' AND review_status = 'pending_review'
RETURNING id
"""

_REJECT_SQL = """
UPDATE posts SET review_status = 'rejected', status = 'rejected', reviewed_by = %s,
                 reviewed_by_name = %s, reviewed_at = now(), rejected_at = now()
WHERE id = %s AND status = 'awaiting_review' AND review_status = 'pending_review'
RETURNING id
"""

# The AI result: the analysed fields, the error and the new state, in one write.
_RECORD_ANALYSIS_SQL = """
UPDATE posts
SET is_relevant = %s, duplicate_of_id = %s, topic = %s, importance = %s, summary_fa = %s,
    key_points = %s, llm_raw_response = %s, ai_error = %s, ai_processed_at = now(),
    status = %s
WHERE id = %s AND status = 'approved'
RETURNING id
"""


@contextmanager
def _connection() -> Iterator[psycopg.Connection]:
    """One short-lived autocommit connection per repository call (no pool)."""
    with psycopg.connect(
        get_settings().database_url, row_factory=dict_row, autocommit=True
    ) as connection:
        yield connection


# --- topics and sources (FR-14: the admin owns them, not the code) ------------------


def list_topics(*, active_only: bool = False) -> list[TopicRecord]:
    """Every configured topic, sorted by key; ``active_only`` filters disabled ones."""
    sql = f"SELECT {_TOPIC_COLUMNS} FROM topics"
    if active_only:
        sql += " WHERE is_active"
    sql += " ORDER BY key"
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return [TopicRecord.model_validate(row) for row in rows]


def topic_exists(key: str) -> bool:
    """Used before creating a source, so the admin gets "unknown topic" and not "duplicate"."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM topics WHERE key = %s", (key,))
        return cursor.fetchone() is not None


def create_topic(key: str, name: str) -> TopicRecord | None:
    """Create a topic; ``None`` when the key is already taken."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO topics (key, name) VALUES (%s, %s)
            ON CONFLICT (key) DO NOTHING
            RETURNING {_TOPIC_COLUMNS}
            """,
            (key, name),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    logger.info("Created topic key=%s name=%s", key, name)
    return TopicRecord.model_validate(row)


def update_topic_name(key: str, name: str) -> bool:
    """Rename a topic (the key stays stable because past posts point at it)."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE topics SET name = %s, updated_at = now() WHERE key = %s RETURNING id",
            (name, key),
        )
        return cursor.fetchone() is not None


def set_topic_active(key: str, is_active: bool) -> bool:
    """Enable/disable a topic (and, transitively, the feeds that belong to it)."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE topics SET is_active = %s, updated_at = now() WHERE key = %s RETURNING id",
            (is_active, key),
        )
        return cursor.fetchone() is not None


def delete_topic(key: str) -> bool:
    """Delete a topic and its sources.

    Already-analysed posts keep their ``source_topic_key`` text, so the audit trail and
    the review history survive the delete (``posts`` has no foreign key to ``topics``).
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM topics WHERE key = %s RETURNING id", (key,))
        return cursor.fetchone() is not None


def list_sources(*, active_only: bool = False) -> list[SourceRecord]:
    """Every feed with its topic; ``active_only`` also drops feeds of disabled topics."""
    sql = f"""
    SELECT {_SOURCE_COLUMNS}
    FROM sources s JOIN topics t ON t.id = s.topic_id
    """
    if active_only:
        sql += " WHERE s.is_active AND t.is_active"
    sql += " ORDER BY t.key, s.rss_url"
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return [SourceRecord.model_validate(row) for row in rows]


def create_source(topic_key: str, rss_url: str, fetch_limit: int | None = None) -> SourceRecord | None:
    """Add a feed to a topic; ``None`` when the URL is already configured.

    Callers check :func:`topic_exists` first when they want to tell the two failure
    reasons apart (the admin commands do).
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO sources (topic_id, rss_url, fetch_limit)
            SELECT t.id, %s, %s FROM topics t WHERE t.key = %s
            ON CONFLICT (rss_url) DO NOTHING
            RETURNING id
            """,
            (rss_url, fetch_limit, topic_key),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    logger.info("Created source rss_url=%s topic=%s fetch_limit=%s", rss_url, topic_key, fetch_limit)
    return SourceRecord(
        id=int(row["id"]),
        topic_key=topic_key,
        rss_url=rss_url,
        fetch_limit=fetch_limit,
        is_active=True,
    )


def set_source_topic(rss_url: str, topic_key: str) -> bool:
    """Re-assign a feed to another (existing) topic."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE sources s SET topic_id = t.id, updated_at = now()
            FROM topics t
            WHERE s.rss_url = %s AND t.key = %s
            RETURNING s.id
            """,
            (rss_url, topic_key),
        )
        return cursor.fetchone() is not None


def set_source_fetch_limit(rss_url: str, fetch_limit: int | None) -> bool:
    """Set a per-source batch cap; ``None`` falls back to ``RSS_FETCH_LIMIT``."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE sources SET fetch_limit = %s, updated_at = now()
            WHERE rss_url = %s
            RETURNING id
            """,
            (fetch_limit, rss_url),
        )
        return cursor.fetchone() is not None


def set_source_active(rss_url: str, is_active: bool) -> bool:
    """Enable/disable one feed."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE sources SET is_active = %s, updated_at = now()
            WHERE rss_url = %s
            RETURNING id
            """,
            (is_active, rss_url),
        )
        return cursor.fetchone() is not None


def delete_source(rss_url: str) -> bool:
    """Delete one feed."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM sources WHERE rss_url = %s RETURNING id", (rss_url,))
        return cursor.fetchone() is not None


# --- posts --------------------------------------------------------------------------


# The three queries this code cannot run without, each using the same column list as the
# real reads. A database created by an earlier phase has the wrong shape here, and
# `docker compose up` cannot fix it: `db/schema.sql` only runs on a *first* boot
# (Invariant 9 — there are no migrations). Hence the startup preflight below.
_SCHEMA_PROBES: tuple[tuple[str, str], ...] = (
    ("topics", f"SELECT {_TOPIC_COLUMNS} FROM topics LIMIT 0"),
    ("sources", f"SELECT {_SOURCE_COLUMNS} FROM sources s JOIN topics t ON t.id = s.topic_id LIMIT 0"),
    ("posts", f"SELECT {_POST_COLUMNS} FROM posts LIMIT 0"),
)


def find_unusable_tables() -> list[str]:
    """Tables this code cannot query with its own column lists (empty list = ready).

    Catches both a table that does not exist and one left over from an earlier phase (right
    name, wrong columns) — which is exactly what a Docker volume from a previous phase looks
    like. Being schema-aware here is what turns a worker that silently logs
    ``UndefinedTable`` every cycle into one clear startup error (Invariant 9, NFR-3).
    """
    problems: list[str] = []
    with _connection() as connection:
        for name, probe in _SCHEMA_PROBES:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(probe)
                    cursor.fetchall()
            except psycopg.Error:
                # Autocommit: the failed probe is its own transaction, so the next one runs.
                problems.append(name)
    return problems


def exists(reddit_id: str) -> bool:
    """FR-2 — cheap exact duplicate check before storing anything."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM posts WHERE reddit_id = %s", (reddit_id,))
        return cursor.fetchone() is not None


def save(post: PostRecord) -> int:
    """FR-9 — persist one fetched post and return its real row id.

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
                post.posted_at,
                post.status,
                post.review_status,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            post_id = int(row["id"])
            logger.info(
                "Stored reddit_id=%s as id=%s status=%s",
                post.reddit_id,
                post_id,
                post.status,
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


def fetch_new_reviews() -> list[PostRecord]:
    """FR-12 — posts stored but whose review message has not been delivered yet."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_NEW_REVIEWS_SQL)
        rows = cursor.fetchall()
    return [PostRecord.model_validate(row) for row in rows]


def mark_review_dispatched(post_id: int, *, channel_id: str, message_id: int) -> bool:
    """Record that the post is now visible in the review channel (``new`` -> ``awaiting_review``).

    Returns ``False`` when the row moved on in the meantime, which keeps a repeated
    dispatch from overwriting a fresher state.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_MARK_DISPATCHED_SQL, (channel_id, message_id, post_id))
        won = cursor.fetchone() is not None
    if won:
        logger.info(
            "id=%s status -> awaiting_review (review message %s in %s)",
            post_id,
            message_id,
            channel_id,
        )
    return won


def decide_review(
    post_id: int,
    *,
    decision: ReviewDecision,
    admin_id: str,
    admin_name: str | None = None,
) -> bool:
    """FR-12/Invariant 12 — record the admin's decision, exactly once.

    ``True`` means this call is the one that made the decision; ``False`` means the post
    was already approved or rejected (the second admin gets told so). ``admin_id`` is the
    authority; ``admin_name`` is the display name the review message shows (phase 6).
    """
    sql = _APPROVE_SQL if decision == "approved" else _REJECT_SQL
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(sql, (admin_id, admin_name, post_id))
        won = cursor.fetchone() is not None
    if won:
        logger.info(
            "id=%s review decision: %s by admin %s (%s)",
            post_id,
            decision,
            admin_id,
            admin_name or "no display name",
        )
    else:
        logger.info("id=%s review decision: %s ignored (already reviewed)", post_id, decision)
    return won


def fetch_approved_for_analysis() -> list[PostRecord]:
    """FR-13 — posts an admin approved, waiting for the single LLM call."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_APPROVED_SQL)
        rows = cursor.fetchall()
    return [PostRecord.model_validate(row) for row in rows]


def record_analysis(
    post_id: int,
    *,
    analysis: LlmAnalysis | None,
    duplicate_of_id: int | None,
    raw_response: dict[str, Any] | None,
    status: str,
    ai_error: str | None = None,
) -> bool:
    """FR-13 — store one LLM result (valid or unusable) and move the post on.

    ``analysis`` is ``None`` when the model answer failed validation: then the analysis
    columns stay empty and only ``status='failed'``/``ai_error`` are written (Invariant 3).
    Returns whether the row was still in the ``approved`` state.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            _RECORD_ANALYSIS_SQL,
            (
                analysis.is_relevant if analysis else None,
                duplicate_of_id,
                analysis.topic if analysis else None,
                analysis.importance if analysis else None,
                analysis.summary_fa if analysis else None,
                Jsonb(list(analysis.key_points)) if analysis else None,
                Jsonb(raw_response) if raw_response else None,
                ai_error,
                status,
                post_id,
            ),
        )
        row = cursor.fetchone()
    if row is None:
        logger.warning("id=%s was no longer approved; the analysis result was dropped", post_id)
        return False
    logger.info("id=%s analysed -> status=%s", post_id, status)
    return True


def fetch_pending_to_send() -> list[PostRecord]:
    """FR-11 — posts ready to publish whose send never completed."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_PENDING_TO_SEND_SQL)
        rows = cursor.fetchall()
    return [PostRecord.model_validate(row) for row in rows]


def claim_for_publish(post_id: int) -> bool:
    """Invariant 12 — take the publication claim (``to_send`` -> ``publishing``).

    Only the caller that flips the row may call Telegram, so a post is never published
    twice even if two runs overlap or a decision is replayed.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_CLAIM_PUBLISH_SQL, (post_id,))
        claimed = cursor.fetchone() is not None
    logger.debug("id=%s publish claim: %s", post_id, "taken" if claimed else "already taken")
    return claimed


def mark_published(post_id: int, *, channel_id: str, message_id: int) -> bool:
    """Finish a claimed publication (``publishing`` -> ``sent``) with its audit ids."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_MARK_PUBLISHED_SQL, (channel_id, message_id, post_id))
        done = cursor.fetchone() is not None
    if done:
        logger.info("id=%s status -> sent (public message %s)", post_id, message_id)
    return done


def release_publish_claim(post_id: int) -> bool:
    """Give a claimed publication back to the queue after Telegram refused it (FR-11)."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_RELEASE_PUBLISH_SQL, (post_id,))
        released = cursor.fetchone() is not None
    if released:
        logger.info("id=%s status -> to_send (the send failed, retrying next run)", post_id)
    return released


def fetch_recent_candidates(limit: int, hours: int) -> list[PostRecord]:
    """FR-4 / Invariant 10 — at most ``limit`` analysed posts of the last ``hours``.

    Newest first, so the indexes handed to the LLM stay stable and bounded. Only analysed
    posts (``summary_fa`` present) are candidates, i.e. posts that cleared the review gate.
    """
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(_CANDIDATES_SQL, (hours, limit))
        rows = cursor.fetchall()

    candidates = [PostRecord.model_validate(row) for row in rows]
    logger.debug("Loaded %d similarity candidate(s)", len(candidates))
    return candidates


def fetch_post(post_id: int) -> PostRecord | None:
    """One post by id (used by the admin/review handlers to re-read stored state)."""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT {_POST_COLUMNS} FROM posts WHERE id = %s", (post_id,))
        row = cursor.fetchone()
    return PostRecord.model_validate(row) if row else None
