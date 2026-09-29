"""FR-2 / FR-4 / FR-9 / FR-11..FR-14 against a real Postgres (`docker compose up -d db`).

Deliberately integration tests rather than cursor mocks: the parametrized SQL, the JSONB
round-trip, the UNIQUE(reddit_id) idempotency, the taxonomy constraints and the atomic
state transitions are exactly what this phase must prove, and none of them can be proven
by a fake cursor. They skip themselves with a clear message when no Postgres is reachable.

The transitions are single conditional statements (Invariant 12), so every one of them is
tested twice: the winning call reports ``True``, the losing call reports ``False`` and
leaves the row where it was.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import pytest

from app import repository
from app.models import LlmAnalysis, PostRecord
from tests.conftest import (
    TEST_REDDIT_ID_PREFIX,
    temp_source_url,
    temp_topic_key,
)

TOPIC_KEY = temp_topic_key("repo")


def _record(reddit_id: str, **overrides: Any) -> PostRecord:
    """A freshly fetched post; ``overrides`` tweak one field per test."""
    fields: dict[str, Any] = {
        "reddit_id": reddit_id,
        "subreddit": "mlops",
        "source_topic_key": TOPIC_KEY,
        "title": "A new open model was released",
        "url": f"https://www.reddit.com/r/mlops/comments/{reddit_id}/x/",
        "author": "somebody",
        "raw_content": "Body text",
        "posted_at": datetime.now(timezone.utc),
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


def _approved(connection: psycopg.Connection, reddit_id: str) -> int:
    """Walk one post from storage to the ``approved`` queue, through the real transitions."""
    post_id = repository.save(_record(reddit_id))
    assert repository.mark_review_dispatched(post_id, channel_id="-100999", message_id=7) is True
    assert repository.decide_review(post_id, decision="approved", admin_id="777") is True
    return post_id


def _analysis(**overrides: Any) -> LlmAnalysis:
    fields: dict[str, Any] = {
        "is_relevant": True,
        "duplicate_of_candidate_index": None,
        "topic": TOPIC_KEY,
        "importance": "high",
        "summary_fa": "خلاصه فارسی پست.",
        "key_points": ["نکته اول", "نکته دوم"],
    }
    fields.update(overrides)
    return LlmAnalysis(**fields)


@pytest.fixture
def topic_row(db_connection: psycopg.Connection) -> str:
    """A test topic created through the repository, removed by the cleanup fixture."""
    topic = repository.create_topic(TOPIC_KEY, "موضوع تست")
    assert topic is not None
    return topic.key


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


def test_save_persists_the_fetched_fields_and_starts_the_review_cycle(
    db_connection: psycopg.Connection,
) -> None:
    record = _record(f"{TEST_REDDIT_ID_PREFIX}fields")

    post_id = repository.save(record)

    row = _row(db_connection, post_id)
    assert row["reddit_id"] == record.reddit_id
    assert row["subreddit"] == "mlops"
    assert row["source_topic_key"] == TOPIC_KEY
    assert row["title"] == record.title
    assert row["url"] == record.url
    assert row["author"] == "somebody"
    assert row["raw_content"] == "Body text"
    assert row["posted_at"] == record.posted_at
    # A saved post is *not* analysed: it waits for a human (FR-12/FR-13).
    assert row["status"] == "new" and row["review_status"] == "pending_review"
    assert row["is_relevant"] is None and row["summary_fa"] is None
    assert row["llm_raw_response"] is None and row["ai_processed_at"] is None
    assert row["published_at"] is None and row["ai_error"] is None


def test_save_is_idempotent_for_the_same_reddit_id(db_connection: psycopg.Connection) -> None:
    """NFR-1 / Invariant 1 — a repeated run never writes a second row."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}dupe"

    first = repository.save(_record(reddit_id, title="اولین عنوان"))
    second = repository.save(_record(reddit_id, title="دومین عنوان"))

    assert first == second
    assert _count(db_connection, reddit_id) == 1
    # The conflicting save must not overwrite what is already stored.
    assert _row(db_connection, first)["title"] == "اولین عنوان"


def test_save_never_overwrites_the_state_of_an_existing_row(
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 1 / NFR-1, the part a re-run could realistically get wrong.

    `save` is an `INSERT ... ON CONFLICT DO NOTHING`, so a second fetch of the same
    ``reddit_id`` cannot move the row backwards: an analysed, published post stays
    ``sent`` with its ids and summary.
    """
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}no_overwrite"
    post_id = _approved(db_connection, reddit_id)
    repository.record_analysis(
        post_id,
        analysis=_analysis(),
        duplicate_of_id=None,
        raw_response={"is_relevant": True},
        status="to_send",
    )
    assert repository.claim_for_publish(post_id) is True
    assert repository.mark_published(post_id, channel_id="-100123", message_id=55) is True

    again = repository.save(_record(reddit_id, title="عنوان دیگر"))

    assert again == post_id
    assert _count(db_connection, reddit_id) == 1
    row = _row(db_connection, post_id)
    assert row["status"] == "sent"  # the re-fetch cannot undo the publication
    assert row["summary_fa"] == "خلاصه فارسی پست."
    assert row["title"] == "A new open model was released"


# --- topics and sources (FR-14) --------------------------------------------------


def test_create_topic_refuses_a_duplicate_key(db_connection: psycopg.Connection) -> None:
    assert repository.create_topic(TOPIC_KEY, "موضوع تست") is not None

    assert repository.create_topic(TOPIC_KEY, "نام تکراری") is None  # the key is the identity


def test_topics_can_be_listed_renamed_toggled_and_deleted(
    db_connection: psycopg.Connection, topic_row: str
) -> None:
    listed = {topic.key: topic for topic in repository.list_topics()}
    assert listed[topic_row].name == "موضوع تست"
    assert listed[topic_row].is_active is True

    assert repository.update_topic_name(topic_row, "نام تازه") is True
    assert repository.set_topic_active(topic_row, False) is True

    refreshed = {topic.key: topic for topic in repository.list_topics()}
    assert refreshed[topic_row].name == "نام تازه"
    assert refreshed[topic_row].is_active is False
    assert topic_row not in {topic.key for topic in repository.list_topics(active_only=True)}

    assert repository.delete_topic(topic_row) is True
    assert topic_row not in {topic.key for topic in repository.list_topics()}
    # Unknown keys are simply reported as "not found", never as an exception.
    assert repository.update_topic_name("t_test_missing", "x") is False
    assert repository.set_topic_active("t_test_missing", True) is False
    assert repository.delete_topic("t_test_missing") is False


def test_a_source_carries_its_topic_and_its_own_fetch_limit(
    db_connection: psycopg.Connection, topic_row: str
) -> None:
    source = repository.create_source(topic_row, temp_source_url("a.rss"), 50)

    assert source is not None
    assert source.topic_key == topic_row and source.fetch_limit == 50
    assert source.is_active is True

    stored = next(s for s in repository.list_sources() if s.rss_url == temp_source_url("a.rss"))
    assert stored.fetch_limit == 50 and stored.topic_key == topic_row

    assert repository.set_source_fetch_limit(stored.rss_url, None) is True
    assert next(
        s for s in repository.list_sources() if s.rss_url == stored.rss_url
    ).fetch_limit is None


def test_create_source_requires_an_existing_topic(db_connection: psycopg.Connection) -> None:
    """A feed cannot hang off a topic that does not exist (`sources.topic_id` is a FK)."""
    assert repository.topic_exists("t_test_absent") is False
    assert repository.create_source("t_test_absent", temp_source_url("orphan.rss")) is None
    assert repository.topic_exists(TOPIC_KEY) is False


def test_a_duplicate_feed_url_is_refused_but_the_topic_may_be_checked_first(
    db_connection: psycopg.Connection, topic_row: str
) -> None:
    url = temp_source_url("dup.rss")
    assert repository.create_source(topic_row, url) is not None

    assert repository.create_source(topic_row, url) is None  # UNIQUE(rss_url)
    assert repository.topic_exists(topic_row) is True


def test_a_source_can_be_moved_to_another_topic(
    db_connection: psycopg.Connection, topic_row: str
) -> None:
    other_key = temp_topic_key("other")
    assert repository.create_topic(other_key, "موضوع دیگر") is not None
    assert repository.create_source(topic_row, temp_source_url("move.rss")) is not None

    assert repository.set_source_topic(temp_source_url("move.rss"), other_key) is True

    moved = next(s for s in repository.list_sources() if s.rss_url == temp_source_url("move.rss"))
    assert moved.topic_key == other_key
    assert repository.set_source_topic(temp_source_url("ghost.rss"), other_key) is False


def test_disabling_a_topic_hides_its_feeds_from_the_worker(
    db_connection: psycopg.Connection,
) -> None:
    """`list_sources(active_only=True)` is what the pipeline fetches (NFR-7)."""
    assert repository.create_topic(TOPIC_KEY, "موضوع تست") is not None
    assert repository.create_source(TOPIC_KEY, temp_source_url("off.rss")) is not None

    assert temp_source_url("off.rss") in {s.rss_url for s in repository.list_sources(active_only=True)}

    assert repository.set_topic_active(TOPIC_KEY, False) is True

    assert temp_source_url("off.rss") not in {
        s.rss_url for s in repository.list_sources(active_only=True)
    }
    # ... while the admin listing still shows it, so it can be turned back on.
    assert temp_source_url("off.rss") in {s.rss_url for s in repository.list_sources()}


def test_a_disabled_source_is_not_fetched_and_can_be_deleted(
    db_connection: psycopg.Connection, topic_row: str
) -> None:
    url = temp_source_url("toggled.rss")
    assert repository.create_source(topic_row, url) is not None

    assert repository.set_source_active(url, False) is True
    assert url not in {s.rss_url for s in repository.list_sources(active_only=True)}
    assert repository.set_source_active("https://ghost.example/feed.rss", True) is False

    assert repository.delete_source(url) is True
    assert repository.delete_source(url) is False


def test_deleting_a_topic_cascades_to_its_sources(db_connection: psycopg.Connection) -> None:
    assert repository.create_topic(TOPIC_KEY, "موضوع تست") is not None
    assert repository.create_source(TOPIC_KEY, temp_source_url("cascade.rss")) is not None

    assert repository.delete_topic(TOPIC_KEY) is True

    assert temp_source_url("cascade.rss") not in {s.rss_url for s in repository.list_sources()}


def test_seeded_topics_are_available_on_a_fresh_database(
    db_connection: psycopg.Connection,
) -> None:
    """The starting taxonomy comes from `db/schema.sql` (Invariant 9), not from code."""
    keys = {topic.key for topic in repository.list_topics()}

    assert {"ai", "startup"} <= keys
    assert any(source.topic_key == "ai" for source in repository.list_sources())


# --- FR-12: the review transitions ------------------------------------------------


def test_a_stored_post_is_dispatched_to_review_exactly_once(
    db_connection: psycopg.Connection,
) -> None:
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}dispatch"))

    assert [row.id for row in repository.fetch_new_reviews()] == [post_id]

    assert repository.mark_review_dispatched(post_id, channel_id="-100999", message_id=31) is True
    # A second dispatch would overwrite the stored message id: it is refused.
    assert repository.mark_review_dispatched(post_id, channel_id="-100999", message_id=32) is False

    row = _row(db_connection, post_id)
    assert row["status"] == "awaiting_review"
    assert row["private_channel_id"] == "-100999" and row["private_message_id"] == 31
    assert repository.fetch_new_reviews() == []


def test_approving_records_who_decided_and_when(db_connection: psycopg.Connection) -> None:
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}approve"))
    repository.mark_review_dispatched(post_id, channel_id="-100999", message_id=1)

    assert repository.decide_review(post_id, decision="approved", admin_id="777") is True

    row = _row(db_connection, post_id)
    assert row["status"] == "approved" and row["review_status"] == "approved"
    assert row["reviewed_by"] == "777"
    assert row["reviewed_at"] is not None and row["approved_at"] is not None
    assert row["rejected_at"] is None
    assert [r.id for r in repository.fetch_approved_for_analysis()] == [post_id]


def test_rejecting_is_terminal_and_stops_the_post(db_connection: psycopg.Connection) -> None:
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}reject"))
    repository.mark_review_dispatched(post_id, channel_id="-100999", message_id=2)

    assert repository.decide_review(post_id, decision="rejected", admin_id="888") is True

    row = _row(db_connection, post_id)
    assert row["status"] == "rejected" and row["review_status"] == "rejected"
    assert row["rejected_at"] is not None and row["approved_at"] is None
    assert repository.fetch_approved_for_analysis() == []
    assert repository.fetch_pending_to_send() == []


def test_a_decision_is_only_recorded_once(db_connection: psycopg.Connection) -> None:
    """Invariant 12: two admins, one winner — at the storage layer."""
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}race"))
    repository.mark_review_dispatched(post_id, channel_id="-100999", message_id=3)

    assert repository.decide_review(post_id, decision="approved", admin_id="777") is True
    assert repository.decide_review(post_id, decision="rejected", admin_id="888") is False

    row = _row(db_connection, post_id)
    assert row["review_status"] == "approved" and row["reviewed_by"] == "777"
    assert row["rejected_at"] is None


def test_a_post_that_was_never_dispatched_cannot_be_decided(
    db_connection: psycopg.Connection,
) -> None:
    """The queue is `awaiting_review`: a row still `new` has no message to answer."""
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}not_dispatched"))

    assert repository.decide_review(post_id, decision="approved", admin_id="777") is False


# --- FR-13: the analysis result ---------------------------------------------------


def test_record_analysis_stores_the_verdict_and_moves_the_post(
    db_connection: psycopg.Connection,
) -> None:
    post_id = _approved(db_connection, f"{TEST_REDDIT_ID_PREFIX}analysed")

    assert repository.record_analysis(
        post_id,
        analysis=_analysis(importance="medium"),
        duplicate_of_id=None,
        raw_response=_analysis().model_dump(mode="json"),
        status="to_send",
    ) is True

    row = _row(db_connection, post_id)
    assert row["status"] == "to_send"
    assert row["is_relevant"] is True and row["importance"] == "medium"
    assert row["summary_fa"] == "خلاصه فارسی پست."
    assert row["key_points"] == ["نکته اول", "نکته دوم"]
    assert row["ai_processed_at"] is not None and row["ai_error"] is None
    assert [r.id for r in repository.fetch_pending_to_send()] == [post_id]


def test_record_analysis_can_mark_a_failed_ai_step(db_connection: psycopg.Connection) -> None:
    """Invariant 3: unusable output is stored with the reason, and no analysis."""
    post_id = _approved(db_connection, f"{TEST_REDDIT_ID_PREFIX}ai_failed")

    assert repository.record_analysis(
        post_id,
        analysis=None,
        duplicate_of_id=None,
        raw_response=None,
        status="failed",
        ai_error="LLM answer was not valid JSON",
    ) is True

    row = _row(db_connection, post_id)
    assert row["status"] == "failed"
    assert row["ai_error"] == "LLM answer was not valid JSON"
    assert row["is_relevant"] is None and row["summary_fa"] is None
    assert row["ai_processed_at"] is not None


def test_record_analysis_is_refused_once_the_post_left_the_queue(
    db_connection: psycopg.Connection,
) -> None:
    """A stale worker must not overwrite a finished post (the state guard)."""
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}stale"))

    assert repository.record_analysis(
        post_id,
        analysis=_analysis(),
        duplicate_of_id=None,
        raw_response=None,
        status="to_send",
    ) is False
    assert _row(db_connection, post_id)["summary_fa"] is None


# --- FR-10 / FR-11: publication ---------------------------------------------------


def test_publication_is_claimed_but_once(db_connection: psycopg.Connection) -> None:
    """Invariant 12: the claim is what makes a double publication impossible."""
    post_id = _approved(db_connection, f"{TEST_REDDIT_ID_PREFIX}claim")
    repository.record_analysis(
        post_id, analysis=_analysis(), duplicate_of_id=None, raw_response=None, status="to_send"
    )

    assert repository.claim_for_publish(post_id) is True
    assert repository.claim_for_publish(post_id) is False  # already claimed

    assert repository.mark_published(post_id, channel_id="-100123", message_id=99) is True
    row = _row(db_connection, post_id)
    assert row["status"] == "sent"
    assert row["public_channel_id"] == "-100123" and row["public_message_id"] == 99
    assert row["published_at"] is not None
    # And it is out of every queue.
    assert repository.fetch_pending_to_send() == []
    assert repository.claim_for_publish(post_id) is False


def test_a_failed_publication_returns_to_the_queue(db_connection: psycopg.Connection) -> None:
    """FR-11: the claim is handed back, so the next cycle retries the send."""
    post_id = _approved(db_connection, f"{TEST_REDDIT_ID_PREFIX}release")
    repository.record_analysis(
        post_id, analysis=_analysis(), duplicate_of_id=None, raw_response=None, status="to_send"
    )
    assert repository.claim_for_publish(post_id) is True

    assert repository.release_publish_claim(post_id) is True
    assert repository.release_publish_claim(post_id) is False  # nothing left to release

    assert [row.id for row in repository.fetch_pending_to_send()] == [post_id]
    assert _row(db_connection, post_id)["published_at"] is None


def test_fetch_pending_to_send_returns_only_to_send_rows_oldest_first(
    db_connection: psycopg.Connection,
) -> None:
    first = _approved(db_connection, f"{TEST_REDDIT_ID_PREFIX}pending_a")
    second = _approved(db_connection, f"{TEST_REDDIT_ID_PREFIX}pending_b")
    other = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}pending_other"))
    for post_id in (second, first):
        repository.record_analysis(
            post_id, analysis=_analysis(), duplicate_of_id=None, raw_response=None, status="to_send"
        )

    assert [row.id for row in repository.fetch_pending_to_send()] == [first, second]
    assert other not in [row.id for row in repository.fetch_pending_to_send()]


# --- FR-4 / Invariant 10: similarity candidates -----------------------------------


def _analysed_at(connection: psycopg.Connection, reddit_id: str, *, minutes_ago: int, summary: str) -> int:
    post_id = _approved(connection, reddit_id)
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE posts SET posted_at = now() - make_interval(mins => %s) WHERE id = %s",
            (minutes_ago, post_id),
        )
    repository.record_analysis(
        post_id,
        analysis=_analysis(summary_fa=summary),
        duplicate_of_id=None,
        raw_response=None,
        status="sent",
    )
    return post_id


def test_fetch_recent_candidates_is_newest_first_within_the_limit(
    db_connection: psycopg.Connection,
) -> None:
    oldest = _analysed_at(db_connection, f"{TEST_REDDIT_ID_PREFIX}c_old", minutes_ago=90, summary="قدیمی")
    middle = _analysed_at(db_connection, f"{TEST_REDDIT_ID_PREFIX}c_mid", minutes_ago=60, summary="میانی")
    newest = _analysed_at(db_connection, f"{TEST_REDDIT_ID_PREFIX}c_new", minutes_ago=30, summary="تازه")

    candidates = repository.fetch_recent_candidates(2, 72)

    assert [candidate.id for candidate in candidates] == [newest, middle]
    assert oldest not in [candidate.id for candidate in candidates]
    assert candidates[0].summary_fa == "تازه"


def test_fetch_recent_candidates_skips_unanalysed_and_old_posts(
    db_connection: psycopg.Connection,
) -> None:
    ancient = _analysed_at(db_connection, f"{TEST_REDDIT_ID_PREFIX}c_ancient", minutes_ago=10_000, summary="کهن")
    fresh = _analysed_at(db_connection, f"{TEST_REDDIT_ID_PREFIX}c_fresh", minutes_ago=5, summary="تازه")
    waiting = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}c_unanalysed"))

    candidates = repository.fetch_recent_candidates(50, 72)

    ids = [candidate.id for candidate in candidates]
    assert fresh in ids
    assert ancient not in ids  # older than the lookback window
    assert waiting not in ids  # never analysed (no summary), so never a candidate


def test_fetch_post_returns_the_stored_row_or_none(db_connection: psycopg.Connection) -> None:
    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}fetch_post"))

    assert repository.fetch_post(post_id) is not None
    assert repository.fetch_post(post_id).reddit_id == f"{TEST_REDDIT_ID_PREFIX}fetch_post"
    assert repository.fetch_post(10**9) is None


def test_posted_at_and_fetched_at_are_both_kept(db_connection: psycopg.Connection) -> None:
    """The RSS timestamp and our own are different facts and must not be conflated."""
    posted = datetime.now(timezone.utc) - timedelta(days=2)

    post_id = repository.save(_record(f"{TEST_REDDIT_ID_PREFIX}timestamps", posted_at=posted))

    row = _row(db_connection, post_id)
    assert row["posted_at"] == posted
    assert row["fetched_at"] > row["posted_at"]
    assert row["published_at"] is None
