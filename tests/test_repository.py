"""FR-2 / FR-9 / FR-11 tests: SQL shape and row mapping, no real database."""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from app.models import AnalyzedPost, RawPost
from app.repository import PostRepository
from tests.conftest import FakeConnection


def _repository(responses: list[list[dict]] | None = None) -> tuple[PostRepository, FakeConnection]:
    connection = FakeConnection(responses)
    return PostRepository(connection), connection


def test_exists_returns_true_when_a_row_is_found() -> None:
    repository, connection = _repository([[{"?column?": 1}]])

    assert repository.exists("t3_1abcde") is True
    sql, params = connection.executed[0]
    assert "FROM posts WHERE reddit_id = %s" in sql
    assert params == ("t3_1abcde",)


def test_exists_returns_false_when_no_row_is_found() -> None:
    repository, _ = _repository([[]])

    assert repository.exists("t3_missing") is False


def test_get_similarity_candidates_maps_rows_and_passes_limits() -> None:
    rows = [
        {
            "id": 7,
            "reddit_id": "t3_new",
            "subreddit": "MachineLearning",
            "source_topic_key": "ai",
            "title": "Newest",
            "url": "https://example.com/7",
            "author": "somebody",
            "topic": "ai",
            "importance": "high",
            "summary_fa": "خلاصه",
            "key_points": ["نکته"],
            "published_at": None,
            "status": "sent",
        },
        {
            "id": 3,
            "reddit_id": "t3_old",
            "subreddit": "MachineLearning",
            "source_topic_key": "ai",
            "title": "Older",
            "url": "https://example.com/3",
            "author": None,
            "topic": None,
            "importance": None,
            "summary_fa": None,
            "key_points": [],
            "published_at": None,
            "status": "sent",
        },
    ]
    repository, connection = _repository([rows])

    candidates = repository.get_similarity_candidates(limit=50, hours=72)

    assert [candidate.id for candidate in candidates] == [7, 3]
    assert candidates[0].summary_fa == "خلاصه"
    _, params = connection.executed[0]
    assert params == (72, 50)


def test_insert_post_sends_jsonb_payload_and_returns_the_new_id() -> None:
    repository, connection = _repository([[{"id": 42}]])
    record = AnalyzedPost(
        reddit_id="t3_1abcde",
        subreddit="MachineLearning",
        source_topic_key="ai",
        title="Title",
        url="https://example.com/post",
        author="somebody",
        raw_content="body",
        published_at=datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc),
        is_relevant=True,
        duplicate_of_id=None,
        topic="ai",
        importance="high",
        summary_fa="خلاصه فارسی",
        key_points=["نکته یک"],
        llm_raw_response={"is_relevant": True},
        status="to_send",
    )

    assert repository.insert_post(record) == 42

    sql, params = connection.executed[0]
    assert "INSERT INTO posts" in sql
    assert "ON CONFLICT (reddit_id) DO NOTHING" in sql
    assert params[0] == "t3_1abcde"
    assert params[6] == "body"
    assert params[9] is None  # duplicate_of_id
    assert params[11] == "high"  # importance
    assert isinstance(params[13], Jsonb) and params[13].obj == ["نکته یک"]
    assert isinstance(params[14], Jsonb)
    assert params[15] == "to_send"


def test_insert_post_returns_none_for_an_already_stored_reddit_id() -> None:
    repository, _ = _repository([[]])
    record = AnalyzedPost(
        reddit_id="t3_dupe",
        subreddit="startups",
        source_topic_key="startup",
        title="Title",
        url="https://example.com/post",
        status="to_send",
    )

    assert repository.insert_post(record) is None


def test_insert_post_sends_null_raw_response_when_absent() -> None:
    repository, connection = _repository([[{"id": 1}]])

    repository.insert_post(
        AnalyzedPost(
            reddit_id="t3_x",
            subreddit="r",
            source_topic_key="ai",
            title="t",
            url="u",
            status="failed",
        )
    )

    _, params = connection.executed[0]
    assert params[14] is None
    assert params[13].obj == []


def test_mark_sent_updates_status_and_sent_at() -> None:
    repository, connection = _repository()

    repository.mark_sent(42)

    sql, params = connection.executed[0]
    assert "SET status = 'sent', sent_at = now()" in sql
    assert params == (42,)


def test_fetch_pending_send_returns_post_records() -> None:
    rows = [
        {
            "id": 5,
            "reddit_id": "t3_pending",
            "subreddit": "MachineLearning",
            "source_topic_key": "ai",
            "title": "Pending",
            "url": "https://example.com/pending",
            "author": "somebody",
            "topic": "ai",
            "importance": "medium",
            "summary_fa": "خلاصه",
            "key_points": ["نکته"],
            "published_at": None,
            "status": "to_send",
        }
    ]
    repository, connection = _repository([rows])

    pending = repository.fetch_pending_send()

    assert len(pending) == 1
    assert pending[0].id == 5
    assert pending[0].key_points == ["نکته"]
    assert pending[0].status == "to_send"
    sql = connection.statements[0]
    assert "WHERE status = 'to_send'" in sql
    assert "ORDER BY id" in sql


def test_analyzed_post_accepts_a_raw_post_without_analysis() -> None:
    """The `failed` path stores the raw post with empty analysis fields (Invariant 3)."""
    raw = RawPost(
        reddit_id="t3_x",
        subreddit="startups",
        source_topic_key="startup",
        title="Title",
        url="https://example.com/x",
    )

    record = AnalyzedPost(**raw.model_dump(), status="failed")

    assert record.is_relevant is None
    assert record.key_points == []
