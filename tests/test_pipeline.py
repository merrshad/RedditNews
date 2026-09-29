"""Pipeline tests: the review gate, the publication state machine and error isolation.

The orchestration is exercised through its module-level entry points (`run_once`,
`process_approved_posts`, `retry_pending_sends`) exactly as `app/main.py` uses them.
Storage is the **real Postgres** (the dedicated test database, see ``conftest``): the
review/publication flow is a state machine expressed in conditional SQL, and a hand-written
fake would test the fake instead of the machine. RSS, the LLM transport and Telegram are
faked (NFR-8) — no test may touch the network or broadcast.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest

from app import pipeline, repository, review
from app.models import LlmAnalysis, PostRecord, RawPost
from app.pipeline import (
    build_post_record,
    determine_status,
    meets_importance_threshold,
    resolve_duplicate_id,
)
from app.settings import Settings
from tests.conftest import (
    TEST_REDDIT_ID_PREFIX,
    FakeChatCompletion,
    FakeTelegram,
    insert_source,
    insert_topic,
    patch_telegram,
    temp_topic_key,
)

TOPIC_KEY = temp_topic_key("ai")  # the key an approved answer must name (FR-5)
TOPIC_NAME = "هوش مصنوعی"
REVIEW_CHANNEL = "-100999"
PUBLIC_CHANNEL = "-100123"
ADMIN_ID = "777"

VALID_ANSWER: dict[str, Any] = {
    "is_relevant": True,
    "duplicate_of_candidate_index": None,
    "topic": TOPIC_KEY,
    "importance": "high",
    "summary_fa": "خلاصه فارسی پست.",
    "key_points": ["نکته اول"],
}


def _answer(**overrides: Any) -> str:
    return json.dumps({**VALID_ANSWER, **overrides}, ensure_ascii=False)


def _post(slug: str = "new", *, title: str = "یک پست تازه") -> RawPost:
    return RawPost(
        reddit_id=f"{TEST_REDDIT_ID_PREFIX}{slug}",
        subreddit="mlops",
        source_topic_key=TOPIC_KEY,
        title=title,
        url=f"https://www.reddit.com/r/mlops/comments/{slug}/x/",
        author="somebody",
        raw_content="متن خام پست",
    )


@pytest.fixture
def taxonomy(db_connection: psycopg.Connection) -> psycopg.Connection:
    """One topic plus one source, so a cycle has a taxonomy to read (FR-14)."""
    insert_topic(db_connection, "ai", TOPIC_NAME)
    insert_source(db_connection, "ai", "r/ai")
    return db_connection


def _candidate(
    connection: psycopg.Connection,
    slug: str,
    *,
    summary: str = "خلاصه پست قبلی.",
    minutes_ago: int = 60,
) -> int:
    """An already-analysed post in the database, i.e. what FR-4 offers the LLM as context.

    ``posted_at`` is set explicitly so the candidate order the LLM sees is deterministic
    (candidates are handed over newest first).
    """
    row = connection.execute(
        """
        INSERT INTO posts (
            reddit_id, subreddit, source_topic_key, title, url, summary_fa, key_points,
            is_relevant, topic, importance, posted_at, status, review_status
        ) VALUES (%s, 'mlops', %s, %s, %s, %s, '["نکته"]'::jsonb, TRUE, %s, 'high',
                  now() - make_interval(mins => %s), 'sent', 'approved')
        RETURNING id
        """,
        (
            f"{TEST_REDDIT_ID_PREFIX}{slug}",
            TOPIC_KEY,
            slug,
            f"https://example.com/{slug}",
            summary,
            TOPIC_KEY,
            minutes_ago,
        ),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    telegram: FakeTelegram | None = None,
    llm: FakeChatCompletion | None = None,
    posts: list[RawPost] | None = None,
) -> tuple[FakeChatCompletion, FakeTelegram]:
    """Wire the real pipeline to the real database, faking only the three boundaries."""
    llm = llm or FakeChatCompletion(_answer())
    telegram = patch_telegram(
        monkeypatch, telegram or FakeTelegram(), default_chat_id=settings.telegram_chat_id
    )
    monkeypatch.setattr("app.analyzer.chat_completion", llm)
    monkeypatch.setattr("app.pipeline.get_settings", lambda: settings)
    monkeypatch.setattr("app.pipeline.fetch_all", lambda *args, **kwargs: list(posts or []))
    return llm, telegram


def _row(connection: psycopg.Connection, reddit_id: str) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM posts WHERE reddit_id = %s", (reddit_id,)).fetchone()
    assert row is not None, f"reddit_id={reddit_id} was not stored"
    return row


def _approve(post_id: int, *, user_id: str = ADMIN_ID, callback_id: str = "cb-approve") -> dict:
    """A Telegram ``callback_query`` for the ✅ button of one review message."""
    return {
        "id": callback_id,
        "data": f"approve:{post_id}",
        "from": {"id": int(user_id)},
        "message": {"message_id": 1, "chat": {"id": int(REVIEW_CHANNEL)}},
    }


def _reject(post_id: int, *, user_id: str = ADMIN_ID, callback_id: str = "cb-reject") -> dict:
    return {
        "id": callback_id,
        "data": f"reject:{post_id}",
        "from": {"id": int(user_id)},
        "message": {"message_id": 1, "chat": {"id": int(REVIEW_CHANNEL)}},
    }


# --- pure decision helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    ("importance", "minimum", "expected"),
    [
        ("low", "low", True),
        ("medium", "low", True),
        ("high", "low", True),
        ("low", "medium", False),
        ("medium", "medium", True),
        ("low", "high", False),
        (None, "low", False),
        ("bogus", "low", False),
    ],
)
def test_meets_importance_threshold(importance: str | None, minimum: str, expected: bool) -> None:
    assert meets_importance_threshold(importance, minimum) is expected


@pytest.mark.parametrize(
    ("relevant", "duplicate_of_id", "importance", "expected"),
    [
        (False, None, "high", "skipped_irrelevant"),
        (True, 42, "high", "skipped_duplicate"),
        (True, None, "low", "skipped_low_importance"),
        (True, None, "high", "to_send"),
    ],
)
def test_determine_status(
    relevant: bool, duplicate_of_id: int | None, importance: str, expected: str
) -> None:
    analysis = LlmAnalysis(**{**VALID_ANSWER, "is_relevant": relevant, "importance": importance})

    status = determine_status(
        analysis, duplicate_of_id=duplicate_of_id, min_importance_to_send="medium"
    )

    assert status == expected


def test_resolve_duplicate_id_maps_the_position_to_the_real_row_id() -> None:
    """Invariant 4: index 2 means the *second* candidate, i.e. its database id."""
    analysis = LlmAnalysis(**{**VALID_ANSWER, "duplicate_of_candidate_index": 2})
    candidates = [
        PostRecord(id=101, reddit_id="t3_a", subreddit="mlops", source_topic_key=TOPIC_KEY, title="a", url="u"),
        PostRecord(id=202, reddit_id="t3_b", subreddit="mlops", source_topic_key=TOPIC_KEY, title="b", url="u"),
    ]

    assert resolve_duplicate_id(analysis, candidates) == 202


def test_resolve_duplicate_id_refuses_an_index_out_of_range() -> None:
    """A defensive assert, not a guess: the analyzer already rejects such an answer."""
    analysis = LlmAnalysis(**{**VALID_ANSWER, "duplicate_of_candidate_index": 3})
    candidate = PostRecord(
        id=101, reddit_id="t3_a", subreddit="mlops", source_topic_key=TOPIC_KEY, title="a", url="u"
    )

    with pytest.raises(AssertionError):
        resolve_duplicate_id(analysis, [candidate])


def test_a_fetched_post_becomes_a_pending_review_row() -> None:
    record = build_post_record(_post())

    assert record.status == "new"
    assert record.review_status == "pending_review"
    assert record.summary_fa is None and record.is_relevant is None


# --- FR-12: the review gate -------------------------------------------------------


def test_a_new_post_waits_for_the_admin_and_never_reaches_the_llm(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """The point of phase 5: RSS -> store -> private channel, and then nothing at all.

    No LLM call may happen before an admin presses ✅, and nothing may be published.
    """
    llm, telegram = _harness(monkeypatch, settings=settings, posts=[_post()])

    pipeline.run_once()

    row = _row(taxonomy, _post().reddit_id)
    assert row["status"] == "awaiting_review"
    assert row["review_status"] == "pending_review"
    assert row["reviewed_by"] is None and row["reviewed_at"] is None
    assert row["summary_fa"] is None and row["is_relevant"] is None and row["ai_error"] is None
    assert llm.calls == []  # no post may enter the LLM before approval
    assert telegram.messages_to(PUBLIC_CHANNEL) == []

    # Exactly one message, in the private channel, for this whole post.
    assert len(telegram.messages_to(REVIEW_CHANNEL)) == 1
    text = telegram.messages_to(REVIEW_CHANNEL)[0]
    assert "یک پست تازه" in text and "mlops" in text and TOPIC_NAME in text

    # ... and it carries the two buttons addressed to the stored row.
    markup = telegram.buttons_to(REVIEW_CHANNEL)[0]
    assert markup is not None
    callbacks = [button["callback_data"] for button in markup["inline_keyboard"][0]]
    assert callbacks == [f"approve:{row['id']}", f"reject:{row['id']}"]
    assert row["private_message_id"] is not None
    assert row["private_channel_id"] == REVIEW_CHANNEL


def test_every_fetched_post_gets_its_own_review_message(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """FR-12: no digest — 3 fetched posts means 3 independent messages with 3 keyboards."""
    posts = [_post(f"one"), _post("two"), _post("three")]
    _, telegram = _harness(monkeypatch, settings=settings, posts=posts)

    pipeline.run_once()

    assert len(telegram.messages_to(REVIEW_CHANNEL)) == 3
    callbacks = {
        button["callback_data"]
        for markup in telegram.buttons_to(REVIEW_CHANNEL)
        for row in markup["inline_keyboard"]
        for button in row
    }
    stored_ids = {
        _row(taxonomy, post.reddit_id)["id"] for post in posts
    }
    assert callbacks == {f"approve:{i}" for i in stored_ids} | {f"reject:{i}" for i in stored_ids}


def test_a_post_whose_review_message_fails_stays_new_and_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """FR-12 crash recovery: nothing is lost when Telegram is down for one cycle."""
    telegram = FakeTelegram()
    telegram.fail_next_sends = 1
    _harness(monkeypatch, settings=settings, telegram=telegram, posts=[_post()])

    pipeline.run_once()

    assert _row(taxonomy, _post().reddit_id)["status"] == "new"

    pipeline.run_once()  # the post is still there, and Telegram works again

    assert _row(taxonomy, _post().reddit_id)["status"] == "awaiting_review"
    assert len(telegram.messages_to(REVIEW_CHANNEL)) == 1


def test_a_known_post_is_not_stored_delivered_or_analysed_again(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 1: the same ``reddit_id`` in a second cycle changes nothing."""
    llm, telegram = _harness(monkeypatch, settings=settings, posts=[_post()])

    pipeline.run_once()
    first = _row(taxonomy, _post().reddit_id)

    pipeline.run_once()

    assert _row(taxonomy, _post().reddit_id) == first
    assert len(telegram.messages_to(REVIEW_CHANNEL)) == 1
    assert llm.calls == []


def test_the_configured_fetch_limit_is_handed_to_the_collector(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """NFR-7: how many posts a cycle may take is configuration, not a constant."""
    captured: list[tuple[Any, int]] = []

    def _record(sources: Any, *, default_fetch_limit: int) -> list[RawPost]:
        captured.append((sources, default_fetch_limit))
        return []

    llm, _ = _harness(monkeypatch, settings=settings)
    monkeypatch.setattr("app.pipeline.fetch_all", _record)

    pipeline.run_once()

    assert captured, "the collector was not called"
    sources, limit = captured[0]
    assert limit == settings.rss_fetch_limit
    assert [source.topic_key for source in sources], "no source was passed to the collector"
    assert llm.calls == []


def test_an_empty_fetch_produces_no_messages_and_no_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    llm, telegram = _harness(monkeypatch, settings=settings, posts=[])

    pipeline.run_once()

    assert llm.calls == [] and telegram.sent == []
    assert taxonomy.execute(
        "SELECT count(*) AS n FROM posts WHERE reddit_id LIKE %s",
        (f"{TEST_REDDIT_ID_PREFIX}%",),
    ).fetchone()["n"] == 0


# --- FR-13: approval, analysis, publication --------------------------------------


def test_approving_a_post_analyses_and_publishes_it_once(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """The happy path end to end: ✅ -> one LLM call -> one public message."""
    llm, telegram = _harness(monkeypatch, settings=settings, posts=[_post()])
    pipeline.run_once()
    post_id = _row(taxonomy, _post().reddit_id)["id"]

    assert review.handle_callback(_approve(post_id)) is True

    approved = _row(taxonomy, _post().reddit_id)
    assert approved["status"] == "approved" and approved["review_status"] == "approved"
    assert approved["reviewed_by"] == ADMIN_ID and approved["reviewed_at"] is not None
    assert approved["approved_at"] is not None and approved["rejected_at"] is None
    assert telegram.edits[-1][0] == review.PROCESSING_TEXT
    assert llm.calls == []  # deciding a post is not analysing it

    # What `main` does right after a decision: no RSS fetch, just the approved queue.
    pipeline.process_approved_posts()

    published = _row(taxonomy, _post().reddit_id)
    assert published["status"] == "sent"
    assert published["published_at"] is not None and published["ai_processed_at"] is not None
    assert published["public_message_id"] is not None
    assert published["public_channel_id"] == PUBLIC_CHANNEL
    assert published["is_relevant"] is True and published["importance"] == "high"
    assert published["summary_fa"] == VALID_ANSWER["summary_fa"]
    assert published["llm_raw_response"] is not None
    assert len(llm.calls) == 1

    public = telegram.messages_to(PUBLIC_CHANNEL)
    assert len(public) == 1
    assert VALID_ANSWER["summary_fa"] in public[0] and "نکته اول" in public[0]
    # The review message stays as the audit trail, its buttons replaced by the outcome.
    assert telegram.edits[-1][0] == review.PUBLISHED_TEXT

    # Invariant 12: a second pass over the same post publishes nothing again.
    pipeline.run_once()

    assert len(telegram.messages_to(PUBLIC_CHANNEL)) == 1
    assert len(llm.calls) == 1
    assert repository.claim_for_publish(post_id) is False


def test_rejecting_a_post_stops_the_flow_before_the_llm(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """FR-13: ❌ is terminal — no LLM, no translation, no summary, no publication."""
    llm, telegram = _harness(monkeypatch, settings=settings, posts=[_post()])
    pipeline.run_once()
    post_id = _row(taxonomy, _post().reddit_id)["id"]

    assert review.handle_callback(_reject(post_id)) is True

    rejected = _row(taxonomy, _post().reddit_id)
    assert rejected["status"] == "rejected" and rejected["review_status"] == "rejected"
    assert rejected["rejected_at"] is not None and rejected["reviewed_by"] == ADMIN_ID
    assert rejected["summary_fa"] is None and rejected["ai_processed_at"] is None
    assert telegram.edits[-1][0] == review.REJECTED_TEXT

    pipeline.run_once()

    assert llm.calls == []  # never analysed, not even in a later cycle
    assert telegram.messages_to(PUBLIC_CHANNEL) == []
    assert _row(taxonomy, _post().reddit_id)["status"] == "rejected"


def test_the_first_of_two_simultaneous_decisions_wins(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 12: ``PENDING -> APPROVED`` succeeds exactly once."""
    telegram = FakeTelegram()
    _harness(monkeypatch, settings=settings, telegram=telegram, posts=[_post()])
    pipeline.run_once()
    post_id = _row(taxonomy, _post().reddit_id)["id"]

    assert review.handle_callback(_approve(post_id, callback_id="first")) is True
    # The second admin presses the same, now stale, button.
    assert review.handle_callback(_reject(post_id, user_id="888", callback_id="second")) is False

    row = _row(taxonomy, _post().reddit_id)
    assert row["review_status"] == "approved" and row["reviewed_by"] == ADMIN_ID
    assert row["rejected_at"] is None
    assert row["status"] == "approved"  # the losing click changed nothing
    assert telegram.answers[-1] == ("second", review.ALREADY_REVIEWED_TEXT)
    # The loser must not rewrite the message the winner already updated.
    assert telegram.edits[-1][0] == review.PROCESSING_TEXT


def test_only_listed_admins_may_decide(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """FR-12: the allow-list is the authorisation; nobody else can approve."""
    telegram = FakeTelegram()
    _harness(monkeypatch, settings=settings, telegram=telegram, posts=[_post()])
    pipeline.run_once()
    post_id = _row(taxonomy, _post().reddit_id)["id"]

    assert review.handle_callback(_approve(post_id, user_id="999")) is False

    row = _row(taxonomy, _post().reddit_id)
    assert row["status"] == "awaiting_review" and row["review_status"] == "pending_review"
    assert row["reviewed_by"] is None
    assert telegram.answers[-1] == ("cb-approve", review.NOT_ADMIN_TEXT)
    assert telegram.edits == []


def test_a_duplicate_is_recorded_against_the_real_id_of_the_second_candidate(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 4 + FR-4: the model points at position 2, the database stores that row's id."""
    older = _candidate(taxonomy, "candidate-one", summary="خلاصه اول", minutes_ago=120)
    newer = _candidate(taxonomy, "candidate-two", summary="خلاصه دوم", minutes_ago=60)

    llm, telegram = _harness(
        monkeypatch,
        settings=settings,
        posts=[_post()],
        llm=FakeChatCompletion(_answer(duplicate_of_candidate_index=2)),
    )
    pipeline.run_once()
    review.handle_callback(_approve(_row(taxonomy, _post().reddit_id)["id"]))

    # What the LLM was shown: newest first, so position 2 is the older candidate.
    listed = repository.fetch_recent_candidates(50, 72)
    assert [candidate.id for candidate in listed] == [newer, older]

    pipeline.process_approved_posts()

    row = _row(taxonomy, _post().reddit_id)
    assert row["status"] == "skipped_duplicate"
    assert row["duplicate_of_id"] == listed[1].id  # a real row id, not the index 2
    assert row["duplicate_of_id"] != 2
    assert telegram.messages_to(PUBLIC_CHANNEL) == []
    assert telegram.edits[-1][0].startswith("✅ تأیید شد\n⛔️ منتشر نشد")
    assert len(llm.calls) == 1


def test_a_low_importance_post_is_not_published_even_after_approval(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """FR-10(c): the admin said yes, the importance threshold still says no."""
    strict = settings.model_copy(update={"min_importance_to_send": "medium"})
    llm, telegram = _harness(
        monkeypatch,
        settings=strict,
        posts=[_post()],
        llm=FakeChatCompletion(_answer(importance="low")),
    )
    pipeline.run_once()
    review.handle_callback(_approve(_row(taxonomy, _post().reddit_id)["id"]))

    pipeline.process_approved_posts()

    row = _row(taxonomy, _post().reddit_id)
    assert row["status"] == "skipped_low_importance"
    assert row["importance"] == "low" and row["published_at"] is None
    assert telegram.messages_to(PUBLIC_CHANNEL) == []
    assert len(llm.calls) == 1


def test_an_invalid_llm_answer_fails_only_that_post(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 3 and 8: unusable output is stored as failed, the queue carries on."""
    bad = _post("bad", title="پست با پاسخ بد")
    good = _post("good", title="پست سالم")

    def _completion(system_prompt: str, user_prompt: str) -> str:
        # The prompt carries the title (never the reddit_id — Invariant 4).
        return "{not json at all" if bad.title in user_prompt else _answer()

    llm = FakeChatCompletion(_answer())
    telegram = patch_telegram(
        monkeypatch, FakeTelegram(), default_chat_id=settings.telegram_chat_id
    )
    monkeypatch.setattr("app.analyzer.chat_completion", _completion)
    monkeypatch.setattr("app.pipeline.get_settings", lambda: settings)
    monkeypatch.setattr("app.pipeline.fetch_all", lambda *a, **k: [bad, good])

    # Both posts are fetched (the LLM is never called while storing) ...
    pipeline.run_once()
    assert llm.calls == []
    for post in (bad, good):
        assert review.handle_callback(_approve(_row(taxonomy, post.reddit_id)["id"])) is True

    pipeline.process_approved_posts()

    failed = _row(taxonomy, bad.reddit_id)
    assert failed["status"] == "failed"
    assert failed["ai_error"] and "JSON" in failed["ai_error"]
    assert failed["is_relevant"] is None and failed["published_at"] is None
    assert failed["ai_processed_at"] is not None

    published = _row(taxonomy, good.reddit_id)
    assert published["status"] == "sent"  # the failure did not stop the rest of the queue
    assert len(telegram.messages_to(PUBLIC_CHANNEL)) == 1


# --- FR-11: publication is retried, never repeated --------------------------------


def test_a_rejected_publication_is_retried_without_re_analysing(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """FR-11: the row is durable, so a failed send is finished by the next run."""
    telegram = FakeTelegram()
    llm, _ = _harness(monkeypatch, settings=settings, telegram=telegram, posts=[_post()])
    pipeline.run_once()
    review.handle_callback(_approve(_row(taxonomy, _post().reddit_id)["id"]))

    telegram.fail_next_sends = 1  # the send to the public channel is refused
    pipeline.process_approved_posts()

    after_failure = _row(taxonomy, _post().reddit_id)
    assert after_failure["status"] == "to_send"  # the claim was handed back
    assert after_failure["published_at"] is None
    assert after_failure["summary_fa"] is not None  # analysed once, and kept
    assert len(llm.calls) == 1

    pipeline.retry_pending_sends()  # what the next cycle starts with

    recovered = _row(taxonomy, _post().reddit_id)
    assert recovered["status"] == "sent"
    assert recovered["published_at"] is not None and recovered["public_message_id"] is not None
    assert len(llm.calls) == 1  # never re-analysed
    assert len(telegram.messages_to(PUBLIC_CHANNEL)) == 1


def test_the_lookback_limit_is_the_one_handed_to_the_repository(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 10: the candidate window is configuration-driven and never open-ended."""
    seen: list[tuple[int, int]] = []

    def _record(limit: int, hours: int) -> list[PostRecord]:
        seen.append((limit, hours))
        return []

    bounded = settings.model_copy(update={"similarity_lookback_limit": 7})
    _harness(monkeypatch, settings=bounded, posts=[_post()])
    monkeypatch.setattr("app.pipeline.repository.fetch_recent_candidates", _record)

    pipeline.run_once()
    review.handle_callback(_approve(_row(taxonomy, _post().reddit_id)["id"]))
    pipeline.process_approved_posts()

    assert seen == [(7, bounded.similarity_lookback_hours)]
