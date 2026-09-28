"""Pipeline tests: ordering, status decisions, idempotency and error isolation.

The orchestration is exercised through its module-level entry points (`run_once`,
`retry_pending_sends`) exactly as `app/main.py` uses them. RSS, the LLM transport and
Telegram are faked (NFR-8); storage is either the in-memory fake or — for the recovery
tests — the real Postgres through `db/schema.sql`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from app import pipeline, repository
from app.models import LlmAnalysis, PostRecord, RawPost
from app.pipeline import (
    build_post_record,
    determine_status,
    meets_importance_threshold,
    resolve_duplicate_id,
)
from app.reddit_source import TopicsConfig
from app.settings import Settings
from tests.conftest import (
    TEST_REDDIT_ID_PREFIX,
    FakeChatCompletion,
    FakeRepository,
    patch_repository,
)

VALID_ANSWER = {
    "is_relevant": True,
    "duplicate_of_candidate_index": None,
    "topic": "ai",
    "importance": "high",
    "summary_fa": "خلاصه فارسی پست.",
    "key_points": ["نکته اول"],
}
VALID_JSON = json.dumps(VALID_ANSWER, ensure_ascii=False)


def _post(reddit_id: str = "t3_1abcde", *, title: str = "A new open model was released") -> RawPost:
    return RawPost(
        reddit_id=reddit_id,
        subreddit="MachineLearning",
        source_topic_key="ai",
        title=title,
        url=f"https://www.reddit.com/r/MachineLearning/comments/{reddit_id}/x/",
        author="somebody",
        raw_content="Body",
    )


def _candidate(post_id: int, reddit_id: str) -> PostRecord:
    """An already-stored post, as `fetch_recent_candidates` would return it."""
    return PostRecord(
        id=post_id,
        reddit_id=reddit_id,
        subreddit="MachineLearning",
        source_topic_key="ai",
        title=reddit_id,
        url=f"https://example.com/{reddit_id}",
        summary_fa="خلاصه پست قبلی.",
    )


def _pending(post_id: int = 5) -> PostRecord:
    """A row an earlier run left behind as `to_send` (FR-11)."""
    return PostRecord(
        id=post_id,
        reddit_id=f"t3_pending{post_id}",
        subreddit="MachineLearning",
        source_topic_key="ai",
        title=f"Pending post {post_id}",
        url=f"https://example.com/pending/{post_id}",
        topic=None,
        importance="medium",
        summary_fa="خلاصه پست معلق.",
        key_points=["نکته"],
        status="to_send",
    )


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


def _raise_runtime_error(system_prompt: str, user_prompt: str) -> str:
    """A ``chat_completion`` stand-in for the "the provider is down" path."""
    raise RuntimeError("provider down")


def _answering_by_title(answers: dict[str, str]) -> Callable[[str, str], str]:
    """A ``chat_completion`` stand-in that answers per post, keyed by its title.

    The title is part of the prompt payload, so this lets one run contain a post whose
    answer is invalid while the next post is analysed normally.
    """

    def _completion(system_prompt: str, user_prompt: str) -> str:
        for title, answer in answers.items():
            if title in user_prompt:
                return answer
        raise AssertionError(f"prompt for an unexpected post: {user_prompt[:200]}")

    return _completion


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    topics_config: TopicsConfig,
    sender: Callable[[str], bool] | None = None,
    llm: Callable[[str, str], str] | None = None,
) -> tuple[FakeSender | Callable[[str], bool], Callable[[str, str], str]]:
    """Replace the pipeline's collaborators with fakes (NFR-8: no network/Telegram).

    The entry points take no arguments by design; tests therefore inject the settings, the
    topics config and the notifier they want instead of letting the real ones be built.
    """
    llm = llm if llm is not None else FakeChatCompletion(VALID_JSON)
    monkeypatch.setattr("app.analyzer.chat_completion", llm)
    sender = sender if sender is not None else FakeSender()
    monkeypatch.setattr("app.pipeline.send_message", sender)
    monkeypatch.setattr("app.pipeline.get_settings", lambda: settings)
    monkeypatch.setattr("app.pipeline.load_topics_config", lambda: topics_config)
    return sender, llm


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    topics_config: TopicsConfig,
    repository: FakeRepository | None = None,
    answer: str = VALID_JSON,
    sender: FakeSender | None = None,
    llm: Callable[[str, str], str] | None = None,
) -> tuple[FakeRepository, FakeSender, Callable[[str, str], str]]:
    """The pipeline wired to in-memory storage, a fake notifier and a stubbed LLM."""
    if llm is None:
        llm = FakeChatCompletion(answer)
    sender, llm = _patch_seams(
        monkeypatch, settings=settings, topics_config=topics_config, sender=sender, llm=llm
    )
    repository = patch_repository(monkeypatch, repository or FakeRepository())
    return repository, sender, llm  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _single_post_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default a run fetches exactly one post; tests may override this."""
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: [_post()])


# --- pure decision helpers -------------------------------------------------------


@pytest.mark.parametrize(
    ("importance", "minimum", "expected"),
    [
        ("low", "low", True),
        ("medium", "low", True),
        ("high", "medium", True),
        ("low", "medium", False),
        ("low", "high", False),
        (None, "low", False),
        ("nonsense", "low", False),
    ],
)
def test_meets_importance_threshold(importance: str | None, minimum: str, expected: bool) -> None:
    assert meets_importance_threshold(importance, minimum) is expected


def test_determine_status_covers_every_outcome() -> None:
    relevant = LlmAnalysis.model_validate(VALID_ANSWER)
    irrelevant = LlmAnalysis.model_validate(
        {**VALID_ANSWER, "is_relevant": False, "summary_fa": ""}
    )
    low_importance = LlmAnalysis.model_validate({**VALID_ANSWER, "importance": "low"})

    assert determine_status(relevant, duplicate_of_id=None, min_importance_to_send="low") == "to_send"
    assert (
        determine_status(relevant, duplicate_of_id=7, min_importance_to_send="low")
        == "skipped_duplicate"
    )
    assert (
        determine_status(irrelevant, duplicate_of_id=None, min_importance_to_send="low")
        == "skipped_irrelevant"
    )
    assert (
        determine_status(low_importance, duplicate_of_id=None, min_importance_to_send="medium")
        == "skipped_low_importance"
    )


def test_resolve_duplicate_id_maps_only_local_candidate_indexes() -> None:
    """Invariant 4 — the index is 1-based and only the ids of that call are reachable."""
    not_duplicate = LlmAnalysis.model_validate(VALID_ANSWER)
    duplicate = LlmAnalysis.model_validate({**VALID_ANSWER, "duplicate_of_candidate_index": 1})

    assert resolve_duplicate_id(not_duplicate, [_candidate(77, "t3_prev")]) is None
    assert resolve_duplicate_id(duplicate, [_candidate(77, "t3_prev")]) == 77


def test_resolve_duplicate_id_fails_loudly_for_an_out_of_range_index() -> None:
    """The analyzer rejects that answer already, so reaching here is a bug, not bad input."""
    analysis = LlmAnalysis.model_validate({**VALID_ANSWER, "duplicate_of_candidate_index": 9})

    with pytest.raises(AssertionError, match="outside"):
        resolve_duplicate_id(analysis, [])


def test_build_post_record_for_the_failed_path_keeps_raw_post_fields() -> None:
    record = build_post_record(_post(), status="failed", raw_response={"is_relevant": True})

    assert record.id is None
    assert record.status == "failed"
    assert record.is_relevant is None
    assert record.key_points == []
    assert record.llm_raw_response == {"is_relevant": True}
    assert record.title == "A new open model was released"


# --- run_once --------------------------------------------------------------------


def test_run_once_sends_a_new_relevant_post_exactly_once(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, sender, _ = _harness(
        monkeypatch, settings=settings, topics_config=topics_config
    )

    pipeline.run_once()

    assert repository.saved[0].status == "to_send"
    # FR-9/NFR-3: the validated analysis JSON is kept for audit.
    assert repository.saved[0].llm_raw_response == LlmAnalysis.model_validate(
        VALID_ANSWER
    ).model_dump(mode="json")
    assert repository.sent_ids == [repository.last_id]
    assert len(sender.messages) == 1
    assert "خلاصه فارسی پست." in sender.messages[0]
    assert "• نکته اول" in sender.messages[0]
    # Invariant 2(d): the row is stored before the send is attempted.
    assert repository.calls.index("save") < repository.calls.index("update_status")
    assert repository.calls.count("update_status") == 1


def test_run_once_does_not_re_analyse_a_post_that_is_already_stored(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Invariant 1: the second run sees the same `reddit_id` and never re-calls the LLM."""
    repository, sender, llm = _harness(
        monkeypatch, settings=settings, topics_config=topics_config
    )

    pipeline.run_once()
    pipeline.run_once()

    assert len(llm.calls) == 1  # type: ignore[attr-defined]
    assert len(repository.saved) == 1
    assert len(sender.messages) == 1


def test_run_once_skips_a_post_that_is_already_stored(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, sender, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(existing=("t3_1abcde",)),
    )

    pipeline.run_once()

    assert repository.saved == []
    assert repository.calls == ["fetch_pending_to_send", "exists"]
    assert sender.messages == []


def test_run_once_stores_irrelevant_posts_without_sending(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, sender, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        answer=json.dumps(
            {**VALID_ANSWER, "is_relevant": False, "summary_fa": ""}, ensure_ascii=False
        ),
    )

    pipeline.run_once()

    assert [record.status for record in repository.saved] == ["skipped_irrelevant"]
    assert sender.messages == []
    assert repository.updates == []


def test_run_once_maps_a_duplicate_candidate_index_to_the_real_database_id(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Invariant 4 — the LLM names candidate 2, and the id stored is that candidate's id."""
    newer = _candidate(77, "t3_newer")
    older = _candidate(41, "t3_older")
    repository, sender, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(candidates=(newer, older)),
        answer=json.dumps({**VALID_ANSWER, "duplicate_of_candidate_index": 2}, ensure_ascii=False),
    )

    pipeline.run_once()

    stored = repository.saved[0]
    assert stored.status == "skipped_duplicate"
    assert stored.duplicate_of_id == 41  # not 2, and not the newest candidate either
    assert sender.messages == []


def test_run_once_skips_a_relevant_post_below_the_importance_threshold(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, sender, _ = _harness(
        monkeypatch,
        settings=settings.model_copy(update={"min_importance_to_send": "medium"}),
        topics_config=topics_config,
        answer=json.dumps({**VALID_ANSWER, "importance": "low"}, ensure_ascii=False),
    )

    pipeline.run_once()

    assert [record.status for record in repository.saved] == ["skipped_low_importance"]
    assert sender.messages == []


def test_run_once_passes_the_configured_lookback_limits(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, _, _ = _harness(monkeypatch, settings=settings, topics_config=topics_config)

    pipeline.run_once()

    assert "fetch_recent_candidates(limit=50,hours=72)" in repository.calls


def test_run_once_records_failed_output_and_still_processes_the_rest(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Invariant 3 + Invariant 8: one invalid answer never stops the other posts."""
    monkeypatch.setattr(
        "app.pipeline.fetch_all",
        lambda topics_config: [
            _post("t3_1abcde", title="Broken post"),
            _post("t3_2fghij", title="Good post"),
        ],
    )
    repository, sender, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        llm=_answering_by_title(
            {"Broken post": "sorry, no json", "Good post": VALID_JSON}
        ),
    )

    pipeline.run_once()

    assert [(record.title, record.status) for record in repository.saved] == [
        ("Broken post", "failed"),
        ("Good post", "to_send"),
    ]
    assert repository.saved[0].llm_raw_response is None
    assert len(sender.messages) == 1
    assert "Good post" in sender.messages[0]
    assert repository.sent_ids == [repository.last_id]


def test_run_once_writes_nothing_when_the_llm_transport_keeps_failing(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """A failed LLM call stores nothing, so the next run analyses the post again."""
    repository, sender, _ = _harness(
        monkeypatch, settings=settings, topics_config=topics_config, llm=_raise_runtime_error
    )

    pipeline.run_once()  # must not raise

    assert repository.saved == []
    assert repository.updates == []
    assert sender.messages == []


def test_run_once_keeps_a_rejected_send_as_to_send(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, _, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        sender=FailingSender(),
    )

    pipeline.run_once()  # must not raise

    assert repository.saved[0].status == "to_send"
    assert repository.sent_ids == []


def test_run_once_wires_real_modules_end_to_end(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Only the HTTP boundaries are faked: RSS parsing, analysis, formatting and the
    store call all run for real (the storage itself is the in-memory fake).
    """
    from app.reddit_source import fetch_all as real_fetch_all
    from tests.test_reddit_source import SAMPLE_FEED

    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: SAMPLE_FEED.encode())
    monkeypatch.setattr("app.pipeline.fetch_all", real_fetch_all)
    repository, sender, _ = _harness(
        monkeypatch, settings=settings, topics_config=topics_config
    )

    pipeline.run_once()

    stored = repository.saved[0]
    assert stored.reddit_id == "t3_1abcde"
    assert stored.status == "to_send"
    assert stored.source_topic_key == "ai"
    assert stored.subreddit == "MachineLearning"
    assert stored.importance == "high"
    assert stored.key_points == ["نکته اول"]

    message = sender.messages[0]
    assert "📌 <b>A new open model was released</b>" in message
    assert "r/MachineLearning • هوش مصنوعی • اهمیت: بالا" in message
    assert "خلاصه فارسی پست." in message
    assert "• نکته اول" in message
    # The URL is the one the real parser read out of the feed entry.
    assert message.endswith(
        "🔗 https://www.reddit.com/r/MachineLearning/comments/1abcde/a_new_open_model/"
    )


def test_send_refuses_a_post_that_was_never_stored(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Invariant 2 — sending before storing must be impossible."""
    repository, sender, _ = _harness(
        monkeypatch, settings=settings, topics_config=topics_config
    )

    with pytest.raises(ValueError, match="never stored"):
        pipeline._send(_pending(5).model_copy(update={"id": None}), topic_names={})

    assert sender.messages == []
    assert repository.updates == []


# --- retry_pending_sends (FR-11) --------------------------------------------------


def test_retry_pending_sends_finishes_pending_rows_without_an_llm_call(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, sender, llm = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(pending=(_pending(5), _pending(6))),
    )

    pipeline.retry_pending_sends()

    assert repository.sent_ids == [5, 6]
    assert len(sender.messages) == 2
    assert "خلاصه پست معلق." in sender.messages[0]
    assert llm.calls == []  # type: ignore[attr-defined]


def test_retry_pending_sends_continues_after_one_rejected_send(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    repository, sender, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(pending=(_pending(5), _pending(6))),
        sender=FakeSender(fail_first=True),
    )

    pipeline.retry_pending_sends()

    assert repository.sent_ids == [6]
    assert len(sender.messages) == 1
    assert "Pending post 6" in sender.messages[0]


def test_a_rejected_send_stays_to_send_and_is_retried_by_the_next_run(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """The whole FR-11 loop with the in-memory store: reject -> keep -> retry -> sent."""
    repository, sender, llm = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        sender=FakeSender(fail_first=True),
    )

    pipeline.run_once()

    assert repository.saved[0].status == "to_send"
    assert repository.sent_ids == []
    assert sender.messages == []

    # The next cycle fetches nothing new, so only the stored 'to_send' row is retried.
    repository.pending = [
        repository.saved[0].model_copy(update={"id": repository.last_id, "status": "to_send"})
    ]
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: [])

    pipeline.run_once()

    assert repository.sent_ids == [repository.last_id]
    assert len(sender.messages) == 1
    assert len(llm.calls) == 1  # type: ignore[attr-defined]  # never re-analysed


# --- real storage boundary --------------------------------------------------------


def test_a_rejected_send_is_recovered_from_the_database_on_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 2 + FR-11 with real Postgres: store first, then send, and recover.

    Only RSS/LLM/Telegram are faked here; the rows really go through ``db/schema.sql``.
    """
    from app.reddit_source import fetch_all as real_fetch_all
    from tests.test_reddit_source import SAMPLE_FEED

    reddit_id = f"{TEST_REDDIT_ID_PREFIX}e2e"
    feed = SAMPLE_FEED.replace("t3_1abcde", reddit_id)
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: feed.encode("utf-8"))
    # The sample feed has two entries; one post is enough to describe the FR-11 loop.
    monkeypatch.setattr(
        "app.pipeline.fetch_all", lambda topics_config: real_fetch_all(topics_config)[:1]
    )
    sender, llm = _patch_seams(
        monkeypatch, settings=settings, topics_config=topics_config, sender=FailingSender()
    )

    def stored_rows() -> list[dict]:
        with db_connection.cursor() as cursor:
            cursor.execute(
                "SELECT reddit_id, status, sent_at, summary_fa FROM posts WHERE reddit_id LIKE %s",
                (f"{TEST_REDDIT_ID_PREFIX}%",),
            )
            return cursor.fetchall()

    # First run: Telegram is down, but the analysed post is already stored (Invariant 2).
    pipeline.run_once()

    assert [row["status"] for row in stored_rows()] == ["to_send"]
    assert stored_rows()[0]["sent_at"] is None
    assert len(llm.calls) == 1  # type: ignore[attr-defined]

    # Second run: no new RSS item, so the stored 'to_send' row is retried (FR-11).
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: [])
    working_sender = FakeSender()
    monkeypatch.setattr("app.pipeline.send_message", working_sender)

    pipeline.run_once()

    assert [row["status"] for row in stored_rows()] == ["sent"]
    assert stored_rows()[0]["sent_at"] is not None
    assert len(llm.calls) == 1  # type: ignore[attr-defined]  # never re-analysed
    assert len(working_sender.messages) == 1
    assert "خلاصه فارسی پست." in working_sender.messages[0]
    assert sender.messages == []


def _patch_http_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    topics_config: TopicsConfig,
    posts: list[RawPost],
    sender: FakeSender | None = None,
    llm: Callable[[str, str], str] | None = None,
) -> FakeSender:
    """Patch only the HTTP boundaries; ``app.repository`` stays real (Postgres).

    ``_patch_seams`` already injects the settings, the topics config and the notifier;
    this adds the RSS items and deliberately leaves the repository untouched, so every
    row really goes through ``db/schema.sql``.
    """
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: list(posts))
    sender, _ = _patch_seams(
        monkeypatch, settings=settings, topics_config=topics_config, sender=sender, llm=llm
    )
    return sender  # type: ignore[return-value]


def _row_for(connection: psycopg.Connection, reddit_id: str) -> dict:
    """The stored row for one ``reddit_id`` (asserts that it exists)."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT reddit_id, status, sent_at, is_relevant, duplicate_of_id, importance "
            "FROM posts WHERE reddit_id = %s",
            (reddit_id,),
        )
        row = cursor.fetchone()
    assert row is not None, f"no stored row for reddit_id={reddit_id}"
    return row


def _insert_candidate(
    connection: psycopg.Connection, *, post_id: int, reddit_id: str, published_at: datetime
) -> None:
    """Seed one analysed row so the real ``fetch_recent_candidates`` can find it.

    The id is set explicitly so a mapping test can prove the 1-based index really became
    *that* row's id rather than the index value itself.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO posts (
                id, reddit_id, subreddit, source_topic_key, title, url, published_at,
                is_relevant, topic, importance, summary_fa, key_points, status
            ) VALUES (
                %s, %s, 'MachineLearning', 'ai', %s, %s, %s,
                TRUE, 'ai', 'medium', %s, '[]'::jsonb, 'sent'
            )
            """,
            (
                post_id,
                reddit_id,
                f"Candidate {reddit_id}",
                f"https://example.com/{reddit_id}",
                published_at,
                f"خلاصه {reddit_id}",
            ),
        )


def test_real_db_sends_a_new_relevant_post_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """FR-9 + FR-10 with real Postgres: store the row, then send exactly one message."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_new"
    llm = FakeChatCompletion(VALID_JSON)
    sender = _patch_http_seams(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        posts=[_post(reddit_id)],
        llm=llm,
    )

    pipeline.run_once()

    row = _row_for(db_connection, reddit_id)
    assert row["status"] == "sent"
    assert row["sent_at"] is not None
    assert row["is_relevant"] is True
    assert row["duplicate_of_id"] is None
    assert len(llm.calls) == 1
    assert len(sender.messages) == 1


def test_real_db_a_second_run_does_not_reanalyse_or_resend(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 1 with real Postgres: the stored ``reddit_id`` short-circuits the next run."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_stored"
    llm = FakeChatCompletion(VALID_JSON)
    sender = _patch_http_seams(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        posts=[_post(reddit_id)],
        llm=llm,
    )

    pipeline.run_once()
    assert len(llm.calls) == 1

    pipeline.run_once()  # the feed still returns the same item

    assert len(llm.calls) == 1  # repository.exists() short-circuits before the LLM
    assert len(sender.messages) == 1
    assert repository.fetch_pending_to_send() == []
    assert _row_for(db_connection, reddit_id)["status"] == "sent"


def test_real_db_maps_a_duplicate_index_to_the_real_candidate_id(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 4 with real Postgres: candidate #2 becomes that row's id, never the number 2."""
    now = datetime.now(timezone.utc)
    first_id, second_id = 771001, 771002
    _insert_candidate(
        db_connection,
        post_id=first_id,
        reddit_id=f"{TEST_REDDIT_ID_PREFIX}pipeline_cand_a",
        published_at=now - timedelta(hours=1),
    )
    _insert_candidate(
        db_connection,
        post_id=second_id,
        reddit_id=f"{TEST_REDDIT_ID_PREFIX}pipeline_cand_b",
        published_at=now - timedelta(hours=2),
    )

    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_dup"
    answer = json.dumps({**VALID_ANSWER, "duplicate_of_candidate_index": 2}, ensure_ascii=False)
    sender = _patch_http_seams(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        posts=[_post(reddit_id)],
        llm=FakeChatCompletion(answer),
    )

    pipeline.run_once()

    row = _row_for(db_connection, reddit_id)
    assert row["status"] == "skipped_duplicate"
    assert row["duplicate_of_id"] == second_id
    assert row["duplicate_of_id"] != 2
    assert sender.messages == []


def test_real_db_stores_a_low_importance_post_without_sending(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """FR-10(c) with real Postgres: kept and labelled, never sent, never left pending."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_low"
    answer = json.dumps({**VALID_ANSWER, "importance": "low"}, ensure_ascii=False)
    sender = _patch_http_seams(
        monkeypatch,
        settings=settings.model_copy(update={"min_importance_to_send": "medium"}),
        topics_config=topics_config,
        posts=[_post(reddit_id)],
        llm=FakeChatCompletion(answer),
    )

    pipeline.run_once()

    row = _row_for(db_connection, reddit_id)
    assert row["status"] == "skipped_low_importance"
    assert row["is_relevant"] is True
    assert row["importance"] == "low"
    assert row["sent_at"] is None
    assert sender.messages == []
    assert repository.fetch_pending_to_send() == []  # nothing for FR-11 either


def test_real_db_records_failed_and_keeps_processing_the_next_post(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """Invariants 3 + 8 with real Postgres: a bad answer is stored as ``failed``, the run goes on."""
    broken = f"{TEST_REDDIT_ID_PREFIX}pipeline_broken"
    fine = f"{TEST_REDDIT_ID_PREFIX}pipeline_fine"
    sender = _patch_http_seams(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        posts=[_post(broken, title="Broken post"), _post(fine, title="Good post")],
        llm=_answering_by_title(
            {"Broken post": "sorry, no json", "Good post": VALID_JSON}
        ),
    )

    pipeline.run_once()

    broken_row = _row_for(db_connection, broken)
    assert broken_row["status"] == "failed"
    assert broken_row["is_relevant"] is None
    assert broken_row["sent_at"] is None

    fine_row = _row_for(db_connection, fine)
    assert fine_row["status"] == "sent"
    assert fine_row["sent_at"] is not None

    assert len(sender.messages) == 1
    assert "Good post" in sender.messages[0]
