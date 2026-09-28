"""Pipeline tests: ordering, status decisions, idempotency and error isolation.

The fast tests swap in an in-memory repository; the ``real storage`` section below
keeps the repository real (Postgres) and proves the same six behaviours end to end.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from app import repository
from app.models import LlmAnalysis, PostRecord, RawPost
from app.pipeline import (
    Pipeline,
    build_post_record,
    determine_status,
    meets_importance_threshold,
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


def _post(reddit_id: str = "t3_1abcde") -> RawPost:
    return RawPost(
        reddit_id=reddit_id,
        subreddit="MachineLearning",
        source_topic_key="ai",
        title="A new open model was released",
        url=f"https://www.reddit.com/r/MachineLearning/comments/{reddit_id}/x/",
        author="somebody",
        raw_content="Body",
    )


def _pending(post_id: int = 5) -> PostRecord:
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
    message — so a rejected send comes back as ``False`` and the pipeline keeps the
    post as ``to_send`` for the next run (FR-10/FR-11).
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


class SequencedChatCompletion(FakeChatCompletion):
    """Returns one answer per call, so a single run can mix valid and invalid output."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__()
        self._answers = list(answers)

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._answers.pop(0) if self._answers else ""


def _raise_runtime_error(system_prompt: str, user_prompt: str) -> str:
    """A ``chat_completion`` stand-in for the "the provider is down" path."""
    raise RuntimeError("provider down")


def _install_llm(
    monkeypatch: pytest.MonkeyPatch, llm: FakeChatCompletion
) -> FakeChatCompletion:
    """Point ``app.analyzer.chat_completion`` at a fake transport (NFR-8: no network)."""
    monkeypatch.setattr("app.analyzer.chat_completion", llm)
    return llm


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeChatCompletion:
    """The fake LLM transport, exposed so a test can assert how often it was called."""
    return _install_llm(monkeypatch, FakeChatCompletion())


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    topics_config: TopicsConfig,
    repository: FakeRepository | None = None,
    answer: str = json.dumps(VALID_ANSWER, ensure_ascii=False),
    sender: FakeSender | None = None,
    llm: FakeChatCompletion | None = None,
) -> tuple[Pipeline, FakeRepository, FakeSender]:
    """A pipeline wired to in-memory storage/Telegram fakes and a stubbed LLM transport."""
    llm = llm if llm is not None else FakeChatCompletion()
    llm.answer = answer
    _install_llm(monkeypatch, llm)
    repository = patch_repository(monkeypatch, repository or FakeRepository())
    sender = sender if sender is not None else FakeSender()
    pipeline = Pipeline(
        settings=settings,
        topics_config=topics_config,
        send_message=sender,
    )
    return pipeline, repository, sender


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


def test_build_post_record_for_the_failed_path_keeps_raw_post_fields() -> None:
    record = build_post_record(_post(), status="failed", raw_response={"is_relevant": True})

    assert record.id is None
    assert record.status == "failed"
    assert record.is_relevant is None
    assert record.key_points == []
    assert record.llm_raw_response == {"is_relevant": True}
    assert record.title == "A new open model was released"


# --- run_once --------------------------------------------------------------------


def test_run_once_skips_a_post_that_is_already_stored(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, repository, sender = _harness(
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
    pipeline, repository, sender = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        answer=json.dumps(
            {**VALID_ANSWER, "is_relevant": False, "summary_fa": ""},
            ensure_ascii=False,
        ),
    )

    pipeline.run_once()

    assert [record.status for record in repository.saved] == ["skipped_irrelevant"]
    assert sender.messages == []
    assert repository.updates == []


def test_run_once_maps_a_duplicate_index_to_the_real_database_id(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    candidate = PostRecord(
        id=77,
        reddit_id="t3_prev",
        subreddit="MachineLearning",
        source_topic_key="ai",
        title="Previous",
        url="https://example.com/prev",
        summary_fa="خلاصه",
    )
    pipeline, repository, sender = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(candidates=(candidate,)),
        answer=json.dumps({**VALID_ANSWER, "duplicate_of_candidate_index": 1}, ensure_ascii=False),
    )

    pipeline.run_once()

    assert repository.saved[0].duplicate_of_id == 77
    assert repository.saved[0].status == "skipped_duplicate"
    assert sender.messages == []


def test_run_once_sends_a_qualifying_post_only_after_storing_it(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, repository, sender = _harness(
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
    assert repository.calls.index("save") < repository.calls.index("update_status")


def test_run_once_passes_the_configured_lookback_limits(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, repository, _ = _harness(
        monkeypatch, settings=settings, topics_config=topics_config
    )

    pipeline.run_once()

    assert "fetch_recent_candidates(limit=50,hours=72)" in repository.calls


def test_run_once_respects_a_higher_importance_threshold(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, repository, sender = _harness(
        monkeypatch,
        settings=settings.model_copy(update={"min_importance_to_send": "high"}),
        topics_config=topics_config,
        answer=json.dumps({**VALID_ANSWER, "importance": "low"}, ensure_ascii=False),
    )

    pipeline.run_once()

    assert [record.status for record in repository.saved] == ["skipped_low_importance"]
    assert sender.messages == []


def test_run_once_records_failed_status_for_invalid_llm_output(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, repository, sender = _harness(
        monkeypatch, settings=settings, topics_config=topics_config, answer="sorry, no json"
    )

    pipeline.run_once()

    assert [record.status for record in repository.saved] == ["failed"]
    assert repository.saved[0].llm_raw_response is None
    assert sender.messages == []


def test_run_once_leaves_the_post_unstored_when_the_llm_call_itself_fails(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    monkeypatch.setattr("app.analyzer.chat_completion", _raise_runtime_error)
    repository = patch_repository(monkeypatch, FakeRepository())
    pipeline = Pipeline(
        settings=settings,
        topics_config=topics_config,
        send_message=FakeSender(),
    )

    pipeline.run_once()

    # No record is written, so the next run analyses the post again (no send either).
    assert repository.saved == []
    assert repository.updates == []


def test_run_once_keeps_processing_after_one_post_fails(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    monkeypatch.setattr(
        "app.pipeline.fetch_all", lambda topics_config: [_post("t3_broken"), _post("t3_fine")]
    )
    pipeline, repository, _ = _harness(
        monkeypatch, settings=settings, topics_config=topics_config, answer="not json"
    )

    pipeline.run_once()

    assert [record.reddit_id for record in repository.saved] == ["t3_broken", "t3_fine"]
    assert all(record.status == "failed" for record in repository.saved)


def test_run_once_retries_pending_sends_without_re_analysing(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    llm: FakeChatCompletion,
) -> None:
    fetched: list[str] = []
    monkeypatch.setattr(
        "app.pipeline.fetch_all", lambda topics_config: fetched.append("fetch") or []
    )
    pipeline, repository, sender = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(pending=(_pending(5),)),
        llm=llm,
    )

    pipeline.run_once()

    assert repository.sent_ids == [5]
    assert "خلاصه پست معلق." in sender.messages[0]
    assert repository.calls.index("fetch_pending_to_send") < repository.calls.index("update_status")
    assert fetched == ["fetch"]
    assert llm.calls == []  # FR-11: no second LLM call


def test_run_once_keeps_a_failed_send_as_to_send(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, repository, _ = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        sender=FailingSender(),
    )

    pipeline.run_once()  # must not raise

    assert repository.saved[0].status == "to_send"
    assert repository.sent_ids == []


def test_run_once_continues_with_the_next_pending_send_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: [])
    sender = FakeSender(fail_first=True)
    pipeline, repository, sender = _harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        repository=FakeRepository(pending=(_pending(5), _pending(6))),
        sender=sender,
    )

    pipeline.run_once()

    assert repository.sent_ids == [6]
    assert len(sender.messages) == 1
    assert "Pending post 6" in sender.messages[0]


def test_send_refuses_a_post_that_was_never_stored(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Invariant 2 — sending before storing must be impossible."""
    pipeline, repository, sender = _harness(
        monkeypatch, settings=settings, topics_config=topics_config
    )

    with pytest.raises(ValueError, match="never stored"):
        pipeline._send(_pending(5).model_copy(update={"id": None}))

    assert sender.messages == []
    assert repository.updates == []


def test_run_once_wires_real_modules_end_to_end(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    """Only the HTTP (RSS/LLM) and storage boundaries are faked, everything else is real."""
    from tests.test_reddit_source import SAMPLE_FEED

    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: SAMPLE_FEED.encode())
    pipeline, repository, sender = _harness(
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
    assert message.endswith(f"🔗 {_post().url}")


def test_resolve_duplicate_ignores_an_out_of_range_index(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, topics_config: TopicsConfig
) -> None:
    pipeline, _, _ = _harness(monkeypatch, settings=settings, topics_config=topics_config)
    analysis = LlmAnalysis.model_validate({**VALID_ANSWER, "duplicate_of_candidate_index": 9})

    assert pipeline._resolve_duplicate(analysis, []) is None


# --- real storage boundary --------------------------------------------------------


def test_a_failed_send_is_recovered_from_the_database_on_the_next_run(
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
    feed = SAMPLE_FEED.replace("t3_1abcde", reddit_id).replace("t3_2fghij", f"{reddit_id}_two")
    monkeypatch.setattr("app.pipeline.fetch_all", real_fetch_all)
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: feed.encode("utf-8"))
    llm = _install_llm(
        monkeypatch, FakeChatCompletion(json.dumps(VALID_ANSWER, ensure_ascii=False))
    )

    def build(sender: FakeSender) -> Pipeline:
        return Pipeline(
            settings=settings,
            topics_config=topics_config,
            send_message=sender,
        )

    def stored_rows() -> list[dict]:
        with db_connection.cursor() as cursor:
            cursor.execute(
                "SELECT reddit_id, status, sent_at, summary_fa FROM posts "
                "WHERE reddit_id LIKE %s ORDER BY reddit_id",
                (f"{TEST_REDDIT_ID_PREFIX}%",),
            )
            return cursor.fetchall()

    # First run: Telegram is down, but both analysed posts are already stored (Invariant 2).
    build(FailingSender()).run_once()

    assert [row["status"] for row in stored_rows()] == ["to_send", "to_send"]
    assert all(row["sent_at"] is None for row in stored_rows())
    assert len(llm.calls) == 2  # one LLM call per fetched post

    # Second run: no new RSS items, so only the pending rows are retried (FR-11).
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: [])
    working_sender = FakeSender()
    build(working_sender).run_once()

    assert [row["status"] for row in stored_rows()] == ["sent", "sent"]
    assert all(row["sent_at"] is not None for row in stored_rows())
    assert len(llm.calls) == 2  # never re-analysed
    assert len(working_sender.messages) == 2
    assert "خلاصه فارسی پست." in working_sender.messages[0]


# --- real storage boundary: the six behaviours the phase-3 core must guarantee ----
#
# Same scenarios as above, but `app.repository` is *not* patched: only RSS, the LLM
# transport and Telegram are faked, so every row really goes through `db/schema.sql`.


def _row_for(connection: psycopg.Connection, reddit_id: str) -> dict:
    """The stored row for one ``reddit_id`` (asserts it exists)."""
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

    The id is set explicitly so a mapping test can prove the 1-based index really was
    translated into *that* row's id rather than the index value itself.
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


def _real_db_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    topics_config: TopicsConfig,
    posts: list[RawPost],
    answer: str | list[str] = json.dumps(VALID_ANSWER, ensure_ascii=False),
    sender: FakeSender | None = None,
) -> tuple[Pipeline, FakeSender, FakeChatCompletion]:
    """A ``Pipeline`` over the real Postgres repository; only RSS/LLM/Telegram are faked."""
    monkeypatch.setattr("app.pipeline.fetch_all", lambda topics_config: list(posts))
    llm: FakeChatCompletion = (
        SequencedChatCompletion(answer)
        if isinstance(answer, list)
        else FakeChatCompletion(answer)
    )
    _install_llm(monkeypatch, llm)
    sender = sender if sender is not None else FakeSender()
    pipeline = Pipeline(
        settings=settings,
        topics_config=topics_config,
        send_message=sender,
    )
    return pipeline, sender, llm


def test_real_db_sends_a_new_relevant_post_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """FR-9 + FR-10 — store, then send, and only once."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_new"
    pipeline, sender, llm = _real_db_harness(
        monkeypatch, settings=settings, topics_config=topics_config, posts=[_post(reddit_id)]
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
    """Invariant 1 — the same feed item on the next poll costs no LLM call and no send."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_stored"
    pipeline, sender, llm = _real_db_harness(
        monkeypatch, settings=settings, topics_config=topics_config, posts=[_post(reddit_id)]
    )

    pipeline.run_once()
    assert len(llm.calls) == 1

    pipeline.run_once()  # the feed still returns the same reddit_id

    assert len(llm.calls) == 1  # repository.exists() short-circuits before the LLM
    assert len(sender.messages) == 1
    assert len(repository.fetch_pending_to_send()) == 0
    row = _row_for(db_connection, reddit_id)
    assert row["status"] == "sent"


def test_real_db_maps_a_duplicate_index_to_the_real_candidate_id(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 4 — candidate #2 becomes that row's id, never the number 2."""
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
    pipeline, sender, _ = _real_db_harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        posts=[_post(reddit_id)],
        answer=json.dumps(
            {**VALID_ANSWER, "duplicate_of_candidate_index": 2}, ensure_ascii=False
        ),
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
    """FR-10(c) — below ``MIN_IMPORTANCE_TO_SEND`` is kept, called out, and not sent."""
    reddit_id = f"{TEST_REDDIT_ID_PREFIX}pipeline_low"
    pipeline, sender, _ = _real_db_harness(
        monkeypatch,
        settings=settings.model_copy(update={"min_importance_to_send": "medium"}),
        topics_config=topics_config,
        posts=[_post(reddit_id)],
        answer=json.dumps({**VALID_ANSWER, "importance": "low"}, ensure_ascii=False),
    )

    pipeline.run_once()

    row = _row_for(db_connection, reddit_id)
    assert row["status"] == "skipped_low_importance"
    assert row["is_relevant"] is True
    assert row["importance"] == "low"
    assert row["sent_at"] is None
    assert sender.messages == []
    assert repository.fetch_pending_to_send() == []  # nothing left for FR-11 either


def test_real_db_records_failed_and_keeps_processing_the_next_post(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 3 + 8 — bad LLM output is stored as ``failed`` and never stops the run."""
    broken = f"{TEST_REDDIT_ID_PREFIX}pipeline_broken"
    fine = f"{TEST_REDDIT_ID_PREFIX}pipeline_fine"
    pipeline, sender, llm = _real_db_harness(
        monkeypatch,
        settings=settings,
        topics_config=topics_config,
        posts=[_post(broken), _post(fine)],
        answer=["sorry, no json", json.dumps(VALID_ANSWER, ensure_ascii=False)],
    )

    pipeline.run_once()

    broken_row = _row_for(db_connection, broken)
    assert broken_row["status"] == "failed"
    assert broken_row["is_relevant"] is None
    assert broken_row["sent_at"] is None

    fine_row = _row_for(db_connection, fine)
    assert fine_row["status"] == "sent"
    assert fine_row["sent_at"] is not None

    assert len(llm.calls) == 2
    assert len(sender.messages) == 1
    assert "خلاصه فارسی پست." in sender.messages[0]
