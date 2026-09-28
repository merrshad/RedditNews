"""Pipeline tests: ordering, status decisions, idempotency and error isolation."""

from __future__ import annotations

import json

import pytest

from app.models import AnalyzedPost, LlmAnalysis, PostRecord, RawPost
from app.pipeline import (
    Pipeline,
    build_analyzed_post,
    determine_status,
    meets_importance_threshold,
    to_post_record,
)
from app.settings import Settings, TopicConfig
from app.telegram_notifier import TelegramError
from tests.conftest import FakeChatCompletion

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
        topic="ai",
        importance="medium",
        summary_fa="خلاصه پست معلق.",
        key_points=["نکته"],
        status="to_send",
    )


def _candidate() -> PostRecord:
    return PostRecord(
        id=77,
        reddit_id="t3_prev",
        subreddit="MachineLearning",
        source_topic_key="ai",
        title="Previous",
        url="https://example.com/prev",
        summary_fa="خلاصه",
        key_points=["نکته"],
        status="sent",
    )


class FakeRepository:
    """In-memory stand-in for :class:`app.repository.PostRepository`."""

    def __init__(
        self,
        *,
        existing: tuple[str, ...] = (),
        candidates: tuple[PostRecord, ...] = (),
        pending: tuple[PostRecord, ...] = (),
    ) -> None:
        self.existing = set(existing)
        self.candidates = list(candidates)
        self.pending = list(pending)
        self.inserted: list[AnalyzedPost] = []
        self.sent_ids: list[int] = []
        self.calls: list[str] = []
        self._next_id = 100

    def exists(self, reddit_id: str) -> bool:
        self.calls.append("exists")
        return reddit_id in self.existing

    def get_similarity_candidates(self, *, limit: int, hours: int) -> list[PostRecord]:
        self.calls.append(f"candidates(limit={limit},hours={hours})")
        return list(self.candidates)

    def insert_post(self, record: AnalyzedPost) -> int | None:
        self.calls.append("insert")
        if record.reddit_id in self.existing:
            return None
        self._next_id += 1
        self.inserted.append(record)
        return self._next_id

    def mark_sent(self, post_id: int) -> None:
        self.calls.append("mark_sent")
        self.sent_ids.append(post_id)

    def fetch_pending_send(self) -> list[PostRecord]:
        self.calls.append("fetch_pending_send")
        return list(self.pending)


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.fail_first = False

    def send_message(self, text: str, **kwargs: object) -> None:
        if self.fail_first:
            self.fail_first = False
            raise TelegramError("sendMessage rejected")
        self.messages.append(text)


class FailingNotifier(FakeNotifier):
    def send_message(self, text: str, **kwargs: object) -> None:
        raise TelegramError("sendMessage rejected")


def _raise_runtime_error(system_prompt: str, user_prompt: str) -> str:
    raise RuntimeError("provider down")


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeChatCompletion:
    """Replace the LLM transport with a canned (valid) answer; tests may change it."""
    fake = FakeChatCompletion(json.dumps(VALID_ANSWER, ensure_ascii=False))
    monkeypatch.setattr("app.analyzer.chat_completion", fake)
    return fake


def _build_pipeline(
    *,
    settings: Settings,
    topics: list[TopicConfig],
    repository: FakeRepository,
    llm: FakeChatCompletion,
    answer: str | None = None,
    notifier: FakeNotifier | None = None,
) -> tuple[Pipeline, FakeNotifier]:
    if answer is not None:
        llm.answer = answer
    notifier = notifier if notifier is not None else FakeNotifier()
    pipeline = Pipeline(
        settings=settings,
        repository=repository,  # type: ignore[arg-type]
        notifier=notifier,  # type: ignore[arg-type]
        topics=topics,
    )
    return pipeline, notifier


@pytest.fixture(autouse=True)
def _single_post_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default a run fetches exactly one post; tests may override this."""
    monkeypatch.setattr("app.pipeline.fetch_posts", lambda topics, *, max_retries: [_post()])


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


def test_build_analyzed_post_for_the_failed_path_keeps_raw_post_fields() -> None:
    record = build_analyzed_post(_post(), status="failed", raw_response={"is_relevant": True})

    assert record.status == "failed"
    assert record.is_relevant is None
    assert record.key_points == []
    assert record.llm_raw_response == {"is_relevant": True}


def test_to_post_record_copies_the_stored_fields() -> None:
    record = build_analyzed_post(_post(), status="to_send")

    post_record = to_post_record(record, 42)

    assert post_record.id == 42
    assert post_record.status == "to_send"
    assert post_record.title == record.title
    assert post_record.key_points == []


# --- run_once --------------------------------------------------------------------


def test_run_once_skips_a_post_that_is_already_stored(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository(existing=("t3_1abcde",))
    pipeline, notifier = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm
    )

    pipeline.run_once()

    assert repository.inserted == []
    assert repository.calls == ["fetch_pending_send", "exists"]
    assert notifier.messages == []


def test_run_once_stores_irrelevant_posts_without_sending(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository()
    pipeline, notifier = _build_pipeline(
        settings=settings,
        topics=topics,
        repository=repository,
        llm=llm,
        answer=json.dumps(
            {**VALID_ANSWER, "is_relevant": False, "summary_fa": ""}, ensure_ascii=False
        ),
    )

    pipeline.run_once()

    assert [record.status for record in repository.inserted] == ["skipped_irrelevant"]
    assert notifier.messages == []
    assert repository.sent_ids == []


def test_run_once_maps_a_duplicate_index_to_the_real_database_id(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository(candidates=(_candidate(),))
    pipeline, notifier = _build_pipeline(
        settings=settings,
        topics=topics,
        repository=repository,
        llm=llm,
        answer=json.dumps({**VALID_ANSWER, "duplicate_of_candidate_index": 1}, ensure_ascii=False),
    )

    pipeline.run_once()

    assert repository.inserted[0].duplicate_of_id == 77
    assert repository.inserted[0].status == "skipped_duplicate"
    assert notifier.messages == []


def test_run_once_sends_a_qualifying_post_only_after_storing_it(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository()
    pipeline, notifier = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm
    )

    pipeline.run_once()

    assert repository.inserted[0].status == "to_send"
    # FR-9/NFR-3: the validated analysis is kept for audit.
    assert repository.inserted[0].llm_raw_response == LlmAnalysis.model_validate(
        VALID_ANSWER
    ).model_dump(mode="json")
    assert repository.sent_ids == [repository._next_id]
    assert len(notifier.messages) == 1
    assert "خلاصه فارسی پست." in notifier.messages[0]
    assert "• نکته اول" in notifier.messages[0]
    assert repository.calls.index("insert") < repository.calls.index("mark_sent")


def test_run_once_respects_a_higher_importance_threshold(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository()
    pipeline, notifier = _build_pipeline(
        settings=settings.model_copy(update={"min_importance_to_send": "high"}),
        topics=topics,
        repository=repository,
        llm=llm,
        answer=json.dumps({**VALID_ANSWER, "importance": "low"}, ensure_ascii=False),
    )

    pipeline.run_once()

    assert [record.status for record in repository.inserted] == ["skipped_low_importance"]
    assert notifier.messages == []


def test_run_once_records_failed_status_for_invalid_llm_output(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository()
    pipeline, notifier = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm, answer="sorry, no json"
    )

    pipeline.run_once()

    assert [record.status for record in repository.inserted] == ["failed"]
    assert repository.inserted[0].llm_raw_response is None
    assert notifier.messages == []


def test_run_once_leaves_the_post_unstored_when_the_llm_call_itself_fails(
    settings: Settings, topics: list[TopicConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.analyzer.chat_completion", _raise_runtime_error)
    repository = FakeRepository()
    pipeline = Pipeline(
        settings=settings,
        repository=repository,  # type: ignore[arg-type]
        notifier=FakeNotifier(),  # type: ignore[arg-type]
        topics=topics,
    )

    pipeline.run_once()

    # No record is written, so the next run analyses the post again (no send either).
    assert repository.inserted == []
    assert repository.sent_ids == []


def test_run_once_keeps_processing_after_one_post_fails(
    settings: Settings,
    topics: list[TopicConfig],
    llm: FakeChatCompletion,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.pipeline.fetch_posts",
        lambda topics, *, max_retries: [_post("t3_broken"), _post("t3_fine")],
    )
    repository = FakeRepository()
    pipeline, _ = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm, answer="not json"
    )

    pipeline.run_once()

    assert [record.reddit_id for record in repository.inserted] == ["t3_broken", "t3_fine"]
    assert all(record.status == "failed" for record in repository.inserted)


def test_run_once_retries_pending_sends_without_re_analysing(
    settings: Settings,
    topics: list[TopicConfig],
    llm: FakeChatCompletion,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched: list[str] = []
    monkeypatch.setattr(
        "app.pipeline.fetch_posts",
        lambda topics, *, max_retries: fetched.append("fetch") or [],
    )
    repository = FakeRepository(pending=(_pending(5),))
    pipeline, notifier = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm
    )

    pipeline.run_once()

    assert repository.sent_ids == [5]
    assert "خلاصه پست معلق." in notifier.messages[0]
    assert repository.calls.index("fetch_pending_send") < repository.calls.index("mark_sent")
    assert fetched == ["fetch"]
    assert llm.calls == []  # FR-11: no second LLM call


def test_run_once_keeps_a_failed_send_as_to_send(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository()
    pipeline, _ = _build_pipeline(
        settings=settings,
        topics=topics,
        repository=repository,
        llm=llm,
        notifier=FailingNotifier(),
    )

    pipeline.run_once()  # must not raise

    assert repository.inserted[0].status == "to_send"
    assert repository.sent_ids == []


def test_run_once_continues_with_the_next_pending_send_after_a_failure(
    settings: Settings,
    topics: list[TopicConfig],
    llm: FakeChatCompletion,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.pipeline.fetch_posts", lambda topics, *, max_retries: [])
    repository = FakeRepository(pending=(_pending(5), _pending(6)))
    notifier = FakeNotifier()
    notifier.fail_first = True
    pipeline, notifier = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm, notifier=notifier
    )

    pipeline.run_once()

    assert repository.sent_ids == [6]
    assert len(notifier.messages) == 1
    assert "Pending post 6" in notifier.messages[0]


def test_run_once_wires_real_modules_end_to_end(
    settings: Settings,
    topics: list[TopicConfig],
    llm: FakeChatCompletion,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the HTTP (RSS/LLM) and storage boundaries are faked, everything else is real."""
    from tests.test_reddit_source import SAMPLE_FEED

    monkeypatch.setattr(
        "app.reddit_source.fetch_feed", lambda feed_url, *, max_retries: SAMPLE_FEED.encode()
    )
    repository = FakeRepository()
    pipeline, notifier = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm
    )

    pipeline.run_once()

    stored = repository.inserted[0]
    assert stored.reddit_id == "t3_1abcde"
    assert stored.status == "to_send"
    assert stored.source_topic_key == "ai"
    assert stored.subreddit == "MachineLearning"
    assert stored.importance == "high"
    assert stored.key_points == ["نکته اول"]

    message = notifier.messages[0]
    assert "A new open model was released" in message
    assert "🗂 موضوع: هوش مصنوعی" in message
    assert "⭐ اهمیت: زیاد" in message
    assert "خلاصه فارسی پست." in message
    assert message.endswith("مشاهده پست اصلی</a>")


def test_resolve_duplicate_ignores_an_out_of_range_index(
    settings: Settings, topics: list[TopicConfig], llm: FakeChatCompletion
) -> None:
    repository = FakeRepository()
    pipeline, _ = _build_pipeline(
        settings=settings, topics=topics, repository=repository, llm=llm
    )
    analysis = LlmAnalysis.model_validate({**VALID_ANSWER, "duplicate_of_candidate_index": 9})

    assert pipeline._resolve_duplicate(analysis, []) is None
