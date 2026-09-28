"""FR-3..FR-8 tests: prompt building, the mocked LLM call and Invariant 3 validation."""

from __future__ import annotations

import json

import pytest

from app.analyzer import AnalysisError, analyze, build_user_prompt, load_system_prompt
from app.models import LlmAnalysis, PostRecord, RawPost
from app.reddit_source import TopicConfig
from tests.conftest import FakeChatCompletion

ALLOWED_TOPICS = [
    TopicConfig(key="ai", name="هوش مصنوعی"),
    TopicConfig(key="startup", name="استارتاپ"),
]

VALID_ANSWER = {
    "is_relevant": True,
    "duplicate_of_candidate_index": None,
    "topic": "ai",
    "importance": "high",
    "summary_fa": "خلاصه فارسی پست.",
    "key_points": ["نکته اول", "نکته دوم"],
}


def _post() -> RawPost:
    return RawPost(
        reddit_id="t3_1abcde",
        subreddit="MachineLearning",
        source_topic_key="ai",
        title="A new open model was released",
        url="https://www.reddit.com/r/MachineLearning/comments/1abcde/x/",
        author="somebody",
        raw_content="Body text",
    )


def _candidates() -> list[PostRecord]:
    return [
        PostRecord(
            id=11,
            reddit_id="t3_prev1",
            subreddit="MachineLearning",
            source_topic_key="ai",
            title="Previous one",
            url="https://example.com/prev1",
            summary_fa="خلاصه ۱",
        ),
        PostRecord(
            id=12,
            reddit_id="t3_prev2",
            subreddit="startups",
            source_topic_key="startup",
            title="Previous two",
            url="https://example.com/prev2",
            summary_fa="خلاصه ۲",
        ),
    ]


def _mock_answer(monkeypatch: pytest.MonkeyPatch, answer: str) -> FakeChatCompletion:
    """Replace the real LLM transport with a canned answer (NFR-8: no network)."""
    fake = FakeChatCompletion(answer)
    monkeypatch.setattr("app.analyzer.chat_completion", fake)
    return fake


# --- prompt material -------------------------------------------------------------


def test_load_system_prompt_describes_the_json_contract() -> None:
    prompt = load_system_prompt()

    assert "is_relevant" in prompt
    assert "duplicate_of_candidate_index" in prompt
    assert "key_points" in prompt
    assert "Persian" in prompt


def test_build_user_prompt_exposes_local_indexes_but_not_database_ids() -> None:
    payload = json.loads(build_user_prompt(_post(), _candidates(), ALLOWED_TOPICS))

    assert [topic["key"] for topic in payload["allowed_topics"]] == ["ai", "startup"]
    assert payload["new_post"]["title"] == "A new open model was released"
    assert [candidate["index"] for candidate in payload["candidates"]] == [1, 2]
    assert payload["candidates"][1]["summary_fa"] == "خلاصه ۲"

    dumped = json.dumps(payload, ensure_ascii=False)
    assert "t3_prev1" not in dumped  # never leak a reddit_id
    assert '"id"' not in dumped  # never leak a database id (Invariant 4)


# --- happy path ------------------------------------------------------------------


def test_analyze_returns_a_validated_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_answer(monkeypatch, json.dumps(VALID_ANSWER, ensure_ascii=False))

    analysis = analyze(_post(), _candidates(), ALLOWED_TOPICS)

    assert isinstance(analysis, LlmAnalysis)
    assert analysis.is_relevant is True
    assert analysis.importance == "high"
    assert analysis.topic == "ai"
    assert analysis.summary_fa == "خلاصه فارسی پست."
    assert analysis.key_points == ["نکته اول", "نکته دوم"]


def test_analyze_makes_one_call_with_the_named_and_indexed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _mock_answer(monkeypatch, json.dumps(VALID_ANSWER, ensure_ascii=False))

    analyze(_post(), _candidates(), ALLOWED_TOPICS)

    assert len(fake.calls) == 1
    system_prompt, user_prompt = fake.calls[0]
    assert system_prompt == load_system_prompt()
    assert "خلاصه ۱" in user_prompt
    assert "t3_prev1" not in user_prompt


def test_analyze_returns_the_raw_candidate_index_without_mapping_it_to_an_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = {**VALID_ANSWER, "duplicate_of_candidate_index": 2}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    analysis = analyze(_post(), _candidates(), ALLOWED_TOPICS)

    # Candidate 2 has database id 12, but analyze must not resolve that (Invariant 4).
    assert analysis.duplicate_of_candidate_index == 2


# --- failure paths (each must surface as AnalysisError) --------------------------


def test_analyze_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_answer(monkeypatch, "I cannot answer that.")

    with pytest.raises(AnalysisError):
        analyze(_post(), _candidates(), ALLOWED_TOPICS)


def test_analyze_rejects_a_missing_required_field(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = {key: value for key, value in VALID_ANSWER.items() if key != "importance"}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    with pytest.raises(AnalysisError):
        analyze(_post(), _candidates(), ALLOWED_TOPICS)


def test_analyze_rejects_a_topic_outside_the_allowed_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = {**VALID_ANSWER, "topic": "crypto"}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    with pytest.raises(AnalysisError):
        analyze(_post(), _candidates(), ALLOWED_TOPICS)


def test_analyze_rejects_a_duplicate_index_outside_the_sent_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = {**VALID_ANSWER, "duplicate_of_candidate_index": 3}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    with pytest.raises(AnalysisError):
        analyze(_post(), _candidates(), ALLOWED_TOPICS)


def test_analyze_rejects_a_duplicate_index_when_no_candidate_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = {**VALID_ANSWER, "duplicate_of_candidate_index": 1}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    with pytest.raises(AnalysisError):
        analyze(_post(), [], ALLOWED_TOPICS)


def test_analyze_accepts_more_key_points_than_the_prompt_asks_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FR-8 asks the model for 2-5 bullets; the code deliberately does not enforce it.

    The prompt is where the 2-5 range is stated, and this test pins the choice to stay
    lenient here: rejecting the whole answer over a sixth bullet would turn a good post into
    ``failed`` (Invariant 3) and lose the summary the reader wanted. ``formatting`` renders
    whatever it is given and keeps the message inside Telegram's limit, so an over-eager
    model costs a slightly longer message, not a lost post.
    """
    answer = {**VALID_ANSWER, "key_points": [f"نکته {index}" for index in range(1, 8)]}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    analysis = analyze(_post(), _candidates(), ALLOWED_TOPICS)

    assert len(analysis.key_points) == 7  # kept as-is, not truncated and not rejected


def test_analyze_accepts_no_key_points_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other end of the range: an empty list is valid and simply renders no section."""
    answer = {**VALID_ANSWER, "key_points": []}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    analysis = analyze(_post(), _candidates(), ALLOWED_TOPICS)

    assert analysis.key_points == []


def test_analysis_error_keeps_the_parsed_answer_for_the_audit_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = {**VALID_ANSWER, "topic": "crypto"}
    _mock_answer(monkeypatch, json.dumps(answer, ensure_ascii=False))

    with pytest.raises(AnalysisError) as excinfo:
        analyze(_post(), _candidates(), ALLOWED_TOPICS)

    assert excinfo.value.raw_response == answer
