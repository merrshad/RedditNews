"""FR-3..FR-8 tests: prompt building, JSON parsing and Invariant 3 validation."""

from __future__ import annotations

import json

import pytest

from app.analyzer import (
    LlmOutputError,
    analyze_post,
    build_user_prompt,
    load_system_prompt,
    parse_analysis,
)
from app.models import PostRecord, RawPost
from tests.conftest import StubLlmClient

ALLOWED_TOPICS = ["ai", "startup"]

VALID_ANSWER = {
    "is_relevant": True,
    "duplicate_of_candidate_index": None,
    "topic": "ai",
    "importance": "high",
    "summary_fa": "خلاصه فارسی پست.",
    "key_points_fa": ["نکته اول", "نکته دوم"],
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
            subreddit="MachineLearning",
            source_topic_key="ai",
            title="Previous two",
            url="https://example.com/prev2",
            summary_fa="خلاصه ۲",
        ),
    ]


def test_load_system_prompt_is_read_from_the_prompt_file() -> None:
    prompt = load_system_prompt()

    assert "is_relevant" in prompt
    assert "duplicate_of_candidate_index" in prompt


def test_build_user_prompt_exposes_local_indexes_but_not_database_ids() -> None:
    payload = json.loads(build_user_prompt(_post(), _candidates(), ALLOWED_TOPICS))

    assert payload["allowed_topics"] == ALLOWED_TOPICS
    assert payload["new_post"]["title"] == "A new open model was released"
    assert [candidate["index"] for candidate in payload["recent_posts"]] == [1, 2]
    assert payload["recent_posts"][1]["summary_fa"] == "خلاصه ۲"
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "t3_prev1" not in dumped
    assert '"id"' not in dumped


def test_parse_analysis_accepts_a_plain_json_answer() -> None:
    outcome = parse_analysis(
        json.dumps(VALID_ANSWER, ensure_ascii=False),
        allowed_topics=ALLOWED_TOPICS,
        candidate_count=2,
    )

    assert outcome.analysis.importance == "high"
    assert outcome.analysis.key_points_fa == ["نکته اول", "نکته دوم"]
    assert outcome.raw_response["topic"] == "ai"


def test_parse_analysis_tolerates_code_fences_and_surrounding_prose() -> None:
    raw = f"Sure, here it is:\n```json\n{json.dumps(VALID_ANSWER, ensure_ascii=False)}\n```"

    outcome = parse_analysis(raw, allowed_topics=ALLOWED_TOPICS, candidate_count=0)

    assert outcome.analysis.is_relevant is True


def test_parse_analysis_normalizes_topic_case() -> None:
    answer = {**VALID_ANSWER, "topic": "AI"}

    outcome = parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=0)

    assert outcome.analysis.topic == "ai"


def test_parse_analysis_rejects_non_json_output() -> None:
    with pytest.raises(LlmOutputError):
        parse_analysis("I cannot answer that.", allowed_topics=ALLOWED_TOPICS, candidate_count=0)


def test_parse_analysis_rejects_a_missing_required_field() -> None:
    answer = {key: value for key, value in VALID_ANSWER.items() if key != "importance"}

    with pytest.raises(LlmOutputError):
        parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=0)


def test_parse_analysis_rejects_an_empty_summary_for_a_relevant_post() -> None:
    answer = {**VALID_ANSWER, "summary_fa": "   "}

    with pytest.raises(LlmOutputError):
        parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=0)


def test_parse_analysis_rejects_a_topic_outside_the_allowed_list() -> None:
    answer = {**VALID_ANSWER, "topic": "crypto"}

    with pytest.raises(LlmOutputError):
        parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=0)


def test_parse_analysis_rejects_a_missing_topic_for_a_relevant_post() -> None:
    answer = {**VALID_ANSWER, "topic": None}

    with pytest.raises(LlmOutputError):
        parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=0)


def test_parse_analysis_drops_an_unknown_topic_for_an_irrelevant_post() -> None:
    answer = {
        **VALID_ANSWER,
        "is_relevant": False,
        "topic": "crypto",
        "summary_fa": "",
    }

    outcome = parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=0)

    assert outcome.analysis.topic is None
    assert outcome.analysis.is_relevant is False


def test_parse_analysis_rejects_an_index_outside_the_sent_candidates() -> None:
    answer = {**VALID_ANSWER, "duplicate_of_candidate_index": 3}

    with pytest.raises(LlmOutputError):
        parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=2)


def test_parse_analysis_accepts_an_index_inside_the_sent_candidates() -> None:
    answer = {**VALID_ANSWER, "duplicate_of_candidate_index": 2}

    outcome = parse_analysis(json.dumps(answer), allowed_topics=ALLOWED_TOPICS, candidate_count=2)

    assert outcome.analysis.duplicate_of_candidate_index == 2


def test_analyze_post_sends_one_call_with_candidates_in_the_prompt() -> None:
    client = StubLlmClient(json.dumps(VALID_ANSWER, ensure_ascii=False))

    outcome = analyze_post(
        _post(),
        _candidates(),
        llm_client=client,
        allowed_topics=ALLOWED_TOPICS,
        system_prompt="SYSTEM",
    )

    assert len(client.calls) == 1
    system_prompt, user_prompt = client.calls[0]
    assert system_prompt == "SYSTEM"
    assert "خلاصه ۱" in user_prompt
    assert outcome.analysis.summary_fa == "خلاصه فارسی پست."


def test_analyze_post_surfaces_invalid_output_as_llm_output_error() -> None:
    client = StubLlmClient("not json at all")

    with pytest.raises(LlmOutputError):
        analyze_post(
            _post(),
            [],
            llm_client=client,
            allowed_topics=ALLOWED_TOPICS,
            system_prompt="SYSTEM",
        )
