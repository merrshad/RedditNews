"""Prompt building, the single LLM call and output validation (FR-3..FR-8).

Invariant 3: an answer that cannot be validated into :class:`LlmAnalysis` always raises
:class:`AnalysisError` — deciding that a post becomes ``failed`` is the pipeline's job
(phase 3), not this module's.
Invariant 4: only the 1-based candidate indexes passed to the very same call are
accepted; mapping an index to a real database id is the caller's job.
Invariant 5: the LLM is instructed to answer ``summary_fa``/``key_points`` in Persian
regardless of the post language.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.llm_client import chat_completion
from app.models import LlmAnalysis, PostRecord, RawPost
from app.reddit_source import TopicConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "analysis_prompt.md"


class AnalysisError(RuntimeError):
    """The LLM answer could not be turned into a trusted :class:`LlmAnalysis` (Invariant 3).

    ``raw_response`` carries the parsed JSON when it was available, so the pipeline can
    still store it in the ``llm_raw_response`` audit column (FR-9). It is ``None`` when
    the answer was not parseable JSON at all.
    """

    def __init__(self, message: str, *, raw_response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def load_system_prompt(path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    """Read the editable prompt template (kept out of the code on purpose)."""
    return Path(path).read_text(encoding="utf-8")


def build_user_prompt(
    post: RawPost,
    candidates: list[PostRecord],
    allowed_topics: list[TopicConfig],
) -> str:
    """Build the single user message: new post + local-indexed candidates + topics.

    A candidate's database id is deliberately never included — only its 1-based position
    (Invariant 4), its title and its Persian summary, which is all the model needs.
    """
    payload = {
        "allowed_topics": [
            {"key": topic.key, "name": topic.name} for topic in allowed_topics
        ],
        "new_post": {
            "title": post.title,
            "subreddit": post.subreddit,
            "author": post.author,
            "url": post.url,
            "content": post.raw_content,
        },
        "candidates": [
            {"index": index, "title": candidate.title, "summary_fa": candidate.summary_fa}
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse the model answer into a JSON object or raise :class:`AnalysisError`."""
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AnalysisError(f"LLM answer was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise AnalysisError("LLM answer was not a JSON object")
    return data


def _validate_schema(data: dict[str, Any]) -> LlmAnalysis:
    """Validate the parsed answer against the Pydantic model (Invariant 3)."""
    try:
        return LlmAnalysis(**data)
    except ValidationError as exc:
        raise AnalysisError(
            f"LLM answer failed schema validation: {exc}", raw_response=data
        ) from exc


def _validate_against_context(
    analysis: LlmAnalysis,
    *,
    allowed_topics: list[TopicConfig],
    candidates: list[PostRecord],
    raw_response: dict[str, Any],
) -> None:
    """Enforce the FR-5 topic allow-list and the Invariant 4 candidate index range."""
    allowed_keys = {topic.key for topic in allowed_topics}
    if analysis.topic not in allowed_keys:
        raise AnalysisError(
            f"topic {analysis.topic!r} is not one of the allowed topics", raw_response=raw_response
        )

    index = analysis.duplicate_of_candidate_index
    if index is not None and not 1 <= index <= len(candidates):
        raise AnalysisError(
            f"duplicate_of_candidate_index {index} is outside 1..{len(candidates)}",
            raw_response=raw_response,
        )


def analyze(
    post: RawPost,
    candidates: list[PostRecord],
    allowed_topics: list[TopicConfig],
) -> LlmAnalysis:
    """Run the single analysis call for one post and return its validated result.

    Returns the *raw, validated* ``duplicate_of_candidate_index``; it never maps that
    index to a database id (Invariant 4).
    """
    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(post, candidates, allowed_topics)

    logger.debug("Analyzing reddit_id=%s with %d candidate(s)", post.reddit_id, len(candidates))
    raw_text = chat_completion(system_prompt, user_prompt)

    data = _parse_json_object(raw_text)
    analysis = _validate_schema(data)
    _validate_against_context(
        analysis,
        allowed_topics=allowed_topics,
        candidates=candidates,
        raw_response=data,
    )

    logger.info(
        "Analyzed reddit_id=%s relevant=%s importance=%s topic=%s duplicate_index=%s",
        post.reddit_id,
        analysis.is_relevant,
        analysis.importance,
        analysis.topic,
        analysis.duplicate_of_candidate_index,
    )
    return analysis
