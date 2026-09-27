"""Prompt building, LLM call and output validation (FR-3..FR-8).

Invariant 3: an invalid or unusable model answer always raises
:class:`LlmOutputError` — it is never silently treated as relevant/non-duplicate.
Invariant 4: only the 1-based candidate indexes sent in this very call are accepted;
mapping an index to a real database id is the caller's job.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from app.llm_client import LlmClient
from app.models import LlmAnalysis, RawPost, SimilarityCandidate

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "analysis_prompt.md"


class LlmOutputError(RuntimeError):
    """The LLM answer could not be parsed/validated into a trusted analysis."""

    def __init__(self, message: str, *, raw_response: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True)
class AnalysisOutcome:
    """A validated analysis plus the parsed raw answer kept for audit (FR-9)."""

    analysis: LlmAnalysis
    raw_response: dict[str, Any]


def load_system_prompt(path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    """Read the editable prompt template (kept out of the code on purpose)."""
    return Path(path).read_text(encoding="utf-8")


def build_user_prompt(
    post: RawPost,
    candidates: Sequence[SimilarityCandidate],
    allowed_topics: Sequence[str],
) -> str:
    """Build the single user message: new post + local-indexed candidates + topics.

    Candidate database ids are deliberately not included (Invariant 4).
    """
    payload = {
        "allowed_topics": list(allowed_topics),
        "new_post": {
            "title": post.title,
            "subreddit": post.subreddit,
            "author": post.author,
            "url": post.url,
            "content": post.raw_content,
        },
        "recent_posts": [
            {
                "index": index,
                "title": candidate.title,
                "summary_fa": candidate.summary_fa,
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse the model answer, tolerating code fences and surrounding prose."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LlmOutputError("LLM answer contained no JSON object") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LlmOutputError(f"LLM answer was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LlmOutputError("LLM answer was not a JSON object")
    return parsed


def validate_against_context(
    analysis: LlmAnalysis,
    *,
    allowed_topics: Sequence[str],
    candidate_count: int,
) -> LlmAnalysis:
    """Enforce the FR-5 topic allow-list and Invariant 4 candidate index range."""
    allowed = {topic.lower() for topic in allowed_topics}
    topic = (analysis.topic or "").strip().lower() or None

    if analysis.is_relevant:
        if topic is None:
            raise LlmOutputError("relevant post returned without a topic")
        if topic not in allowed:
            raise LlmOutputError(f"topic {topic!r} is not in the allowed topic list")
    elif topic is not None and topic not in allowed:
        topic = None

    duplicate_index = analysis.duplicate_of_candidate_index
    if duplicate_index is not None and duplicate_index > candidate_count:
        raise LlmOutputError(
            f"duplicate index {duplicate_index} is outside the {candidate_count} sent candidates"
        )

    return analysis.model_copy(update={"topic": topic})


def parse_analysis(
    raw_text: str,
    *,
    allowed_topics: Sequence[str],
    candidate_count: int,
) -> AnalysisOutcome:
    """Parse + validate one raw LLM answer (Invariant 3)."""
    raw_response = _extract_json_object(raw_text)
    try:
        analysis = LlmAnalysis.model_validate(raw_response)
    except ValidationError as exc:
        raise LlmOutputError(f"LLM answer failed schema validation: {exc}", raw_response=raw_response) from exc

    analysis = validate_against_context(
        analysis, allowed_topics=allowed_topics, candidate_count=candidate_count
    )
    return AnalysisOutcome(analysis=analysis, raw_response=raw_response)


def analyze_post(
    post: RawPost,
    candidates: Sequence[SimilarityCandidate],
    *,
    llm_client: LlmClient,
    allowed_topics: Sequence[str],
    system_prompt: str | None = None,
) -> AnalysisOutcome:
    """Build the prompt, make the single LLM call and validate its answer."""
    prompt = system_prompt if system_prompt is not None else load_system_prompt()
    user_prompt = build_user_prompt(post, candidates, allowed_topics)

    logger.debug("Analyzing reddit_id=%s with %d candidate(s)", post.reddit_id, len(candidates))
    raw_text = llm_client.complete(system_prompt=prompt, user_prompt=user_prompt)

    outcome = parse_analysis(
        raw_text, allowed_topics=allowed_topics, candidate_count=len(candidates)
    )
    logger.info(
        "Analyzed reddit_id=%s relevant=%s importance=%s duplicate_index=%s",
        post.reddit_id,
        outcome.analysis.is_relevant,
        outcome.analysis.importance,
        outcome.analysis.duplicate_of_candidate_index,
    )
    return outcome
