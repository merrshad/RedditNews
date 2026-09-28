"""Pydantic models shared across the pipeline.

Invariant 3: the LLM output is *always* validated against :class:`LlmAnalysis`
before anything else in the pipeline trusts it.

Three models, one per stage of the data: the RSS item (:class:`RawPost`), the model
answer (:class:`LlmAnalysis`) and the ``posts`` row (:class:`PostRecord`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ImportanceLevel = Literal["low", "medium", "high"]
PostStatus = Literal[
    "new",
    "skipped_irrelevant",
    "skipped_duplicate",
    "skipped_low_importance",
    "to_send",
    "sent",
    "failed",
]

# Ordering used for the MIN_IMPORTANCE_TO_SEND comparison (FR-10).
IMPORTANCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class RawPost(BaseModel):
    """A single RSS item, before any LLM analysis (FR-1)."""

    reddit_id: str
    subreddit: str
    source_topic_key: str
    title: str
    url: str
    author: str | None = None
    raw_content: str | None = None
    published_at: datetime | None = None


class LlmAnalysis(BaseModel):
    """Validated output of the single analysis LLM call (FR-3..FR-8)."""

    is_relevant: bool
    duplicate_of_candidate_index: int | None = Field(
        default=None,
        description="1-based index into the candidate list, or null when not a duplicate.",
    )
    topic: str | None = None
    importance: ImportanceLevel
    summary_fa: str
    key_points_fa: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "LlmAnalysis":
        if self.duplicate_of_candidate_index is not None and self.duplicate_of_candidate_index < 1:
            raise ValueError("duplicate_of_candidate_index must be a 1-based index or null")
        if self.is_relevant and not self.summary_fa.strip():
            raise ValueError("summary_fa must not be empty for a relevant post (Invariant 5)")
        return self


class PostRecord(BaseModel):
    """One ``posts`` row, from "built by the pipeline" to "read back from Postgres".

    The same model is used on both sides of the database: the pipeline fills it and
    ``repository.save`` inserts it (``id`` is ``None`` until then), and the candidate
    / pending-send queries map rows back into it. ``reddit_id``/``id`` are never sent
    to the LLM, only the 1-based position of a candidate is (Invariant 4).
    """

    # None until the row exists in Postgres.
    id: int | None = None
    reddit_id: str
    subreddit: str
    source_topic_key: str
    title: str
    url: str
    author: str | None = None
    raw_content: str | None = None
    published_at: datetime | None = None

    is_relevant: bool | None = None
    duplicate_of_id: int | None = None
    topic: str | None = None
    importance: ImportanceLevel | None = None
    summary_fa: str | None = None
    key_points: list[str] = Field(default_factory=list)
    llm_raw_response: dict[str, Any] | None = None

    status: PostStatus = "new"
