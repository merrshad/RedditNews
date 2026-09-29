"""Pydantic models shared across the pipeline.

Invariant 3: the LLM output is *always* validated against :class:`LlmAnalysis`
before anything else in the pipeline trusts it.

One model per artefact: an RSS item (:class:`RawPost`), a taxonomy row
(:class:`TopicRecord`, :class:`SourceRecord`), the model answer (:class:`LlmAnalysis`)
and the ``posts`` row (:class:`PostRecord`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ImportanceLevel = Literal["low", "medium", "high"]

# Machine state of a post (AGENTS.md section 10). It walks:
# new -> awaiting_review -> approved -> (to_send) -> publishing -> sent, and a rejected or
# skipped post stops earlier. `failed` means the AI answer was unusable.
PostStatus = Literal[
    "new",
    "awaiting_review",
    "approved",
    "rejected",
    "skipped_irrelevant",
    "skipped_duplicate",
    "skipped_low_importance",
    "to_send",
    "publishing",
    "sent",
    "failed",
]

# The human decision, kept apart from the machine state so an audit can read it alone
# (FR-12). `pending_review` until an admin presses a button, then terminal.
ReviewStatus = Literal["pending_review", "approved", "rejected"]
# What a button press may ask for: the two terminal decisions, never "pending".
ReviewDecision = Literal["approved", "rejected"]

# Ordering used for the MIN_IMPORTANCE_TO_SEND comparison (FR-10).
IMPORTANCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class TopicRecord(BaseModel):
    """One row of ``topics`` (FR-5, FR-14): the allowed topics, owned by the admin."""

    id: int | None = None
    key: str
    name: str
    is_active: bool = True


class SourceRecord(BaseModel):
    """One row of ``sources`` (FR-1, FR-14): a feed, its topic and its own batch cap.

    ``fetch_limit`` is ``None`` when the source uses the global ``RSS_FETCH_LIMIT``.
    """

    id: int | None = None
    topic_key: str
    rss_url: str
    fetch_limit: int | None = None
    is_active: bool = True


class RawPost(BaseModel):
    """A single RSS item, before any review or LLM analysis (FR-1)."""

    reddit_id: str
    subreddit: str
    source_topic_key: str
    title: str
    url: str
    author: str | None = None
    raw_content: str | None = None
    posted_at: datetime | None = None


class LlmAnalysis(BaseModel):
    """Validated output of the single analysis LLM call (FR-3..FR-8).

    Duplicate candidates are referenced only by their 1-based position in the list sent
    to the model — never by a database id (Invariant 4).
    """

    is_relevant: bool
    duplicate_of_candidate_index: int | None = Field(
        default=None,
        description="1-based index into the candidate list, or null when not a duplicate.",
    )
    topic: str
    importance: ImportanceLevel
    summary_fa: str
    key_points: list[str] = Field(default_factory=list)

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
    ``repository.save`` inserts it (``id`` is ``None`` until then), and the review,
    candidate and pending-publication queries map rows back into it. ``reddit_id``/``id``
    are never sent to the LLM, only the 1-based position of a candidate is (Invariant 4).

    Two timestamps that are easy to confuse: ``posted_at`` is when *Reddit* published the
    post, ``published_at`` is when *we* published it to the public channel.
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
    posted_at: datetime | None = None

    status: PostStatus = "new"
    review_status: ReviewStatus = "pending_review"
    reviewed_by: str | None = None
    # The admin's Telegram display name at decision time, so the review message can show
    # *who* decided without another API call (phase 6). `reviewed_by` stays the authority.
    reviewed_by_name: str | None = None
    reviewed_at: datetime | None = None
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    private_channel_id: str | None = None
    private_message_id: int | None = None

    is_relevant: bool | None = None
    duplicate_of_id: int | None = None
    topic: str | None = None
    importance: ImportanceLevel | None = None
    summary_fa: str | None = None
    key_points: list[str] = Field(default_factory=list)
    llm_raw_response: dict[str, Any] | None = None
    ai_processed_at: datetime | None = None
    ai_error: str | None = None

    @field_validator("key_points", mode="before")
    @classmethod
    def _empty_key_points(cls, value: Any) -> Any:
        """A NULL JSONB column means "no key points yet", not an invalid row.

        A post is stored before it is analysed, so ``key_points`` is NULL for as long as it
        waits in the review channel; reading that row back must not fail validation.
        """
        return [] if value is None else value

    public_channel_id: str | None = None
    public_message_id: int | None = None
    published_at: datetime | None = None
