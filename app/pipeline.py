"""Orchestration: the one place where the stages of the pipeline are wired together.

AGENTS.md section 4.1 defines the order of one cycle:

1. FR-11 — retry the sends an earlier run left as ``to_send`` (without re-analysing).
2. FR-1  — fetch/parse every configured RSS feed.
3. per post: FR-2 exact duplicate check -> FR-3..FR-8 one LLM call -> FR-9 store ->
   FR-10 send, but only when the stored record qualifies.

Two module-level entry points, both callable without arguments because they read their own
configuration (`get_settings()` / `load_topics_config()`) — the same shape as the rest of
the project, where `repository`, `analyzer` and `telegram_notifier` are plain functions.

The collaborators are imported into this namespace on purpose: a test swaps
``app.pipeline.fetch_all`` / ``analyze`` / ``send_message`` for a fake instead of touching
the network or Telegram (NFR-8). Every failure is contained per post, so one bad post can
never stop the rest of the run (Invariant 8), and the module holds no state between runs
(Invariant 7).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from app import repository
from app.analyzer import AnalysisError, analyze
from app.formatting import format_message
from app.models import IMPORTANCE_ORDER, LlmAnalysis, PostRecord, PostStatus, RawPost
from app.reddit_source import (
    TopicConfig,
    TopicsConfig,
    fetch_all,
    load_topics_config,
    topic_display_names,
)
from app.settings import Settings, get_settings
from app.telegram_notifier import send_message

logger = logging.getLogger(__name__)


# --- pure decisions (no I/O, so they are unit-testable on their own) --------------


def meets_importance_threshold(importance: str | None, minimum: str) -> bool:
    """FR-10(c) — is ``importance`` at least ``minimum``?"""
    if importance not in IMPORTANCE_ORDER or minimum not in IMPORTANCE_ORDER:
        return False
    return IMPORTANCE_ORDER[importance] >= IMPORTANCE_ORDER[minimum]


def determine_status(
    analysis: LlmAnalysis,
    *,
    duplicate_of_id: int | None,
    min_importance_to_send: str,
) -> PostStatus:
    """Decide the persisted status of an analysed post (FR-3, FR-10, Invariant 2).

    ``skipped_low_importance`` is deliberately distinct from ``skipped_irrelevant``: the
    post *is* relevant, it just did not clear ``MIN_IMPORTANCE_TO_SEND``.
    """
    if not analysis.is_relevant:
        return "skipped_irrelevant"
    if duplicate_of_id is not None:
        return "skipped_duplicate"
    if not meets_importance_threshold(analysis.importance, min_importance_to_send):
        return "skipped_low_importance"
    return "to_send"


def resolve_duplicate_id(
    analysis: LlmAnalysis, candidates: Sequence[PostRecord]
) -> int | None:
    """Invariant 4 — map the LLM's local 1-based index to a real database id.

    This is the only place in the project where that mapping happens: the model never sees
    a database id, it only ever picks a position from the candidate list it was given.
    """
    index = analysis.duplicate_of_candidate_index
    if index is None:
        return None

    # Defensive: the analyzer already rejects an out-of-range index (Invariant 3), so
    # reaching this point with one means a bug, not bad model output — fail loudly rather
    # than silently guessing a foreign key.
    assert 1 <= index <= len(candidates), (
        f"duplicate_of_candidate_index {index} is outside the "
        f"{len(candidates)} candidate(s) sent to the LLM"
    )
    return candidates[index - 1].id


def build_post_record(
    post: RawPost,
    *,
    status: PostStatus,
    analysis: LlmAnalysis | None = None,
    duplicate_of_id: int | None = None,
    raw_response: dict[str, Any] | None = None,
) -> PostRecord:
    """Flatten a raw post (+ optional analysis) into the row to store (FR-9).

    Used both for analysed posts and for rows recorded as ``failed`` because the LLM answer
    was invalid (Invariant 3); ``id`` stays ``None`` until ``repository.save``.
    """
    return PostRecord(
        reddit_id=post.reddit_id,
        subreddit=post.subreddit,
        source_topic_key=post.source_topic_key,
        title=post.title,
        url=post.url,
        author=post.author,
        raw_content=post.raw_content,
        published_at=post.published_at,
        is_relevant=analysis.is_relevant if analysis else None,
        duplicate_of_id=duplicate_of_id,
        topic=analysis.topic if analysis else None,
        importance=analysis.importance if analysis else None,
        summary_fa=analysis.summary_fa if analysis else None,
        key_points=list(analysis.key_points) if analysis else [],
        llm_raw_response=raw_response,
        status=status,
    )


# --- entry points -----------------------------------------------------------------


def retry_pending_sends() -> None:
    """FR-11 — try again the sends an earlier run left as ``to_send``.

    ``repository.fetch_pending_to_send`` only returns rows that were already analysed and
    stored, so the LLM is never called here (Invariant 7). A send that fails again simply
    stays ``to_send`` for the next run.
    """
    pending = repository.fetch_pending_to_send()
    if not pending:
        return

    logger.info("Retrying %d pending send(s) from earlier runs", len(pending))
    topic_names = topic_display_names(load_topics_config().topics)
    for record in pending:
        try:
            _send(record, topic_names=topic_names)
        except Exception:
            # Expected rejections come back as False instead; only the unexpected lands here.
            logger.exception(
                "Unexpected failure while retrying the send of reddit_id=%s; continuing",
                record.reddit_id,
            )


def run_once() -> None:
    """One full pipeline cycle; one bad post never stops the others (Invariant 8)."""
    logger.info("Pipeline run started")

    # FR-11 first: finish what an earlier run could not send, before fetching anything new.
    retry_pending_sends()

    settings = get_settings()
    topics_config = load_topics_config()
    allowed_topics: list[TopicConfig] = list(topics_config.topics)
    topic_names = topic_display_names(allowed_topics)

    posts = fetch_all(topics_config)
    for post in posts:
        try:
            _process_post(
                post,
                settings=settings,
                allowed_topics=allowed_topics,
                topic_names=topic_names,
            )
        except Exception:
            logger.exception(
                "Unexpected failure while processing reddit_id=%s; continuing", post.reddit_id
            )

    logger.info("Pipeline run finished (%d fetched post(s))", len(posts))


# --- per-post stages --------------------------------------------------------------


def _process_post(
    post: RawPost,
    *,
    settings: Settings,
    allowed_topics: list[TopicConfig],
    topic_names: Mapping[str, str],
) -> None:
    """FR-2 -> FR-3..FR-8 -> FR-9 -> FR-10 for one fetched post."""
    if repository.exists(post.reddit_id):
        logger.info("reddit_id=%s already stored, skipping the LLM call", post.reddit_id)
        return

    # FR-4 / Invariant 10: the lookback size is bounded by config and passed as-is.
    candidates = repository.fetch_recent_candidates(
        settings.similarity_lookback_limit,
        settings.similarity_lookback_hours,
    )

    try:
        analysis = analyze(post, list(candidates), allowed_topics)
    except AnalysisError as exc:
        # Invariant 3: an answer that failed validation is stored as failed and never
        # treated as relevant/not-a-duplicate.
        logger.error("Invalid LLM output for reddit_id=%s: %s", post.reddit_id, exc)
        repository.save(
            build_post_record(post, status="failed", raw_response=exc.raw_response)
        )
        return

    # Any other exception (the LLM/network still down after HTTP_MAX_RETRIES) is left to
    # bubble up: `run_once` logs it and carries on, and no row is written, so the next run
    # fetches and analyses the post again instead of losing it (AGENTS.md section 4.1).

    duplicate_of_id = resolve_duplicate_id(analysis, candidates)
    status = determine_status(
        analysis,
        duplicate_of_id=duplicate_of_id,
        min_importance_to_send=settings.min_importance_to_send,
    )
    record = build_post_record(
        post,
        status=status,
        analysis=analysis,
        duplicate_of_id=duplicate_of_id,
        # FR-9/NFR-3: the validated analysis JSON is kept for audit/debugging.
        raw_response=analysis.model_dump(mode="json"),
    )

    # Invariant 2(d): the post is completely stored before any send attempt.
    post_id = repository.save(record)
    if status != "to_send":
        return

    _send(record.model_copy(update={"id": post_id}), topic_names=topic_names)


def _send(record: PostRecord, *, topic_names: Mapping[str, str]) -> bool:
    """Format and send one *stored* post (FR-10).

    Returns whether Telegram accepted the message. A rejection is not an exception any
    more: the row keeps ``status='to_send'`` and the next run's FR-11 retry picks it up.
    """
    if record.id is None:
        raise ValueError("refusing to send a post that was never stored (Invariant 2)")

    text = format_message(record, topic_name=_topic_label(record, topic_names))
    if not send_message(text):
        logger.error(
            "Telegram rejected the message for reddit_id=%s; keeping status='to_send'",
            record.reddit_id,
        )
        return False

    # FR-10: only a successful send moves the row to 'sent' and records sent_at.
    repository.update_status(record.id, "sent", datetime.now(timezone.utc))
    logger.info("Sent reddit_id=%s to Telegram (id=%s)", record.reddit_id, record.id)
    return True


def _topic_label(record: PostRecord, topic_names: Mapping[str, str]) -> str | None:
    """Persian topic name for the message; falls back to the stored source topic key.

    Invariant 5: the reader should see «هوش مصنوعی», not the raw YAML key ``ai``.
    """
    return topic_names.get(record.topic or "") or topic_names.get(record.source_topic_key)
