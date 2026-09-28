"""Orchestration of the one-way pipeline (AGENTS.md section 4.1).

Order per run:
1. FR-11 — retry posts left as ``to_send`` by an earlier run.
2. FR-1  — fetch/parse all configured RSS feeds.
3. per post: FR-2 duplicate check -> FR-3..FR-8 single LLM call -> FR-9 store ->
   FR-10 send when the record qualifies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from app import repository
from app.analyzer import AnalysisError, analyze
from app.formatting import format_message
from app.models import IMPORTANCE_ORDER, LlmAnalysis, PostRecord, PostStatus, RawPost
from app.reddit_source import TopicsConfig, fetch_all, topic_display_names
from app.settings import Settings
from app.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


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
    """Decide the persisted status of an analysed post (FR-2/FR-10/Invariant 2)."""
    if not analysis.is_relevant:
        return "skipped_irrelevant"
    if duplicate_of_id is not None:
        return "skipped_duplicate"
    if not meets_importance_threshold(analysis.importance, min_importance_to_send):
        return "skipped_low_importance"
    return "to_send"


def build_post_record(
    post: RawPost,
    *,
    status: PostStatus,
    analysis: LlmAnalysis | None = None,
    duplicate_of_id: int | None = None,
    raw_response: dict[str, Any] | None = None,
) -> PostRecord:
    """Flatten a raw post (+ optional analysis) into the row to store (FR-9).

    Used both for analysed posts and for rows recorded as ``failed`` because the LLM
    answer was invalid (Invariant 3); ``id`` stays ``None`` until ``repository.save``.
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


class Pipeline:
    """Wires the modules together; holds no I/O logic of its own beyond orchestration."""

    def __init__(
        self,
        *,
        settings: Settings,
        notifier: TelegramNotifier,
        topics_config: TopicsConfig,
    ) -> None:
        self._settings = settings
        self._notifier = notifier
        self._topics_config = topics_config
        # The analyzer takes the topic objects themselves (key + Persian name).
        self._topics = list(topics_config.topics)
        self._topic_names = topic_display_names(self._topics)

    def run_once(self) -> None:
        """One full pipeline cycle; never raises for a single bad post/feed."""
        logger.info("Pipeline run started")
        self.retry_pending_sends()

        posts = fetch_all(self._topics_config)
        for post in posts:
            try:
                self.process_post(post)
            except Exception:
                logger.exception(
                    "Unexpected failure while processing reddit_id=%s; continuing",
                    post.reddit_id,
                )

        logger.info("Pipeline run finished (%d fetched post(s))", len(posts))

    def retry_pending_sends(self) -> None:
        """FR-11 — finish sends from earlier runs without re-analysing them."""
        pending = repository.fetch_pending_to_send()
        if not pending:
            return
        logger.info("Retrying %d pending send(s) from earlier runs", len(pending))
        for record in pending:
            try:
                self._send(record)
            except Exception as exc:
                logger.error(
                    "Pending send still failing for reddit_id=%s: %s", record.reddit_id, exc
                )

    def process_post(self, post: RawPost) -> None:
        """FR-2 -> FR-3..FR-8 -> FR-9 -> FR-10 for a single fetched post."""
        if repository.exists(post.reddit_id):
            logger.info("reddit_id=%s already stored, skipping LLM call", post.reddit_id)
            return

        candidates = repository.fetch_recent_candidates(
            self._settings.similarity_lookback_limit,
            self._settings.similarity_lookback_hours,
        )

        try:
            analysis = analyze(post, candidates, self._topics)
        except AnalysisError as exc:
            # Invariant 3: invalid output is recorded as failed, never assumed valid.
            logger.error("Invalid LLM output for reddit_id=%s: %s", post.reddit_id, exc)
            repository.save(
                build_post_record(post, status="failed", raw_response=exc.raw_response)
            )
            return
        except Exception as exc:
            # Transient LLM/network trouble: leave the post unstored so the next run
            # analyses it again (no record, no send, no duplicate).
            logger.error("LLM call failed for reddit_id=%s: %s", post.reddit_id, exc)
            return

        duplicate_of_id = self._resolve_duplicate(analysis, candidates)
        status = determine_status(
            analysis,
            duplicate_of_id=duplicate_of_id,
            min_importance_to_send=self._settings.min_importance_to_send,
        )
        record = build_post_record(
            post,
            status=status,
            analysis=analysis,
            duplicate_of_id=duplicate_of_id,
            # FR-9/NFR-3: keep the validated analysis JSON for audit/debugging.
            raw_response=analysis.model_dump(mode="json"),
        )

        # Invariant 2: store first, send only afterwards.
        post_id = repository.save(record)
        if status != "to_send":
            return

        try:
            self._send(record.model_copy(update={"id": post_id}))
        except Exception as exc:
            # The row stays 'to_send' and FR-11 retries it on the next run.
            logger.error("Sending failed for reddit_id=%s: %s", post.reddit_id, exc)

    @staticmethod
    def _resolve_duplicate(
        analysis: LlmAnalysis, candidates: Sequence[PostRecord]
    ) -> int | None:
        """Invariant 4 — map the LLM's local 1-based index to a real database id."""
        index = analysis.duplicate_of_candidate_index
        if index is None:
            return None
        if 1 <= index <= len(candidates):
            return candidates[index - 1].id
        logger.warning("Ignoring out-of-range duplicate index %s returned by the LLM", index)
        return None

    def _topic_label(self, record: PostRecord) -> str | None:
        return self._topic_names.get(record.topic or "") or self._topic_names.get(
            record.source_topic_key
        )

    def _send(self, record: PostRecord) -> None:
        """Format and send one stored post, then mark it as sent (FR-10)."""
        if record.id is None:
            raise ValueError("refusing to send a post that was never stored (Invariant 2)")

        self._notifier.send_message(
            format_message(record, topic_name=self._topic_label(record))
        )
        repository.update_status(record.id, "sent", datetime.now(timezone.utc))
