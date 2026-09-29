"""Orchestration: the one place where the stages of the pipeline are wired together.

AGENTS.md section 4.1 defines the order of one cycle:

1. FR-11 — publish the posts an earlier run left as ``to_send`` (no re-analysis).
2. FR-13 — run the single LLM call for the posts an admin approved, then publish the
   ones that clear the relevance/duplicate/importance gates.
3. FR-1  — fetch every active source, capped per source, and store what is new as ``new``.
4. FR-12 — deliver every stored-but-undelivered post to the private review channel.

The order matters: nothing new is fetched before the older work is finished, and the LLM
is only ever reached through step 2, whose queue only grows when an admin presses ✅.

Module-level functions that read their own configuration (`get_settings()`), the same
shape as the rest of the project. The collaborators are imported into this namespace on
purpose: a test swaps ``app.pipeline.fetch_all`` / ``analyze`` / ``send_message`` for a
fake instead of touching the network or Telegram (NFR-8). Every failure is contained per
post, so one bad post can never stop the rest of the run (Invariant 8), and the module
holds no state between runs (Invariant 7).

Everything here is awaitable (phase 6). The stages that were a plain ``for`` loop are now
``asyncio.gather`` fan-outs bounded by a small semaphore — a post's analysis or publication
is an independent network round trip, and doing 20 of them one after another was the main
reason a busy cycle felt slow. The order of the *stages* is untouched, and the per-post
try/except that Invariant 8 relies on is still there, just inside the concurrent worker.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeVar

from app import repository, review, telegram_notifier
from app.analyzer import AnalysisError, analyze
from app.formatting import format_message
from app.models import (
    IMPORTANCE_ORDER,
    LlmAnalysis,
    PostRecord,
    PostStatus,
    RawPost,
    ReviewStatus,
    SourceRecord,
    TopicRecord,
)
from app.reddit_source import fetch_all, topic_display_names
from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# How many posts may be in flight at once. Kept small on purpose: the free LLM endpoint
# rate-limits, and Telegram throttles a burst of sends, so a handful of overlapping calls
# is where the latency win stops and the 429s start (NFR-6).
MAX_CONCURRENT_ANALYSES = 3
MAX_CONCURRENT_PUBLICATIONS = 3

T = TypeVar("T")


async def _run_bounded(
    items: Sequence[T], worker: Callable[[T], Awaitable[None]], *, limit: int
) -> None:
    """Run ``worker`` over every item concurrently, at most ``limit`` at a time.

    ``worker`` owns the per-item error handling, so this helper never sees a failure: one
    bad post still cannot stop the others (Invariant 8). An empty ``items`` is a no-op, and
    with a fake Telegram client that never yields, the items are visited in order — which is
    what keeps the tests deterministic.
    """
    if not items:
        return
    gate = asyncio.Semaphore(limit)

    async def _gated(item: T) -> None:
        async with gate:
            await worker(item)

    await asyncio.gather(*(_gated(item) for item in items))


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
    """Decide the persisted status of an *approved* post (FR-3, FR-10, Invariant 2).

    The admin's ✅ is the gate in front of the LLM; these gates run afterwards and decide
    whether the analysed post may actually be published. ``skipped_low_importance`` is
    deliberately distinct from ``skipped_irrelevant``: the post *is* relevant, it just did
    not clear ``MIN_IMPORTANCE_TO_SEND``.
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
    post: RawPost, *, status: PostStatus = "new", review_status: ReviewStatus = "pending_review"
) -> PostRecord:
    """Flatten a fetched post into the row to store (FR-9).

    Used for the pre-review insert only: the analysis columns are filled much later, by
    :func:`repository.record_analysis`, once an admin approved the post (FR-13).
    """
    return PostRecord(
        reddit_id=post.reddit_id,
        subreddit=post.subreddit,
        source_topic_key=post.source_topic_key,
        title=post.title,
        url=post.url,
        author=post.author,
        raw_content=post.raw_content,
        posted_at=post.posted_at,
        status=status,
        review_status=review_status,
    )


# --- entry points -----------------------------------------------------------------


async def run_once() -> None:
    """One full pipeline cycle; one bad post never stops the others (Invariant 8)."""
    logger.info("Pipeline run started")

    settings = get_settings()
    topics = repository.list_topics(active_only=True)
    topic_names = topic_display_names(topics)

    # FR-11 and FR-13 before anything new: finish what an earlier run left behind.
    await process_approved_posts(topic_names=topic_names, topics=topics, settings=settings)

    # FR-1/FR-2: fetch the active sources and store what is new, capped per source.
    sources = repository.list_sources(active_only=True)
    if sources:
        stored = await store_new_posts(sources, default_fetch_limit=settings.rss_fetch_limit)
        logger.info("Stored %d new post(s)", stored)
    else:
        logger.warning("No active source is configured; nothing to fetch (FR-14)")

    # FR-12: every stored post waits for a human, one message per post.
    await review.dispatch_pending_reviews(topic_names)

    logger.info("Pipeline run finished")


async def process_approved_posts(
    *,
    topic_names: Mapping[str, str] | None = None,
    topics: Sequence[TopicRecord] | None = None,
    settings: Settings | None = None,
) -> None:
    """FR-11 + FR-13 — publish the pending sends and analyse the approved posts.

    ``main`` calls this right after an admin decision, so a click is acted on in seconds
    instead of waiting for the next scheduled cycle; the arguments exist only so one run
    can pass down the configuration and taxonomy it already read.
    """
    settings = settings or get_settings()
    topic_names = _topic_names(topic_names)
    topics = list(topics) if topics is not None else repository.list_topics(active_only=True)

    # FR-11 first: retrying finished publications is pure Telegram traffic and must not
    # queue behind a slow batch of model calls.
    await retry_pending_sends(topic_names=topic_names)

    if not topics:
        logger.warning("No active topic is configured; the approved queue cannot be analysed")
        return
    await analyze_approved_posts(topics=list(topics), settings=settings, topic_names=topic_names)


async def retry_pending_sends(*, topic_names: Mapping[str, str] | None = None) -> None:
    """FR-11 — publish the posts an earlier run left as ``to_send``.

    ``repository.fetch_pending_to_send`` only returns rows that were already analysed and
    stored, so the LLM is never called here (Invariant 7).
    """
    pending = repository.fetch_pending_to_send()
    if not pending:
        return

    logger.info("Retrying %d pending publication(s) from earlier runs", len(pending))
    resolved = _topic_names(topic_names)

    async def _retry(record: PostRecord) -> None:
        try:
            await _publish(record, topic_names=resolved)
        except Exception:
            # Expected rejections come back as False instead; only the unexpected lands here.
            logger.exception(
                "Unexpected failure while publishing reddit_id=%s; continuing", record.reddit_id
            )

    await _run_bounded(pending, _retry, limit=MAX_CONCURRENT_PUBLICATIONS)


async def analyze_approved_posts(
    *, topics: list[TopicRecord], settings: Settings, topic_names: Mapping[str, str]
) -> None:
    """FR-13 — one LLM call per approved post, then publish what qualifies.

    The model calls run a few at a time: a free endpoint answers in seconds but a slow one
    answers in minutes, and five approved posts waiting on each other was most of the delay
    an admin noticed after pressing ✅ a few times.
    """
    approved = repository.fetch_approved_for_analysis()
    if not approved:
        return

    logger.info("Analysing %d approved post(s)", len(approved))

    async def _analyse(record: PostRecord) -> None:
        try:
            await _analyze_one(record, topics=topics, settings=settings, topic_names=topic_names)
        except Exception:
            logger.exception(
                "Unexpected failure while analysing reddit_id=%s; continuing", record.reddit_id
            )

    await _run_bounded(approved, _analyse, limit=MAX_CONCURRENT_ANALYSES)


async def store_new_posts(
    sources: Sequence[SourceRecord], *, default_fetch_limit: int
) -> int:
    """FR-1/FR-2 — fetch, drop what is already stored, and insert the rest as ``new``.

    Nothing here touches the LLM: a freshly stored post goes to the review channel first
    (FR-12), and only an admin's ✅ puts it in the analysis queue (FR-13).
    """
    posts = await fetch_all(sources, default_fetch_limit=default_fetch_limit)
    stored = 0

    for post in posts:
        try:
            if repository.exists(post.reddit_id):
                logger.info("reddit_id=%s already stored, skipping", post.reddit_id)
                continue
            repository.save(build_post_record(post))
            stored += 1
        except Exception:
            logger.exception(
                "Unexpected failure while storing reddit_id=%s; continuing", post.reddit_id
            )
    return stored


# --- per-post stages --------------------------------------------------------------


async def _analyze_one(
    record: PostRecord,
    *,
    topics: list[TopicRecord],
    settings: Settings,
    topic_names: Mapping[str, str],
) -> None:
    """FR-3..FR-8 -> FR-9 -> FR-10 for one approved post."""
    if record.id is None:  # pragma: no cover - only stored rows are ever approved
        raise ValueError("refusing to analyse a post that was never stored")

    # FR-4 / Invariant 10: the lookback size is bounded by config and passed as-is.
    candidates = repository.fetch_recent_candidates(
        settings.similarity_lookback_limit,
        settings.similarity_lookback_hours,
    )

    try:
        analysis = await analyze(_as_raw_post(record), list(candidates), topics)
    except AnalysisError as exc:
        # Invariant 3: an answer that failed validation is stored as failed and never
        # treated as relevant/not-a-duplicate.
        logger.error("Invalid LLM output for reddit_id=%s: %s", record.reddit_id, exc)
        repository.record_analysis(
            record.id,
            analysis=None,
            duplicate_of_id=None,
            raw_response=exc.raw_response,
            status="failed",
            ai_error=str(exc),
        )
        await review.announce_analysis_failed(record)
        return

    # Any other exception (the LLM/network still down after HTTP_MAX_RETRIES) is left to
    # bubble up: the caller logs it and carries on, and the row stays `approved`, so the
    # next run analyses it again instead of losing it (AGENTS.md section 4.1).

    duplicate_of_id = resolve_duplicate_id(analysis, candidates)
    status = determine_status(
        analysis,
        duplicate_of_id=duplicate_of_id,
        min_importance_to_send=settings.min_importance_to_send,
    )
    repository.record_analysis(
        record.id,
        analysis=analysis,
        duplicate_of_id=duplicate_of_id,
        # FR-9/NFR-3: the validated analysis is kept for audit/debugging.
        raw_response=analysis.model_dump(mode="json"),
        status=status,
    )

    logger.info("id=%s reviewed and analysed -> status=%s", record.id, status)
    if status != "to_send":
        await review.announce_skipped(record, status=status)
        return

    await _publish(
        _analysed(record, analysis, duplicate_of_id=duplicate_of_id), topic_names=topic_names
    )


async def _publish(record: PostRecord, *, topic_names: Mapping[str, str]) -> bool:
    """Publish one *analysed* post to the public channel (FR-10, Invariant 12).

    The publication claim is taken first, so exactly one caller may talk to Telegram for a
    given post: a replayed decision or an overlapping run cannot publish it twice. A
    rejected send hands the claim back, which keeps ``to_send`` as the retry queue
    (FR-11).
    """
    if record.id is None:
        raise ValueError("refusing to publish a post that was never stored (Invariant 2)")

    if not repository.claim_for_publish(record.id):
        logger.info("reddit_id=%s was already claimed for publication; skipping", record.reddit_id)
        return False

    text = format_message(record, topic_name=_topic_label(record, topic_names))
    channel_id = get_settings().telegram_chat_id
    message_id = await telegram_notifier.send_message(text)
    if message_id is None:
        repository.release_publish_claim(record.id)
        logger.error(
            "Telegram rejected the message for reddit_id=%s; keeping it for a later run",
            record.reddit_id,
        )
        return False

    repository.mark_published(record.id, channel_id=channel_id, message_id=message_id)
    logger.info("Published reddit_id=%s (id=%s)", record.reddit_id, record.id)
    await review.announce_published(record)
    return True


def _as_raw_post(record: PostRecord) -> RawPost:
    """The analysed-stage view of a stored row: what ``analyzer`` needs and nothing more."""
    return RawPost(
        reddit_id=record.reddit_id,
        subreddit=record.subreddit,
        source_topic_key=record.source_topic_key,
        title=record.title,
        url=record.url,
        author=record.author,
        raw_content=record.raw_content,
        posted_at=record.posted_at,
    )


def _analysed(
    record: PostRecord, analysis: LlmAnalysis, *, duplicate_of_id: int | None
) -> PostRecord:
    """The row as it now is in the database, with the fresh analysis merged in.

    Only called when the gates said ``to_send``, so the publication path gets the analysed
    content without a second read.
    """
    return record.model_copy(
        update={
            "status": "to_send",
            "is_relevant": analysis.is_relevant,
            "duplicate_of_id": duplicate_of_id,
            "topic": analysis.topic,
            "importance": analysis.importance,
            "summary_fa": analysis.summary_fa,
            "key_points": list(analysis.key_points),
        }
    )


def _topic_names(topic_names: Mapping[str, str] | None) -> Mapping[str, str]:
    """The caller's mapping, or a fresh one read from the database."""
    return topic_names if topic_names is not None else topic_display_names(repository.list_topics())


def _topic_label(record: PostRecord, topic_names: Mapping[str, str]) -> str | None:
    """Persian topic name for the message; falls back to the stored source topic key.

    Invariant 5: the reader should see «هوش مصنوعی», not the raw key ``ai``.
    """
    return topic_names.get(record.topic or "") or topic_names.get(record.source_topic_key)
