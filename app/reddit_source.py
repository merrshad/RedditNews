"""RSS fetching and parsing for the configured Reddit feeds (FR-1).

This module owns everything that turns a feed document into a :class:`RawPost`, and
nothing else: feeds, their topic and their batch cap are *data* now (``sources`` in
Postgres, managed by the admin), so the caller passes :class:`SourceRecord` rows in.

The batch cap is ``min(source.fetch_limit or default_fetch_limit, available items)``:
a source set to 50 whose document holds 17 items yields 17, and the number 25 is only
ever the configured default (``RSS_FETCH_LIMIT``), never a property of the code.

Phase 6 made the fetching side awaitable: every feed is an independent HTTP request, so
:func:`fetch_all` runs up to ``MAX_CONCURRENT_FEEDS`` of them at the same time instead of
one after another, while parsing stays pure and synchronous.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any, Iterable, Sequence

import feedparser
import httpx

from app.models import RawPost, SourceRecord, TopicRecord
from app.retry import PermanentError, async_sleep, retryable

logger = logging.getLogger(__name__)

USER_AGENT = "reddit-telegram-digest/0.1 (+RSS reader)"
REQUEST_TIMEOUT_SECONDS = 30.0

# How many feeds may be in flight at once. A handful is enough to hide the latency of a
# slow feed without turning a cycle into a burst that Reddit rate-limits (NFR-6).
MAX_CONCURRENT_FEEDS = 4

# Reddit answers 429 (not 403) when it rate-limits the `.rss` endpoints, and sends a
# `Retry-After` header with it. Honouring that header is the difference between backing
# off and hammering: a real phase-4 run got 429 after a handful of quick requests and the
# plain 1s/2s backoff was not enough. The wait is capped so one slow feed can never stall
# a whole cycle for minutes (NFR-2).
RATE_LIMIT_STATUS = 429
MAX_RETRY_AFTER_SECONDS = 30.0

_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_SUBREDDIT_IN_URL_RE = re.compile(r"/r/([^/]+)/", re.IGNORECASE)
_ID_IN_LINK_RE = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)


class FeedError(RuntimeError):
    """Raised when a feed cannot be fetched or parsed."""


def topic_display_names(topics: Sequence[TopicRecord]) -> dict[str, str]:
    """Map topic key -> Persian display name for the Telegram messages (FR-10, Invariant 5)."""
    return {topic.key: topic.name for topic in topics}


def strip_html(value: str) -> str:
    """Turn the HTML blob Reddit puts in RSS summaries into plain text."""
    text = unescape(_TAG_RE.sub(" ", value))
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def subreddit_from_feed_url(feed_url: str) -> str | None:
    """Extract ``MachineLearning`` from ``.../r/MachineLearning/new/.rss``."""
    match = _SUBREDDIT_IN_URL_RE.search(feed_url or "")
    return match.group(1) if match else None


def subreddit_from_entry(entry: Any) -> str | None:
    """Reddit RSS entries carry the subreddit as a ``r/<name>`` tag."""
    for tag in entry.get("tags") or []:
        term = (tag.get("term") or "").strip()
        if term.lower().startswith("r/") and len(term) > 2:
            return term[2:]
    return None


def _reddit_id_from_entry(entry: Any) -> str | None:
    for field in ("id", "guid"):
        value = (entry.get(field) or "").strip()
        if value:
            return value
    link = (entry.get("link") or "").strip()
    match = _ID_IN_LINK_RE.search(link)
    return f"t3_{match.group(1)}" if match else None


def _posted_at_from_entry(entry: Any) -> datetime | None:
    for field in ("published_parsed", "updated_parsed"):
        struct_time = entry.get(field)
        if struct_time:
            return datetime.fromtimestamp(calendar.timegm(struct_time), tz=timezone.utc)
    return None


def _raw_content_from_entry(entry: Any) -> str | None:
    content_blocks: Iterable[Any] = entry.get("content") or []
    for block in content_blocks:
        value = block.get("value") if isinstance(block, dict) else None
        if value:
            return strip_html(value) or None
    summary = entry.get("summary") or entry.get("description")
    if summary:
        return strip_html(summary) or None
    return None


def _author_from_entry(entry: Any) -> str | None:
    author = (entry.get("author") or "").strip()
    if not author:
        return None
    for prefix in ("/u/", "u/", "/user/", "user/"):
        if author.lower().startswith(prefix):
            return author[len(prefix):]
    return author


def parse_entry(entry: Any, *, source_topic_key: str, feed_url: str = "") -> RawPost | None:
    """Convert one feedparser entry into a :class:`RawPost` (pure, FR-1).

    Returns ``None`` for unusable entries (e.g. no id or no link) instead of raising,
    so a single malformed item cannot stop the run.
    """
    reddit_id = _reddit_id_from_entry(entry)
    url = (entry.get("link") or "").strip()
    if not reddit_id or not url:
        logger.warning("Skipping unusable entry from feed %s", feed_url or "<inline>")
        return None

    title = (entry.get("title") or "").strip()
    subreddit = subreddit_from_entry(entry) or subreddit_from_feed_url(feed_url) or "unknown"

    return RawPost(
        reddit_id=reddit_id,
        subreddit=subreddit,
        source_topic_key=source_topic_key,
        title=title or url,
        url=url,
        author=_author_from_entry(entry),
        raw_content=_raw_content_from_entry(entry),
        posted_at=_posted_at_from_entry(entry),
    )


def parse_feed(content: bytes | str, *, source_topic_key: str, feed_url: str = "") -> list[RawPost]:
    """Parse raw RSS document bytes into posts (pure, FR-1)."""
    parsed = feedparser.parse(content)
    if parsed.get("bozo") and not parsed.entries:
        raise FeedError(
            f"could not parse feed {feed_url or '<inline>'}: "
            f"{type(parsed.get('bozo_exception')).__name__}"
        )

    posts: list[RawPost] = []
    for entry in parsed.entries:
        post = parse_entry(entry, source_topic_key=source_topic_key, feed_url=feed_url)
        if post is not None:
            posts.append(post)
    return posts


def _retry_after_seconds(response: httpx.Response) -> float:
    """How long the server asked us to wait, capped at ``MAX_RETRY_AFTER_SECONDS``.

    Absent or unparsable header means ``0.0``: the normal exponential backoff applies.
    """
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return 0.0
    return max(0.0, min(seconds, MAX_RETRY_AFTER_SECONDS))


async def _get(feed_url: str) -> httpx.Response:
    """One HTTP GET — the only network call in this module (tests replace this seam)."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        return await client.get(feed_url)


@retryable(description=lambda feed_url: f"fetch RSS feed {feed_url}")
async def fetch_feed(feed_url: str) -> bytes:
    """Download one feed document with bounded retries and backoff (NFR-2).

    The attempt budget comes from ``HTTP_MAX_RETRIES`` (``app/retry.py``). Feed URLs
    are public configuration, never secrets, so they are safe to log (Invariant 6).

    Only the rate-limit status is worth another attempt: any other 4xx (moved, blocked,
    gone) would fail identically forever, so it raises :class:`PermanentError` and the
    run moves on to the next feed at once (Invariant 8).
    """
    response = await _get(feed_url)

    status = response.status_code
    if status == RATE_LIMIT_STATUS:
        delay = _retry_after_seconds(response)
        if delay:
            logger.warning(
                "Feed %s is rate-limited; waiting %.0fs before retrying", feed_url, delay
            )
            await async_sleep(delay)
    elif 400 <= status < 500:
        # 429 is handled above, so anything left here is the request's own fault (404 feed
        # deleted, 403 blocked, 400 malformed) and would fail identically forever.
        raise PermanentError(f"feed {feed_url} answered HTTP {status}")

    # 429 and every 5xx keep their place in the retry budget; the caller backs off.
    response.raise_for_status()
    return response.content


async def fetch_source(source: SourceRecord, *, default_fetch_limit: int) -> list[RawPost]:
    """Fetch and parse one source, capped at its own (or the global) limit (FR-1).

    Returns ``[]`` when the feed is unreachable or unparsable: one dead feed must never
    stop the run (NFR-2 / Invariant 8). Reddit's documents are newest-first, so the cap
    keeps the newest items.
    """
    limit = source.fetch_limit or default_fetch_limit
    try:
        content = await fetch_feed(source.rss_url)
        parsed_posts = parse_feed(
            content, source_topic_key=source.topic_key, feed_url=source.rss_url
        )
    except Exception as exc:  # NFR-2: a bad feed must not stop the run
        logger.error("Skipping feed %s after failure: %s", source.rss_url, exc)
        return []

    kept = parsed_posts[:limit]
    logger.info(
        "Feed %s parsed: %d item(s), %d kept (limit %d)",
        source.rss_url,
        len(parsed_posts),
        len(kept),
        limit,
    )
    return kept


async def fetch_all(
    sources: Sequence[SourceRecord], *, default_fetch_limit: int
) -> list[RawPost]:
    """Fetch and parse every configured source, a few of them at a time (FR-1).

    Items are de-duplicated by ``reddit_id`` within the batch, keeping the first
    occurrence (Invariant 1): the same post can legitimately appear in two feeds (a
    subreddit and a cross-post), and the exact duplicate check in the database can only
    see what was stored before this run.

    The sources are independent, so they are fetched concurrently (bounded by
    :data:`MAX_CONCURRENT_FEEDS`) and then flattened **in the configured order**, which
    keeps the result — and therefore the review messages — deterministic.
    """
    if not sources:
        return []

    gate = asyncio.Semaphore(MAX_CONCURRENT_FEEDS)

    async def _gated(source: SourceRecord) -> list[RawPost]:
        async with gate:
            return await fetch_source(source, default_fetch_limit=default_fetch_limit)

    batches = await asyncio.gather(*(_gated(source) for source in sources))

    posts: list[RawPost] = []
    seen: set[str] = set()
    for batch in batches:
        for post in batch:
            if post.reddit_id in seen:
                continue
            seen.add(post.reddit_id)
            posts.append(post)

    logger.info("Fetched %d candidate post(s) in total", len(posts))
    return posts
