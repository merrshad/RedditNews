"""RSS fetching and parsing for the configured Reddit feeds (FR-1)."""

from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any, Iterable, Sequence

import feedparser
import httpx

from app.models import RawPost
from app.retry import call_with_retries
from app.settings import TopicConfig

logger = logging.getLogger(__name__)

USER_AGENT = "reddit-telegram-digest/0.1 (+RSS reader)"
REQUEST_TIMEOUT_SECONDS = 30.0

_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_SUBREDDIT_IN_URL_RE = re.compile(r"/r/([^/]+)/", re.IGNORECASE)
_ID_IN_LINK_RE = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)


class FeedError(RuntimeError):
    """Raised when a feed cannot be fetched or parsed."""


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


def _published_at_from_entry(entry: Any) -> datetime | None:
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
        published_at=_published_at_from_entry(entry),
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


def fetch_feed(feed_url: str, *, max_retries: int) -> bytes:
    """Download one feed with bounded retries/backoff (NFR-2)."""

    def _get() -> bytes:
        response = httpx.get(
            feed_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.content

    # Feed URLs are public configuration, never secrets, so they are safe to log.
    return call_with_retries(
        _get, attempts=max_retries, description=f"fetch RSS feed {feed_url}"
    )


def fetch_posts(topics: Sequence[TopicConfig], *, max_retries: int) -> list[RawPost]:
    """Fetch and parse every feed of every topic, skipping the ones that fail.

    Items are de-duplicated by ``reddit_id`` inside the batch (Invariant 1) while
    keeping the first occurrence.
    """
    posts: list[RawPost] = []
    seen: set[str] = set()

    for topic in topics:
        for feed_url in topic.feeds:
            try:
                content = fetch_feed(feed_url, max_retries=max_retries)
                parsed_posts = parse_feed(
                    content, source_topic_key=topic.key, feed_url=feed_url
                )
            except Exception as exc:  # NFR-2: a bad feed must not stop the run
                logger.error("Skipping feed %s after failure: %s", feed_url, exc)
                continue

            new_posts = [post for post in parsed_posts if post.reddit_id not in seen]
            seen.update(post.reddit_id for post in new_posts)
            posts.extend(new_posts)
            logger.info(
                "Feed %s parsed: %d item(s), %d new in this batch",
                feed_url,
                len(parsed_posts),
                len(new_posts),
            )

    logger.info("Fetched %d candidate post(s) in total", len(posts))
    return posts
