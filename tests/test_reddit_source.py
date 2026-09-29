"""FR-1 tests: the per-source batch cap, RSS parsing (pure) and fetch resilience (NFR-2)."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import chain
from typing import Any

import httpx
import pytest

from app.models import SourceRecord, TopicRecord
from app.reddit_source import (
    MAX_RETRY_AFTER_SECONDS,
    FeedError,
    fetch_all,
    fetch_feed,
    parse_entry,
    parse_feed,
    strip_html,
    subreddit_from_feed_url,
    topic_display_names,
)
from app.retry import PermanentError

FEED_URL = "https://www.reddit.com/r/MachineLearning/new/.rss"

# One Atom document as Reddit serves it: a first entry with an HTML content block and
# a subreddit tag, a second one falling back to <summary>.
SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>r/MachineLearning</title>
  <entry>
    <id>t3_1abcde</id>
    <title>A new open model was released</title>
    <link href="https://www.reddit.com/r/MachineLearning/comments/1abcde/a_new_open_model/" />
    <author><name>/u/somebody</name></author>
    <published>2026-09-27T10:00:00+00:00</published>
    <updated>2026-09-27T10:05:00+00:00</updated>
    <category term="r/MachineLearning" label="r/MachineLearning" />
    <content type="html">&lt;p&gt;Full &lt;b&gt;html&lt;/b&gt; body &amp;amp; more&lt;/p&gt;</content>
  </entry>
  <entry>
    <id>t3_2fghij</id>
    <title>Second post</title>
    <link href="https://www.reddit.com/r/MachineLearning/comments/2fghij/second_post/" />
    <author><name>/u/someone_else</name></author>
    <summary>&lt;div&gt;summary &lt;i&gt;text&lt;/i&gt;&lt;/div&gt;</summary>
  </entry>
</feed>
"""

# The RSS 2.0 flavour: <guid> carries the id and <description> the body.
RSS20_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>r/startups</title>
    <item>
      <title>How we launched</title>
      <guid>t3_rss20item</guid>
      <link>https://www.reddit.com/r/startups/comments/rss20item/how_we_launched/</link>
      <description>&lt;p&gt;body &amp;amp; notes&lt;/p&gt;</description>
      <pubDate>Sun, 27 Sep 2026 09:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.fixture
def sample_feed() -> bytes:
    """The sample XML feed document used by the parsing and fetching tests."""
    return SAMPLE_FEED.encode("utf-8")


def _sources(
    *, feeds: dict[str, list[str]], fetch_limit: int | None = None
) -> list[SourceRecord]:
    """The rows ``repository.list_sources`` would hand the collector (FR-1)."""
    return [
        SourceRecord(topic_key=key, rss_url=url, fetch_limit=fetch_limit)
        for key, urls in feeds.items()
        for url in urls
    ]


def _feed_with(ids: list[str]) -> bytes:
    """A minimal Atom document holding one entry per id, newest-first as Reddit sends it."""
    entries = "".join(
        f"<entry><id>{entry_id}</id><title>{entry_id}</title>"
        f"<link href=\"https://example.com/{entry_id}\" /></entry>"
        for entry_id in ids
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">' + entries + "</feed>"
    ).encode("utf-8")


# --- the batch cap is configuration, not a constant (FR-1) ------------------------


def test_topic_display_names_maps_keys_to_persian_names() -> None:
    """Invariant 5: the message shows «هوش مصنوعی», never the raw key."""
    names = topic_display_names(
        [TopicRecord(key="ai", name="هوش مصنوعی"), TopicRecord(key="startup", name="استارتاپ")]
    )

    assert names == {"ai": "هوش مصنوعی", "startup": "استارتاپ"}


def test_the_default_cap_limits_how_many_items_one_source_yields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`min(configured_limit, available)`: 3 items available, cap 2 -> 2 items."""
    document = _feed_with(["t3_one", "t3_two", "t3_three"])
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: document)

    posts = fetch_all(
        _sources(feeds={"ai": ["https://a.example/feed.rss"]}), default_fetch_limit=2
    )

    assert [post.reddit_id for post in posts] == ["t3_one", "t3_two"]


def test_a_cap_larger_than_the_feed_keeps_every_available_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured 50 with a 3-item document yields 3: nothing is invented or dropped."""
    document = _feed_with(["t3_one", "t3_two", "t3_three"])
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: document)

    posts = fetch_all(
        _sources(feeds={"ai": ["https://a.example/feed.rss"]}, fetch_limit=50),
        default_fetch_limit=50,
    )

    assert len(posts) == 3


def test_a_source_cap_overrides_the_global_default(monkeypatch: pytest.MonkeyPatch) -> None:
    document = _feed_with(["t3_one", "t3_two", "t3_three"])
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: document)

    posts = fetch_all(
        _sources(feeds={"ai": ["https://a.example/feed.rss"]}, fetch_limit=1),
        default_fetch_limit=25,
    )

    assert [post.reddit_id for post in posts] == ["t3_one"]


def test_each_source_is_capped_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two feeds in one run keep their own caps (the reason the field is per source)."""
    documents = {
        "https://a.example/feed.rss": _feed_with(["t3_a1", "t3_a2", "t3_a3"]),
        "https://b.example/feed.rss": _feed_with(["t3_b1", "t3_b2", "t3_b3"]),
    }
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: documents[feed_url])
    sources = [
        SourceRecord(topic_key="ai", rss_url="https://a.example/feed.rss", fetch_limit=1),
        SourceRecord(topic_key="startup", rss_url="https://b.example/feed.rss", fetch_limit=3),
    ]

    posts = fetch_all(sources, default_fetch_limit=25)

    assert [post.reddit_id for post in posts] == ["t3_a1", "t3_b1", "t3_b2", "t3_b3"]
    assert {post.source_topic_key for post in posts} == {"ai", "startup"}


def test_a_source_without_its_own_cap_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _feed_with(["t3_one", "t3_two", "t3_three"])
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: document)
    sources = _sources(feeds={"ai": ["https://a.example/feed.rss"]})  # fetch_limit=None

    posts = fetch_all(sources, default_fetch_limit=1)

    assert [post.reddit_id for post in posts] == ["t3_one"]


# --- parsing (pure, FR-1) ---------------------------------------------------------


def test_parse_entry_falls_back_to_the_updated_date() -> None:
    """FR-1: an entry may carry only ``<updated>`` — the date must not be lost."""
    document = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>t3_updatedonly</id>
        <title>Only an updated stamp</title>
        <link href="https://www.reddit.com/r/mlops/comments/updatedonly/x/" />
        <updated>2026-09-27T11:30:00+00:00</updated>
      </entry>
    </feed>"""

    (post,) = parse_feed(document, source_topic_key="ai")

    assert post.posted_at == datetime(2026, 9, 27, 11, 30, tzinfo=timezone.utc)


def test_parse_entry_without_any_date_still_parses() -> None:
    """FR-1: a missing date is not a reason to drop a post (``posted_at`` is nullable).

    The column is nullable on purpose and ``fetch_recent_candidates`` falls back to
    ``fetched_at``, so the post is stored and analyses normally; only the recency window
    is decided by the fetch time instead of the post time.
    """
    document = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>t3_nodate</id>
        <title>No date at all</title>
        <link href="https://www.reddit.com/r/mlops/comments/nodate/x/" />
        <content type="html">body</content>
      </entry>
    </feed>"""

    (post,) = parse_feed(document, source_topic_key="ai")

    assert post.reddit_id == "t3_nodate"
    assert post.posted_at is None
    assert post.title == "No date at all"
    assert post.subreddit == "unknown"  # no r/<name> tag and no feed URL to fall back to


def test_parse_feed_maps_every_required_field(sample_feed: bytes) -> None:
    posts = parse_feed(sample_feed, source_topic_key="ai", feed_url=FEED_URL)

    assert len(posts) == 2
    first = posts[0]
    assert first.reddit_id == "t3_1abcde"
    assert first.subreddit == "MachineLearning"
    assert first.source_topic_key == "ai"
    assert first.title == "A new open model was released"
    assert first.url.endswith("/comments/1abcde/a_new_open_model/")
    assert first.author == "somebody"
    assert first.raw_content == "Full html body & more"
    assert first.posted_at == datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)


def test_parse_feed_falls_back_to_summary_content(sample_feed: bytes) -> None:
    posts = parse_feed(sample_feed, source_topic_key="ai", feed_url=FEED_URL)

    second = posts[1]
    assert second.raw_content == "summary text"
    assert second.posted_at is None


def test_parse_feed_reads_rss2_guid_and_description() -> None:
    posts = parse_feed(RSS20_FEED, source_topic_key="startup", feed_url=FEED_URL)

    assert len(posts) == 1
    post = posts[0]
    assert post.reddit_id == "t3_rss20item"
    assert post.title == "How we launched"
    assert post.raw_content == "body & notes"
    assert post.source_topic_key == "startup"
    assert post.posted_at == datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)


def test_parse_feed_skips_unusable_entries() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>No id and no link</title></entry>
      <entry><id>t3_ok</id><link href="https://example.com/ok" /></entry>
    </feed>
    """

    posts = parse_feed(feed, source_topic_key="ai", feed_url=FEED_URL)

    assert [post.reddit_id for post in posts] == ["t3_ok"]
    # No title in the feed -> the link is used so the NOT NULL column stays useful.
    assert posts[0].title == "https://example.com/ok"


def test_parse_feed_raises_for_unparsable_document() -> None:
    with pytest.raises(FeedError):
        parse_feed(b"this is definitely <<< not >>> a feed", source_topic_key="ai")


def test_parse_entry_derives_id_from_link_when_id_missing() -> None:
    entry = {"link": "https://www.reddit.com/r/startups/comments/9zzzz/hello/"}

    post = parse_entry(entry, source_topic_key="startup", feed_url=FEED_URL)

    assert post is not None
    assert post.reddit_id == "t3_9zzzz"
    # No subreddit tag -> derived from the feed URL.
    assert post.subreddit == "MachineLearning"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>hello <b>world</b></p>", "hello world"),
        ("a &amp; b", "a & b"),
        ("<div>line1</div>\n\n<div>line2</div>", "line1\nline2"),
        ("", ""),
    ],
)
def test_strip_html(raw: str, expected: str) -> None:
    assert strip_html(raw) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.reddit.com/r/MachineLearning/new/.rss", "MachineLearning"),
        ("https://www.reddit.com/search.rss?q=ai", None),
    ],
)
def test_subreddit_from_feed_url(url: str, expected: str | None) -> None:
    assert subreddit_from_feed_url(url) == expected


# --- fetch_feed (retry + User-Agent, NFR-2/Invariant 8) ---------------------------


class _FakeResponse:
    """Minimal ``httpx.Response`` stand-in: only what the status handling reads."""

    def __init__(
        self,
        status_code: int = 200,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", FEED_URL),
                response=None,  # type: ignore[arg-type]
            )


def _explode_first(
    calls: list[str], *, failures: int = 1, payload: bytes = b""
) -> Callable[..., Any]:
    """An ``httpx.get`` stand-in that raises for the first ``failures`` calls."""

    def fake_get(
        url: str, *, headers: dict[str, str], timeout: float, follow_redirects: bool
    ) -> Any:
        calls.append(headers["User-Agent"])
        if len(calls) <= failures:
            raise httpx.ConnectError("connection reset by peer")
        return _FakeResponse(content=payload)

    return fake_get


def test_fetch_feed_sends_a_descriptive_user_agent_and_retries(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("app.reddit_source.httpx.get", _explode_first(calls, payload=sample_feed))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")

    assert fetch_feed(FEED_URL) == sample_feed

    assert len(calls) == 2  # first attempt failed, the retry succeeded
    assert calls[0].startswith("reddit-telegram-digest/")


def test_fetch_feed_gives_up_after_the_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("app.reddit_source.httpx.get", _explode_first(calls, failures=99))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")

    with pytest.raises(httpx.ConnectError):
        fetch_feed(FEED_URL)

    assert len(calls) == 3


def test_fetch_feed_waits_as_long_as_the_rate_limit_asks(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    """A real phase-4 run got HTTP 429 from Reddit: `Retry-After` must be honoured.

    The plain 1s/2s backoff re-hit the rate-limited endpoint almost immediately, which is
    what kept it rate-limited.
    """
    slept: list[float] = []
    responses = chain(
        [_FakeResponse(429, headers={"Retry-After": "7"})],
        [_FakeResponse(content=sample_feed)],
    )
    monkeypatch.setattr("app.reddit_source.httpx.get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(time, "sleep", slept.append)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")

    assert fetch_feed(FEED_URL) == sample_feed

    assert slept[0] == 7  # what the server asked for, before the engine's own backoff
    assert len(slept) == 2  # the rate-limit wait plus one retry backoff


def test_fetch_feed_caps_the_rate_limit_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """One impatient server must not be able to stall a whole cycle (NFR-2)."""
    slept: list[float] = []
    monkeypatch.setattr(
        "app.reddit_source.httpx.get", lambda *args, **kwargs: _FakeResponse(429, headers={"Retry-After": "9999"})
    )
    monkeypatch.setattr(time, "sleep", slept.append)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "1")

    with pytest.raises(httpx.HTTPStatusError):
        fetch_feed(FEED_URL)

    assert slept == [MAX_RETRY_AFTER_SECONDS]


def test_fetch_feed_stops_at_once_for_a_permanent_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted or forbidden feed answers the same forever, so attempts are not wasted."""
    calls: list[int] = []

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        calls.append(1)
        return _FakeResponse(404)

    monkeypatch.setattr("app.reddit_source.httpx.get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")

    with pytest.raises(PermanentError):
        fetch_feed(FEED_URL)

    assert len(calls) == 1


def test_fetch_feed_retries_a_timeout_and_gives_up_after_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NFR-2: a feed that never answers is an ordinary bounded failure.

    A timeout is the realistic "this feed is having a bad day" case, and it must behave
    exactly like any other transport error: retried up to ``HTTP_MAX_RETRIES`` and then
    raised, so ``fetch_all`` can skip that feed and keep the rest of the run.
    """
    calls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append(url)
        raise httpx.ReadTimeout("timed out while reading the feed")

    monkeypatch.setattr("app.reddit_source.httpx.get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")

    with pytest.raises(httpx.ReadTimeout):
        fetch_feed(FEED_URL)

    assert len(calls) == 3


def test_fetch_all_skips_a_timed_out_feed_and_parses_the_rest(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    """Invariant 8: one unresponsive feed must never stop the other feeds."""
    sources = _sources(
        feeds={"ai": ["https://slow.example/feed.rss"], "startup": ["https://ok.example/feed.rss"]}
    )

    def fake_fetch_feed(feed_url: str) -> bytes:
        if "slow" in feed_url:
            raise httpx.ReadTimeout("timed out")
        return sample_feed

    monkeypatch.setattr("app.reddit_source.fetch_feed", fake_fetch_feed)

    posts = fetch_all(sources, default_fetch_limit=25)

    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert {post.source_topic_key for post in posts} == {"startup"}


# --- fetch_all (FR-1) -------------------------------------------------------------


def test_fetch_all_maps_every_feed_to_its_topic(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    sources = _sources(
        feeds={"ai": ["https://a.example/feed.rss"], "startup": ["https://b.example/feed.rss"]}
    )
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: sample_feed)

    posts = fetch_all(sources, default_fetch_limit=25)

    # Two feeds with the same document: the second one is dropped as a duplicate.
    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert {post.source_topic_key for post in posts} == {"ai"}


def test_fetch_all_keeps_going_when_a_feed_fails(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    sources = _sources(
        feeds={"ai": ["https://good.example/feed.rss"], "startup": ["https://bad.example/feed.rss"]}
    )

    def fake_fetch_feed(feed_url: str) -> bytes:
        if feed_url.startswith("https://bad"):
            raise httpx.ConnectError("feed is down")
        return sample_feed

    monkeypatch.setattr("app.reddit_source.fetch_feed", fake_fetch_feed)

    posts = fetch_all(sources, default_fetch_limit=25)

    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert all(post.source_topic_key == "ai" for post in posts)


def test_fetch_all_reads_every_source_of_a_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _sources(
        feeds={"ai": ["https://a.example/feed.rss", "https://b.example/feed.rss"]}
    )
    seen: list[str] = []

    def fake_fetch_feed(feed_url: str) -> bytes:
        seen.append(feed_url)
        return RSS20_FEED.replace("t3_rss20item", f"t3_{feed_url[8]}").encode("utf-8")

    monkeypatch.setattr("app.reddit_source.fetch_feed", fake_fetch_feed)

    posts = fetch_all(sources, default_fetch_limit=25)

    assert seen == ["https://a.example/feed.rss", "https://b.example/feed.rss"]
    assert [post.reddit_id for post in posts] == ["t3_a", "t3_b"]
    assert {post.source_topic_key for post in posts} == {"ai"}


def test_fetch_all_returns_nothing_when_no_source_is_configured() -> None:
    assert fetch_all([], default_fetch_limit=25) == []
