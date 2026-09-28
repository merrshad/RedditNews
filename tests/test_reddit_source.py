"""FR-1 tests: topics config, RSS parsing (pure) and fetch resilience (NFR-2)."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.reddit_source import (
    DEFAULT_TOPICS_PATH,
    FeedError,
    TopicConfig,
    TopicsConfig,
    fetch_all,
    fetch_feed,
    load_topics_config,
    parse_entry,
    parse_feed,
    strip_html,
    subreddit_from_feed_url,
)

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


def _topics_config(*, feeds: dict[str, list[str]]) -> TopicsConfig:
    return TopicsConfig(
        topics=[TopicConfig(key=key, name=key, feeds=urls) for key, urls in feeds.items()]
    )


# --- load_topics_config -----------------------------------------------------------


def test_load_topics_config_reads_topics_and_feeds(tmp_path) -> None:
    path = tmp_path / "topics.yaml"
    path.write_text(
        """
topics:
  - key: ai
    name: "هوش مصنوعی"
    feeds:
      - "https://example.com/ai.rss"
      - "https://example.com/ml.rss"
  - key: startup
    name: "استارتاپ"
    feeds:
      - "https://example.com/startups.rss"
""",
        encoding="utf-8",
    )

    config = load_topics_config(path)

    assert [topic.key for topic in config.topics] == ["ai", "startup"]
    assert config.topics[0].name == "هوش مصنوعی"
    assert config.topics[0].feeds == ["https://example.com/ai.rss", "https://example.com/ml.rss"]


def test_load_topics_config_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "topics.yaml"
    path.write_text(
        """
topics:
  - key: ai
    name: "هوش مصنوعی"
    feeds: ["https://example.com/a.rss"]
  - key: ai
    name: "dup"
    feeds: ["https://example.com/b.rss"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate topic keys"):
        load_topics_config(path)


def test_load_topics_config_rejects_an_empty_topic_list(tmp_path) -> None:
    path = tmp_path / "topics.yaml"
    path.write_text("topics: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one topic"):
        load_topics_config(path)


def test_load_topics_config_reads_the_shipped_default_config() -> None:
    """The YAML we ship must parse and give every topic at least one feed (section 11)."""
    config = load_topics_config()

    assert DEFAULT_TOPICS_PATH.is_file()
    assert config.topics
    assert all(topic.feeds for topic in config.topics)


# --- parsing (pure, FR-1) ---------------------------------------------------------


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
    assert first.published_at == datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)


def test_parse_feed_falls_back_to_summary_content(sample_feed: bytes) -> None:
    posts = parse_feed(sample_feed, source_topic_key="ai", feed_url=FEED_URL)

    second = posts[1]
    assert second.raw_content == "summary text"
    assert second.published_at is None


def test_parse_feed_reads_rss2_guid_and_description() -> None:
    posts = parse_feed(RSS20_FEED, source_topic_key="startup", feed_url=FEED_URL)

    assert len(posts) == 1
    post = posts[0]
    assert post.reddit_id == "t3_rss20item"
    assert post.title == "How we launched"
    assert post.raw_content == "body & notes"
    assert post.source_topic_key == "startup"
    assert post.published_at == datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)


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


def _explode_first(
    calls: list[str], *, failures: int = 1, payload: bytes = b""
) -> Callable[..., Any]:
    """An ``httpx.get`` stand-in that raises for the first ``failures`` calls."""

    class FakeResponse:
        content = payload

        def raise_for_status(self) -> None:
            return None

    def fake_get(
        url: str, *, headers: dict[str, str], timeout: float, follow_redirects: bool
    ) -> Any:
        calls.append(headers["User-Agent"])
        if len(calls) <= failures:
            raise httpx.ConnectError("connection reset by peer")
        return FakeResponse()

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


# --- fetch_all (FR-1) -------------------------------------------------------------


def test_fetch_all_maps_every_feed_to_its_topic(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    config = _topics_config(
        feeds={"ai": ["https://a.example/feed.rss"], "startup": ["https://b.example/feed.rss"]}
    )
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: sample_feed)

    posts = fetch_all(config)

    # Two feeds with the same document: the second one is dropped as a duplicate.
    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert {post.source_topic_key for post in posts} == {"ai"}


def test_fetch_all_keeps_going_when_a_feed_fails(
    monkeypatch: pytest.MonkeyPatch, sample_feed: bytes
) -> None:
    config = _topics_config(
        feeds={"ai": ["https://good.example/feed.rss"], "startup": ["https://bad.example/feed.rss"]}
    )

    def fake_fetch_feed(feed_url: str) -> bytes:
        if feed_url.startswith("https://bad"):
            raise httpx.ConnectError("feed is down")
        return sample_feed

    monkeypatch.setattr("app.reddit_source.fetch_feed", fake_fetch_feed)

    posts = fetch_all(config)

    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert all(post.source_topic_key == "ai" for post in posts)


def test_fetch_all_reads_every_feed_of_a_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _topics_config(
        feeds={"ai": ["https://a.example/feed.rss", "https://b.example/feed.rss"]}
    )
    seen: list[str] = []

    def fake_fetch_feed(feed_url: str) -> bytes:
        seen.append(feed_url)
        return RSS20_FEED.replace("t3_rss20item", f"t3_{feed_url[8]}").encode("utf-8")

    monkeypatch.setattr("app.reddit_source.fetch_feed", fake_fetch_feed)

    posts = fetch_all(config)

    assert seen == ["https://a.example/feed.rss", "https://b.example/feed.rss"]
    assert [post.reddit_id for post in posts] == ["t3_a", "t3_b"]
    assert {post.source_topic_key for post in posts} == {"ai"}


def test_fetch_all_returns_nothing_for_a_config_without_feeds() -> None:
    assert fetch_all(TopicsConfig(topics=[TopicConfig(key="ai", name="هوش مصنوعی")])) == []
