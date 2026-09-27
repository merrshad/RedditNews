"""FR-1 tests: RSS parsing is pure, so no network is involved."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.reddit_source import (
    FeedError,
    fetch_posts,
    parse_entry,
    parse_feed,
    strip_html,
    subreddit_from_feed_url,
)
from app.settings import TopicConfig

FEED_URL = "https://www.reddit.com/r/MachineLearning/new/.rss"

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


def test_parse_feed_maps_every_required_field() -> None:
    posts = parse_feed(SAMPLE_FEED, source_topic_key="ai", feed_url=FEED_URL)

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


def test_parse_feed_falls_back_to_summary_content() -> None:
    posts = parse_feed(SAMPLE_FEED, source_topic_key="ai", feed_url=FEED_URL)

    second = posts[1]
    assert second.raw_content == "summary text"
    assert second.published_at is None


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


def test_fetch_posts_dedupes_and_survives_a_failing_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [
        TopicConfig(key="ai", name="هوش مصنوعی", feeds=["https://good.example/feed.rss"]),
        TopicConfig(key="startup", name="استارتاپ", feeds=["https://bad.example/feed.rss"]),
    ]

    def fake_fetch_feed(feed_url: str, *, max_retries: int) -> bytes:
        if feed_url.startswith("https://bad"):
            raise RuntimeError("boom")
        return SAMPLE_FEED.encode("utf-8")

    monkeypatch.setattr("app.reddit_source.fetch_feed", fake_fetch_feed)

    posts = fetch_posts(topics, max_retries=1)

    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert all(post.source_topic_key == "ai" for post in posts)


def test_fetch_posts_keeps_only_the_first_occurrence_of_a_reddit_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [
        TopicConfig(key="ai", name="هوش مصنوعی", feeds=["https://a.example/feed.rss"]),
        TopicConfig(key="startup", name="استارتاپ", feeds=["https://b.example/feed.rss"]),
    ]

    monkeypatch.setattr(
        "app.reddit_source.fetch_feed",
        lambda feed_url, *, max_retries: SAMPLE_FEED.encode("utf-8"),
    )

    posts = fetch_posts(topics, max_retries=1)

    assert [post.reddit_id for post in posts] == ["t3_1abcde", "t3_2fghij"]
    assert {post.source_topic_key for post in posts} == {"ai"}
