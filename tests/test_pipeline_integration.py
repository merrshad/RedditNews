"""End-to-end cycles with only the LLM and Telegram faked.

`pipeline` is driven through real consecutive cycles against a real Postgres (the
dedicated test database, see ``conftest``) and against a *real capture* of a Reddit Atom
feed (``tests/fixtures/reddit_learnmachinelearning.rss``, 55 KB of live bytes from
``r/learnmachinelearning``). The RSS parser, the taxonomy queries, the prompt builder and
the entire `repository`/`review` layer therefore run for real; only the two outbound HTTP
boundaries that a test cannot own are replaced:

* the LLM transport — there is no API key in a test environment, and no test may spend
  someone's quota;
* Telegram — there is no bot token either, and a test must never broadcast.

Every test names the invariant it proves. Reading the fixture through the real parser is
deliberate: hand-written samples had already drifted from the real document (Reddit omits
the ``r/<name>`` tag, and the whole feed arrives on one line), so a real capture is what
keeps this suite honest about FR-1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import psycopg
import pytest

from app import pipeline, review
from app.models import RawPost
from app.reddit_source import fetch_all as real_fetch_all
from app.settings import Settings
from tests.conftest import (
    TEST_REDDIT_ID_PREFIX,
    FakeChatCompletion,
    FakeTelegram,
    insert_source,
    insert_topic,
    patch_telegram,
    temp_topic_key,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "reddit_learnmachinelearning.rss"

# The captured document is a single long line, so entries are matched across newlines.
_ENTRY_RE = re.compile(rb"<entry>.*?</entry>", re.DOTALL)

TOPIC_KEY = temp_topic_key("itest")
REVIEW_CHANNEL = "-100999"
PUBLIC_CHANNEL = "-100123"
ADMIN_ID = "777"


def _answer(**overrides: Any) -> str:
    """A schema-valid analysis answer for the canned LLM transport."""
    payload: dict[str, Any] = {
        "is_relevant": True,
        "duplicate_of_candidate_index": None,
        "topic": TOPIC_KEY,
        "importance": "high",
        "summary_fa": "خلاصه فارسی تولیدشده در تست یکپارچه.",
        "key_points": ["نکته یکپارچه اول", "نکته یکپارچه دوم"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


@pytest.fixture
def taxonomy(db_connection: psycopg.Connection) -> psycopg.Connection:
    """A topic plus one source whose URL is the one the fake transport answers for."""
    insert_topic(db_connection, "itest", "آزمون یکپارچه")
    insert_source(db_connection, "itest", "r/integration")
    return db_connection


def _feed_document(*, entries: int | None = None) -> bytes:
    """The captured feed, with every entry id moved under the test prefix.

    Rewriting ``<id>t3_xyz</id>`` into ``<id>t3_test_xyz</id>`` keeps the real bytes —
    titles, HTML bodies, timestamps, the missing subreddit tag — while guaranteeing that
    everything the tests write can be removed again by the ``t3_test_%`` cleanup.
    """
    document = FIXTURE_PATH.read_bytes()
    document = re.sub(rb"<id>t3_([a-z0-9]+)</id>", b"<id>t3_test_\\1</id>", document)
    if entries is None:
        return document
    head = document.split(b"<entry>", 1)[0]
    return head + b"".join(_ENTRY_RE.findall(document)[:entries]) + b"</feed>"


def _patch_feed(
    monkeypatch: pytest.MonkeyPatch, *, entries: int | None = None
) -> list[RawPost]:
    """Serve a captured Reddit document instead of the network (the only RSS fake).

    Returns what the *real* parser makes of it, so a test can assert against the same
    posts the pipeline will see.
    """
    document = _feed_document(entries=entries)
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: document)
    return real_fetch_all(
        _active_sources(), default_fetch_limit=25
    )


def _active_sources():
    """The sources the real pipeline would fetch, through the real repository."""
    from app import repository

    return repository.list_sources(active_only=True)


def _patch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    telegram: FakeTelegram | None = None,
    llm: FakeChatCompletion | None = None,
) -> tuple[FakeTelegram, FakeChatCompletion]:
    """Wire the real pipeline to the real database, faking only LLM + Telegram."""
    llm = llm or FakeChatCompletion(_answer())
    telegram = patch_telegram(
        monkeypatch, telegram or FakeTelegram(), default_chat_id=settings.telegram_chat_id
    )
    monkeypatch.setattr("app.analyzer.chat_completion", llm)
    monkeypatch.setattr("app.pipeline.get_settings", lambda: settings)
    return telegram, llm


def _rows(connection: psycopg.Connection) -> list[dict[str, Any]]:
    """Every row these tests wrote, with the columns the invariants talk about."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, reddit_id, status, review_status, published_at, is_relevant, "
            "importance, duplicate_of_id "
            "FROM posts WHERE reddit_id LIKE %s ORDER BY id",
            (f"{TEST_REDDIT_ID_PREFIX}%",),
        )
        return cursor.fetchall()


def _approve_everything(connection: psycopg.Connection) -> None:
    """Approve every post that is really waiting for a decision."""
    for row in _rows(connection):
        if row["status"] != "awaiting_review":
            continue
        callback = {
            "id": f"cb-{row['id']}",
            "data": f"approve:{row['id']}",
            "from": {"id": int(ADMIN_ID)},
            "message": {"message_id": 1, "chat": {"id": int(REVIEW_CHANNEL)}},
        }
        assert review.handle_callback(callback) is True


# --- Invariant 1 + FR-12/FR-13: the whole path, twice ------------------------------


def test_two_cycles_store_review_approve_publish_and_never_repeat_themselves(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 1, 12 and FR-9..FR-13 over real Reddit bytes and a real `posts` table."""
    fetched = _patch_feed(monkeypatch, entries=5)
    telegram, llm = _patch_boundaries(monkeypatch, settings=settings)

    pipeline.run_once()
    first_run = _rows(taxonomy)

    assert len(fetched) == 5
    assert [row["status"] for row in first_run] == ["awaiting_review"] * 5
    assert llm.calls == []  # nothing reached the LLM before a human said yes
    assert telegram.messages_to(PUBLIC_CHANNEL) == []
    review_messages = telegram.messages_to(REVIEW_CHANNEL)
    assert len(review_messages) == 5  # one independent message per post, never a digest
    # The message carries the real title from the feed plus the Persian topic name.
    assert any("<b>" in message and "learnmachinelearning" in message for message in review_messages)
    assert all(
        markup is not None and len(markup["inline_keyboard"][0]) == 2
        for markup in telegram.buttons_to(REVIEW_CHANNEL)
    )

    # The second cycle sees the identical feed document again — a normal situation, since
    # a polling worker keeps reading the same feed. Nothing may change.
    pipeline.run_once()

    assert len(llm.calls) == 0
    assert len(telegram.messages_to(REVIEW_CHANNEL)) == 5
    assert _rows(taxonomy) == first_run

    # An admin approves all five; the AI step runs on the next `main` tick.
    _approve_everything(taxonomy)
    pipeline.process_approved_posts()

    assert len(llm.calls) == 5  # exactly one call per approved post
    published = telegram.messages_to(PUBLIC_CHANNEL)
    assert len(published) == 5
    sent_rows = _rows(taxonomy)
    assert [row["status"] for row in sent_rows] == ["sent"] * 5
    assert all(row["published_at"] is not None for row in sent_rows)
    assert all(row["review_status"] == "approved" for row in sent_rows)
    # The public message is the Persian digest the reader expects.
    assert any("اهمیت: بالا" in message and "🔑 نکات کلیدی:" in message for message in published)

    # A third cycle changes nothing at all: analysed once, published once.
    pipeline.run_once()

    assert len(llm.calls) == 5
    assert len(telegram.messages_to(PUBLIC_CHANNEL)) == 5
    assert len({row["id"] for row in _rows(taxonomy)}) == 5


# --- Invariant 10: the candidate list handed to the LLM stays bounded --------------


def test_the_llm_never_receives_more_candidates_than_the_lookback_limit(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 10, asserted on the prompt the LLM really receives.

    The database deliberately holds more analysed posts than the limit, so the assertion
    can only pass if the `LIMIT` in `fetch_recent_candidates` is doing the work.
    """
    limit = 3
    for index in range(limit + 2):
        taxonomy.execute(
            """
            INSERT INTO posts (reddit_id, subreddit, source_topic_key, title, url, summary_fa,
                               key_points, is_relevant, topic, importance, status, review_status)
            VALUES (%s, 'mlops', %s, %s, %s, %s, '["نکته"]'::jsonb, TRUE, %s, 'high',
                    'sent', 'approved')
            """,
            (
                f"{TEST_REDDIT_ID_PREFIX}candidate_{index}",
                TOPIC_KEY,
                f"Earlier post {index}",
                f"https://example.com/{index}",
                f"خلاصه پست قبلی {index}",
                TOPIC_KEY,
            ),
        )

    _patch_feed(monkeypatch, entries=1)
    bounded = settings.model_copy(update={"similarity_lookback_limit": limit})
    telegram, llm = _patch_boundaries(monkeypatch, settings=bounded)

    pipeline.run_once()
    _approve_everything(taxonomy)
    pipeline.process_approved_posts()

    assert len(llm.calls) == 1
    _, user_prompt = llm.calls[0]
    assert user_prompt.count('"index"') == limit  # exactly the bound, never more
    assert '"candidates"' in user_prompt

    # The bound only trims context: the post itself was still published.
    assert len(telegram.messages_to(PUBLIC_CHANNEL)) == 1


# --- Invariant 7 / FR-11: a rejected publication is finished later ------------------


def test_a_rejected_publication_is_recovered_by_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    taxonomy: psycopg.Connection,
) -> None:
    """Invariant 7 + FR-11: the row is durable, so the next cycle finishes the job.

    The failing send stands in for a crash between "analysed" and "published": the row is
    already durable, so the next run must deliver it *without* asking the LLM again.
    """
    _patch_feed(monkeypatch, entries=3)
    telegram = FakeTelegram()
    _, llm = _patch_boundaries(monkeypatch, settings=settings, telegram=telegram)

    pipeline.run_once()  # the three review messages do get delivered
    _approve_everything(taxonomy)
    assert len(telegram.messages_to(REVIEW_CHANNEL)) == 3

    telegram.fail_next_sends = 3  # ... and now Telegram refuses the publications
    pipeline.process_approved_posts()

    assert [row["status"] for row in _rows(taxonomy)] == ["to_send"] * 3
    assert len(llm.calls) == 3
    assert telegram.messages_to(PUBLIC_CHANNEL) == []

    # The next cycle brings no new RSS item at all; only the leftover rows are retried.
    monkeypatch.setattr(
        "app.reddit_source.fetch_feed",
        lambda feed_url: b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>',
    )
    recovery = FakeTelegram()
    _patch_boundaries(monkeypatch, settings=settings, telegram=recovery, llm=llm)

    pipeline.run_once()

    recovered = _rows(taxonomy)
    assert [row["status"] for row in recovered] == ["sent"] * 3
    assert all(row["published_at"] is not None for row in recovered)
    assert len(recovery.messages_to(PUBLIC_CHANNEL)) == 3
    assert len(llm.calls) == 3  # never re-analysed
