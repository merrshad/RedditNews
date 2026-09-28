"""Phase 4: a whole cycle end to end, with only the LLM and Telegram faked.

`pipeline.run_once()` is driven twice in a row against a real Postgres (the dedicated
test database, see ``conftest``) and against a *real capture* of a Reddit Atom feed
(``tests/fixtures/reddit_learnmachinelearning.rss``, 55 KB of live bytes from
``r/learnmachinelearning``). The RSS parser, the prompt builder and the entire
`repository` layer therefore run for real; only the two outbound HTTP boundaries that a
test cannot own are replaced:

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
import pytest

from app import pipeline, repository
from app.models import PostRecord, RawPost
from app.reddit_source import fetch_all as real_fetch_all
from app.settings import Settings
from tests.conftest import (
    TEST_REDDIT_ID_PREFIX,
    FakeChatCompletion,
    FakeSender,
    FailingSender,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "reddit_learnmachinelearning.rss"
)

# The captured document is a single long line, so entries are matched across newlines.
_ENTRY_RE = re.compile(rb"<entry>.*?</entry>", re.DOTALL)

# A feed document with no entries, used for "this cycle brings nothing new".
EMPTY_FEED = b'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _answer(**overrides: Any) -> str:
    """A schema-valid analysis answer for the canned LLM transport."""
    payload: dict[str, Any] = {
        "is_relevant": True,
        "duplicate_of_candidate_index": None,
        "topic": "ai",
        "importance": "high",
        "summary_fa": "خلاصه فارسی تولیدشده در تست یکپارچه.",
        "key_points": ["نکته یکپارچه اول", "نکته یکپارچه دوم"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def _feed_document(*, entries: int | None = None) -> bytes:
    """The captured feed, with every entry id moved under the test prefix.

    Rewriting ``<id>t3_xyz</id>`` into ``<id>t3_test_xyz</id>`` keeps the real bytes —
    titles, HTML bodies, timestamps, the missing subreddit tag — while guaranteeing that
    everything the tests write can be removed again by the ``t3_test_%`` cleanup. The
    document is otherwise untouched, so the parser is exercised exactly as Reddit sends it.
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
    return real_fetch_all(pipeline.load_topics_config())


def _patch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    sender: FakeSender | None = None,
    answer: str | None = None,
) -> tuple[FakeSender, FakeChatCompletion]:
    """Wire the real pipeline to the real database, faking only LLM + Telegram."""
    llm = FakeChatCompletion(answer if answer is not None else _answer())
    monkeypatch.setattr("app.analyzer.chat_completion", llm)
    sender = sender if sender is not None else FakeSender()
    monkeypatch.setattr("app.pipeline.send_message", sender)
    monkeypatch.setattr("app.pipeline.get_settings", lambda: settings)
    return sender, llm


def _rows(connection: psycopg.Connection) -> list[dict[str, Any]]:
    """Every row these tests wrote, with the columns the invariants talk about."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT reddit_id, status, sent_at, is_relevant, importance, duplicate_of_id "
            "FROM posts WHERE reddit_id LIKE %s ORDER BY id",
            (f"{TEST_REDDIT_ID_PREFIX}%",),
        )
        return cursor.fetchall()


# --- Invariant 1: analysed once, sent once ----------------------------------------


def test_two_consecutive_runs_analyse_and_send_every_post_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 1 + FR-9/FR-10, over real Reddit bytes and a real `posts` table.

    The second run sees the identical feed document again — a normal situation, since a
    polling worker keeps reading the same feed. Nothing may be re-analysed or re-sent.
    """
    fetched = _patch_feed(monkeypatch, entries=5)
    sender, llm = _patch_boundaries(monkeypatch, settings=settings)

    pipeline.run_once()
    first_run_rows = _rows(db_connection)

    assert len(fetched) == 5
    assert len(llm.calls) == 5  # one LLM call per fetched post
    assert len(sender.messages) == 5  # every post cleared the threshold and was sent
    assert [row["status"] for row in first_run_rows] == ["sent"] * 5
    assert all(row["sent_at"] is not None for row in first_run_rows)
    # The message carries the real title from the feed plus the Persian metadata row.
    assert any("<b>" in message and "r/learnmachinelearning" in message for message in sender.messages)
    assert any("اهمیت: بالا" in message for message in sender.messages)

    pipeline.run_once()

    assert len(llm.calls) == 5  # `repository.exists()` short-circuits before the LLM
    assert len(sender.messages) == 5  # and Telegram is not called a second time
    assert _rows(db_connection) == first_run_rows  # nothing new, nothing changed

    # Both runs together must still have produced exactly one row per feed entry.
    assert len({row["reddit_id"] for row in first_run_rows}) == 5


# --- Invariant 7 / FR-11: a send that fails is retried, never re-analysed ----------


def test_a_rejected_send_is_recovered_by_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 7 + FR-11: store first, keep `to_send` on failure, finish it later.

    The failing send stands in for a crash between "stored" and "sent": the row is
    already durable, so the next run must deliver it *without* asking the LLM again.
    """
    _patch_feed(monkeypatch, entries=3)
    rejected, llm = _patch_boundaries(
        monkeypatch, settings=settings, sender=FailingSender()
    )

    pipeline.run_once()

    rows = _rows(db_connection)
    assert [row["status"] for row in rows] == ["to_send"] * 3
    assert all(row["sent_at"] is None for row in rows)
    assert len(llm.calls) == 3
    assert rejected.messages == []  # nothing was actually delivered

    # The next cycle brings no new RSS item at all; only the leftover rows are retried.
    monkeypatch.setattr("app.reddit_source.fetch_feed", lambda feed_url: EMPTY_FEED)
    working = FakeSender()
    monkeypatch.setattr("app.pipeline.send_message", working)

    pipeline.run_once()

    recovered = _rows(db_connection)
    assert [row["status"] for row in recovered] == ["sent"] * 3
    assert all(row["sent_at"] is not None for row in recovered)
    assert len(working.messages) == 3  # delivered by the FR-11 retry path
    assert len(llm.calls) == 3  # never re-analysed


# --- Invariant 10: the candidate list handed to the LLM stays bounded --------------


def test_the_llm_never_receives_more_candidates_than_the_lookback_limit(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    db_connection: psycopg.Connection,
) -> None:
    """Invariant 10, asserted on the prompt the LLM really receives.

    The database deliberately holds more analysed posts than the limit, so the assertion
    can only pass if the `LIMIT` in `fetch_recent_candidates` is doing the work.
    """
    limit = 3
    now = datetime.now(timezone.utc)
    for index in range(limit + 2):
        repository.save(
            PostRecord(
                reddit_id=f"{TEST_REDDIT_ID_PREFIX}integration_candidate_{index}",
                subreddit="learnmachinelearning",
                source_topic_key="ai",
                title=f"Earlier post {index}",
                url=f"https://example.com/{index}",
                published_at=now - timedelta(minutes=index),
                summary_fa=f"خلاصه پست قبلی {index}",
                key_points=["نکته"],
                status="sent",
            )
        )

    # More analysed posts exist than the bound, so the bound genuinely has to cut.
    assert len(repository.fetch_recent_candidates(50, 72)) == limit + 2

    _patch_feed(monkeypatch, entries=1)
    bounded = settings.model_copy(update={"similarity_lookback_limit": limit})
    _, llm = _patch_boundaries(monkeypatch, settings=bounded)

    pipeline.run_once()

    assert len(llm.calls) == 1
    _, user_prompt = llm.calls[0]
    assert user_prompt.count('"index"') == limit  # exactly the bound, never more
    assert '"candidates"' in user_prompt

    # The post itself was still stored and sent normally (the bound only trims context).
    assert [row["status"] for row in _rows(db_connection)].count("sent") == limit + 2 + 1
