"""FR-14: the admin command surface, against the real test database.

The "admin panel" is the bot itself, so these tests drive the same entry point Telegram
does (``telegram_updates.route`` → ``admin.handle_command``) with fake messages, a fake
Telegram boundary (NFR-8) and the real ``repository``. Each test asserts both halves of a
command: the Persian reply the admin reads *and* the row that actually changed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from app import repository, telegram_updates
from app.settings import Settings
from tests.conftest import (
    FakeTelegram,
    patch_telegram,
    temp_source_url,
    temp_topic_key,
)

ADMIN_ID = 777
OTHER_ID = 999

# A post id that cannot exist in the test database, for the "already reviewed" path.
MISSING_POST_ID = 999_999


@pytest.fixture
def telegram(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FakeTelegram:
    """A fake Telegram boundary; ``admin.py`` reads its own (env-based) settings."""
    return patch_telegram(
        monkeypatch, FakeTelegram(), default_chat_id=settings.telegram_chat_id
    )


@pytest.fixture
def send(telegram: FakeTelegram) -> Callable[..., str]:
    """Run one command through the update router and return the single reply text."""

    def _send(text: str, *, user_id: int = ADMIN_ID) -> str:
        before = len(telegram.sent)
        handled = telegram_updates.route(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "chat": {"id": user_id},
                    "from": {"id": user_id, "username": "admin"},
                    "text": text,
                },
            }
        )
        replies = telegram.messages_to(str(user_id))
        if not handled:
            assert len(telegram.sent) == before, "an ignored message must not be answered"
            return ""
        assert len(telegram.sent) == before + 1, "a command must answer exactly once"
        return replies[-1]

    return _send


def _stored_source(url: str) -> Any:
    return next(source for source in repository.list_sources() if source.rss_url == url)


# --- parsing, routing and authorisation -------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/help", ("/help", [])),
        ("/help@RedditDigestBot", ("/help", [])),
        # Arguments are whitespace-split; the handlers rejoin multi-word names.
        ("  /addtopic ai هوش مصنوعی  ", ("/addtopic", ["ai", "هوش", "مصنوعی"])),
        ("/HELP", ("/help", [])),
        ("not a command", None),
        ("", None),
    ],
)
def test_parse_command(text: str, expected: tuple[str, list[str]] | None) -> None:
    """Pure, so the grammar can be pinned without any transport."""
    from app.admin import parse_command

    assert parse_command(text) == expected


def test_a_command_from_a_non_admin_is_ignored(telegram: FakeTelegram, send) -> None:
    """FR-12: the allow-list *is* the authorisation, and it is not confirmed to strangers."""
    assert send("/topics", user_id=OTHER_ID) == ""
    assert telegram.answers == []


def test_a_plain_message_is_not_a_command(telegram: FakeTelegram) -> None:
    handled = telegram_updates.route(
        {
            "update_id": 1,
            "message": {"message_id": 1, "chat": {"id": ADMIN_ID}, "text": "سلام"},
        }
    )

    assert handled is False
    assert telegram.sent == []


def test_a_callback_query_goes_to_the_review_flow_not_the_command_surface(
    telegram: FakeTelegram,
) -> None:
    handled = telegram_updates.route(
        {
            "update_id": 1,
            "callback_query": {
                "id": "cb-1",
                "data": f"approve:{MISSING_POST_ID}",
                "from": {"id": ADMIN_ID},
            },
        }
    )

    assert handled is False  # there is no such post, so the decision was refused
    assert telegram.answers == [("cb-1", "این پست قبلاً بررسی شده است.")]


# --- topics ----------------------------------------------------------------------


def test_help_lists_the_topic_and_source_commands(send) -> None:
    reply = send("/help")

    assert "/addtopic" in reply and "/deltopic" in reply
    assert "/addsource" in reply and "/setsourcelimit" in reply
    assert "RSS_FETCH_LIMIT" in reply


def test_start_is_an_alias_for_help(send) -> None:
    assert send("/start") == send("/help")


def test_addtopic_creates_a_topic_the_pipeline_will_use(send) -> None:
    key = temp_topic_key("admin")

    reply = send(f"/addtopic {key} هوش مصنوعی")

    assert key in reply and "هوش مصنوعی" in reply
    assert key in {topic.key for topic in repository.list_topics(active_only=True)}


def test_addtopic_validates_the_key_and_requires_a_name(send) -> None:
    assert "نامعتبر" in send("/addtopic AI موضوع")
    assert "استفاده" in send("/addtopic onlykey")
    assert "استفاده" in send("/addtopic")


def test_addtopic_refuses_a_duplicate_key(send) -> None:
    key = temp_topic_key("dupe")
    assert key in send(f"/addtopic {key} اول")

    assert "وجود دارد" in send(f"/addtopic {key} دوم")


def test_topics_lists_state_and_source_count(send) -> None:
    key = temp_topic_key("listed")
    send(f"/addtopic {key} موضوع فهرست")
    send(f"/addsource {key} {temp_source_url('listed.rss')}")

    reply = send("/topics")

    assert f"<code>{key}</code>" in reply
    assert "1 منبع" in reply and "فعال" in reply


def test_toggletopic_disables_and_reenables(send) -> None:
    key = temp_topic_key("toggle")
    send(f"/addtopic {key} موضوع")

    assert "غیرفعال شد" in send(f"/toggletopic {key}")
    assert key not in {topic.key for topic in repository.list_topics(active_only=True)}

    assert "فعال شد" in send(f"/toggletopic {key}")
    assert key in {topic.key for topic in repository.list_topics(active_only=True)}


def test_renametopic_and_unknown_keys(send) -> None:
    key = temp_topic_key("rename")
    send(f"/addtopic {key} نام اول")

    assert "نام تازه" in send(f"/renametopic {key} نام تازه")
    assert {topic.key: topic.name for topic in repository.list_topics()}[key] == "نام تازه"

    assert "پیدا نشد" in send(f"/toggletopic {temp_topic_key('absent')}")
    assert "پیدا نشد" in send(f"/renametopic {temp_topic_key('absent')} نام")


def test_deltopic_removes_the_topic_and_its_sources_but_keeps_the_audit_note(send) -> None:
    key = temp_topic_key("delete")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("gone.rss")
    send(f"/addsource {key} {url}")

    reply = send(f"/deltopic {key}")

    assert "حذف شدند" in reply and "سابقه" in reply
    assert key not in {topic.key for topic in repository.list_topics()}
    assert url not in {source.rss_url for source in repository.list_sources()}


# --- sources ---------------------------------------------------------------------


def test_addsource_sets_a_per_source_fetch_limit(send) -> None:
    """FR-1/FR-14: the cap lives on the source, so each feed gets its own batch size."""
    key = temp_topic_key("limit")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("limit.rss")

    assert "سقف: 50" in send(f"/addsource {key} {url} 50")

    stored = _stored_source(url)
    assert stored.fetch_limit == 50 and stored.topic_key == key


def test_addsource_without_a_limit_falls_back_to_the_global_default(send) -> None:
    key = temp_topic_key("default")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("default.rss")

    assert "سقف: پیش‌فرض" in send(f"/addsource {key} {url} default")

    assert _stored_source(url).fetch_limit is None


def test_addsource_validates_topic_url_and_limit(send) -> None:
    key = temp_topic_key("validate")
    send(f"/addtopic {key} موضوع")

    assert "پیدا نشد" in send(f"/addsource {temp_topic_key('absent')} {temp_source_url('x.rss')}")
    assert "نامعتبر" in send(f"/addsource {key} not-a-url")
    assert "عدد مثبت" in send(f"/addsource {key} {temp_source_url('y.rss')} -3")
    assert "استفاده" in send(f"/addsource {key}")
    # Nothing above was actually stored.
    assert temp_source_url("y.rss") not in {s.rss_url for s in repository.list_sources()}


def test_addsource_refuses_a_duplicate_feed(send) -> None:
    key = temp_topic_key("dupfeed")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("dup.rss")
    send(f"/addsource {key} {url}")

    assert "از قبل ثبت شده" in send(f"/addsource {key} {url}")
    assert len([source for source in repository.list_sources() if source.rss_url == url]) == 1


def test_sources_lists_only_the_requested_topic(send) -> None:
    first, second = temp_topic_key("s1"), temp_topic_key("s2")
    send(f"/addtopic {first} موضوع یک")
    send(f"/addtopic {second} موضوع دو")
    send(f"/addsource {first} {temp_source_url('one.rss')}")
    send(f"/addsource {second} {temp_source_url('two.rss')}")

    reply = send(f"/sources {first}")

    assert temp_source_url("one.rss") in reply
    assert temp_source_url("two.rss") not in reply


def test_setsourcelimit_switches_between_a_number_and_the_default(send) -> None:
    key = temp_topic_key("slimit")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("slimit.rss")
    send(f"/addsource {key} {url}")

    assert "روی 100 تنظیم شد" in send(f"/setsourcelimit {url} 100")
    assert _stored_source(url).fetch_limit == 100

    assert "پیش‌فرض" in send(f"/setsourcelimit {url} default")
    assert _stored_source(url).fetch_limit is None

    assert "عدد مثبت" in send(f"/setsourcelimit {url} lots")
    assert "منبع پیدا نشد" in send("/setsourcelimit https://test.example/ghost.rss 5")


def test_setsourcetopic_moves_a_feed_between_topics(send) -> None:
    first, second = temp_topic_key("mv1"), temp_topic_key("mv2")
    send(f"/addtopic {first} موضوع یک")
    send(f"/addtopic {second} موضوع دو")
    url = temp_source_url("move.rss")
    send(f"/addsource {first} {url}")

    assert "تغییر کرد" in send(f"/setsourcetopic {url} {second}")

    assert _stored_source(url).topic_key == second
    assert "پیدا نشد" in send(f"/setsourcetopic {url} {temp_topic_key('absent')}")


def test_togglesource_hides_a_feed_from_the_collector(send) -> None:
    key = temp_topic_key("togglesrc")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("togglesrc.rss")
    send(f"/addsource {key} {url}")

    assert "غیرفعال شد" in send(f"/togglesource {url}")
    assert url not in {source.rss_url for source in repository.list_sources(active_only=True)}

    assert "فعال شد" in send(f"/togglesource {url}")
    assert url in {source.rss_url for source in repository.list_sources(active_only=True)}


def test_delsource_removes_it_and_reports_a_second_attempt(send) -> None:
    key = temp_topic_key("delsrc")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("delsrc.rss")
    send(f"/addsource {key} {url}")

    assert "حذف شد" in send(f"/delsource {url}")
    assert url not in {source.rss_url for source in repository.list_sources()}
    assert "منبع پیدا نشد" in send(f"/delsource {url}")


def test_an_unknown_command_points_at_the_help(send) -> None:
    assert "دستور ناشناخته" in send("/nonsense")


def test_a_topic_disabled_in_the_admin_panel_stops_feeding_the_collector(
    send, db_connection: psycopg.Connection, settings: Settings
) -> None:
    """FR-1 + FR-14, end to end: toggling a topic really removes its feed from a run."""
    key = temp_topic_key("enforce")
    send(f"/addtopic {key} موضوع")
    url = temp_source_url("enforce.rss")
    send(f"/addsource {key} {url}")
    assert url in {source.rss_url for source in repository.list_sources(active_only=True)}

    send(f"/toggletopic {key}")

    assert url not in {source.rss_url for source in repository.list_sources(active_only=True)}
