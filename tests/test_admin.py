"""FR-14: the inline admin panel, against the real test database.

Phase 5 drove a typed command grammar; phase 6 replaced it with inline ("glass") keyboards,
so these tests press the real buttons through the same entry point Telegram uses
(``telegram_updates.route`` → ``admin.handle_callback`` / ``admin.handle_message``) with a
fake Telegram boundary (NFR-8) and the real ``repository``.

Two properties get their own tests because they are what the change was about:

* no command is ever typed — a message from an admin opens the panel, and the only free-text
  input is the *value* of a field the panel has just asked for;
* two half-finished actions cannot interfere: a button press always clears a pending prompt,
  and pending prompts are per chat.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from app import admin, repository, telegram_updates
from app.settings import Settings
from tests.conftest import (
    FakeTelegram,
    patch_telegram,
    run,
    temp_source_url,
    temp_topic_key,
)

ADMIN_ID = 777
OTHER_ID = 999

# A post id that cannot exist in the test database, for the review namespace check.
MISSING_POST_ID = 999_999

BUTTON_NEW_TOPIC = admin.BUTTON_NEW_TOPIC
BUTTON_TOPICS = admin.BUTTON_TOPICS
BUTTON_SOURCES = admin.BUTTON_SOURCES
BUTTON_CONFIRM = admin.BUTTON_CONFIRM
BUTTON_CANCEL = admin.BUTTON_CANCEL
BUTTON_HOME = admin.BUTTON_HOME


def _message(text: str, *, user_id: int = ADMIN_ID) -> dict[str, Any]:
    """A plain Telegram message from one user."""
    return {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "chat": {"id": user_id},
            "from": {"id": user_id, "first_name": "ادمین"},
            "text": text,
        },
    }


def _callback(data: str, *, user_id: int = ADMIN_ID) -> dict[str, Any]:
    """An inline button press: the payload plus the chat the panel message lives in."""
    return {
        "update_id": 2,
        "callback_query": {
            "id": "cb-1",
            "data": data,
            "from": {"id": user_id, "first_name": "ادمین"},
            "message": {"message_id": 1, "chat": {"id": user_id}},
        },
    }


@pytest.fixture(autouse=True)
def _no_pending_prompt_left_over() -> None:
    """``admin`` keeps one pending field per chat; no test may inherit another's."""
    admin.reset_pending()


@pytest.fixture
def telegram(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FakeTelegram:
    """A fake Telegram boundary; ``admin.py`` reads its own (env-based) settings."""
    return patch_telegram(
        monkeypatch, FakeTelegram(), default_chat_id=settings.telegram_chat_id
    )


class Panel:
    """The panel as Telegram drives it: one press or one message, one reply message."""

    def __init__(self, telegram: FakeTelegram, chat_id: int = ADMIN_ID) -> None:
        self.telegram = telegram
        self.chat_id = chat_id

    # --- driving -------------------------------------------------------------------

    def press(self, data: str, *, user_id: int | None = None) -> str:
        """Press one inline button and return the message the panel answers with."""
        handled, text = self.try_press(data, user_id=user_id)
        assert handled is True, f"the panel did not answer `{data}`"
        assert text is not None
        return text

    def try_press(
        self, data: str, *, user_id: int | None = None
    ) -> tuple[bool, str | None]:
        """Press one button without asserting; returns (handled, reply text)."""
        before = len(self.telegram.sent)
        handled = run(
            telegram_updates.route(
                _callback(f"{admin.PANEL_CALLBACK_PREFIX}{data}", user_id=user_id or self.chat_id)
            )
        )
        if len(self.telegram.sent) == before:
            return handled, None
        return handled, self.telegram.sent[-1][0]

    def send(self, text: str, *, user_id: int | None = None) -> str:
        """Send one plain message (the home screen, or the answer to a pending field)."""
        handled = run(telegram_updates.route(_message(text, user_id=user_id or self.chat_id)))
        assert handled is True, "a message from an admin must be answered"
        return self.telegram.sent[-1][0]

    def try_send(self, text: str, *, user_id: int | None = None) -> bool:
        """Send one plain message without asserting; returns whether it was answered."""
        return run(telegram_updates.route(_message(text, user_id=user_id or self.chat_id)))

    # --- reading the last screen ---------------------------------------------------

    @property
    def text(self) -> str:
        return self.telegram.sent[-1][0]

    @property
    def markup(self) -> dict[str, Any] | None:
        return self.telegram.sent[-1][2]

    def labels(self) -> list[str]:
        markup = self.markup
        if markup is None:
            return []
        return [button["text"] for row in markup["inline_keyboard"] for button in row]

    def callbacks(self) -> list[str]:
        markup = self.markup
        if markup is None:
            return []
        return [button["callback_data"] for row in markup["inline_keyboard"] for button in row]

    def press_label(self, label: str) -> str:
        """Press the button whose *label* matches, wherever it is on the screen."""
        labels = self.labels()
        assert label in labels, f"`{label}` is not on this screen: {labels}"
        index = labels.index(label)
        return self.press(self.callbacks()[index][len(admin.PANEL_CALLBACK_PREFIX) :])


@pytest.fixture
def panel(telegram: FakeTelegram) -> Panel:
    return Panel(telegram)


def _stored_source(url: str) -> Any:
    return next(source for source in repository.list_sources() if source.rss_url == url)


# --- authorisation and routing ----------------------------------------------------


def test_any_message_from_an_admin_opens_the_panel(panel: Panel) -> None:
    """FR-14: there is no command to remember — Telegram's message *is* the door."""
    assert admin.HOME_TEXT in panel.send("سلام")
    assert panel.labels() == [
        BUTTON_TOPICS,
        BUTTON_SOURCES,
        BUTTON_NEW_TOPIC,
    ]


def test_a_message_from_a_non_admin_is_ignored_silently(
    panel: Panel, telegram: FakeTelegram
) -> None:
    """FR-12: the allow-list *is* the authorisation, and it is not confirmed to strangers."""
    assert panel.try_send("سلام", user_id=OTHER_ID) is False
    assert telegram.sent == []


def test_a_panel_press_from_a_non_admin_is_refused(
    panel: Panel, telegram: FakeTelegram
) -> None:
    handled, text = panel.try_press("topics", user_id=OTHER_ID)

    assert handled is False and text is None
    assert telegram.answers[-1] == ("cb-1", admin.NOT_ADMIN_TEXT)


def test_a_review_callback_never_reaches_the_panel(telegram: FakeTelegram) -> None:
    """The ``panel:`` namespace is what separates the two callback surfaces."""
    handled = run(
        telegram_updates.route(_callback(f"approve:{MISSING_POST_ID}"))
    )

    assert handled is False  # there is no such post, so the decision was refused
    assert telegram.answers[-1] == ("cb-1", "این پست قبلاً بررسی شده است.")
    assert telegram.sent == []


def test_an_unknown_panel_button_falls_back_to_the_home_screen(panel: Panel) -> None:
    """A stale button from an older message must not leave the admin stuck."""
    reply = panel.press("totally-unknown")

    assert admin.UNKNOWN_PANEL_BUTTON in reply
    assert panel.labels() == [BUTTON_TOPICS, BUTTON_SOURCES, BUTTON_NEW_TOPIC]


# --- topics -----------------------------------------------------------------------


def test_pressing_topics_lists_state_and_source_count(panel: Panel) -> None:
    key = temp_topic_key("listed")
    panel.press("newtopic")
    panel.send(f"{key} موضوع فهرست")
    panel.press("addsource:" + key)
    panel.send(temp_source_url("listed.rss"))

    panel.press("topics")

    assert f"{key}</code> — موضوع فهرست (فعال، 1 منبع)" in panel.text
    # Each topic is one button (the seeded ones are listed too), and there is a way back.
    assert f"{admin.PANEL_CALLBACK_PREFIX}topic:{key}" in panel.callbacks()
    assert any("موضوع فهرست" in label for label in panel.labels())
    assert BUTTON_HOME in panel.labels()


def test_a_topic_is_created_by_answering_the_prompt(panel: Panel) -> None:
    """The only free-text entry is a field the panel asked for — never a command."""
    key = temp_topic_key("panel")

    prompt = panel.press("newtopic")
    assert "کلید" in prompt and BUTTON_CANCEL in panel.labels()

    reply = panel.send(f"{key} هوش مصنوعی")

    assert f"{key}" in reply and "هوش مصنوعی" in reply and admin.DONE_TEXT in reply
    assert key in {topic.key for topic in repository.list_topics(active_only=True)}
    # And the screen it lands on is the new topic's own detail page.
    assert admin.BUTTON_RENAME in panel.labels()


def test_a_missing_name_repeats_the_prompt(panel: Panel) -> None:
    panel.press("newtopic")

    reply = panel.send("فقط‌کلید")  # no name part at all

    assert admin.NEW_TOPIC_PROMPT in reply
    assert BUTTON_CANCEL in panel.labels()
    assert not [
        topic for topic in repository.list_topics() if topic.key.startswith("فقط")
    ]


def test_an_invalid_key_repeats_the_prompt(panel: Panel) -> None:
    """The key is the value the LLM must return as `topic`, so it is validated (FR-5)."""
    panel.press("newtopic")

    reply = panel.send("نامعتبر! نام")

    assert "نامعتبر" in reply
    assert admin.NEW_TOPIC_PROMPT in reply
    assert not [topic for topic in repository.list_topics() if topic.name == "نام"]


def test_a_duplicate_key_is_reported_and_the_field_stays_open(panel: Panel) -> None:
    key = temp_topic_key("dupe")
    panel.press("newtopic")
    panel.send(f"{key} اول")

    panel.press("newtopic")
    reply = panel.send(f"{key} دوم")

    assert "وجود دارد" in reply
    assert [topic.name for topic in repository.list_topics() if topic.key == key] == ["اول"]
    # The prompt is still waiting, so a corrected answer lands on the same screen.
    assert BUTTON_CANCEL in panel.labels()


def test_toggling_a_topic_from_its_screen_flips_the_database(panel: Panel) -> None:
    key = temp_topic_key("toggle")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")

    assert "غیرفعال شد" in panel.press_label(admin.BUTTON_TOGGLE)
    assert key not in {topic.key for topic in repository.list_topics(active_only=True)}

    assert "فعال شد" in panel.press_label(admin.BUTTON_TOGGLE)
    assert key in {topic.key for topic in repository.list_topics(active_only=True)}


def test_renaming_a_topic_uses_a_prompt_and_keeps_the_key(panel: Panel) -> None:
    key = temp_topic_key("rename")
    panel.press("newtopic")
    panel.send(f"{key} نام اول")

    panel.press_label(admin.BUTTON_RENAME)
    reply = panel.send("نام تازه")

    assert "نام تازه" in reply
    assert {topic.key: topic.name for topic in repository.list_topics()}[key] == "نام تازه"


def test_renaming_rejects_an_over_long_name(panel: Panel) -> None:
    key = temp_topic_key("longname")
    panel.press("newtopic")
    panel.send(f"{key} نام")

    panel.press_label(admin.BUTTON_RENAME)
    reply = panel.send("ط" * (admin.MAX_TOPIC_NAME_LENGTH + 1))

    assert "بلندتر" in reply
    assert {topic.key: topic.name for topic in repository.list_topics()}[key] == "نام"


def test_deleting_a_topic_needs_the_confirmation_button(panel: Panel) -> None:
    """A destructive action is a second press, and the first one only asks."""
    key = temp_topic_key("delete")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("gone.rss")
    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)

    panel.press("topic:" + key)
    warning = panel.press_label(admin.BUTTON_DELETE_TOPIC)

    assert "حذف شوند؟" in warning and BUTTON_CONFIRM in panel.labels()
    assert key in {topic.key for topic in repository.list_topics()}  # not deleted yet

    reply = panel.press_label(BUTTON_CONFIRM)

    assert "حذف شد" in reply and "1 منبعش" in reply
    assert key not in {topic.key for topic in repository.list_topics()}
    assert url not in {source.rss_url for source in repository.list_sources()}


def test_cancelling_a_delete_keeps_the_topic(panel: Panel) -> None:
    key = temp_topic_key("keep")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")

    panel.press("deltopic:" + key)
    assert BUTTON_CANCEL in panel.labels()
    panel.press_label(BUTTON_CANCEL)

    assert key in {topic.key for topic in repository.list_topics()}


# --- sources ----------------------------------------------------------------------


def test_a_source_is_added_to_the_topic_the_panel_came_from(panel: Panel) -> None:
    key = temp_topic_key("addsrc")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("added.rss")

    panel.press_label(admin.BUTTON_ADD_SOURCE)
    reply = panel.send(url)

    assert "اضافه شد" in reply
    stored = _stored_source(url)
    assert stored.topic_key == key and stored.fetch_limit is None
    assert admin.BUTTON_LIMIT in panel.labels()  # the detail screen of the new feed


def test_an_invalid_feed_url_repeats_the_prompt(panel: Panel) -> None:
    key = temp_topic_key("badurl")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")

    panel.press_label(admin.BUTTON_ADD_SOURCE)
    reply = panel.send("not-a-url")

    assert "نامعتبر" in reply
    assert not [s for s in repository.list_sources() if s.rss_url == "not-a-url"]
    assert BUTTON_CANCEL in panel.labels()


def test_a_duplicate_feed_is_reported(panel: Panel) -> None:
    key = temp_topic_key("dupfeed")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("dup.rss")

    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)
    panel.press("addsource:" + key)
    reply = panel.send(url)

    assert "از قبل ثبت شده" in reply
    assert len([s for s in repository.list_sources() if s.rss_url == url]) == 1


def test_the_fetch_limit_is_set_with_one_preset_press(panel: Panel) -> None:
    """FR-1/FR-14: the cap lives on the source, so each feed gets its own batch size."""
    key = temp_topic_key("limit")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("limit.rss")
    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)

    panel.press_label(admin.BUTTON_LIMIT)
    assert panel.labels()[:3] == [str(value) for value in admin.FETCH_LIMIT_PRESETS]

    reply = panel.press_label("50")

    assert "50" in reply
    assert _stored_source(url).fetch_limit == 50


def test_the_fetch_limit_can_be_typed_and_reset_to_the_default(panel: Panel) -> None:
    key = temp_topic_key("custom")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("custom.rss")
    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)

    panel.press_label(admin.BUTTON_LIMIT)
    panel.press_label(admin.BUTTON_CUSTOM_LIMIT)
    assert "عدد مثبت" in panel.send("lots")  # the prompt rejects it and stays open

    assert "12" in panel.send("12")
    assert _stored_source(url).fetch_limit == 12

    panel.press_label(admin.BUTTON_LIMIT)
    assert "پیش‌فرض" in panel.press_label(admin.BUTTON_DEFAULT_LIMIT)
    assert _stored_source(url).fetch_limit is None


def test_toggling_a_source_hides_it_from_the_collector(panel: Panel) -> None:
    key = temp_topic_key("togglesrc")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("togglesrc.rss")
    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)

    assert "غیرفعال شد" in panel.press_label(admin.BUTTON_TOGGLE)
    assert url not in {source.rss_url for source in repository.list_sources(active_only=True)}

    assert "فعال شد" in panel.press_label(admin.BUTTON_TOGGLE)
    assert url in {source.rss_url for source in repository.list_sources(active_only=True)}


def test_moving_a_source_to_the_right_topic(panel: Panel) -> None:
    """The picker asks by name; the click carries the key, which is what is stored."""
    first, second = temp_topic_key("mv3"), temp_topic_key("mv4")
    panel.press("newtopic")
    panel.send(f"{first} آلفا")
    panel.press("newtopic")
    panel.send(f"{second} بتا")
    url = temp_source_url("move2.rss")
    panel.press("addsource:" + first)
    panel.send(url)

    panel.press_label(admin.BUTTON_MOVE)

    # The picker lists every topic by its Persian name.
    assert "آلفا" in panel.labels() and "بتا" in panel.labels()
    panel.press_label("بتا")

    assert _stored_source(url).topic_key == second


def test_deleting_a_source_needs_the_confirmation_button(panel: Panel) -> None:
    key = temp_topic_key("delsrc")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("delsrc.rss")
    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)

    assert "حذف شود؟" in panel.press_label(admin.BUTTON_DELETE_SOURCE)
    assert url in {source.rss_url for source in repository.list_sources()}  # still there

    reply = panel.press_label(BUTTON_CONFIRM)

    assert "حذف شد" in reply
    assert url not in {source.rss_url for source in repository.list_sources()}


def test_only_the_sources_of_the_pressed_topic_are_listed(panel: Panel) -> None:
    first, second = temp_topic_key("one"), temp_topic_key("two")
    panel.press("newtopic")
    panel.send(f"{first} موضوع یک")
    panel.press("newtopic")
    panel.send(f"{second} موضوع دو")
    panel.press("addsource:" + first)
    panel.send(temp_source_url("one.rss"))
    panel.press("addsource:" + second)
    panel.send(temp_source_url("two.rss"))

    panel.press("topic:" + first)
    panel.press_label(admin.BUTTON_TOPIC_SOURCES)

    assert temp_source_url("one.rss") in panel.text
    assert temp_source_url("two.rss") not in panel.text


# --- no interference between two actions ------------------------------------------


def test_a_button_press_clears_a_pending_field(panel: Panel) -> None:
    """Observed failure mode this prevents: a stale prompt eating the next message."""
    panel.press("newtopic")  # a prompt is now open
    panel.press("home")  # ... and the admin changes their mind

    reply = panel.send("این متن نباید موضوع بسازد")

    assert admin.HOME_TEXT in reply
    assert not [
        topic for topic in repository.list_topics() if "این" in topic.name
    ]


def test_two_chats_do_not_share_a_pending_field(telegram: FakeTelegram) -> None:
    first = Panel(telegram, chat_id=ADMIN_ID)
    second = Panel(telegram, chat_id=888)  # the other admin id in the test settings
    key = temp_topic_key("shared")

    first.press("newtopic")

    assert admin.HOME_TEXT in second.send("سلام")  # the second admin just gets the panel
    assert key not in {topic.key for topic in repository.list_topics()}

    assert "ساخته شد" in first.send(f"{key} موضوع")  # the first admin's answers are untouched


def test_a_cancelled_prompt_leaves_nothing_behind(panel: Panel) -> None:
    key = temp_topic_key("cancelled")

    panel.press("newtopic")
    assert panel.press_label(BUTTON_CANCEL) == admin.CANCEL_TEXT

    # The field is gone, so the next message is treated as a fresh panel visit.
    assert admin.HOME_TEXT in panel.send(key)
    assert key not in {topic.key for topic in repository.list_topics()}
    assert panel.labels() == [BUTTON_TOPICS, BUTTON_SOURCES, BUTTON_NEW_TOPIC]


# --- the panel really drives the collector (FR-1 + FR-14, end to end) -------------


def test_a_topic_disabled_in_the_panel_stops_feeding_the_collector(
    panel: Panel, db_connection: psycopg.Connection, settings: Settings
) -> None:
    key = temp_topic_key("enforce")
    panel.press("newtopic")
    panel.send(f"{key} موضوع")
    url = temp_source_url("enforce.rss")
    panel.press_label(admin.BUTTON_ADD_SOURCE)
    panel.send(url)
    assert url in {source.rss_url for source in repository.list_sources(active_only=True)}

    panel.press("topic:" + key)
    panel.press_label(admin.BUTTON_TOGGLE)

    assert url not in {source.rss_url for source in repository.list_sources(active_only=True)}
