"""The admin panel: topics and sources as data, driven from Telegram (FR-14).

There is no web UI and no separate login (AGENTS.md section 2): the "panel" is the bot
itself. Only ids listed in ``TELEGRAM_ADMIN_IDS`` are answered, so Telegram identity *is*
the authorisation.

**No command grammar.** Phase 5 asked the admin to type ``/addtopic ai هوش مصنوعی``; phase 6
replaced that surface with inline ("glass") keyboards, because a typed command is both hard
to discover and impossible to keep free of interference between two commands typed in a
row. Every screen is a message whose buttons are the whole operation set:

* ``panel:home`` → ``panel:topics`` / ``panel:sources`` — the two lists;
* ``panel:topic:<key>`` / ``panel:source:<id>`` — one row's detail screen with rename,
  toggle, delete (behind a confirmation button), move to another topic, and its fetch cap;
* creating something is a *prompt*, not a command: the bot asks for exactly one value and
  the admin's next message is the answer to that field (that is why a chat may have at most
  one pending field at a time). Pressing any panel button clears it, so two half-finished
  actions can never interfere with each other.

``_PENDING`` is the only state in the module: one pending field per chat, deliberately in
memory — the worker is a single process, and losing a half-typed value on restart only means
pressing the button again (KISS; the data itself always lives in Postgres).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, NamedTuple

from app import repository, telegram_notifier
from app.formatting import escape_html, truncate
from app.models import SourceRecord, TopicRecord
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Every callback this module owns lives in this namespace; ``app/review.py`` owns
# ``approve:``/``reject:``/``status:`` and ``app/telegram_updates.py`` routes by prefix.
PANEL_CALLBACK_PREFIX = "panel:"

# A destructive action is two states: ``panel:deltopic:<key>`` asks, and the confirmation
# button carries ``:yes``, so the second press is a different callback from the first.
CONFIRM_SUFFIX = "yes"

HOME_TEXT = (
    "<b>پنل مدیریت</b>\n"
    "موضوعات و منابع را از همین دکمه‌ها مدیریت کنید؛ هیچ دستوری برای تایپ‌کردن وجود ندارد."
)
TOPICS_TITLE = "<b>موضوعات</b>"
SOURCES_TITLE = "<b>منابع</b>"
CANCEL_TEXT = "لغو شد."
NO_PENDING_TEXT = "کار نیمه‌تمام قبلی پاک شد؛ یکی از دکمه‌های پنل را بزنید."
DONE_TEXT = "انجام شد."

BUTTON_HOME = "🏠 خانه"
BUTTON_TOPICS = "📚 موضوعات"
BUTTON_SOURCES = "📡 منابع"
BUTTON_NEW_TOPIC = "➕ موضوع جدید"
BUTTON_BACK_TOPICS = "⬅️ موضوعات"
BUTTON_BACK_SOURCES = "⬅️ منابع"
BUTTON_RENAME = "✏️ تغییر نام"
BUTTON_TOGGLE = "🔁 فعال/غیرفعال"
BUTTON_ADD_SOURCE = "➕ افزودن منبع"
BUTTON_TOPIC_SOURCES = "📡 منابع این موضوع"
BUTTON_DELETE_TOPIC = "🗑 حذف موضوع"
BUTTON_LIMIT = "🔢 سقف واکشی"
BUTTON_MOVE = "🔀 تغییر موضوع"
BUTTON_DELETE_SOURCE = "🗑 حذف منبع"
BUTTON_CONFIRM = "✅ بله، حذف کن"
BUTTON_CANCEL = "🗑 لغو"
BUTTON_DEFAULT_LIMIT = "پیش‌فرض"
BUTTON_CUSTOM_LIMIT = "✏️ عدد دلخواه"

# Quick picks for a per-source cap; ``RSS_FETCH_LIMIT`` stays the default (section 11).
FETCH_LIMIT_PRESETS: tuple[int, ...] = (25, 50, 100)

# The topic key is what the LLM may return as `topic` (FR-5), so it stays a short, stable,
# URL-safe token: lowercase letters, digits, dash and underscore.
TOPIC_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
MAX_TOPIC_NAME_LENGTH = 100

# How much of a feed URL is readable on a button.
SOURCE_LABEL_LIMIT = 40

NEW_TOPIC_PROMPT = (
    "➕ <b>موضوع جدید</b>\n"
    "کلید و نام موضوع را در یک پیام بفرستید، به شکل «کلید نام».\n"
    "کلید فقط حروف کوچک، رقم، خط تیره و زیرخط (۲ تا ۳۲ کاراکتر، با حرف یا رقم شروع شود)."
)
RENAME_PROMPT = "✏️ <b>تغییر نام</b>\nنام تازهٔ موضوع را بفرستید."
ADD_SOURCE_PROMPT = (
    "➕ <b>افزودن منبع</b>\n"
    "آدرس فید RSS را بفرستید (با http:// یا https:// شروع شود).\n"
    "سقف واکشی بعداً از همین صفحه قابل تغییر است."
)
LIMIT_PROMPT = "🔢 <b>سقف واکشی</b>\nیک عدد مثبت بفرستید، یا کلمهٔ «default» برای بازگشت به پیش‌فرض."

INVALID_KEY = (
    "کلید موضوع نامعتبر است: فقط حروف کوچک، رقم، خط تیره و زیرخط (۲ تا ۳۲ کاراکتر، "
    "با حرف یا رقم شروع شود)."
)
INVALID_URL = "آدرس فید نامعتبر است؛ باید با http:// یا https:// شروع شود."
INVALID_LIMIT = "سقف واکشی باید یک عدد مثبت باشد (یا کلمهٔ default)."
UNKNOWN_PANEL_BUTTON = "این دکمه دیگر معتبر نیست؛ از منو یکی را بزنید."
NOT_ADMIN_TEXT = "این عملیات فقط برای ادمین‌هاست."


class Pending(NamedTuple):
    """The one field the panel is waiting for from a chat."""

    action: str
    target: str | None = None


class PanelReply(NamedTuple):
    """One message to send back, plus whether the field must stay open.

    A rejected value (an invalid key, a URL that is not a URL) is answered with the same
    prompt, and the field has to stay armed — otherwise the admin's correction would be
    read as a fresh panel visit and silently do nothing.
    """

    text: str
    keyboard: Keyboard | None = None
    keep_pending: bool = False


# action names of :class:`Pending`
PENDING_NEW_TOPIC = "newtopic"
PENDING_RENAME_TOPIC = "rename"
PENDING_ADD_SOURCE = "addsource"
PENDING_SET_LIMIT = "limit"

_PENDING: dict[str, Pending] = {}

Keyboard = dict[str, Any]


# --- authorisation -----------------------------------------------------------------


def is_admin(sender_id: Any) -> bool:
    """The same allow-list the review buttons use (FR-12), read fresh from settings."""
    admin_ids = get_settings().admin_ids
    return bool(admin_ids) and sender_id is not None and str(sender_id) in admin_ids


# --- rendering ---------------------------------------------------------------------


def _button(text: str, callback_data: str) -> dict[str, str]:
    return {"text": text, "callback_data": f"{PANEL_CALLBACK_PREFIX}{callback_data}"}


def _keyboard(*rows: Sequence[dict[str, str]]) -> Keyboard:
    return {"inline_keyboard": [list(row) for row in rows]}


def home_keyboard() -> Keyboard:
    return _keyboard([_button(BUTTON_TOPICS, "topics"), _button(BUTTON_SOURCES, "sources")],
                     [_button(BUTTON_NEW_TOPIC, "newtopic")])


def topics_keyboard(topics: Sequence[TopicRecord], counts: dict[str, int]) -> Keyboard:
    rows = [
        [_button(_topic_button_label(topic, counts.get(topic.key, 0)), f"topic:{topic.key}")]
        for topic in topics
    ]
    rows.append([_button(BUTTON_NEW_TOPIC, "newtopic")])
    rows.append([_button(BUTTON_HOME, "home")])
    return _keyboard(*rows)


def topic_keyboard(topic: TopicRecord, *, source_count: int) -> Keyboard:
    return _keyboard(
        [_button(BUTTON_RENAME, f"rename:{topic.key}"), _button(BUTTON_TOGGLE, f"toggletopic:{topic.key}")],
        [_button(BUTTON_ADD_SOURCE, f"addsource:{topic.key}")],
        [_button(BUTTON_TOPIC_SOURCES, f"topic-sources:{topic.key}")],
        [_button(BUTTON_DELETE_TOPIC, f"deltopic:{topic.key}")],
        [_button(BUTTON_BACK_TOPICS, "topics"), _button(BUTTON_HOME, "home")],
    )


def sources_keyboard(sources: Sequence[SourceRecord]) -> Keyboard:
    rows = [
        [_button(_source_button_label(source), f"source:{source.id}")] for source in sources
    ]
    rows.append([_button(BUTTON_HOME, "home")])
    return _keyboard(*rows)


def source_keyboard(source: SourceRecord) -> Keyboard:
    return _keyboard(
        [_button(BUTTON_TOGGLE, f"togglesource:{source.id}"), _button(BUTTON_LIMIT, f"limit:{source.id}")],
        [_button(BUTTON_MOVE, f"movetopic:{source.id}")],
        [_button(BUTTON_DELETE_SOURCE, f"delsource:{source.id}")],
        [_button(BUTTON_BACK_SOURCES, "sources"), _button(BUTTON_HOME, "home")],
    )


def limit_keyboard(source_id: int) -> Keyboard:
    presets = [
        _button(str(value), f"limit:{source_id}:{value}") for value in FETCH_LIMIT_PRESETS
    ]
    return _keyboard(
        presets,
        [_button(BUTTON_DEFAULT_LIMIT, f"limit:{source_id}:default"), _button(BUTTON_CUSTOM_LIMIT, f"sourcelimit:{source_id}")],
        [_button(BUTTON_BACK_SOURCES, f"source:{source_id}")],
    )


def move_keyboard(source_id: int, topics: Sequence[TopicRecord]) -> Keyboard:
    rows = [
        [_button(topic.name, f"movetopic:{source_id}:{topic.key}")] for topic in topics
    ]
    rows.append([_button(BUTTON_BACK_SOURCES, f"source:{source_id}")])
    return _keyboard(*rows)


def confirm_keyboard(action: str, target: str, *, back: str) -> Keyboard:
    """A destructive action is always a second press, never the first one."""
    return _keyboard(
        [_button(BUTTON_CONFIRM, f"{action}:{target}:{CONFIRM_SUFFIX}")],
        [_button(BUTTON_CANCEL, back)],
    )


def prompt_keyboard() -> Keyboard:
    return _keyboard([_button(BUTTON_CANCEL, "cancel")])


def _topic_button_label(topic: TopicRecord, source_count: int) -> str:
    state = "فعال" if topic.is_active else "غیرفعال"
    return f"{topic.key} — {topic.name} ({state}، {source_count} منبع)"


def _source_button_label(source: SourceRecord) -> str:
    limit = source.fetch_limit if source.fetch_limit is not None else "پیش‌فرض"
    state = "فعال" if source.is_active else "غیرفعال"
    return f"{truncate(_short_url(source.rss_url), SOURCE_LABEL_LIMIT)} • {limit} • {state}"


def _short_url(rss_url: str) -> str:
    """``https://www.reddit.com/r/mlops/new/.rss`` -> ``reddit.com/r/mlops/new/.rss``."""
    return rss_url.split("://", 1)[-1]


def _topic_lines(topics: Sequence[TopicRecord], counts: dict[str, int]) -> list[str]:
    if not topics:
        return ["هیچ موضوعی تعریف نشده است؛ با «موضوع جدید» یکی بسازید."]
    return [
        f"• <code>{escape_html(topic.key)}</code> — {escape_html(topic.name)} "
        f"({'فعال' if topic.is_active else 'غیرفعال'}، {counts.get(topic.key, 0)} منبع)"
        for topic in topics
    ]


def _source_lines(sources: Sequence[SourceRecord]) -> list[str]:
    if not sources:
        return ["منبعی ثبت نشده است."]
    lines = []
    for source in sources:
        limit = source.fetch_limit if source.fetch_limit is not None else "پیش‌فرض"
        lines.append(
            f"• <code>{escape_html(source.rss_url)}</code>\n"
            f"  موضوع: <code>{escape_html(source.topic_key)}</code> • "
            f"سقف: {escape_html(str(limit))} • "
            f"{'فعال' if source.is_active else 'غیرفعال'}"
        )
    return lines


def _source_counts(topics: Sequence[TopicRecord], sources: Sequence[SourceRecord]) -> dict[str, int]:
    counts: dict[str, int] = {topic.key: 0 for topic in topics}
    for source in sources:
        counts[source.topic_key] = counts.get(source.topic_key, 0) + 1
    return counts


def _find_topic(key: str) -> TopicRecord | None:
    return next((topic for topic in repository.list_topics() if topic.key == key), None)


def _find_source(source_id: str) -> SourceRecord | None:
    return next(
        (source for source in repository.list_sources() if str(source.id) == source_id), None
    )


# --- screens -----------------------------------------------------------------------


def topics_screen() -> tuple[str, Keyboard]:
    topics = repository.list_topics()
    sources = repository.list_sources()
    text = "\n".join([TOPICS_TITLE, *_topic_lines(topics, _source_counts(topics, sources))])
    return text, topics_keyboard(topics, _source_counts(topics, sources))


def sources_screen(topic_key: str | None = None) -> tuple[str, Keyboard]:
    sources = [
        source
        for source in repository.list_sources()
        if topic_key is None or source.topic_key == topic_key
    ]
    heading = SOURCES_TITLE if topic_key is None else f"{SOURCES_TITLE} — <code>{escape_html(topic_key)}</code>"
    return "\n".join([heading, *_source_lines(sources)]), sources_keyboard(sources)


def topic_screen(key: str) -> tuple[str, Keyboard]:
    topic = _find_topic(key)
    if topic is None:
        return topics_screen()
    sources = [source for source in repository.list_sources() if source.topic_key == key]
    text = "\n".join(
        [
            f"<b>موضوع</b> <code>{escape_html(topic.key)}</code>",
            f"نام: {escape_html(topic.name)}",
            f"وضعیت: {'فعال' if topic.is_active else 'غیرفعال'}",
            f"تعداد منابع: {len(sources)}",
        ]
    )
    return text, topic_keyboard(topic, source_count=len(sources))


def source_screen(source_id: str) -> tuple[str, Keyboard]:
    source = _find_source(source_id)
    if source is None:
        return sources_screen()
    limit = source.fetch_limit if source.fetch_limit is not None else "پیش‌فرض (RSS_FETCH_LIMIT)"
    text = "\n".join(
        [
            "<b>منبع</b>",
            f"آدرس: <code>{escape_html(source.rss_url)}</code>",
            f"موضوع: <code>{escape_html(source.topic_key)}</code>",
            f"سقف واکشی: {escape_html(str(limit))}",
            f"وضعیت: {'فعال' if source.is_active else 'غیرفعال'}",
        ]
    )
    return text, source_keyboard(source)


# --- panel routing -----------------------------------------------------------------


def is_panel_callback(data: str) -> bool:
    """Whether one callback belongs to the admin panel rather than the review flow."""
    return data.startswith(PANEL_CALLBACK_PREFIX)


async def handle_callback(callback_query: dict[str, Any]) -> bool:
    """Answer one panel button press. Returns whether a panel screen was produced."""
    sender = callback_query.get("from")
    sender_id = (sender or {}).get("id") if isinstance(sender, dict) else None
    callback_id = str(callback_query.get("id") or "")
    chat_id = _callback_chat_id(callback_query)

    if not is_admin(sender_id):
        logger.warning("Ignoring an admin panel press from a non-admin (id=%s)", sender_id)
        if callback_id:
            await telegram_notifier.answer_callback_query(callback_id, text=NOT_ADMIN_TEXT)
        return False

    if callback_id:
        await telegram_notifier.answer_callback_query(callback_id)

    # A button press always wins over a half-finished prompt: that is what keeps two
    # actions from interfering with each other.
    _PENDING.pop(_chat_key(chat_id), None)

    text, keyboard = _screen(str(callback_query.get("data") or ""), _chat_key(chat_id))
    await _send(chat_id, text, keyboard)
    return True


def _screen(data: str, chat_key: str) -> tuple[str, Keyboard | None]:
    """Turn one panel callback into the message to send back (the only side effects here)."""
    parts = data[len(PANEL_CALLBACK_PREFIX) :].split(":")
    action, rest = parts[0], parts[1:]

    if action == "home":
        return HOME_TEXT, home_keyboard()
    if action == "topics":
        return topics_screen()
    if action == "sources":
        return sources_screen(rest[0] if rest else None)
    if action == "topic":
        return topic_screen(rest[0]) if rest else topics_screen()
    if action == "source":
        return source_screen(rest[0]) if rest else sources_screen()
    if action == "topic-sources":
        return sources_screen(rest[0]) if rest else sources_screen()
    if action == "newtopic":
        _PENDING[chat_key] = Pending(PENDING_NEW_TOPIC)
        return NEW_TOPIC_PROMPT, prompt_keyboard()
    if action == "rename" and rest:
        _PENDING[chat_key] = Pending(PENDING_RENAME_TOPIC, rest[0])
        return RENAME_PROMPT, prompt_keyboard()
    if action == "addsource" and rest:
        _PENDING[chat_key] = Pending(PENDING_ADD_SOURCE, rest[0])
        return ADD_SOURCE_PROMPT, prompt_keyboard()
    if action == "sourcelimit" and rest:
        _PENDING[chat_key] = Pending(PENDING_SET_LIMIT, rest[0])
        return LIMIT_PROMPT, prompt_keyboard()
    if action == "cancel":
        return CANCEL_TEXT, home_keyboard()

    if action == "toggletopic" and rest:
        return _toggle_topic(rest[0])
    if action == "togglesource" and rest:
        return _toggle_source(rest[0])
    if action == "deltopic" and rest:
        return _confirm_delete_topic(rest[0], parts)
    if action == "delsource" and rest:
        return _confirm_delete_source(rest[0], parts)
    if action == "limit" and rest:
        return _set_limit(rest, parts)
    if action == "movetopic" and rest:
        return _move_source(rest, parts)

    logger.warning("Unknown admin panel callback: %s", data)
    return UNKNOWN_PANEL_BUTTON, home_keyboard()


def _toggle_topic(key: str) -> tuple[str, Keyboard]:
    topic = _find_topic(key)
    if topic is None:
        return topics_screen()
    repository.set_topic_active(key, not topic.is_active)
    state = "غیرفعال" if topic.is_active else "فعال"
    text, keyboard = topic_screen(key)
    return f"{DONE_TEXT} موضوع <code>{escape_html(key)}</code> {state} شد.\n\n{text}", keyboard


def _toggle_source(source_id: str) -> tuple[str, Keyboard]:
    source = _find_source(source_id)
    if source is None:
        return sources_screen()
    repository.set_source_active(source.rss_url, not source.is_active)
    state = "غیرفعال" if source.is_active else "فعال"
    text, keyboard = source_screen(source_id)
    return f"{DONE_TEXT} منبع {state} شد.\n\n{text}", keyboard


def _confirm_delete_topic(key: str, parts: list[str]) -> tuple[str, Keyboard]:
    confirmed = len(parts) > 2 and parts[2] == CONFIRM_SUFFIX
    topic = _find_topic(key)
    if topic is None:
        return topics_screen()
    if not confirmed:
        return (
            f"موضوع <code>{escape_html(key)}</code> و منابعش حذف شوند؟ "
            "پست‌های ذخیره‌شده دست‌نخورده می‌مانند (سابقه).",
            confirm_keyboard("deltopic", key, back="topics"),
        )
    sources = [source for source in repository.list_sources() if source.topic_key == key]
    repository.delete_topic(key)
    text, keyboard = topics_screen()
    return f"{DONE_TEXT} موضوع حذف شد به همراه {len(sources)} منبعش.\n\n{text}", keyboard


def _confirm_delete_source(source_id: str, parts: list[str]) -> tuple[str, Keyboard]:
    confirmed = len(parts) > 2 and parts[2] == CONFIRM_SUFFIX
    source = _find_source(source_id)
    if source is None:
        return sources_screen()
    if not confirmed:
        return (
            f"منبع <code>{escape_html(source.rss_url)}</code> حذف شود؟",
            confirm_keyboard("delsource", source_id, back=f"source:{source_id}"),
        )
    repository.delete_source(source.rss_url)
    return f"{DONE_TEXT} منبع حذف شد.", sources_keyboard(repository.list_sources())


def _set_limit(rest: list[str], parts: list[str]) -> tuple[str, Keyboard]:
    source = _find_source(rest[0])
    if source is None:
        return sources_screen()
    if len(parts) > 2:
        raw = parts[2]
        value = None if raw == "default" else int(raw)
        repository.set_source_fetch_limit(source.rss_url, value)
        shown = "پیش‌فرض" if value is None else str(value)
        text, keyboard = source_screen(rest[0])
        return f"{DONE_TEXT} سقف واکشی روی {shown} تنظیم شد.\n\n{text}", keyboard
    return (
        f"سقف واکشی برای <code>{escape_html(source.rss_url)}</code> چقدر باشد؟",
        limit_keyboard(int(rest[0])),
    )


def _move_source(rest: list[str], parts: list[str]) -> tuple[str, Keyboard]:
    source = _find_source(rest[0])
    if source is None:
        return sources_screen()
    if len(parts) > 2:
        topic_key = parts[2]
        if not repository.set_source_topic(source.rss_url, topic_key):
            return "موضوع پیدا نشد.", sources_screen()[1]
        text, keyboard = source_screen(rest[0])
        return (
            f"{DONE_TEXT} موضوع منبع به <code>{escape_html(topic_key)}</code> تغییر کرد.\n\n{text}",
            keyboard,
        )
    return (
        "منبع به کدام موضوع منتقل شود؟",
        move_keyboard(int(rest[0]), repository.list_topics()),
    )


# --- prompted values (the only free-text entry points) -----------------------------


async def handle_message(message: dict[str, Any]) -> bool:
    """Answer one admin message: either the pending field, or the panel's home screen.

    Any message from an admin opens the panel — there is no command to remember and no
    command to type. A message from anyone else is ignored without a reply: the bot never
    confirms its own existence to a stranger.
    """
    chat_id = (message.get("chat") or {}).get("id")
    sender = message.get("from") or {}
    sender_id = sender.get("id")

    if not is_admin(sender_id):
        logger.warning("Ignoring a message from a non-admin (id=%s)", sender_id)
        return False

    chat_key = _chat_key(chat_id)
    pending = _PENDING.pop(chat_key, None)
    text = str(message.get("text") or "").strip()
    if pending is None:
        await _send(chat_id, HOME_TEXT, home_keyboard())
        return True

    reply = _apply_pending(pending, text)
    if reply.keep_pending:
        _PENDING[chat_key] = pending
    await _send(chat_id, reply.text, reply.keyboard)
    logger.info("Admin id=%s answered the %s prompt", sender_id, pending.action)
    return True


def _apply_pending(pending: Pending, text: str) -> PanelReply:
    if pending.action == PENDING_NEW_TOPIC:
        return _create_topic(text)
    if pending.action == PENDING_RENAME_TOPIC:
        return _rename_topic(pending.target or "", text)
    if pending.action == PENDING_ADD_SOURCE:
        return _create_source(pending.target or "", text)
    if pending.action == PENDING_SET_LIMIT:
        return _apply_limit(pending.target or "", text)
    return PanelReply(NO_PENDING_TEXT, home_keyboard())  # pragma: no cover


def _create_topic(text: str) -> PanelReply:
    """``"ai هوش مصنوعی"`` -> a new topic.

    The key is lower-cased for the admin (Telegram keyboards encourage capitals); every
    other rule is the one the LLM output is held to (FR-5).
    """
    key, _, name = text.partition(" ")
    key, name = key.strip().lower(), name.strip()
    if not key or not name:
        return _reprompt(NEW_TOPIC_PROMPT)
    if not TOPIC_KEY_RE.match(key):
        return _reprompt(f"{INVALID_KEY}\n\n{NEW_TOPIC_PROMPT}")
    if len(name) > MAX_TOPIC_NAME_LENGTH:
        return _reprompt(
            f"نام موضوع بلندتر از {MAX_TOPIC_NAME_LENGTH} کاراکتر است.\n\n{NEW_TOPIC_PROMPT}"
        )
    topic = repository.create_topic(key, name)
    if topic is None:
        return _reprompt(
            f"موضوع <code>{escape_html(key)}</code> از قبل وجود دارد.\n\n{NEW_TOPIC_PROMPT}"
        )
    text_out, keyboard = topic_screen(key)
    return PanelReply(
        f"{DONE_TEXT} موضوع <code>{escape_html(key)}</code> ساخته شد: "
        f"{escape_html(name)}\n\n{text_out}",
        keyboard,
    )


def _rename_topic(key: str, name: str) -> PanelReply:
    if not name:
        return _reprompt(RENAME_PROMPT)
    if len(name) > MAX_TOPIC_NAME_LENGTH:
        return _reprompt(
            f"نام موضوع بلندتر از {MAX_TOPIC_NAME_LENGTH} کاراکتر است.\n\n{RENAME_PROMPT}"
        )
    if not repository.update_topic_name(key, name):
        return PanelReply(
            f"موضوع <code>{escape_html(key)}</code> پیدا نشد.", topics_screen()[1]
        )
    text, keyboard = topic_screen(key)
    return PanelReply(
        f"{DONE_TEXT} نام موضوع به «{escape_html(name)}» تغییر کرد.\n\n{text}", keyboard
    )


def _create_source(topic_key: str, rss_url: str) -> PanelReply:
    if not rss_url.startswith(("http://", "https://")):
        return _reprompt(f"{INVALID_URL}\n\n{ADD_SOURCE_PROMPT}")
    if not repository.topic_exists(topic_key):
        return PanelReply(
            f"موضوع <code>{escape_html(topic_key)}</code> پیدا نشد.", topics_screen()[1]
        )
    source = repository.create_source(topic_key, rss_url)
    if source is None:
        # Re-adding the same feed is an answer to this prompt, not a new panel visit, so
        # the field stays open and the admin can correct the URL.
        return _reprompt(
            f"این فید از قبل ثبت شده است: <code>{escape_html(rss_url)}</code>\n\n"
            f"{ADD_SOURCE_PROMPT}"
        )
    text, keyboard = source_screen(str(source.id))
    return PanelReply(
        f"{DONE_TEXT} منبع به <code>{escape_html(topic_key)}</code> اضافه شد.\n\n{text}",
        keyboard,
    )


def _apply_limit(source_id: str, text: str) -> PanelReply:
    source = _find_source(source_id)
    if source is None:
        text_out, keyboard = sources_screen()
        return PanelReply(text_out, keyboard)
    if text.strip().lower() in {"", "default"}:
        repository.set_source_fetch_limit(source.rss_url, None)
        shown = "پیش‌فرض"
    else:
        try:
            value = int(text.strip())
        except ValueError:
            return _reprompt(f"{INVALID_LIMIT}\n\n{LIMIT_PROMPT}")
        if value <= 0:
            return _reprompt(f"{INVALID_LIMIT}\n\n{LIMIT_PROMPT}")
        repository.set_source_fetch_limit(source.rss_url, value)
        shown = str(value)
    text_out, keyboard = source_screen(source_id)
    return PanelReply(f"{DONE_TEXT} سقف واکشی روی {shown} تنظیم شد.\n\n{text_out}", keyboard)


def _reprompt(text: str) -> PanelReply:
    """Answer a rejected value with the same prompt, keeping the field armed."""
    return PanelReply(text, prompt_keyboard(), keep_pending=True)


# --- plumbing ----------------------------------------------------------------------


def _chat_key(chat_id: Any) -> str:
    return str(chat_id)


def _callback_chat_id(callback_query: dict[str, Any]) -> Any:
    message = callback_query.get("message")
    if isinstance(message, dict):
        chat = message.get("chat")
        if isinstance(chat, dict):
            return chat.get("id")
    return None


async def _send(chat_id: Any, text: str, keyboard: Keyboard | None) -> bool:
    """Send one panel message (never raises, NFR-2)."""
    return (
        await telegram_notifier.send_message(
            text, chat_id=str(chat_id), reply_markup=keyboard
        )
        is not None
    )


def reset_pending() -> None:
    """Drop every half-finished prompt (used by tests and after a restart)."""
    _PENDING.clear()


__all__ = [
    "HOME_TEXT",
    "PANEL_CALLBACK_PREFIX",
    "handle_callback",
    "handle_message",
    "home_keyboard",
    "is_admin",
    "is_panel_callback",
    "reset_pending",
]
