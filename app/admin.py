"""The admin panel: topics and sources as data, driven from Telegram (FR-14).

There is no web UI and no separate login (AGENTS.md section 2): the "panel" is the bot
itself. Commands arrive as plain ``message`` updates through ``app.telegram_updates`` and
only ids listed in ``TELEGRAM_ADMIN_IDS`` are answered, so Telegram identity *is* the
authorisation.

The grammar is deliberately one line per command — no conversation state, no menus, no
pending-reply bookkeeping (KISS):

``/help`` · ``/topics`` · ``/addtopic`` · ``/renametopic`` · ``/toggletopic`` · ``/deltopic``
``/sources`` · ``/addsource`` · ``/setsourcetopic`` · ``/setsourcelimit`` ·
``/togglesource`` · ``/delsource``

Every command answers with a short Persian summary of what changed, and every answer that
echoes stored data is HTML-escaped, because the transport sends ``parse_mode=HTML``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

from app import repository, telegram_notifier
from app.formatting import escape_html
from app.models import SourceRecord, TopicRecord
from app.settings import get_settings

logger = logging.getLogger(__name__)

# The topic key is what the LLM may return as `topic` (FR-5), so it stays a short, stable,
# URL-safe token: lowercase letters, digits, dash and underscore.
TOPIC_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
MAX_TOPIC_NAME_LENGTH = 100

HELP_TEXT = """<b>دستورهای مدیریت</b>

<b>موضوعات</b>
/topics — فهرست موضوعات
/addtopic &lt;key&gt; &lt;نام&gt; — ساخت موضوع
/renametopic &lt;key&gt; &lt;نام&gt; — تغییر نام
/toggletopic &lt;key&gt; — فعال/غیرفعال
/deltopic &lt;key&gt; — حذف (به همراه منابعش)

<b>منابع</b>
/sources [key] — فهرست منابع
/addsource &lt;key&gt; &lt;rss_url&gt; [limit] — افزودن منبع
/setsourcetopic &lt;rss_url&gt; &lt;key&gt; — تغییر موضوع منبع
/setsourcelimit &lt;rss_url&gt; &lt;limit|default&gt; — سقف واکشی
/togglesource &lt;rss_url&gt; — فعال/غیرفعال
/delsource &lt;rss_url&gt; — حذف

اگر `limit` یک عدد نباشد، منبع از `RSS_FETCH_LIMIT` استفاده می‌کند.
"""

USAGE = "دستور ناقص است. برای راهنما /help را بزنید."
UNKNOWN_COMMAND = "دستور ناشناخته. برای راهنما /help را بزنید."
INVALID_KEY = (
    "کلید موضوع نامعتبر است: فقط حروف کوچک، رقم، خط تیره و زیرخط (۲ تا ۳۲ کاراکتر، "
    "با حرف یا رقم شروع شود)."
)
INVALID_URL = "آدرس فید نامعتبر است؛ باید با http:// یا https:// شروع شود."
INVALID_LIMIT = "سقف واکشی باید یک عدد مثبت باشد (یا کلمهٔ default)."


# --- helpers ----------------------------------------------------------------------


def _reply(chat_id: Any, text: str) -> bool:
    """Send one HTML reply back to the admin's chat (never raises, NFR-2)."""
    return telegram_notifier.send_message(text, chat_id=str(chat_id)) is not None


def _topic_lines(topics: Sequence[TopicRecord], sources: Sequence[SourceRecord]) -> list[str]:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.topic_key] = counts.get(source.topic_key, 0) + 1
    if not topics:
        return ["هیچ موضوعی تعریف نشده است. با /addtopic یکی بسازید."]
    lines = ["<b>موضوعات</b>"]
    for topic in topics:
        state = "فعال" if topic.is_active else "غیرفعال"
        lines.append(
            f"• <code>{escape_html(topic.key)}</code> — {escape_html(topic.name)} "
            f"({state}، {counts.get(topic.key, 0)} منبع)"
        )
    return lines


def _source_lines(sources: Sequence[SourceRecord], *, topic_keys: set[str]) -> list[str]:
    """List the feeds, optionally only those of one topic."""
    selected = [source for source in sources if not topic_keys or source.topic_key in topic_keys]
    if not selected:
        return ["منبعی برای نمایش نیست."]
    lines = ["<b>منابع</b>"]
    for source in selected:
        limit = source.fetch_limit if source.fetch_limit is not None else "پیش‌فرض"
        state = "فعال" if source.is_active else "غیرفعال"
        lines.append(
            f"• <code>{escape_html(source.rss_url)}</code>\n"
            f"  موضوع: <code>{escape_html(source.topic_key)}</code> • "
            f"سقف: {escape_html(str(limit))} • {state}"
        )
    return lines


def _parse_fetch_limit(raw: str | None) -> int | None | str:
    """``None``/``"default"`` -> use the global default; a positive int; or an error string."""
    if raw is None or raw.strip().lower() in {"", "default"}:
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return INVALID_LIMIT
    return value if value > 0 else INVALID_LIMIT


# --- handlers (one per command; each returns the reply text) ----------------------


def _help(_args: list[str]) -> str:
    return HELP_TEXT


def _topics(_args: list[str]) -> str:
    return "\n".join(
        _topic_lines(repository.list_topics(), repository.list_sources())
    )


def _addtopic(args: list[str]) -> str:
    if len(args) < 2:
        return "استفاده: /addtopic <key> <نام>"
    key, name = args[0], " ".join(args[1:])
    if not TOPIC_KEY_RE.match(key):
        return INVALID_KEY
    if len(name) > MAX_TOPIC_NAME_LENGTH:
        return f"نام موضوع بلندتر از {MAX_TOPIC_NAME_LENGTH} کاراکتر است."
    topic = repository.create_topic(key, name)
    if topic is None:
        return f"موضوع <code>{escape_html(key)}</code> از قبل وجود دارد."
    return f"موضوع <code>{escape_html(key)}</code> ساخته شد: {escape_html(name)}"


def _renametopic(args: list[str]) -> str:
    if len(args) < 2:
        return "استفاده: /renametopic <key> <نام>"
    key, name = args[0], " ".join(args[1:])
    if not repository.update_topic_name(key, name):
        return f"موضوع <code>{escape_html(key)}</code> پیدا نشد."
    return f"نام <code>{escape_html(key)}</code> به «{escape_html(name)}» تغییر کرد."


def _toggletopic(args: list[str]) -> str:
    if len(args) != 1:
        return "استفاده: /toggletopic <key>"
    key = args[0]
    current = {topic.key: topic for topic in repository.list_topics()}.get(key)
    if current is None:
        return f"موضوع <code>{escape_html(key)}</code> پیدا نشد."
    if not repository.set_topic_active(key, not current.is_active):
        return f"موضوع <code>{escape_html(key)}</code> پیدا نشد."
    state = "غیرفعال" if current.is_active else "فعال"
    return f"موضوع <code>{escape_html(key)}</code> {state} شد."


def _deltopic(args: list[str]) -> str:
    if len(args) != 1:
        return "استفاده: /deltopic <key>"
    key = args[0]
    sources = [s for s in repository.list_sources() if s.topic_key == key]
    if not repository.delete_topic(key):
        return f"موضوع <code>{escape_html(key)}</code> پیدا نشد."
    return (
        f"موضوع <code>{escape_html(key)}</code> و {len(sources)} منبعش حذف شدند. "
        "پست‌های ذخیره‌شده دست‌نخورده می‌مانند (سابقه)."
    )


def _sources(args: list[str]) -> str:
    return "\n".join(
        _source_lines(repository.list_sources(), topic_keys=set(args[:1]))
    )


def _addsource(args: list[str]) -> str:
    if len(args) < 2:
        return "استفاده: /addsource <topic_key> <rss_url> [limit]"
    topic_key, rss_url = args[0], args[1]
    limit = _parse_fetch_limit(args[2] if len(args) > 2 else None)
    if isinstance(limit, str):
        return limit
    if not rss_url.startswith(("http://", "https://")):
        return INVALID_URL
    if not repository.topic_exists(topic_key):
        return f"موضوع <code>{escape_html(topic_key)}</code> پیدا نشد؛ اول بسازیدش."
    source = repository.create_source(topic_key, rss_url, limit)
    if source is None:
        return f"این فید از قبل ثبت شده است: <code>{escape_html(rss_url)}</code>"
    shown = source.fetch_limit if source.fetch_limit is not None else "پیش‌فرض"
    return (
        f"منبع <code>{escape_html(rss_url)}</code> به "
        f"<code>{escape_html(topic_key)}</code> اضافه شد (سقف: {shown})."
    )


def _setsourcetopic(args: list[str]) -> str:
    if len(args) != 2:
        return "استفاده: /setsourcetopic <rss_url> <topic_key>"
    rss_url, topic_key = args
    if not repository.set_source_topic(rss_url, topic_key):
        return "منبع یا موضوع پیدا نشد."
    return (
        f"موضوع <code>{escape_html(rss_url)}</code> به "
        f"<code>{escape_html(topic_key)}</code> تغییر کرد."
    )


def _setsourcelimit(args: list[str]) -> str:
    if len(args) != 2:
        return "استفاده: /setsourcelimit <rss_url> <limit|default>"
    rss_url = args[0]
    limit = _parse_fetch_limit(args[1])
    if isinstance(limit, str):
        return limit
    if not repository.set_source_fetch_limit(rss_url, limit):
        return "منبع پیدا نشد."
    shown = limit if limit is not None else "پیش‌فرض (RSS_FETCH_LIMIT)"
    return f"سقف <code>{escape_html(rss_url)}</code> روی {shown} تنظیم شد."


def _togglesource(args: list[str]) -> str:
    if len(args) != 1:
        return "استفاده: /togglesource <rss_url>"
    rss_url = args[0]
    current = {source.rss_url: source for source in repository.list_sources()}.get(rss_url)
    if current is None:
        return "منبع پیدا نشد."
    repository.set_source_active(rss_url, not current.is_active)
    state = "غیرفعال" if current.is_active else "فعال"
    return f"منبع <code>{escape_html(rss_url)}</code> {state} شد."


def _delsource(args: list[str]) -> str:
    if len(args) != 1:
        return "استفاده: /delsource <rss_url>"
    rss_url = args[0]
    if not repository.delete_source(rss_url):
        return "منبع پیدا نشد."
    return f"منبع <code>{escape_html(rss_url)}</code> حذف شد."


HANDLERS: dict[str, Callable[[list[str]], str]] = {
    "/help": _help,
    "/start": _help,
    "/topics": _topics,
    "/addtopic": _addtopic,
    "/renametopic": _renametopic,
    "/toggletopic": _toggletopic,
    "/deltopic": _deltopic,
    "/sources": _sources,
    "/addsource": _addsource,
    "/setsourcetopic": _setsourcetopic,
    "/setsourcelimit": _setsourcelimit,
    "/togglesource": _togglesource,
    "/delsource": _delsource,
}


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """``"/addtopic ai هوش مصنوعی"`` -> ``("/addtopic", ["ai", "هوش مصنوعی"])`` (pure).

    ``/command@TheBot`` (the form Telegram uses in group chats) is accepted too.
    """
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    command = parts[0].split("@", 1)[0].lower()
    return (command, parts[1:]) if command else None


def handle_command(message: dict[str, Any]) -> bool:
    """Answer one admin message. Returns whether a command was handled.

    A message from anyone outside ``TELEGRAM_ADMIN_IDS`` is ignored without a reply: the
    bot never confirms its own existence to a stranger, and every ignored attempt is
    logged so an operator can see it.
    """
    parsed = parse_command(str(message.get("text") or ""))
    if parsed is None:
        return False

    command, args = parsed
    chat_id = (message.get("chat") or {}).get("id")
    sender = message.get("from") or {}
    sender_id = sender.get("id")

    if not _is_admin(sender_id):
        logger.warning("Ignoring admin command %s from a non-admin (id=%s)", command, sender_id)
        return False

    handler = HANDLERS.get(command)
    reply = handler(args) if handler else UNKNOWN_COMMAND
    if reply:
        _reply(chat_id, reply)
    logger.info("Admin command %s from id=%s handled", command, sender_id)
    return True


def _is_admin(sender_id: Any) -> bool:
    """The same allow-list the review buttons use (FR-12), read fresh from settings."""
    admin_ids = get_settings().admin_ids
    return bool(admin_ids) and sender_id is not None and str(sender_id) in admin_ids
