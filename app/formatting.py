"""Format a stored post into the Persian Telegram message text (FR-10).

Pure functions only — no I/O, so this module is trivially unit-testable (NFR-8).
Invariant 5: everything shown to the reader is Persian, regardless of the post
language. The exact layout is documented in AGENTS.md section 12, and this module is
also the single place that guarantees the message fits Telegram's length limit (FR-10).
"""

from __future__ import annotations

from html import escape as html_escape

from app.models import PostRecord

# Telegram's hard limit for the text of one message.
TELEGRAM_MESSAGE_LIMIT = 4096

# Importance -> Persian label for the reader, plus the fallback for a missing value.
IMPORTANCE_LABELS_FA: dict[str, str] = {
    "low": "کم",
    "medium": "متوسط",
    "high": "بالا",
}
UNKNOWN_TOPIC_LABEL = "نامشخص"
UNKNOWN_IMPORTANCE_LABEL = "نامشخص"
MISSING_SUMMARY_LABEL = "خلاصه‌ای برای این پست ثبت نشده است."

KEY_POINTS_HEADING = "🔑 نکات کلیدی:"
LINK_PREFIX = "🔗 "

TRUNCATION_SUFFIX = "…"


def escape_html(text: str) -> str:
    """Escape the three characters Telegram's HTML parse mode cares about.

    ``quote=False`` leaves ``"`` and ``'`` readable: the text never ends up inside an
    HTML attribute, so escaping them would only make the message uglier.
    """
    return html_escape(text, quote=False)


def importance_label(importance: str | None) -> str:
    """Persian label for an importance value; ``نامشخص`` when it is absent (FR-10)."""
    return IMPORTANCE_LABELS_FA.get(
        (importance or "").strip().lower(), UNKNOWN_IMPORTANCE_LABEL
    )


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` down to at most ``limit`` characters, marking the cut with ``…``."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - len(TRUNCATION_SUFFIX)].rstrip() + TRUNCATION_SUFFIX


def _topic_label(post: PostRecord, topic_name: str | None) -> str:
    """Persian name when the caller resolved one, otherwise the stored topic key."""
    return topic_name or post.topic or UNKNOWN_TOPIC_LABEL


def _header(post: PostRecord, topic_name: str | None) -> str:
    """The title plus the single metadata row (FR-10)."""
    title = escape_html(post.title)
    meta = (
        f"r/{escape_html(post.subreddit)}"
        f" • {escape_html(_topic_label(post, topic_name))}"
        f" • اهمیت: {escape_html(importance_label(post.importance))}"
    )
    return f"📌 <b>{title}</b>\n{meta}"


def _summary(post: PostRecord) -> str:
    return escape_html((post.summary_fa or "").strip()) or MISSING_SUMMARY_LABEL


def _key_points(post: PostRecord) -> str:
    points = [point.strip() for point in (post.key_points or []) if point and point.strip()]
    if not points:
        return ""
    bullets = "\n".join(f"• {escape_html(point)}" for point in points)
    return f"{KEY_POINTS_HEADING}\n{bullets}"


def _link(post: PostRecord) -> str:
    return f"{LINK_PREFIX}{escape_html(post.url)}"


def _render(post: PostRecord, topic_name: str | None, summary: str) -> str:
    sections = [
        _header(post, topic_name),
        summary,
        _key_points(post),
        _link(post),
    ]
    return "\n\n".join(section for section in sections if section)


def format_message(post: PostRecord, *, topic_name: str | None = None) -> str:
    """Render the Persian Telegram message for one stored post (FR-10).

    ``topic_name`` is the Persian display name resolved from ``config/topics.yaml``
    (``pipeline`` already has the mapping); when a caller has none, the stored topic key
    is shown instead of a raw English key leaking into the message. The layout:

        📌 <b>{title}</b>
        r/{subreddit} • {topic} • اهمیت: {importance_fa}

        {summary_fa}

        🔑 نکات کلیدی:
        • {key_point}

        🔗 {url}

    The result always fits Telegram's ``TELEGRAM_MESSAGE_LIMIT``: the summary is the
    only unbounded, non-essential section, so it is shortened first; the final hard cut
    only covers the pathological case where the fixed sections alone are already too
    long (e.g. a list of enormous key points).
    """
    message = _render(post, topic_name, _summary(post))
    if len(message) <= TELEGRAM_MESSAGE_LIMIT:
        return message

    # Shorten the summary by exactly the overflow; truncate() also adds the "…".
    summary = _summary(post)
    room = len(summary) - (len(message) - TELEGRAM_MESSAGE_LIMIT)
    message = _render(post, topic_name, truncate(summary, room))
    if len(message) <= TELEGRAM_MESSAGE_LIMIT:
        return message

    # The fixed sections do not fit on their own, so cut the whole message.
    return truncate(message, TELEGRAM_MESSAGE_LIMIT)
