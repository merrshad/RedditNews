"""Format a stored post into the Persian Telegram message text (FR-10).

Pure functions only — no I/O, so this module is trivially unit-testable (NFR-8).
Invariant 5: everything shown to the reader is Persian, regardless of the post
language.
"""

from __future__ import annotations

from app.models import PostRecord

# Telegram's hard limit is 4096 characters; stay below it to leave room for the
# truncation suffix and for Telegram's own entity handling.
TELEGRAM_MESSAGE_LIMIT = 4096
SAFE_MESSAGE_LIMIT = 4000

MAX_TITLE_CHARS = 300
MAX_SUMMARY_CHARS = 1500
MAX_KEY_POINT_CHARS = 300

TRUNCATION_SUFFIX = "…"

IMPORTANCE_LABELS_FA: dict[str, str] = {
    "low": "کم",
    "medium": "متوسط",
    "high": "زیاد",
}
UNKNOWN_TOPIC_LABEL = "نامشخص"
MISSING_SUMMARY_LABEL = "خلاصه‌ای برای این پست ثبت نشده است."


def escape_html(text: str) -> str:
    """Escape the three characters Telegram's HTML parse mode cares about."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def importance_label(importance: str | None) -> str:
    """Persian label for the importance value (FR-10)."""
    return IMPORTANCE_LABELS_FA.get((importance or "").lower(), IMPORTANCE_LABELS_FA["low"])


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: max(limit - len(TRUNCATION_SUFFIX), 0)].rstrip() + TRUNCATION_SUFFIX


def _format_header(post: PostRecord, topic_name: str | None) -> str:
    lines = [
        f"<b>{escape_html(truncate(post.title, MAX_TITLE_CHARS))}</b>",
        f"📌 سابردیت: r/{escape_html(post.subreddit)}",
        f"🗂 موضوع: {escape_html(topic_name or post.topic or UNKNOWN_TOPIC_LABEL)}",
        f"⭐ اهمیت: {escape_html(importance_label(post.importance))}",
    ]
    return "\n".join(lines)


def _format_summary(post: PostRecord) -> str:
    summary = truncate(
        escape_html((post.summary_fa or "").strip()) or MISSING_SUMMARY_LABEL,
        MAX_SUMMARY_CHARS,
    )
    return f"📝 خلاصه:\n{summary}"


def _format_key_points(post: PostRecord) -> str:
    points = [point.strip() for point in (post.key_points or []) if point and point.strip()]
    if not points:
        return ""
    bullets = "\n".join(
        f"• {escape_html(truncate(point, MAX_KEY_POINT_CHARS))}" for point in points
    )
    return f"🔑 نکات کلیدی:\n{bullets}"


def _format_link(post: PostRecord) -> str:
    # Quotes are stripped so the URL can never break out of the href attribute.
    href = escape_html(post.url.replace('"', "%22").replace("'", "%27"))
    return f'🔗 <a href="{href}">مشاهده پست اصلی</a>'


def format_post_message(post: PostRecord, *, topic_name: str | None = None) -> str:
    """Render the full Persian Telegram message for a post (FR-10)."""
    sections = [
        _format_header(post, topic_name),
        _format_summary(post),
        _format_key_points(post),
        _format_link(post),
    ]
    message = "\n\n".join(section for section in sections if section)

    # Safety net for pathological inputs: always stay inside Telegram's limit.
    if len(message) > SAFE_MESSAGE_LIMIT:
        message = truncate(message, SAFE_MESSAGE_LIMIT)
    return message
