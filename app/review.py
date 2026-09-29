"""The human review gate: private channel in, admin decision, message lifecycle (FR-12).

Nothing here calls the LLM. This module only moves a stored post through the *human*
part of the flow:

1. :func:`dispatch_pending_reviews` sends every post that was just stored (``new``) to the
   private review channel — one message per post, each with ✅/❌ buttons (FR-12).
2. :func:`handle_callback` applies a button press. The state change is one conditional
   SQL statement (``repository.decide_review``), so of two admins clicking at the same
   time exactly one wins and the other is told the post was already reviewed
   (Invariant 12).
3. The review message is **never deleted**: :func:`announce_processing`,
   :func:`announce_published`, :func:`announce_skipped` and
   :func:`announce_analysis_failed` rewrite its text and drop the buttons, which is what
   leaves the channel usable as an audit trail.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from app import repository, telegram_notifier
from app.formatting import escape_html, importance_label
from app.models import PostRecord, ReviewDecision
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Callback payloads. Telegram caps callback_data at 64 bytes, so they stay this short.
CALLBACK_APPROVE_PREFIX = "approve:"
CALLBACK_REJECT_PREFIX = "reject:"

APPROVE_BUTTON_TEXT = "✅ تأیید"
REJECT_BUTTON_TEXT = "❌ عدم تأیید"

NEW_POST_HEADING = "🆕 پست جدید برای بررسی"
# What the admin sees once the decision is in, and after the public channel got it.
PROCESSING_TEXT = "✅ تأیید شد\n📤 در حال پردازش..."
PUBLISHED_TEXT = "✅ تأیید شد\n📤 با موفقیت در کانال عمومی منتشر شد"
REJECTED_TEXT = "❌ این پست تأیید نشد."

# The raw RSS text is shown to the human, so it does not need to be complete: a couple of
# paragraphs are enough to judge the post, and Telegram's 4096-char limit is a real cap.
REVIEW_CONTENT_LIMIT = 1200

NOT_ADMIN_TEXT = "این عملیات فقط برای ادمین‌هاست."
ALREADY_REVIEWED_TEXT = "این پست قبلاً بررسی شده است."
UNKNOWN_BUTTON_TEXT = "دکمهٔ ناشناخته"

STATUS_REASONS_FA: dict[str, str] = {
    "skipped_irrelevant": "نامرتبط تشخیص داده شد",
    "skipped_duplicate": "تکراری با یک پست قبلی است",
    "skipped_low_importance": "اهمیت آن زیر آستانهٔ ارسال است",
}


def review_keyboard(post_id: int) -> dict[str, object]:
    """The two inline buttons of one review message (FR-12)."""
    return {
        "inline_keyboard": [
            [
                {"text": APPROVE_BUTTON_TEXT, "callback_data": f"{CALLBACK_APPROVE_PREFIX}{post_id}"},
                {"text": REJECT_BUTTON_TEXT, "callback_data": f"{CALLBACK_REJECT_PREFIX}{post_id}"},
            ]
        ]
    }


_CALLBACK_PREFIXES: tuple[tuple[str, ReviewDecision], ...] = (
    (CALLBACK_APPROVE_PREFIX, "approved"),
    (CALLBACK_REJECT_PREFIX, "rejected"),
)


def parse_callback_data(data: str) -> tuple[ReviewDecision, int] | None:
    """``"approve:42"`` -> ``("approved", 42)``; anything else -> ``None`` (pure)."""
    for prefix, decision in _CALLBACK_PREFIXES:
        if data.startswith(prefix):
            raw_id = data[len(prefix):].strip()
            if raw_id.isdigit() and int(raw_id) > 0:
                return decision, int(raw_id)
    return None


def is_admin(user_id: str | None) -> bool:
    """FR-12 — only listed Telegram users may decide; an empty list allows nobody."""
    admin_ids = get_settings().admin_ids
    return bool(admin_ids) and user_id is not None and user_id in admin_ids


def format_review_message(post: PostRecord, *, topic_name: str | None = None) -> str:
    """Render the pre-LLM review message: enough for a human to say yes or no (FR-12)."""
    topic = topic_name or post.topic or post.source_topic_key
    lines = [
        NEW_POST_HEADING,
        f"📌 <b>{escape_html(post.title)}</b>",
        f"r/{escape_html(post.subreddit)} • {escape_html(topic)} • 🆔 <code>{post.id}</code>",
    ]
    if post.author:
        lines.append(f"✍️ u/{escape_html(post.author)}")
    if post.posted_at:
        lines.append(f"🕒 {post.posted_at:%Y-%m-%d %H:%M} UTC")

    content = (post.raw_content or "").strip()
    if content:
        # Escape first, then cut: cutting HTML could split an entity.
        lines.append(escape_html(content)[:REVIEW_CONTENT_LIMIT])

    lines.append(f"🔗 {escape_html(post.url)}")
    return "\n\n".join(lines)


def dispatch_pending_reviews(topic_names: Mapping[str, str] | None = None) -> int:
    """Send every stored-but-undelivered post to the review channel, one message each.

    A post whose message could not be sent stays ``new`` and is retried by the next run
    (the same crash-recovery idea as FR-11). Returns how many messages went out.
    """
    new_posts = repository.fetch_new_reviews()
    if not new_posts:
        return 0

    channel_id = get_settings().telegram_review_channel_id
    if not channel_id:
        # Fail loudly instead of sending to nowhere: without a review channel no post can
        # ever be approved, so the fetched posts would pile up as `new` forever.
        logger.error(
            "TELEGRAM_REVIEW_CHANNEL_ID is not set; %d post(s) cannot be reviewed",
            len(new_posts),
        )
        return 0

    sent = 0

    for post in new_posts:
        if post.id is None:  # pragma: no cover - a stored row always has an id
            continue
        try:
            message_id = telegram_notifier.send_message(
                format_review_message(post, topic_name=_topic_name(post, topic_names)),
                chat_id=channel_id,
                reply_markup=review_keyboard(post.id),
            )
            if message_id is None:
                logger.error(
                    "Review message for reddit_id=%s was not delivered; it stays 'new'",
                    post.reddit_id,
                )
                continue
            repository.mark_review_dispatched(
                post.id, channel_id=channel_id, message_id=message_id
            )
            sent += 1
        except Exception:
            # Invariant 8: one bad post must not stop the others.
            logger.exception(
                "Unexpected failure while dispatching reddit_id=%s; continuing", post.reddit_id
            )

    if sent:
        logger.info("Delivered %d post(s) to the review channel", sent)
    return sent


def handle_callback(callback_query: dict[str, object]) -> bool:
    """Apply one ✅/❌ press: decide atomically, answer the client, rewrite the message.

    Returns ``True`` when this call is the one that made the decision.
    """
    callback_id = str(callback_query.get("id") or "")
    data = str(callback_query.get("data") or "")
    admin_id = _user_id(callback_query.get("from"))

    parsed = parse_callback_data(data)
    if parsed is None:
        _answer(callback_id, UNKNOWN_BUTTON_TEXT)
        return False

    decision, post_id = parsed
    if not is_admin(admin_id):
        logger.warning("Ignoring a review decision from a non-admin (id=%s)", admin_id)
        _answer(callback_id, NOT_ADMIN_TEXT)
        return False

    if not repository.decide_review(post_id, decision=decision, admin_id=admin_id or ""):
        _answer(callback_id, ALREADY_REVIEWED_TEXT)
        return False

    _answer(callback_id, APPROVE_BUTTON_TEXT if decision == "approved" else REJECT_BUTTON_TEXT)

    post = repository.fetch_post(post_id)
    if post is None:  # pragma: no cover - the row was just updated
        return True
    update_review_message(post, PROCESSING_TEXT if decision == "approved" else REJECTED_TEXT)
    return True


def update_review_message(post: PostRecord, text: str) -> bool:
    """Rewrite the review message and take its buttons away (FR-12: never delete it)."""
    if not post.private_channel_id or not post.private_message_id:
        logger.warning(
            "reddit_id=%s has no review message to update (it was never dispatched)",
            post.reddit_id,
        )
        return False
    return telegram_notifier.edit_message_text(
        text, chat_id=post.private_channel_id, message_id=post.private_message_id
    )


def announce_processing(post: PostRecord) -> bool:
    """The admin's ✅ was recorded; the LLM is about to run (FR-13)."""
    return update_review_message(post, PROCESSING_TEXT)


def announce_published(post: PostRecord) -> bool:
    """The post reached the public channel; the review message becomes the proof."""
    return update_review_message(post, PUBLISHED_TEXT)


def announce_skipped(post: PostRecord, *, status: str) -> bool:
    """The admin said yes but the LLM gates said no — say why, leave the record."""
    reason = STATUS_REASONS_FA.get(status, "شرایط انتشار را نداشت")
    return update_review_message(post, f"✅ تأیید شد\n⛔️ منتشر نشد: {reason}")


def announce_analysis_failed(post: PostRecord) -> bool:
    """The model answer was unusable, so nothing was published (Invariant 3)."""
    return update_review_message(
        post, "✅ تأیید شد\n⚠️ تحلیل ناموفق بود؛ برای بررسی به لاگ مراجعه کنید."
    )


def describe_decision(post: PostRecord) -> str:
    """One-line summary for the worker log: who decided, what, and how important it was."""
    return (
        f"reddit_id={post.reddit_id} review={post.review_status} "
        f"by={post.reviewed_by} importance={importance_label(post.importance)}"
    )


def _topic_name(post: PostRecord, topic_names: Mapping[str, str] | None) -> str | None:
    if not topic_names:
        return None
    return topic_names.get(post.source_topic_key)


def _user_id(sender: object) -> str | None:
    """Telegram's numeric user id as a string, from a ``from`` object."""
    if not isinstance(sender, dict):
        return None
    raw = sender.get("id")
    return None if raw is None else str(raw)


def _answer(callback_query_id: str, text: str) -> None:
    if callback_query_id:
        telegram_notifier.answer_callback_query(callback_query_id, text=text)


__all__ = [
    "ALREADY_REVIEWED_TEXT",
    "announce_analysis_failed",
    "announce_processing",
    "announce_published",
    "announce_skipped",
    "describe_decision",
    "dispatch_pending_reviews",
    "format_review_message",
    "handle_callback",
    "is_admin",
    "parse_callback_data",
    "review_keyboard",
    "update_review_message",
]
