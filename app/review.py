"""The human review gate: private channel in, admin decision, message lifecycle (FR-12).

Nothing here calls the LLM. This module only moves a stored post through the *human*
part of the flow:

1. :func:`dispatch_pending_reviews` sends every post that was just stored (``new``) to the
   private review channel — one message per post, each with ✅/❌ buttons (FR-12). The
   messages go out concurrently, bounded by :data:`MAX_CONCURRENT_REVIEW_MESSAGES`.
2. :func:`handle_callback` applies a button press. The state change is one conditional
   SQL statement (``repository.decide_review``), so of two admins clicking at the same
   time exactly one wins and the other is told the post was already reviewed
   (Invariant 12).
3. The review message **keeps its original text forever**. Only its keyboard changes:
   before a decision it carries ✅/❌, afterwards a status button that names the admin who
   decided (and, as the post moves on, what happened to it). That is what leaves the
   channel usable as an audit trail, and it is what the phase-6 fix is about: editing the
   *text* used to replace the whole post with a one-line status, which read as if the post
   had been deleted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

from app import repository, telegram_notifier
from app.formatting import escape_html, importance_label, truncate
from app.models import PostRecord, ReviewDecision
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Callback payloads. Telegram caps callback_data at 64 bytes, so they stay this short.
# Three namespaces, all owned by this module: the two decisions and the status button the
# decided message keeps (app/admin.py uses the separate `panel:` namespace).
CALLBACK_APPROVE_PREFIX = "approve:"
CALLBACK_REJECT_PREFIX = "reject:"
CALLBACK_STATUS_PREFIX = "status:"

CallbackKind = Literal["approve", "reject", "status"]

APPROVE_BUTTON_TEXT = "✅ تأیید"
REJECT_BUTTON_TEXT = "❌ عدم تأیید"

# What the decided message shows instead of the two buttons.
APPROVED_STATUS_BUTTON_TEXT = "✅ تأیید شد"
REJECTED_STATUS_BUTTON_TEXT = "❌ تأیید نشد"
PROCESSING_BUTTON_TEXT = "⏳ در حال پردازش…"
PUBLISHING_BUTTON_TEXT = "⏳ در حال انتشار…"
PUBLISHED_BUTTON_TEXT = "📤 در کانال عمومی منتشر شد"
FAILED_BUTTON_TEXT = "⚠️ تحلیل ناموفق بود"

NEW_POST_HEADING = "🆕 پست جدید برای بررسی"

# Every field of the review message is printed with its own label, so a missing value reads
# as "not available" instead of silently disappearing (phase 6).
UNKNOWN_VALUE = "نامشخص"
TITLE_LABEL = "📌 <b>عنوان:</b>"
SUBREDDIT_LABEL = "🌐 <b>سابردیت:</b>"
TOPIC_LABEL = "🏷 <b>موضوع:</b>"
AUTHOR_LABEL = "✍️ <b>نویسنده:</b>"
POSTED_AT_LABEL = "🕒 <b>زمان انتشار:</b>"
CONTENT_LABEL = "📝 <b>متن پست:</b>"
LINK_LABEL = "🔗 <b>لینک پست:</b>"

# The raw RSS text is shown to the human, so it does not need to be complete: a couple of
# paragraphs are enough to judge the post, and Telegram's 4096-char limit is a real cap.
REVIEW_CONTENT_LIMIT = 1200
# A headline is unbounded in RSS; the review message only has room for a readable one.
REVIEW_TITLE_LIMIT = 300

# How many review messages may be in flight at once. Telegram throttles a burst, and the
# 429 path waits the announced `retry_after`, so a small window is both faster than one
# message at a time and safer than all of them at once (NFR-6).
MAX_CONCURRENT_REVIEW_MESSAGES = 4

NOT_ADMIN_TEXT = "این عملیات فقط برای ادمین‌هاست."
ALREADY_REVIEWED_TEXT = "این پست قبلاً بررسی شده است."
UNKNOWN_BUTTON_TEXT = "دکمهٔ ناشناخته"
UNKNOWN_POST_TEXT = "این پست پیدا نشد."

STATUS_REASONS_FA: dict[str, str] = {
    "skipped_irrelevant": "نامرتبط تشخیص داده شد",
    "skipped_duplicate": "تکراری با یک پست قبلی است",
    "skipped_low_importance": "اهمیت آن زیر آستانهٔ ارسال است",
}

# The prefix an admin press resolves to, and the state it writes.
_DECISION_BY_KIND: dict[str, ReviewDecision] = {"approve": "approved", "reject": "rejected"}
_CALLBACK_PREFIXES: tuple[tuple[str, CallbackKind], ...] = (
    (CALLBACK_APPROVE_PREFIX, "approve"),
    (CALLBACK_REJECT_PREFIX, "reject"),
    (CALLBACK_STATUS_PREFIX, "status"),
)


class Callback(NamedTuple):
    """A parsed callback payload: which button, and which post it belongs to."""

    kind: CallbackKind
    post_id: int


# --- keyboards ---------------------------------------------------------------------


def review_keyboard(post_id: int) -> dict[str, Any]:
    """The two inline buttons of a post that still waits for a decision (FR-12)."""
    return {
        "inline_keyboard": [
            [
                {
                    "text": APPROVE_BUTTON_TEXT,
                    "callback_data": f"{CALLBACK_APPROVE_PREFIX}{post_id}",
                },
                {
                    "text": REJECT_BUTTON_TEXT,
                    "callback_data": f"{CALLBACK_REJECT_PREFIX}{post_id}",
                },
            ]
        ]
    }


def decision_maker(post: PostRecord) -> str:
    """Who decided, by human name when Telegram gave one (phase 6).

    The id is always stored (``reviewed_by``); the display name is what the status button
    shows, and it falls back to the id so an old row never renders empty.
    """
    return post.reviewed_by_name or post.reviewed_by or UNKNOWN_VALUE


def decision_button_text(post: PostRecord) -> str:
    """The status button label: the decision *and* who made it."""
    label = (
        APPROVED_STATUS_BUTTON_TEXT
        if post.review_status == "approved"
        else REJECTED_STATUS_BUTTON_TEXT
    )
    return f"{label} — {decision_maker(post)}"


def _outcome_button_text(post: PostRecord) -> str:
    """The second status row: what happened to the post after the decision (or "")."""
    if post.status == "sent":
        return PUBLISHED_BUTTON_TEXT
    if post.status == "approved":
        return PROCESSING_BUTTON_TEXT
    if post.status in {"to_send", "publishing"}:
        return PUBLISHING_BUTTON_TEXT
    if post.status == "failed":
        return FAILED_BUTTON_TEXT
    reason = STATUS_REASONS_FA.get(post.status)
    return f"⛔️ منتشر نشد: {reason}" if reason else ""


def keyboard_for(post: PostRecord) -> dict[str, Any]:
    """The keyboard the review message should carry right now (FR-12).

    Still undecided -> the ✅/❌ pair. Decided -> a status button naming the admin, plus a
    second row describing the outcome. Both rows point at the ``status:`` namespace, so
    pressing one only asks for the decision again; it never re-opens the decision.
    """
    if post.id is None:  # pragma: no cover - only stored rows are ever dispatched
        raise ValueError("refusing to build a keyboard for a post that was never stored")
    if post.review_status == "pending_review":
        return review_keyboard(post.id)

    callback_data = f"{CALLBACK_STATUS_PREFIX}{post.id}"
    rows = [[{"text": decision_button_text(post), "callback_data": callback_data}]]
    outcome = _outcome_button_text(post)
    if outcome:
        rows.append([{"text": outcome, "callback_data": callback_data}])
    return {"inline_keyboard": rows}


# --- parsing and authorisation -----------------------------------------------------


def parse_callback_data(data: str) -> Callback | None:
    """``"approve:42"`` -> ``Callback("approve", 42)``; anything else -> ``None`` (pure)."""
    for prefix, kind in _CALLBACK_PREFIXES:
        if data.startswith(prefix):
            raw_id = data[len(prefix) :].strip()
            if raw_id.isdigit() and int(raw_id) > 0:
                return Callback(kind=kind, post_id=int(raw_id))
    return None


def is_admin(user_id: str | None) -> bool:
    """FR-12 — only listed Telegram users may decide; an empty list allows nobody."""
    admin_ids = get_settings().admin_ids
    return bool(admin_ids) and user_id is not None and user_id in admin_ids


# --- the review message ------------------------------------------------------------


def _field(label: str, value: str) -> str:
    """One labelled section of the review message."""
    return f"{label} {value}"


def _author_value(post: PostRecord) -> str:
    author = (post.author or "").strip()
    return f"u/{escape_html(author)}" if author else UNKNOWN_VALUE


def _posted_at_value(post: PostRecord) -> str:
    if post.posted_at is None:
        return UNKNOWN_VALUE
    return f"{post.posted_at:%Y-%m-%d %H:%M} UTC"


def _content_value(post: PostRecord) -> str:
    """The body text, cut first and escaped afterwards (cutting HTML can split an entity)."""
    content = (post.raw_content or "").strip()
    if not content:
        return UNKNOWN_VALUE
    return f"<blockquote>{escape_html(truncate(content, REVIEW_CONTENT_LIMIT))}</blockquote>"


def format_review_message(post: PostRecord, *, topic_name: str | None = None) -> str:
    """Render the pre-LLM review message: enough for a human to say yes or no (FR-12).

    Every field is labelled and every field is always present — title, subreddit, topic,
    author, publish time, body text and link — with ``نامشخص`` where the feed gave nothing,
    so an admin never has to guess which line is which.
    """
    topic = topic_name or post.topic or post.source_topic_key
    post_id = UNKNOWN_VALUE if post.id is None else str(post.id)

    sections = [
        f"{NEW_POST_HEADING} — 🆔 <code>{escape_html(post_id)}</code>",
        _field(TITLE_LABEL, f"\n<b>{escape_html(truncate(post.title, REVIEW_TITLE_LIMIT))}</b>"),
        "\n".join(
            (
                _field(SUBREDDIT_LABEL, f"r/{escape_html(post.subreddit or UNKNOWN_VALUE)}"),
                _field(TOPIC_LABEL, escape_html(topic or UNKNOWN_VALUE)),
                _field(AUTHOR_LABEL, _author_value(post)),
                _field(POSTED_AT_LABEL, _posted_at_value(post)),
            )
        ),
        _field(CONTENT_LABEL, f"\n{_content_value(post)}"),
        _field(LINK_LABEL, f"\n{escape_html(post.url or UNKNOWN_VALUE)}"),
    ]
    return "\n\n".join(sections)


async def update_review_message(post: PostRecord) -> bool:
    """Re-draw the *buttons* of a post's review message and leave its text untouched.

    ``editMessageReplyMarkup`` is what makes this exact: the original post body stays
    byte-identical in the channel while the keyboard turns into the decision status. The
    message is never deleted and never rewritten, so the channel is a real audit trail
    (FR-12 / Invariant 13).
    """
    if not post.private_channel_id or not post.private_message_id:
        logger.warning(
            "reddit_id=%s has no review message to update (it was never dispatched)",
            post.reddit_id,
        )
        return False
    return await telegram_notifier.edit_message_reply_markup(
        chat_id=post.private_channel_id,
        message_id=post.private_message_id,
        reply_markup=keyboard_for(post),
    )


async def announce_processing(post: PostRecord) -> bool:
    """The admin's ✅ was recorded; the LLM is about to run (FR-13)."""
    return await update_review_message(post)


async def announce_published(post: PostRecord) -> bool:
    """The post reached the public channel; the status button says so."""
    return await update_review_message(post.model_copy(update={"status": "sent"}))


async def announce_skipped(post: PostRecord, *, status: str) -> bool:
    """The admin said yes but the LLM gates said no — say why, leave the record."""
    return await update_review_message(post.model_copy(update={"status": status}))


async def announce_analysis_failed(post: PostRecord) -> bool:
    """The model answer was unusable, so nothing was published (Invariant 3)."""
    return await update_review_message(post.model_copy(update={"status": "failed"}))


# --- dispatch ----------------------------------------------------------------------


async def _dispatch_one(post: PostRecord, *, channel_id: str, topic_name: str | None) -> bool:
    """Send one review message and record where it landed. Never raises (Invariant 8)."""
    if post.id is None:  # pragma: no cover - a stored row always has an id
        return False
    try:
        message_id = await telegram_notifier.send_message(
            format_review_message(post, topic_name=topic_name),
            chat_id=channel_id,
            reply_markup=review_keyboard(post.id),
        )
        if message_id is None:
            logger.error(
                "Review message for reddit_id=%s was not delivered; it stays 'new'",
                post.reddit_id,
            )
            return False
        repository.mark_review_dispatched(post.id, channel_id=channel_id, message_id=message_id)
        return True
    except Exception:
        # Invariant 8: one bad post must not stop the others.
        logger.exception(
            "Unexpected failure while dispatching reddit_id=%s; continuing", post.reddit_id
        )
        return False


async def dispatch_pending_reviews(topic_names: Mapping[str, str] | None = None) -> int:
    """Send every stored-but-undelivered post to the review channel, one message each.

    The messages are independent, so they are delivered concurrently (bounded by
    :data:`MAX_CONCURRENT_REVIEW_MESSAGES`) instead of one HTTP round trip after another —
    a 50-post batch used to take a minute of pure waiting. A post whose message could not
    be sent stays ``new`` and is retried by the next run (the same crash-recovery idea as
    FR-11). Returns how many messages went out.
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

    gate = asyncio.Semaphore(MAX_CONCURRENT_REVIEW_MESSAGES)
    names = topic_names or {}

    async def _gated(post: PostRecord) -> bool:
        async with gate:
            return await _dispatch_one(
                post, channel_id=channel_id, topic_name=names.get(post.source_topic_key)
            )

    results = await asyncio.gather(*(_gated(post) for post in new_posts))
    sent = sum(1 for delivered in results if delivered)
    if sent:
        logger.info("Delivered %d post(s) to the review channel", sent)
    return sent


# --- decisions ---------------------------------------------------------------------


async def handle_callback(callback_query: dict[str, object]) -> bool:
    """Apply one ✅/❌ press: decide atomically, answer the client, re-draw the buttons.

    Returns ``True`` when this call is the one that made the decision. A press on the
    status button of an already-decided post only answers with the stored decision.
    """
    callback_id = str(callback_query.get("id") or "")
    data = str(callback_query.get("data") or "")
    sender = callback_query.get("from")
    admin_id = _user_id(sender)

    parsed = parse_callback_data(data)
    if parsed is None:
        await _answer(callback_id, UNKNOWN_BUTTON_TEXT)
        return False

    if not is_admin(admin_id):
        logger.warning("Ignoring a review decision from a non-admin (id=%s)", admin_id)
        await _answer(callback_id, NOT_ADMIN_TEXT)
        return False

    if parsed.kind == "status":
        await _answer(callback_id, _status_alert(repository.fetch_post(parsed.post_id)))
        return False

    decision = _DECISION_BY_KIND[parsed.kind]
    if not repository.decide_review(
        parsed.post_id,
        decision=decision,
        admin_id=admin_id or "",
        admin_name=_admin_name(sender),
    ):
        await _answer(callback_id, ALREADY_REVIEWED_TEXT)
        return False

    await _answer(callback_id, APPROVE_BUTTON_TEXT if decision == "approved" else REJECT_BUTTON_TEXT)

    post = repository.fetch_post(parsed.post_id)
    if post is None:  # pragma: no cover - the row was just updated
        return True
    await update_review_message(post)
    return True


def describe_decision(post: PostRecord) -> str:
    """One-line summary for the worker log: who decided, what, and how important it was."""
    return (
        f"reddit_id={post.reddit_id} review={post.review_status} "
        f"by={decision_maker(post)} importance={importance_label(post.importance)}"
    )


def _status_alert(post: PostRecord | None) -> str:
    """What the status button answers when it is pressed."""
    if post is None:
        return UNKNOWN_POST_TEXT
    if post.review_status == "approved":
        return f"✅ این پست توسط {decision_maker(post)} تأیید شد."
    if post.review_status == "rejected":
        return f"❌ این پست توسط {decision_maker(post)} رد شد."
    return "این پست هنوز بررسی نشده است."


def _user_id(sender: object) -> str | None:
    """Telegram's numeric user id as a string, from a ``from`` object."""
    if not isinstance(sender, dict):
        return None
    raw = sender.get("id")
    return None if raw is None else str(raw)


def _admin_name(sender: object) -> str | None:
    """A human-readable name for the deciding admin, as Telegram reports it (phase 6)."""
    if not isinstance(sender, dict):
        return None
    full_name = " ".join(
        str(sender.get(part) or "").strip() for part in ("first_name", "last_name")
    ).strip()
    if full_name:
        return full_name
    username = str(sender.get("username") or "").strip()
    return f"@{username}" if username else None


async def _answer(callback_query_id: str, text: str) -> None:
    if callback_query_id:
        await telegram_notifier.answer_callback_query(callback_query_id, text=text)


__all__ = [
    "ALREADY_REVIEWED_TEXT",
    "announce_analysis_failed",
    "announce_processing",
    "announce_published",
    "announce_skipped",
    "decision_button_text",
    "decision_maker",
    "describe_decision",
    "dispatch_pending_reviews",
    "format_review_message",
    "handle_callback",
    "is_admin",
    "keyboard_for",
    "parse_callback_data",
    "review_keyboard",
    "update_review_message",
]
