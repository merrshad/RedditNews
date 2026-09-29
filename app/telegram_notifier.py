"""The whole Telegram Bot API boundary for this project (AGENTS.md sections 8 and 9).

Still no bot framework (YAGNI): plain HTTP calls through ``httpx``. What changed in
phase 5 is that the bot is no longer one-way — the private review channel carries inline
buttons, so the same module also performs the one inbound call, ``getUpdates``, which
``app.telegram_updates`` long-polls.

Every function in the public API answers with a value instead of raising, so the caller
decides what a failure means: ``None``/``False`` for send/edit/answer, ``[]`` for updates.
Only a genuinely unexpected error propagates.

Invariant 6: the bot token must never reach a log line. The token sits in the request
URL, so failures are described with a sanitised message instead of httpx's own text
(which embeds that URL), and the low-level call is logged by label only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

from app.retry import PermanentError, retryable
from app.settings import get_settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 30.0

# Server-side wait for `getUpdates` (long polling) and the client-side timeout that must
# be a bit larger, or httpx gives up just before Telegram answers.
LONG_POLL_SECONDS = 25
LONG_POLL_HTTP_TIMEOUT_SECONDS = 35.0

# Telegram answers 429 when it throttles us (worth waiting for) and 400/401/403 when the
# request itself is wrong (bad token, unknown chat, malformed HTML — never worth it).
RATE_LIMIT_STATUS = 429

# A 429 carries how long to wait (`parameters.retry_after`, and usually the `Retry-After`
# header). Honouring it is what makes a burst of review messages actually land: a real run
# pushing 25 messages to one channel got `retry after 30`, and the plain 1s/2s backoff
# could not cover that, so 3 posts were left for the next cycle. Capped for the same reason
# as the RSS side — one throttled call must not stall a whole cycle for minutes (NFR-2).
MAX_RETRY_AFTER_SECONDS = 30.0

# Telegram's hard limit for a single message. Truncating is formatting.py's job
# (FR-10), so this module only reports the anomaly (see :func:`send_message`).
TELEGRAM_MESSAGE_LIMIT = 4096

# An inline keyboard with zero rows: what Telegram expects when the buttons must go while
# the message itself stays (FR-12 — the review message is the audit trail, never deleted).
REMOVE_KEYBOARD: dict[str, Any] = {"inline_keyboard": []}

INLINE_BUTTON_CALLBACK = "callback_query"
PLAIN_MESSAGE = "message"


class TelegramError(RuntimeError):
    """Telegram rejected the call or could not be reached (never carries the token)."""


class TelegramPermanentError(TelegramError, PermanentError):
    """Telegram said the request itself is wrong, so no attempt can succeed (phase 4).

    Still a :class:`TelegramError`, which keeps the public contract intact: the failure
    becomes ``None``/``False`` — it just stops burning the retry budget and its backoff
    sleeps first. A real run with a rejected token wasted ~3.3s per post (≈80s per
    25-post cycle) retrying an answer that could never change.
    """


def _describe_call(*_args: Any, **_kwargs: Any) -> str:
    """Retry-log label for one attempt: the method name, never the token (Invariant 6)."""
    method = _args[0] if _args else "?"
    return f"telegram {method}"


def _retry_after_seconds(response: httpx.Response, body: dict[str, Any]) -> float:
    """How long Telegram asked us to wait, capped at ``MAX_RETRY_AFTER_SECONDS``.

    The value normally arrives in the JSON body (``parameters.retry_after``) and often in the
    ``Retry-After`` header too; absent or unparsable means ``0.0``, i.e. the normal
    exponential backoff applies.
    """
    parameters = body.get("parameters")
    raw: Any = parameters.get("retry_after") if isinstance(parameters, dict) else None
    if raw is None:
        raw = getattr(response, "headers", {}).get("Retry-After")
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(seconds, MAX_RETRY_AFTER_SECONDS))


@retryable(description=_describe_call)
def _call_once(method: str, payload: dict[str, Any], *, timeout_seconds: float) -> Any:
    """One Bot API call; raises :class:`TelegramError` on any failure.

    ``retryable`` wraps *this* call because only here does a failure raise; the public
    functions below convert it into a value.
    """
    token = get_settings().telegram_bot_token
    try:
        response = httpx.post(
            f"{TELEGRAM_API_BASE}/bot{token}/{method}",
            json=payload,
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        # httpx embeds the request URL (and therefore the token) in its message.
        raise TelegramError(f"{method} transport failure ({type(exc).__name__})") from exc

    try:
        body = response.json()
    except ValueError:
        body = {}

    if not isinstance(body, dict):
        body = {}

    if response.status_code >= 400 or not body.get("ok"):
        description = str(body.get("description") or "unknown")
        message = f"{method} rejected (HTTP {response.status_code}, description: {description})"
        if 400 <= response.status_code < 500 and response.status_code != RATE_LIMIT_STATUS:
            raise TelegramPermanentError(message)
        if response.status_code == RATE_LIMIT_STATUS:
            delay = _retry_after_seconds(response, body)
            if delay:
                logger.warning(
                    "Telegram is throttling %s; waiting %.0fs before trying again",
                    method,
                    delay,
                )
                time.sleep(delay)
        raise TelegramError(message)

    return body.get("result")


def send_message(
    text: str,
    *,
    chat_id: str | None = None,
    reply_markup: dict[str, Any] | None = None,
) -> int | None:
    """Send one HTML message to ``chat_id`` (default: the public channel) — FR-10.

    Returns Telegram's ``message_id`` (the audit value FR-12 wants) or ``None`` when the
    message was not sent. A rejected send is not an exception: the caller keeps the post
    for a later run (FR-11).
    """
    target = chat_id or get_settings().telegram_chat_id

    if len(text) > TELEGRAM_MESSAGE_LIMIT:
        # Truncation is formatting.py's responsibility; here we only report it.
        logger.warning(
            "Telegram message is %d chars, above the %d-char API limit; sending as-is",
            len(text),
            TELEGRAM_MESSAGE_LIMIT,
        )

    payload: dict[str, Any] = {
        "chat_id": target,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    try:
        result = _call_once("sendMessage", payload, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    except TelegramError as exc:  # includes TelegramPermanentError: same answer, no retries
        logger.error("Telegram send failed, keeping the post for a later run: %s", exc)
        return None

    message_id = result.get("message_id") if isinstance(result, dict) else None
    if message_id is None:
        logger.error("Telegram sendMessage answered without a message_id; treating it as failed")
        return None

    logger.info(
        "Telegram message sent to chat %s (%d chars, message_id=%s)",
        target,
        len(text),
        message_id,
    )
    return int(message_id)


def edit_message_text(
    text: str,
    *,
    chat_id: str,
    message_id: int,
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Rewrite a message we already sent, buttons included — FR-12.

    The review message is edited in place and never deleted: ``reply_markup`` defaults to
    :data:`REMOVE_KEYBOARD` so that deciding a post also takes its buttons away.
    """
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
        "reply_markup": REMOVE_KEYBOARD if reply_markup is None else reply_markup,
    }

    try:
        _call_once("editMessageText", payload, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    except TelegramError as exc:
        logger.error("Telegram message edit failed (message_id=%s): %s", message_id, exc)
        return False

    logger.info("Telegram message %s updated", message_id)
    return True


def answer_callback_query(callback_query_id: str, *, text: str | None = None) -> bool:
    """Close the spinner on a pressed button, optionally with a short note (FR-12).

    Telegram requires an answer for every callback query, otherwise the client keeps
    showing a progress indicator; a failure here must never stop the decision itself.
    """
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
        payload["show_alert"] = False

    try:
        _call_once("answerCallbackQuery", payload, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    except TelegramError as exc:
        logger.warning("Could not answer a callback query: %s", exc)
        return False
    return True


def get_updates(
    *,
    offset: int | None = None,
    long_poll_seconds: int = LONG_POLL_SECONDS,
    allowed_updates: Sequence[str] = (INLINE_BUTTON_CALLBACK, PLAIN_MESSAGE),
) -> list[dict[str, Any]]:
    """Long-poll for admin decisions and commands — FR-12/FR-15.

    ``offset`` is Telegram's "give me updates after this id" cursor. A failure (or a
    Telegram outage) answers ``[]`` so the worker loop keeps running (NFR-2): nothing is
    lost, because Telegram keeps undelivered updates for 24 hours and the next poll asks
    for the same offset again.
    """
    payload: dict[str, Any] = {
        "timeout": long_poll_seconds,
        "allowed_updates": list(allowed_updates),
    }
    if offset is not None:
        payload["offset"] = offset

    try:
        result = _call_once(
            "getUpdates", payload, timeout_seconds=LONG_POLL_HTTP_TIMEOUT_SECONDS
        )
    except TelegramError as exc:
        logger.error("Telegram getUpdates failed: %s", exc)
        return []

    if not isinstance(result, list):
        return []
    return [update for update in result if isinstance(update, dict)]
