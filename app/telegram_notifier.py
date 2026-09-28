"""One-way Telegram Bot API broadcast (`sendMessage`) — no bot framework (YAGNI).

Public API: :func:`send_message`. Like ``repository``/``llm_client`` this module is a
single module-level function: it reads the token/chat from :mod:`app.settings`, wraps
the HTTP call with :func:`app.retry.retryable` (NFR-2/Invariant 8) and answers with a
boolean instead of raising, so ``pipeline`` can choose between ``status='sent'`` and
keeping the post as ``to_send`` for the next run (FR-10/FR-11).

Invariant 6: the bot token must never reach a log line, so every failure is described
with a sanitised message instead of forwarding httpx's own (which embeds the token-bearing
request URL).
"""

from __future__ import annotations

import logging

import httpx

from app.retry import retryable
from app.settings import get_settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 30.0

# Telegram's hard limit for a single message. Truncating is formatting.py's job
# (FR-10), so this module only reports the anomaly (see :func:`send_message`).
TELEGRAM_MESSAGE_LIMIT = 4096


class TelegramError(RuntimeError):
    """Telegram rejected the message or could not be reached (never carries the token)."""


def _describe_call(text: str) -> str:
    """Retry-log label for one attempt: the target chat, never the token (Invariant 6)."""
    return f"telegram sendMessage to chat {get_settings().telegram_chat_id}"


@retryable(description=_describe_call)
def _send_message_once(text: str) -> None:
    """One ``sendMessage`` attempt; raises :class:`TelegramError` on any failure.

    ``retryable`` (not a decorator on :func:`send_message`) wraps *this* call because
    only here does a failure raise — the public function converts it into ``False``.
    """
    settings = get_settings()
    try:
        response = httpx.post(
            f"{TELEGRAM_API_BASE}/bot{settings.telegram_bot_token}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        # httpx embeds the request URL (and therefore the token) in its message.
        raise TelegramError(f"sendMessage transport failure ({type(exc).__name__})") from exc

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code >= 400 or not (isinstance(body, dict) and body.get("ok")):
        description = "unknown"
        if isinstance(body, dict):
            description = str(body.get("description") or description)
        raise TelegramError(
            f"sendMessage rejected (HTTP {response.status_code}, description: {description})"
        )


def send_message(text: str) -> bool:
    """Send one HTML-formatted message to the configured chat (FR-10).

    Returns ``True`` only when Telegram accepted the message. An ordinary failure
    (rejected, or unreachable after ``HTTP_MAX_RETRIES`` attempts) is logged and
    reported as ``False`` so the caller keeps the post for the next run (FR-11);
    genuinely unexpected errors still propagate to the pipeline's own guard.
    """
    if len(text) > TELEGRAM_MESSAGE_LIMIT:
        # Truncation is formatting.py's responsibility; here we only report it.
        logger.warning(
            "Telegram message is %d chars, above the %d-char API limit; sending as-is",
            len(text),
            TELEGRAM_MESSAGE_LIMIT,
        )

    try:
        _send_message_once(text)
    except TelegramError as exc:
        logger.error("Telegram send failed, keeping the post for the next run: %s", exc)
        return False

    logger.info(
        "Telegram message sent to chat %s (%d chars)",
        get_settings().telegram_chat_id,
        len(text),
    )
    return True
