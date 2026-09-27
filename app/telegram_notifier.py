"""One-way Telegram Bot API broadcast (`sendMessage`) — no bot framework (YAGNI).

Invariant 6: the bot token must never reach a log line, so every error raised here is
sanitized instead of forwarding httpx's message (which embeds the request URL).
"""

from __future__ import annotations

import logging

import httpx

from app.retry import call_with_retries

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 30.0


class TelegramError(RuntimeError):
    """Telegram rejected or could not receive the message."""


class TelegramNotifier:
    """Sends formatted messages to the configured chat/channel (FR-10)."""

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        max_retries: int,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._max_retries = max_retries
        self._timeout = timeout

    def send_message(self, text: str, *, disable_notification: bool = False) -> None:
        """Send one HTML-formatted message; raises :class:`TelegramError` on failure."""

        def _send() -> None:
            response = httpx.post(
                f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                    "disable_notification": disable_notification,
                },
                timeout=self._timeout,
            )
            description = "unknown"
            try:
                body = response.json()
            except ValueError:
                body = {}
            if isinstance(body, dict):
                description = str(body.get("description") or description)

            if response.status_code >= 400 or not (isinstance(body, dict) and body.get("ok")):
                raise TelegramError(
                    f"sendMessage rejected (HTTP {response.status_code}, description: {description})"
                )

        def _guarded_send() -> None:
            try:
                _send()
            except TelegramError:
                raise
            except httpx.HTTPError as exc:
                # httpx embeds the request URL (and therefore the token) in its message.
                raise TelegramError(f"sendMessage transport failure ({type(exc).__name__})") from exc

        call_with_retries(
            _guarded_send,
            attempts=self._max_retries,
            description=f"telegram sendMessage to chat {self._chat_id}",
        )
        logger.info("Telegram message sent to chat %s (%d chars)", self._chat_id, len(text))
