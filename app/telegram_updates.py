"""Intake of Telegram updates: the cursor and the routing (FR-12, FR-14).

The bot is two-way, and this is the only place that knows about Telegram's ``update_id``
cursor. Since phase 6 the worker has a task of its own for this (``main._updates_loop``),
so updates are consumed continuously while the RSS/analysis cycle runs in the same event
loop — a button press never waits for a cycle to finish.

Telegram keeps undelivered updates for 24 hours, and the cursor only advances once an
update was handled, so a restart or an outage loses nothing.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from app import admin, review
from app.telegram_notifier import (
    INLINE_BUTTON_CALLBACK,
    LONG_POLL_SECONDS,
    PLAIN_MESSAGE,
    get_updates,
)

logger = logging.getLogger(__name__)

# Update kinds the worker listens to: button presses (review decisions and the admin
# panel) and plain messages (which open the panel, or answer its prompt).
ALLOWED_UPDATES: tuple[str, ...] = (INLINE_BUTTON_CALLBACK, PLAIN_MESSAGE)


class PollResult(NamedTuple):
    """What one long-poll produced: the next cursor and how many decisions it applied."""

    offset: int | None
    decisions: int


async def poll_once(
    offset: int | None = None, *, long_poll_seconds: int = LONG_POLL_SECONDS
) -> PollResult:
    """Fetch pending updates, handle them, and return the next cursor.

    ``decisions`` counts the review decisions that really changed a row, so ``main`` can
    run the analysis/publication step immediately instead of waiting for the next
    scheduled cycle.
    """
    updates = await get_updates(
        offset=offset, long_poll_seconds=long_poll_seconds, allowed_updates=ALLOWED_UPDATES
    )

    decisions = 0
    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            # Offset is "give me updates after this id"; advance past what we just saw.
            offset = update_id + 1 if offset is None else max(offset, update_id + 1)
        try:
            if await route(update):
                decisions += 1
        except Exception:
            # One malformed update must not end the loop (NFR-2).
            logger.exception("Failed to handle Telegram update %s; continuing", update_id)

    return PollResult(offset=offset, decisions=decisions)


async def route(update: dict[str, Any]) -> bool:
    """Hand one update to the module that owns it; anything unknown is ignored.

    Two callback namespaces live side by side, so the prefix — not the order of attempts —
    decides who answers: ``panel:`` is the admin panel (FR-14) and everything else is a
    review button (FR-12).
    """
    callback_query = update.get("callback_query")
    if isinstance(callback_query, dict):
        if admin.is_panel_callback(str(callback_query.get("data") or "")):
            return await admin.handle_callback(callback_query)
        return await review.handle_callback(callback_query)

    message = update.get("message")
    if isinstance(message, dict):
        # A chat message: the panel's home screen, or the answer to a pending prompt.
        return await admin.handle_message(message)

    return False
