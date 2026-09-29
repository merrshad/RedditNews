"""Intake of Telegram updates: the cursor and the routing (FR-12).

Phase 5 makes the bot two-way, and this is the only place that knows about Telegram's
``update_id`` cursor. The worker loop itself long-polls it (``main.run_forever``), so a
button press is acted on within seconds while the pipeline keeps its own
``POLL_INTERVAL_SECONDS`` schedule — one process, no queue and no threading
(AGENTS.md section 8).

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

# Update kinds the worker listens to: button presses (the review decisions) and plain
# messages (the admin panel, app/admin.py).
ALLOWED_UPDATES: tuple[str, ...] = (INLINE_BUTTON_CALLBACK, PLAIN_MESSAGE)


class PollResult(NamedTuple):
    """What one long-poll produced: the next cursor and how many rows it changed."""

    offset: int | None
    decisions: int


def poll_once(
    offset: int | None = None, *, long_poll_seconds: int = LONG_POLL_SECONDS
) -> PollResult:
    """Fetch pending updates, handle them, and return the next cursor.

    ``decisions`` counts the review decisions that really changed a row, so ``main`` can
    run the analysis/publication step immediately instead of waiting for the next
    scheduled cycle.
    """
    updates = get_updates(
        offset=offset, long_poll_seconds=long_poll_seconds, allowed_updates=ALLOWED_UPDATES
    )

    decisions = 0
    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            # Offset is "give me updates after this id"; advance past what we just saw.
            offset = update_id + 1 if offset is None else max(offset, update_id + 1)
        try:
            if route(update):
                decisions += 1
        except Exception:
            # One malformed update must not end the loop (NFR-2).
            logger.exception("Failed to handle Telegram update %s; continuing", update_id)

    return PollResult(offset=offset, decisions=decisions)


def route(update: dict[str, Any]) -> bool:
    """Hand one update to the module that owns it; anything unknown is ignored."""
    callback_query = update.get("callback_query")
    if isinstance(callback_query, dict):
        # A button press: the review decision and its message update.
        return review.handle_callback(callback_query)

    message = update.get("message")
    if isinstance(message, dict):
        # A chat message: the admin command surface.
        return admin.handle_command(message)

    return False
