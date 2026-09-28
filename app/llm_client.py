"""The one raw LLM call (OpenAI-compatible SDK) — transport only (section 8, NFR-2).

``chat_completion`` builds the client from the configured ``settings``, sends the two
messages and returns the assistant text. Parsing/validation of that text is *not* done
here — that belongs to ``analyzer.py`` (AGENTS.md section 12, separation of concerns).
"""

from __future__ import annotations

import logging

from openai import OpenAI

from app.retry import call_with_retries
from app.settings import get_settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 120.0


class LlmError(RuntimeError):
    """Raised when the LLM call does not succeed within the attempt budget."""


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    """Send one system+user prompt and return the raw text of the first choice.

    The whole call runs behind the shared retry/backoff helper (Invariant 8, NFR-2).
    Provider errors are re-raised as :class:`LlmError` with only the error type/message,
    never the API key (Invariant 6, NFR-4).
    """
    settings = get_settings()
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    def _call() -> str:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not response.choices:
            raise LlmError("LLM returned no choices")
        return (response.choices[0].message.content or "").strip()

    try:
        return call_with_retries(
            _call,
            attempts=settings.http_max_retries,
            description=f"LLM chat completion (model={settings.openai_model})",
        )
    except Exception as exc:
        # The model name is configuration, not a secret; the provider message is kept
        # short so a request URL (which may embed the key) can never reach a log line.
        raise LlmError(f"{type(exc).__name__}: {exc}") from exc
