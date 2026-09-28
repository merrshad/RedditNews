"""The one raw LLM call (OpenAI-compatible SDK) — transport only (section 8, NFR-2).

``chat_completion`` builds the client from the configured ``settings``, sends the two
messages and returns the assistant text. Parsing/validation of that text is *not* done
here — that belongs to ``analyzer.py`` (AGENTS.md section 12, separation of concerns).
"""

from __future__ import annotations

import logging

from openai import OpenAI

from app.retry import PermanentError, call_with_retries
from app.settings import get_settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 120.0

# Provider statuses that mean "this request will never succeed": an invalid key, an
# unknown model, a malformed body. 429 (throttled) and 5xx are the ones worth retrying.
RATE_LIMIT_STATUS = 429


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
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != RATE_LIMIT_STATUS:
                # A rejected key or an unknown model fails identically every time; measured
                # in a real phase-4 run, retrying it cost ~6s per post instead of ~1.5s.
                raise PermanentError(f"{type(exc).__name__}: {exc}") from exc
            raise
        if not response.choices:
            raise LlmError("LLM returned no choices")
        return (response.choices[0].message.content or "").strip()

    try:
        return call_with_retries(
            _call,
            attempts=settings.http_max_retries,
            description=f"LLM chat completion (model={settings.openai_model})",
        )
    except PermanentError:
        # Already carries the type/status; wrapping it in LlmError would lose that marker.
        raise
    except Exception as exc:
        # The model name is configuration, not a secret; the provider message is kept
        # short so a request URL (which may embed the key) can never reach a log line.
        raise LlmError(f"{type(exc).__name__}: {exc}") from exc
