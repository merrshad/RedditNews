"""Thin wrapper over the OpenAI-compatible SDK (NFR-2, provider-agnostic).

Only raw HTTP/SDK work lives here; prompt building and output interpretation live in
``analyzer.py`` (AGENTS.md section 12 — separation of concerns).
"""

from __future__ import annotations

import logging

from openai import OpenAI

from app.retry import call_with_retries

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 120.0


class LlmError(RuntimeError):
    """Raised when the LLM call fails after all attempts."""


class LlmClient:
    """Single call: system prompt + user prompt -> raw text response."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._max_retries = max_retries

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the raw assistant message; retries are handled by ``retry.py``."""

        def _call() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
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
                attempts=self._max_retries,
                description=f"LLM completion (model={self._model})",
            )
        except Exception as exc:
            # Never log the API key: only the error type and message of the provider.
            raise LlmError(f"{type(exc).__name__}: {exc}") from exc
