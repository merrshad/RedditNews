"""`llm_client` transport tests: attempt budget, error shape and log hygiene.

`analyzer` tests fake `chat_completion` entirely, so this is the only place that covers
the retry policy of the real call. It matters because a *permanent* provider rejection
(an invalid key, an unknown model) used to consume the whole retry budget with backoff
sleeps — measured in a real phase-4 run at ~6s per post instead of ~1.5s.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from app import llm_client
from app.retry import PermanentError


class _ProviderError(Exception):
    """What the OpenAI SDK raises for a non-2xx answer: it carries ``status_code``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


def _fake_client(monkeypatch: pytest.MonkeyPatch, error: Exception) -> list[int]:
    """Point `llm_client.OpenAI` at a client whose only job is to raise ``error``."""
    calls: list[int] = []

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(1)
            raise error

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = _Chat()

    monkeypatch.setattr(llm_client, "OpenAI", _Client)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")
    return calls


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_a_rejected_request_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    calls = _fake_client(monkeypatch, _ProviderError(status_code))

    with pytest.raises(PermanentError):
        llm_client.chat_completion("system", "user")

    assert len(calls) == 1  # the key will not become valid on the second attempt


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_a_temporary_provider_problem_is_still_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    calls = _fake_client(monkeypatch, _ProviderError(status_code))

    with pytest.raises(llm_client.LlmError):  # ordinary failure, wrapped as before
        llm_client.chat_completion("system", "user")

    assert len(calls) == 3  # HTTP_MAX_RETRIES


def test_the_returned_text_is_the_first_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Message:
        content = "  the answer  "

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs: Any) -> _Response:
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = _Chat()

    monkeypatch.setattr(llm_client, "OpenAI", _Client)

    assert llm_client.chat_completion("system", "user") == "the answer"
