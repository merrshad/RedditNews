"""`llm_client` transport tests: attempt budget, error shape and log hygiene.

`analyzer` tests fake `chat_completion` entirely, so this is the only place that covers
the retry policy of the real call. It matters because a *permanent* provider rejection
(an invalid key, an unknown model) used to consume the whole retry budget with backoff
sleeps — measured in a real phase-4 run at ~6s per post instead of ~1.5s.
"""

from __future__ import annotations

import os
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


def test_the_failure_after_the_last_attempt_keeps_the_provider_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator needs to know *why* it failed, without the key leaking (Invariant 6).

    A transient provider outage is the case ``pipeline`` relies on: `analyze` raises, no row
    is written, and the next run fetches and analyses the post again (AGENTS.md section 4.1).
    """
    calls = _fake_client(monkeypatch, _ProviderError(503))
    monkeypatch.setenv("HTTP_MAX_RETRIES", "2")

    with pytest.raises(llm_client.LlmError) as excinfo:
        llm_client.chat_completion("system", "user")

    assert len(calls) == 2  # the configured budget, not more and not fewer
    message = str(excinfo.value)
    assert "_ProviderError" in message  # the exception type is preserved
    assert "503" in message  # and the provider's status
    assert os.environ["OPENAI_API_KEY"] not in message  # Invariant 6: never the key


def test_a_single_configured_attempt_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR-7: ``HTTP_MAX_RETRIES=1`` really means one call."""
    calls = _fake_client(monkeypatch, _ProviderError(500))
    monkeypatch.setenv("HTTP_MAX_RETRIES", "1")

    with pytest.raises(llm_client.LlmError):
        llm_client.chat_completion("system", "user")

    assert len(calls) == 1


def test_an_answer_without_choices_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that answers with an empty ``choices`` array must not return "".

    Returning an empty string would reach `analyzer` and become a `failed` row with a
    confusing "not valid JSON" reason; failing here names the real problem instead.
    """

    class _Response:
        choices: list[Any] = []

    class _Completions:
        def create(self, **kwargs: Any) -> _Response:
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = _Chat()

    monkeypatch.setattr(llm_client, "OpenAI", _Client)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "1")

    with pytest.raises(llm_client.LlmError, match="no choices"):
        llm_client.chat_completion("system", "user")


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
