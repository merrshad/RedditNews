"""Telegram notifier tests — ``httpx`` is faked, so nothing here touches the network (NFR-8).

The contract under test (FR-10/FR-11): ``send_message`` answers ``True`` only when
Telegram accepted the message, reports an ordinary failure as ``False`` after the
``HTTP_MAX_RETRIES`` budget is spent, and never lets the bot token reach a log line
(Invariant 6).
"""

from __future__ import annotations

import logging
import time

import httpx
import pytest

from app import telegram_notifier

TOKEN = "123456:AASECRET-TELEGRAM-TOKEN"
CHAT_ID = "-100987654"
MESSAGE_TEXT = "<b>عنوان پست</b>\n\n📝 خلاصه:\nخلاصه فارسی پست.\n\n🔗 لینک"


class FakeResponse:
    """The parts of ``httpx.Response`` the notifier reads (status + JSON body)."""

    def __init__(
        self, status_code: int = 200, body: object = None, *, valid_json: bool = True
    ) -> None:
        self.status_code = status_code
        self._body = {"ok": True, "result": {"message_id": 1}} if body is None else body
        self._valid_json = valid_json

    def json(self) -> object:
        if not self._valid_json:
            raise ValueError("response body is not JSON")
        return self._body


class FakeHttp:
    """Records every attempt and replays queued outcomes (a response or an exception)."""

    def __init__(self, default: object | None = None) -> None:
        self.default: object = FakeResponse() if default is None else default
        self.queue: list[object] = []
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, *, json: dict[str, object], timeout: float) -> object:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        outcome = self.queue.pop(0) if self.queue else self.default
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def attempts(self) -> int:
        """How many HTTP attempts were made (the retry budget is readable from this)."""
        return len(self.calls)


@pytest.fixture(autouse=True)
def _telegram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known token/chat and retry budget; ``conftest`` clears the settings cache."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)
    monkeypatch.setenv("HTTP_MAX_RETRIES", "3")
    # `retryable` resolves `time.sleep` at call time, so tests never really wait.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
    """Replace ``httpx.post`` for the duration of one test."""
    fake = FakeHttp()
    monkeypatch.setattr(telegram_notifier.httpx, "post", fake)
    return fake


# --- success ----------------------------------------------------------------------


def test_send_message_returns_true_and_posts_the_expected_payload(http: FakeHttp) -> None:
    assert telegram_notifier.send_message(MESSAGE_TEXT) is True

    assert http.attempts == 1
    call = http.calls[0]
    assert call["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert call["json"] == {
        "chat_id": CHAT_ID,
        "text": MESSAGE_TEXT,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    assert call["timeout"] == telegram_notifier.REQUEST_TIMEOUT_SECONDS


# --- ordinary failures become False, never an exception ---------------------------


@pytest.mark.parametrize("status_code", [400, 401, 404, 429, 500, 502])
def test_send_message_returns_false_after_retries_when_telegram_rejects(
    http: FakeHttp, status_code: int
) -> None:
    http.default = FakeResponse(
        status_code=status_code, body={"ok": False, "description": "Bad Request: chat not found"}
    )

    # No exception may escape: the pipeline decides 'sent' vs keep for the next run.
    assert telegram_notifier.send_message(MESSAGE_TEXT) is False
    assert http.attempts == 3  # HTTP_MAX_RETRIES


def test_send_message_returns_false_when_the_body_reports_not_ok(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=200, body={"ok": False, "description": "chat not found"})

    assert telegram_notifier.send_message(MESSAGE_TEXT) is False
    assert http.attempts == 3


def test_send_message_returns_false_for_a_non_json_response(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=502, valid_json=False)

    assert telegram_notifier.send_message(MESSAGE_TEXT) is False
    assert http.attempts == 3


def test_send_message_returns_false_when_the_transport_keeps_failing(http: FakeHttp) -> None:
    http.default = httpx.ConnectError(
        f"Failed to connect to api.telegram.org/bot{TOKEN}/sendMessage"
    )

    assert telegram_notifier.send_message(MESSAGE_TEXT) is False
    assert http.attempts == 3


def test_the_attempt_budget_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp
) -> None:
    """NFR-7: `HTTP_MAX_RETRIES` is read at call time, not baked in at import."""
    monkeypatch.setenv("HTTP_MAX_RETRIES", "1")
    http.default = FakeResponse(status_code=500)

    assert telegram_notifier.send_message(MESSAGE_TEXT) is False
    assert http.attempts == 1


def test_a_failed_send_is_logged_so_the_post_can_be_retried(http: FakeHttp, caplog) -> None:
    http.default = FakeResponse(
        status_code=400, body={"ok": False, "description": "Bad Request: chat not found"}
    )

    with caplog.at_level(logging.ERROR, logger="app.telegram_notifier"):
        assert telegram_notifier.send_message(MESSAGE_TEXT) is False

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert errors
    assert "chat not found" in errors[0].getMessage()


def test_an_unexpected_error_is_not_swallowed(http: FakeHttp) -> None:
    """Only *ordinary* Telegram failures become False; a programming error must surface."""
    http.default = TypeError("bug in the payload builder")

    with pytest.raises(TypeError):
        telegram_notifier.send_message(MESSAGE_TEXT)


# --- Invariant 6: the token never reaches a log line ------------------------------


@pytest.mark.parametrize(
    "outcome",
    [
        FakeResponse(),
        FakeResponse(status_code=400, body={"ok": False, "description": "Bad Request"}),
        FakeResponse(status_code=500, valid_json=False),
        httpx.ConnectError(f"Failed to connect to api.telegram.org/bot{TOKEN}/sendMessage"),
    ],
    ids=["success", "rejected", "non-json", "transport"],
)
def test_the_token_never_reaches_a_log_line(http: FakeHttp, caplog, outcome: object) -> None:
    http.default = outcome
    caplog.set_level(logging.DEBUG)

    telegram_notifier.send_message(MESSAGE_TEXT)

    # httpx puts the token-bearing URL in its own messages, so this is a real check.
    assert TOKEN not in caplog.text


def test_the_retry_log_names_the_target_chat(http: FakeHttp, caplog) -> None:
    http.default = FakeResponse(status_code=500)

    with caplog.at_level(logging.DEBUG):
        telegram_notifier.send_message(MESSAGE_TEXT)

    assert any(CHAT_ID in record.getMessage() for record in caplog.records)


# --- the 4096-character limit is formatting.py's problem, not this module's --------


def test_an_over_long_message_is_warned_about_but_sent_unchanged(
    http: FakeHttp, caplog
) -> None:
    over_long = "ا" * (telegram_notifier.TELEGRAM_MESSAGE_LIMIT + 1)

    with caplog.at_level(logging.WARNING, logger="app.telegram_notifier"):
        assert telegram_notifier.send_message(over_long) is True

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(telegram_notifier.TELEGRAM_MESSAGE_LIMIT) in warnings[0].getMessage()
    # Truncating is formatting.py's job (FR-10): the text goes out untouched.
    assert http.calls[0]["json"]["text"] == over_long


def test_a_message_at_the_limit_is_not_warned_about(http: FakeHttp, caplog) -> None:
    at_limit = "x" * telegram_notifier.TELEGRAM_MESSAGE_LIMIT

    with caplog.at_level(logging.WARNING, logger="app.telegram_notifier"):
        assert telegram_notifier.send_message(at_limit) is True

    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []
