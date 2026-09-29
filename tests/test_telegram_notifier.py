"""Telegram client tests — ``httpx`` is faked, so nothing here touches the network (NFR-8).

The contract under test: every function answers with a value instead of raising, and the
bot token never reaches a log line (Invariant 6). ``send_message`` (FR-10) returns
Telegram's ``message_id`` or ``None``; ``edit_message_text``/``answer_callback_query``
(FR-12) return a boolean; ``get_updates`` returns a list (empty on failure, so the worker
loop keeps running). An ordinary failure spends the ``HTTP_MAX_RETRIES`` budget; a
rejected request (4xx) does not.
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
        self,
        status_code: int = 200,
        body: object = None,
        *,
        valid_json: bool = True,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = {"ok": True, "result": {"message_id": 1}} if body is None else body
        self._valid_json = valid_json
        self.headers = headers or {}

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


def test_send_message_returns_the_message_id_and_posts_the_expected_payload(http: FakeHttp) -> None:
    assert telegram_notifier.send_message(MESSAGE_TEXT) == 1

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


@pytest.mark.parametrize("status_code", [429, 500, 502])
def test_send_message_returns_false_after_retries_when_telegram_rejects(
    http: FakeHttp, status_code: int
) -> None:
    http.default = FakeResponse(
        status_code=status_code, body={"ok": False, "description": "Too Many Requests"}
    )

    # No exception may escape: the pipeline decides 'sent' vs keep for the next run.
    assert telegram_notifier.send_message(MESSAGE_TEXT) is None
    assert http.attempts == 3  # HTTP_MAX_RETRIES: throttling and 5xx may pass later


# --- 429: honour the wait Telegram announces -------------------------------------------


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture every wait, so a test can tell the announced delay from the backoff."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def _too_many_requests(*, parameters: dict[str, object] | None = None, headers=None) -> FakeResponse:
    body: dict[str, object] = {"ok": False, "description": "Too Many Requests: retry after 30"}
    if parameters is not None:
        body["parameters"] = parameters
    return FakeResponse(status_code=429, body=body, headers=headers)


def test_a_rate_limit_is_waited_out_for_the_announced_time(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed for real: 25 review messages in one burst got `retry after 30` and the plain
    1s/2s backoff could not cover it, so 3 posts were left for the next cycle. Waiting the
    announced time is what makes the batch actually land.
    """
    sleeps = _record_sleeps(monkeypatch)
    http.queue = [_too_many_requests(parameters={"retry_after": 30})]
    http.default = FakeResponse()

    assert telegram_notifier.send_message(MESSAGE_TEXT) == 1
    assert sleeps[0] == 30


def test_the_announced_wait_is_capped(http: FakeHttp, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _record_sleeps(monkeypatch)
    http.queue = [_too_many_requests(parameters={"retry_after": 600})]
    http.default = FakeResponse()

    assert telegram_notifier.send_message(MESSAGE_TEXT) == 1
    assert sleeps[0] == telegram_notifier.MAX_RETRY_AFTER_SECONDS


def test_the_retry_after_header_is_used_when_the_body_has_no_hint(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = _record_sleeps(monkeypatch)
    http.queue = [_too_many_requests(headers={"Retry-After": "12"})]
    http.default = FakeResponse()

    assert telegram_notifier.send_message(MESSAGE_TEXT) == 1
    assert sleeps[0] == 12


def test_without_any_hint_only_the_normal_backoff_applies(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = _record_sleeps(monkeypatch)
    http.queue = [_too_many_requests(), _too_many_requests(headers={"Retry-After": "nonsense"})]
    http.default = FakeResponse()

    assert telegram_notifier.send_message(MESSAGE_TEXT) == 1
    assert all(seconds < 5 for seconds in sleeps)


def test_the_throttle_wait_is_logged_without_the_token(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _record_sleeps(monkeypatch)
    http.queue = [_too_many_requests(parameters={"retry_after": 30})]
    http.default = FakeResponse()

    with caplog.at_level(logging.WARNING):
        assert telegram_notifier.send_message(MESSAGE_TEXT) == 1

    assert "throttling sendMessage" in caplog.text
    assert TOKEN not in caplog.text


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_a_rejected_request_is_not_retried(http: FakeHttp, status_code: int) -> None:
    """A bad token/chat/HTML fails identically every time, so the budget is not spent.

    Measured in a real phase-4 run: retrying a rejected token cost ~3.3s per post, about
    80 seconds of a 25-post cycle, on every cycle.
    """
    http.default = FakeResponse(
        status_code=status_code, body={"ok": False, "description": "Bad Request: chat not found"}
    )

    assert telegram_notifier.send_message(MESSAGE_TEXT) is None
    assert http.attempts == 1


def test_a_rejected_request_is_logged_as_permanent(http: FakeHttp, caplog) -> None:
    """The operator must be able to tell a bad token apart from a flaky network."""
    http.default = FakeResponse(
        status_code=401, body={"ok": False, "description": "Unauthorized"}
    )

    with caplog.at_level(logging.ERROR):
        assert telegram_notifier.send_message(MESSAGE_TEXT) is None

    assert "failed permanently" in caplog.text


def test_send_message_returns_false_when_the_body_reports_not_ok(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=200, body={"ok": False, "description": "chat not found"})

    assert telegram_notifier.send_message(MESSAGE_TEXT) is None
    assert http.attempts == 3


def test_send_message_returns_false_for_a_non_json_response(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=502, valid_json=False)

    assert telegram_notifier.send_message(MESSAGE_TEXT) is None
    assert http.attempts == 3


def test_send_message_returns_false_when_the_transport_keeps_failing(http: FakeHttp) -> None:
    http.default = httpx.ConnectError(
        f"Failed to connect to api.telegram.org/bot{TOKEN}/sendMessage"
    )

    assert telegram_notifier.send_message(MESSAGE_TEXT) is None
    assert http.attempts == 3


def test_the_attempt_budget_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp
) -> None:
    """NFR-7: `HTTP_MAX_RETRIES` is read at call time, not baked in at import."""
    monkeypatch.setenv("HTTP_MAX_RETRIES", "1")
    http.default = FakeResponse(status_code=500)

    assert telegram_notifier.send_message(MESSAGE_TEXT) is None
    assert http.attempts == 1


def test_a_failed_send_is_logged_so_the_post_can_be_retried(http: FakeHttp, caplog) -> None:
    http.default = FakeResponse(
        status_code=400, body={"ok": False, "description": "Bad Request: chat not found"}
    )

    with caplog.at_level(logging.ERROR, logger="app.telegram_notifier"):
        assert telegram_notifier.send_message(MESSAGE_TEXT) is None

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


def test_the_retry_log_names_the_method_but_never_the_token(http: FakeHttp, caplog) -> None:
    """The retry label has to say which call failed without touching the credentials."""
    http.default = FakeResponse(status_code=500)

    with caplog.at_level(logging.DEBUG):
        telegram_notifier.send_message(MESSAGE_TEXT)

    assert any("sendMessage" in record.getMessage() for record in caplog.records)
    assert TOKEN not in caplog.text


# --- the 4096-character limit is formatting.py's problem, not this module's --------


def test_an_over_long_message_is_warned_about_but_sent_unchanged(
    http: FakeHttp, caplog
) -> None:
    over_long = "ا" * (telegram_notifier.TELEGRAM_MESSAGE_LIMIT + 1)

    with caplog.at_level(logging.WARNING, logger="app.telegram_notifier"):
        assert telegram_notifier.send_message(over_long) == 1

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(telegram_notifier.TELEGRAM_MESSAGE_LIMIT) in warnings[0].getMessage()
    # Truncating is formatting.py's job (FR-10): the text goes out untouched.
    assert http.calls[0]["json"]["text"] == over_long


def test_a_message_at_the_limit_is_not_warned_about(http: FakeHttp, caplog) -> None:
    at_limit = "x" * telegram_notifier.TELEGRAM_MESSAGE_LIMIT

    with caplog.at_level(logging.WARNING, logger="app.telegram_notifier"):
        assert telegram_notifier.send_message(at_limit) == 1

    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


# --- the review channel needs keyboards, edits and an answer for every press -------


REVIEW_CHANNEL = "-100999"
KEYBOARD = {"inline_keyboard": [[{"text": "✅ تأیید", "callback_data": "approve:7"}]]}


def test_send_message_can_target_another_chat_with_an_inline_keyboard(http: FakeHttp) -> None:
    """FR-12: the review message goes somewhere else and carries the ✅/❌ buttons."""
    assert telegram_notifier.send_message(
        MESSAGE_TEXT, chat_id=REVIEW_CHANNEL, reply_markup=KEYBOARD
    ) == 1

    payload = http.calls[0]["json"]
    assert payload["chat_id"] == REVIEW_CHANNEL
    assert payload["reply_markup"] == KEYBOARD


def test_send_message_reports_failure_when_telegram_answers_without_a_message_id(
    http: FakeHttp,
) -> None:
    http.default = FakeResponse(status_code=200, body={"ok": True, "result": True})

    assert telegram_notifier.send_message(MESSAGE_TEXT) is None


def test_edit_message_text_removes_the_buttons_by_default(http: FakeHttp) -> None:
    """FR-12: a decided post keeps its message but loses the buttons."""
    assert telegram_notifier.edit_message_text(
        "✅ تأیید شد", chat_id=REVIEW_CHANNEL, message_id=42
    ) is True

    call = http.calls[0]
    assert call["url"].endswith("/editMessageText")
    assert call["json"]["message_id"] == 42
    assert call["json"]["reply_markup"] == telegram_notifier.REMOVE_KEYBOARD


def test_edit_message_text_reports_a_rejected_edit(http: FakeHttp) -> None:
    http.default = FakeResponse(
        status_code=400, body={"ok": False, "description": "message to edit not found"}
    )

    assert telegram_notifier.edit_message_text(
        "متن", chat_id=REVIEW_CHANNEL, message_id=42
    ) is False
    assert http.attempts == 1  # a rejected edit is permanent, like a rejected send


def test_answer_callback_query_stops_the_spinner(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=200, body={"ok": True, "result": True})

    assert telegram_notifier.answer_callback_query("cb-1", text="تأیید شد") is True

    call = http.calls[0]
    assert call["url"].endswith("/answerCallbackQuery")
    assert call["json"] == {"callback_query_id": "cb-1", "text": "تأیید شد", "show_alert": False}


def test_answer_callback_query_never_raises_on_failure(http: FakeHttp) -> None:
    """The decision itself must stand even if the client cannot be told."""
    http.default = httpx.ConnectError("boom")

    assert telegram_notifier.answer_callback_query("cb-1") is False


def test_get_updates_returns_the_updates_telegram_holds(http: FakeHttp) -> None:
    updates = [{"update_id": 11, "callback_query": {"id": "cb-1", "data": "approve:1"}}]
    http.default = FakeResponse(status_code=200, body={"ok": True, "result": updates})

    assert telegram_notifier.get_updates(offset=5) == updates

    call = http.calls[0]
    assert call["url"].endswith("/getUpdates")
    assert call["json"]["offset"] == 5
    assert call["json"]["timeout"] == telegram_notifier.LONG_POLL_SECONDS
    # The client timeout must outlast the server-side long poll, or httpx gives up first.
    assert call["timeout"] > telegram_notifier.LONG_POLL_SECONDS


def test_a_short_poll_is_used_when_a_cycle_is_due(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=200, body={"ok": True, "result": []})

    assert telegram_notifier.get_updates(long_poll_seconds=0) == []
    assert http.calls[0]["json"]["timeout"] == 0
    assert "offset" not in http.calls[0]["json"]  # no cursor yet: take what is pending


def test_get_updates_returns_an_empty_list_instead_of_raising(http: FakeHttp) -> None:
    """NFR-2: a Telegram outage must not end the worker loop."""
    http.default = httpx.ConnectError("api.telegram.org unreachable")

    assert telegram_notifier.get_updates() == []


def test_get_updates_ignores_a_non_list_result(http: FakeHttp) -> None:
    http.default = FakeResponse(status_code=200, body={"ok": True, "result": {"nope": 1}})

    assert telegram_notifier.get_updates() == []
