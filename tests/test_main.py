"""Tests for the worker in ``app/main.py``.

Phase 6 turned the single ``while True`` into one asyncio event loop with two tasks: one
long-polls Telegram, the other keeps the pipeline schedule. The loop is the last safety net
of the service — a failing cycle must never end it (NFR-2), a press must be handled *while* a
cycle is still running, and a shutdown signal must end it cleanly.

The tests drive the real ``main()`` with the two waits replaced (``app.main._sleep`` and
``app.main._wait_for_decision``), so they are about the scheduling rules and never about
really waiting.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

from app import main, repository
from app.settings import Settings
from app.telegram_updates import PollResult
from tests.conftest import TEST_DATABASE_URL, run


class _Clock:
    """The loop's virtual schedule: every wait is counted, and the nth one ends the loop.

    ``_sleep`` is the update loop's own floor (it must never really wait) and
    ``_wait_for_decision`` is the pipeline's wait between two cycles. Neither may sleep for
    real, and the nth wait raises what a shutdown signal raises under the hood.
    """

    def __init__(self, stop_after_waits: int) -> None:
        self.slept: list[float] = []
        self.waits = 0
        self._stop_after = stop_after_waits

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        # Yield, like the real `asyncio.sleep` does: a wait that never hands control back
        # would starve the other task and turn these tests into a hang.
        await asyncio.sleep(0)

    async def wait_for_decision(self, decision: asyncio.Event, timeout: float) -> bool:
        await asyncio.sleep(0)
        self.waits += 1
        if self.waits >= self._stop_after:
            raise KeyboardInterrupt
        return False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "_Clock":
        monkeypatch.setattr("app.main._sleep", self.sleep)
        monkeypatch.setattr("app.main._wait_for_decision", self.wait_for_decision)
        return self


def _quiet(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr("app.main._install_signal_handlers", lambda: None)
    # The loop tests are not about the database, and the preflight has its own tests below.
    monkeypatch.setattr("app.main.preflight", lambda: None)


def _polling(decisions: list[int], long_polls: list[int] | None = None):
    """A `telegram_updates.poll_once` stand-in replaying one ``decisions`` count per call."""
    remaining = list(decisions)

    async def _poll_once(offset: int | None = None, *, long_poll_seconds: int = 0) -> PollResult:
        if long_polls is not None:
            long_polls.append(long_poll_seconds)
        count = remaining.pop(0) if remaining else 0
        return PollResult(offset=(offset or 0) + 1, decisions=count)

    return _poll_once


def test_main_keeps_cycling_after_a_failed_run_and_stops_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    cycles: list[int] = []

    async def _run_once() -> None:
        cycles.append(len(cycles) + 1)
        if len(cycles) == 2:
            raise RuntimeError("the LLM provider exploded")

    _quiet(monkeypatch, settings)
    monkeypatch.setattr("app.main.telegram_updates.poll_once", _polling([]))
    monkeypatch.setattr("app.pipeline.run_once", _run_once)
    # The virtual clock ends the loop on the second inter-cycle wait, i.e. after two cycles.
    clock = _Clock(stop_after_waits=2).install(monkeypatch)

    main.main()  # returns normally instead of propagating the failure

    assert cycles == [1, 2]
    # When Telegram answers immediately there is nothing to wait for, so the update loop
    # floors itself instead of spinning; every entry is that floor.
    assert clock.slept
    assert all(0 < seconds <= main.MIN_POLL_INTERVAL_SECONDS for seconds in clock.slept)


def test_the_update_loop_long_polls_without_shortening_for_a_due_cycle(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """With two tasks there is no reason to interrupt a long poll when a cycle is due.

    Phase 5 asked for a zero-second poll whenever a cycle was pending, because the poll and
    the cycle were the same thread; now the cycle simply proceeds.
    """
    long_polls: list[int] = []

    _quiet(monkeypatch, settings)
    monkeypatch.setattr("app.main.telegram_updates.poll_once", _polling([], long_polls))
    monkeypatch.setattr("app.pipeline.run_once", _record([], "cycle"))
    _Clock(stop_after_waits=1).install(monkeypatch)

    main.main()

    assert long_polls and set(long_polls) == {main.LONG_POLL_SECONDS}


def test_a_decision_runs_the_approved_queue_without_a_new_cycle(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A click must be answered in seconds, not after `POLL_INTERVAL_SECONDS`."""
    slow_cycles = settings.model_copy(update={"poll_interval_seconds": 3600})
    processed: list[str] = []
    answers = [True]

    async def _wait_for_decision(_decision: asyncio.Event, _timeout: float) -> bool:
        await asyncio.sleep(0)
        if not answers:
            raise KeyboardInterrupt
        return answers.pop(0)

    _quiet(monkeypatch, slow_cycles)
    monkeypatch.setattr("app.main.telegram_updates.poll_once", _polling([1]))
    monkeypatch.setattr("app.pipeline.run_once", _record(processed, "cycle"))
    monkeypatch.setattr(
        "app.pipeline.process_approved_posts", _record(processed, "decision")
    )
    _Clock(stop_after_waits=1).install(monkeypatch)
    monkeypatch.setattr("app.main._wait_for_decision", _wait_for_decision)

    main.main()

    # The click is served right after the cycle that was in progress, not
    # POLL_INTERVAL_SECONDS (3600s) later; the extra cycle is the loop going round again.
    assert processed[:2] == ["cycle", "decision"]


def _record(target: list[str], value: str):
    """An awaitable stand-in that records one step (``run_once`` is async since phase 6)."""

    async def _step() -> None:
        target.append(value)

    return _step


def test_a_press_is_handled_while_a_cycle_is_still_running(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """The reason for two tasks: a long cycle no longer delays the admin (phase 6).

    The cycle blocks until the update loop has seen the press, which is exactly the
    situation the old single-threaded loop could not express — there, the poll could not run
    at all until `run_once()` returned, and this test would hang.
    """
    events: list[str] = []
    cycle_running = asyncio.Event()
    pressed = asyncio.Event()

    async def _slow_cycle() -> None:
        events.append("cycle-started")
        cycle_running.set()
        await pressed.wait()  # the cycle stays busy until the press has been seen
        events.append("cycle-finished")

    async def _poll_once(offset: int | None = None, **_kwargs: Any) -> PollResult:
        await cycle_running.wait()  # do not "arrive" before the cycle really started
        events.append("poll")
        pressed.set()
        return PollResult(offset=1, decisions=1)

    _quiet(monkeypatch, settings)
    monkeypatch.setattr("app.main.telegram_updates.poll_once", _poll_once)
    monkeypatch.setattr("app.pipeline.run_once", _slow_cycle)
    _Clock(stop_after_waits=1).install(monkeypatch)

    main.main()

    # The press is seen while the cycle is still busy (the extra "poll" is the update loop
    # going round again after the cycle ended, which is what it is supposed to do).
    assert events[:3] == ["cycle-started", "poll", "cycle-finished"]


def test_run_safely_logs_and_swallows_a_failing_step(caplog) -> None:
    async def _boom() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        run(main._run_safely(_boom))  # must not raise: the loop owns the retry

    assert "boom" in caplog.text


def test_run_safely_reports_a_successful_step(caplog) -> None:
    called: list[str] = []

    async def _ok() -> None:
        called.append("ok")

    run(main._run_safely(_ok))

    assert called == ["ok"]
    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


def test_the_signal_handler_takes_the_same_path_as_ctrl_c() -> None:
    """``docker stop`` (SIGTERM) must interrupt the poll and exit cleanly."""
    with pytest.raises(KeyboardInterrupt):
        main._request_shutdown(signal.SIGTERM, None)


def _point_repository_at(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """Make ``repository`` (which reads its own settings) talk to a specific database."""
    stale = Settings(
        database_url=url,
        openai_api_key="test-key",
        openai_base_url="https://llm.example/v1",
        openai_model="test-model",
        telegram_bot_token="000000:test-token",
        telegram_chat_id="-100123",
        telegram_review_channel_id="-100999",
        telegram_admin_ids="777",
        poll_interval_seconds=1,
        rss_fetch_limit=25,
        similarity_lookback_limit=50,
        similarity_lookback_hours=72,
        min_importance_to_send="low",
        http_max_retries=2,
        log_level="INFO",
    )
    monkeypatch.setattr("app.repository.get_settings", lambda: stale)


def test_preflight_accepts_the_database_built_from_schema_sql(
    postgres_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real test database *is* ``db/schema.sql``, so the preflight must pass on it."""
    _point_repository_at(monkeypatch, postgres_database)

    assert repository.find_unusable_tables() == []
    assert main.preflight() is None


def test_preflight_rejects_a_database_left_over_from_an_earlier_phase(
    postgres_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed for real: a Docker volume from phase 4 makes every cycle die with
    ``UndefinedTable`` while the worker still looks alive. The preflight must turn that
    into one actionable startup error instead (Invariant 9, NFR-3).

    The maintenance database of the same server stands in for such a volume: it exists and
    has none of the app tables.
    """
    parts = urlsplit(postgres_database)
    _point_repository_at(monkeypatch, urlunsplit(parts._replace(path="/postgres")))

    assert set(repository.find_unusable_tables()) == {"topics", "sources", "posts"}
    with pytest.raises(SystemExit) as excinfo:
        main.preflight()
    message = str(excinfo.value)
    assert "topics" in message and "down -v" in message


def test_main_refuses_to_start_instead_of_looping_on_a_broken_schema(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    cycles: list[str] = []
    _quiet(monkeypatch, settings)
    monkeypatch.setattr("app.main.preflight", _raise_schema_error)
    monkeypatch.setattr("app.pipeline.run_once", _record(cycles, "cycle"))

    with pytest.raises(SystemExit):
        main.main()

    assert cycles == []


def _raise_schema_error() -> None:
    raise SystemExit("database schema is not ready")


def test_setup_logging_keeps_the_secret_bearing_transport_loggers_quiet() -> None:
    """Invariant 6: the transport libraries log requests, and requests carry secrets.

    A real phase-4 run showed the bot token in `docker logs` at the default level, because
    httpx prints the full URL (`.../bot<TOKEN>/sendMessage`); at DEBUG it also prints the
    headers, which hold the LLM bearer key. Capping those loggers is what makes
    `LOG_LEVEL` safe to raise.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        main.setup_logging("DEBUG")

        for name in main.SECRET_BEARING_LOGGERS:
            assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
        assert root.level == logging.DEBUG  # the app's own logs still follow LOG_LEVEL
    finally:
        # `basicConfig(force=True)` replaces the root handlers, so put them back.
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for name in main.SECRET_BEARING_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
