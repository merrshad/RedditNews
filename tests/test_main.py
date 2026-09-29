"""Tests for the worker loop in ``app/main.py``.

Phase 5 gave the worker two jobs in one loop: long-poll Telegram for admin decisions, and
run a full pipeline cycle on its own schedule. The loop is the last safety net of the
service — a failing cycle must never end it (NFR-2), a decision must not have to wait for
the next scheduled cycle, and a shutdown signal must end it cleanly.
"""

from __future__ import annotations

import logging
import signal
from typing import Any

import pytest

from app import main
from app.settings import Settings
from app.telegram_updates import PollResult


class _Clock:
    """A virtual clock: the loop's schedule only moves when it sleeps.

    ``time.sleep`` is faked anyway (the loop must not really wait), and a fake that does
    not advance ``time.monotonic`` would make ``next_cycle_at`` unreachable, so the two are
    faked together. The nth sleep raises what the signal handler raises under the hood.
    """

    def __init__(self, stop_after_sleeps: int) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self._stop_after = stop_after_sleeps

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        if len(self.slept) >= self._stop_after:
            raise KeyboardInterrupt

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "_Clock":
        monkeypatch.setattr("app.main.time.monotonic", self.monotonic)
        monkeypatch.setattr("app.main.time.sleep", self.sleep)
        return self


def _quiet(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr("app.main._install_signal_handlers", lambda: None)


def test_main_keeps_cycling_after_a_failed_run_and_stops_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    cycles: list[int] = []

    def _run_once() -> None:
        cycles.append(len(cycles) + 1)
        if len(cycles) == 2:
            raise RuntimeError("the LLM provider exploded")

    _quiet(monkeypatch, settings)
    monkeypatch.setattr("app.main.telegram_updates.poll_once", _polling([]))
    monkeypatch.setattr("app.pipeline.run_once", _run_once)
    clock = _Clock(stop_after_sleeps=2).install(monkeypatch)

    main.main()  # returns normally instead of propagating the failure

    assert cycles == [1, 2]
    # When Telegram answers immediately there is nothing to wait for, so the loop floors
    # itself instead of spinning; every entry is that floor.
    assert clock.slept and all(0 < seconds <= main.MIN_POLL_INTERVAL_SECONDS for seconds in clock.slept)


def test_a_decision_is_processed_without_waiting_for_the_next_cycle(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A click must be answered in seconds, not after `POLL_INTERVAL_SECONDS`."""
    slow_cycles = settings.model_copy(update={"poll_interval_seconds": 3600})
    long_polls: list[int] = []
    processed: list[str] = []

    _quiet(monkeypatch, slow_cycles)
    monkeypatch.setattr("app.main.telegram_updates.poll_once", _polling([0, 1], long_polls))
    monkeypatch.setattr("app.pipeline.run_once", lambda: processed.append("cycle"))
    monkeypatch.setattr(
        "app.pipeline.process_approved_posts", lambda: processed.append("decision")
    )
    _Clock(stop_after_sleeps=2).install(monkeypatch)

    main.main()

    assert processed == ["cycle", "decision"]
    # The first iteration had a cycle due, so it asked for updates without blocking; the
    # second one could afford to wait for a click.
    assert long_polls[0] == 0
    assert long_polls[1] == main.LONG_POLL_SECONDS


def _polling(decisions: list[int], long_polls: list[int] | None = None):
    """A `telegram_updates.poll_once` stand-in replaying one ``decisions`` count per call."""
    remaining = list(decisions)

    def _poll_once(offset: int | None = None, *, long_poll_seconds: int = 0) -> PollResult:
        if long_polls is not None:
            long_polls.append(long_poll_seconds)
        count = remaining.pop(0) if remaining else 0
        return PollResult(offset=(offset or 0) + 1, decisions=count)

    return _poll_once


def test_run_safely_logs_and_swallows_a_failing_step(caplog) -> None:
    def _boom() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        main._run_safely(_boom)  # must not raise: the loop owns the retry

    assert "boom" in caplog.text


def test_run_safely_reports_a_successful_step(caplog) -> None:
    called: list[str] = []

    main._run_safely(lambda: called.append("ok"))

    assert called == ["ok"]
    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []


def test_the_signal_handler_takes_the_same_path_as_ctrl_c() -> None:
    """``docker stop`` (SIGTERM) must interrupt the poll and exit cleanly."""
    with pytest.raises(KeyboardInterrupt):
        main._request_shutdown(signal.SIGTERM, None)


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
