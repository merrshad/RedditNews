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

from urllib.parse import urlsplit, urlunsplit

from app import main, repository
from app.settings import Settings
from app.telegram_updates import PollResult
from tests.conftest import TEST_DATABASE_URL


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
    # The loop tests are not about the database, and the preflight has its own tests below.
    monkeypatch.setattr("app.main.preflight", lambda: None)


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
    monkeypatch.setattr("app.pipeline.run_once", lambda: cycles.append("cycle"))

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
