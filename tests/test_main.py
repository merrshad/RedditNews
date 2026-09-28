"""Tests for the worker loop in ``app/main.py``.

The loop is the last safety net of the service: a failing pipeline cycle must never end
it (NFR-2), and a shutdown signal must end it cleanly.
"""

from __future__ import annotations

import logging
import signal

import pytest

from app import main
from app.settings import Settings


def test_main_keeps_cycling_after_a_failed_run_and_stops_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    cycles: list[int] = []
    slept: list[float] = []

    def _run_once() -> None:
        cycles.append(len(cycles) + 1)
        if len(cycles) == 2:
            raise RuntimeError("the LLM provider exploded")

    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 2:
            raise KeyboardInterrupt  # what the signal handler does under the hood

    monkeypatch.setattr("app.pipeline.run_once", _run_once)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.setup_logging", lambda level: None)
    monkeypatch.setattr("app.main._install_signal_handlers", lambda: None)
    monkeypatch.setattr("app.main.time.sleep", _sleep)

    main.main()  # returns normally instead of propagating the failure

    assert cycles == [1, 2]
    assert slept == [settings.poll_interval_seconds] * 2


def test_the_signal_handler_takes_the_same_path_as_ctrl_c() -> None:
    """``docker stop`` (SIGTERM) must interrupt the sleep and exit cleanly."""
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
