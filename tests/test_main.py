"""Entry-point tests: the poll loop survives a bad cycle and stops cleanly."""

from __future__ import annotations

import signal

import pytest

from app import main as main_module
from app.reddit_source import TopicsConfig
from app.settings import Settings


class _StopLoop(Exception):
    """Sentinel raised by the fake clock to break ``run_forever`` after one cycle."""


@pytest.fixture
def fake_time(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the module's clock so ``run_forever`` can be observed in one cycle."""
    sleeps: list[int] = []

    class FakeTime:
        @staticmethod
        def sleep(seconds: int) -> None:
            sleeps.append(seconds)
            raise _StopLoop

    monkeypatch.setattr(main_module, "time", FakeTime)
    return sleeps


def test_sigterm_is_turned_into_keyboard_interrupt() -> None:
    """NFR-2 — SIGTERM follows the same path as Ctrl-C, so ``docker stop`` is clean."""
    with pytest.raises(KeyboardInterrupt):
        main_module._exit_on_sigterm(signal.SIGTERM, None)


def test_install_signal_handlers_registers_the_sigterm_handler() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    try:
        main_module.install_signal_handlers()

        assert signal.getsignal(signal.SIGTERM) is main_module._exit_on_sigterm
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_run_forever_survives_a_failed_cycle(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    topics_config: TopicsConfig,
    fake_time: list[int],
) -> None:
    """NFR-2 — one exploding cycle must not kill the worker."""
    calls: list[str] = []

    def boom(settings: Settings, *, topics_config: TopicsConfig) -> None:
        calls.append("run_once")
        raise RuntimeError("cycle exploded")

    monkeypatch.setattr(main_module, "run_once", boom)

    with pytest.raises(_StopLoop):
        main_module.run_forever(settings, topics_config=topics_config)

    assert calls == ["run_once"]
    assert fake_time == [settings.poll_interval_seconds]


def test_main_installs_the_handler_and_exits_on_the_shutdown_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` wires the SIGTERM handler and swallows the shutdown it triggers."""
    installed: list[bool] = []

    def request_shutdown(settings: Settings, *, topics_config: TopicsConfig) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "setup_logging", lambda level: None)
    monkeypatch.setattr(main_module, "install_signal_handlers", lambda: installed.append(True))
    monkeypatch.setattr(main_module, "run_forever", request_shutdown)

    main_module.main()  # must return rather than propagate the shutdown

    assert installed == [True]
