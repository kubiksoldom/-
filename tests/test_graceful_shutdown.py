import os
import signal
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("PAPER_MODE", "1")
os.environ.setdefault("SAFE_MODE", "1")

import main  # noqa: E402


class DummyPaperBroker:
    def __init__(self):
        self.positions = {
            "BTCUSDT": {"symbol": "BTCUSDT", "side": "Buy", "size": "1"},
        }
        self.force_close_calls = 0
        self.strategy_close_calls = 0

    def get_positions(self):
        return {"result": {"list": list(self.positions.values())}}

    def has_open_position(self, symbol):
        return symbol in self.positions

    def close_position_by_market(self, symbol):
        self.strategy_close_calls += 1
        self.positions.pop(symbol, None)
        return True

    def force_close_all_positions_absolute(self):
        self.force_close_calls += 1
        self.positions.clear()


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    original = (
        main.PAUSE_ENTRIES,
        main.SESSION_BREAK_ACTIVE,
        main.SOFT_STOP_ACTIVE,
        main.GRACEFUL_STOP_REQUESTED,
        main.GRACEFUL_STOP_ACTIVE,
        main.GRACEFUL_STOP_COMPLETE_LOGGED,
        main.SCHEDULE_ALLOWED,
        main.SCHEDULE_REASON,
        main.loss_streak,
        main.last_loss_time,
        main.SESSION_STATE.get("shutdown_requested", False),
    )
    main.PAUSE_ENTRIES = False
    main.SESSION_BREAK_ACTIVE = False
    main.SOFT_STOP_ACTIVE = False
    main.GRACEFUL_STOP_REQUESTED = False
    main.GRACEFUL_STOP_ACTIVE = False
    main.GRACEFUL_STOP_COMPLETE_LOGGED = False
    main.SCHEDULE_ALLOWED = True
    main.SCHEDULE_REASON = ""
    main.loss_streak = 0
    main.last_loss_time = 0.0
    main.SESSION_STATE["shutdown_requested"] = False
    yield
    (
        main.PAUSE_ENTRIES,
        main.SESSION_BREAK_ACTIVE,
        main.SOFT_STOP_ACTIVE,
        main.GRACEFUL_STOP_REQUESTED,
        main.GRACEFUL_STOP_ACTIVE,
        main.GRACEFUL_STOP_COMPLETE_LOGGED,
        main.SCHEDULE_ALLOWED,
        main.SCHEDULE_REASON,
        main.loss_streak,
        main.last_loss_time,
        main.SESSION_STATE["shutdown_requested"],
    ) = original


def test_sigint_keeps_managing_until_paper_position_closes(monkeypatch):
    broker = DummyPaperBroker()
    logs = []
    monkeypatch.setattr(main, "log", lambda message, level="INFO": logs.append(message))
    monkeypatch.setattr(main, "get_margin_state", lambda: {"im_pct": 0.0, "frozen": False})

    main._handle_sigterm(signal.SIGINT, None)

    assert main.GRACEFUL_STOP_REQUESTED is True
    assert main.PAUSE_ENTRIES is True
    assert broker.has_open_position("BTCUSDT") is True

    action = main.advance_graceful_shutdown(
        broker,
        ["BTCUSDT"],
        do_trade=True,
        force_close_on_exit=False,
    )

    assert action == "managing"
    assert main.GRACEFUL_STOP_ACTIVE is True
    assert broker.force_close_calls == 0
    assert main.can_enter_now() == (False, "pause_entries")

    broker.close_position_by_market("BTCUSDT")
    action = main.advance_graceful_shutdown(
        broker,
        ["BTCUSDT"],
        do_trade=True,
        force_close_on_exit=False,
    )

    assert broker.strategy_close_calls == 1
    assert action == "graceful_exit"
    assert logs == [
        "[SHUTDOWN] graceful shutdown requested",
        "[SHUTDOWN] entries disabled, managing open positions",
        "[SHUTDOWN] all positions closed, exiting",
    ]


def test_session_transition_cannot_reenable_entries_after_sigint():
    main._handle_sigterm(signal.SIGINT, None)

    entries_allowed = main.set_session_entries_allowed(True, log_change=False)

    assert entries_allowed is False
    assert main.PAUSE_ENTRIES is True


def test_force_close_on_exit_remains_separate_from_graceful_stop(monkeypatch):
    broker = DummyPaperBroker()
    logs = []
    monkeypatch.setattr(main, "log", lambda message, level="INFO": logs.append(message))

    main._handle_sigterm(signal.SIGINT, None)
    action = main.advance_graceful_shutdown(
        broker,
        ["BTCUSDT"],
        do_trade=True,
        force_close_on_exit=True,
    )

    assert action == "forced_exit"
    assert broker.force_close_calls == 1
    assert broker.positions == {}
    assert main.GRACEFUL_STOP_ACTIVE is False
    assert logs == ["[SHUTDOWN] FORCE_CLOSE_ON_EXIT=1, force closing positions"]
