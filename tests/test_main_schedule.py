import datetime as dt
import os
import time
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("PAPER_MODE", "1")
os.environ.setdefault("SAFE_MODE", "1")

import main  # noqa: E402


def test_schedule_status_off_hours(monkeypatch):
    monkeypatch.setenv("FORCE_SCHEDULE_OFF", "0")
    monkeypatch.setenv("EXCLUDE_WEEKENDS", "0")
    monkeypatch.setenv("TRADE_HOURS_LOCAL", "10:00-12:00")

    fake_now = dt.datetime(2024, 5, 13, 8, 30)
    monkeypatch.setattr(main, "_now_local", lambda: fake_now)

    allowed, reason, next_ts = main.schedule_status()
    assert allowed is False
    assert reason == "off_hours"
    assert isinstance(next_ts, float)


def test_schedule_status_weekend(monkeypatch):
    monkeypatch.setenv("FORCE_SCHEDULE_OFF", "0")
    monkeypatch.setenv("EXCLUDE_WEEKENDS", "1")
    monkeypatch.setenv("TRADE_HOURS_LOCAL", "08:00-18:00")

    fake_now = dt.datetime(2024, 5, 12, 9, 0)  # Sunday
    monkeypatch.setattr(main, "_now_local", lambda: fake_now)

    allowed, reason, next_ts = main.schedule_status()
    assert allowed is False
    assert reason == "weekend"
    assert isinstance(next_ts, float)


def test_can_enter_now_loss_cooldown(monkeypatch):
    monkeypatch.setattr(main, "get_margin_state", lambda: {"im_pct": 5.0, "frozen": False})
    main.SCHEDULE_ALLOWED = True
    main.PAUSE_ENTRIES = False
    main.SESSION_BREAK_ACTIVE = False
    main.SOFT_STOP_ACTIVE = False
    main.loss_streak = main.LOSS_STREAK_MAX
    main.last_loss_time = time.time()

    ok, reason = main.can_enter_now()
    assert not ok
    assert reason == "loss_cooldown"

    # reset globals for other tests
    main.loss_streak = 0
    main.last_loss_time = 0.0


def test_break_disables_entries_but_management_stays_active(monkeypatch):
    monkeypatch.setattr(main, "get_margin_state", lambda: {"im_pct": 5.0, "frozen": False})
    main.SCHEDULE_ALLOWED = True
    main.SCHEDULE_REASON = ""
    main.loss_streak = 0
    main.last_loss_time = 0.0
    main.SOFT_STOP_ACTIVE = False

    entries_allowed = main.set_session_entries_allowed(False, log_change=False)
    ok, reason = main.can_enter_now()

    assert entries_allowed is False
    assert ok is False
    assert reason == "session_closed"
    assert main.position_management_active(has_open_position=True) is True

    main.set_session_entries_allowed(True, log_change=False)


def test_handle_graceful_shutdown_soft_stop_does_not_force_close():
    class DummyBroker:
        def __init__(self):
            self.force_close_calls = 0

        def force_close_all_positions_absolute(self):
            self.force_close_calls += 1

    broker = DummyBroker()

    did_force = main.handle_graceful_shutdown(broker, do_trade=True, force_close_on_exit=False)

    assert did_force is False
    assert broker.force_close_calls == 0
    assert main.PAUSE_ENTRIES is True
    main.PAUSE_ENTRIES = False


def test_handle_graceful_shutdown_force_close_when_enabled():
    class DummyBroker:
        def __init__(self):
            self.force_close_calls = 0

        def force_close_all_positions_absolute(self):
            self.force_close_calls += 1

    broker = DummyBroker()

    did_force = main.handle_graceful_shutdown(broker, do_trade=True, force_close_on_exit=True)

    assert did_force is True
    assert broker.force_close_calls == 1
    main.PAUSE_ENTRIES = False


def test_max_runtime_state_blocks_entries_and_exits_only_without_positions():
    entries_allowed, can_exit = main.max_runtime_state(
        start_time=100.0,
        now=200.0,
        max_runtime_sec=50,
        has_open_positions=True,
    )
    assert entries_allowed is False
    assert can_exit is False

    entries_allowed, can_exit = main.max_runtime_state(
        start_time=100.0,
        now=200.0,
        max_runtime_sec=50,
        has_open_positions=False,
    )
    assert entries_allowed is False
    assert can_exit is True


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("schedule:off_hours", "schedule_off"),
        ("margin:33.2%", "margin"),
        ("qty_adjust", "qty_below_min"),
        ("min_notional", "qty_below_min"),
        ("ML veto", "ml_veto"),
        ("position already open", "open_position_exists"),
    ],
)
def test_normalize_skip_reason(raw, expected):
    assert main.normalize_skip_reason(raw) == expected


def test_ml_entry_mode_labels_and_entry_debug(monkeypatch):
    messages = []
    monkeypatch.setattr(main.config, "DEBUG_DECISIONS", 1, raising=False)
    monkeypatch.setattr(main, "log", lambda message, level="INFO": messages.append(message))

    assert main.ml_entry_mode_label(
        ml_trading_enabled=False,
        apply_new_ml=False,
        ml_block_disabled=True,
        ml_veto_enabled=False,
        ml_shadow_enabled=True,
    ) == "UNAVAILABLE_SHADOW"
    assert main.ml_entry_mode_label(
        ml_trading_enabled=True,
        apply_new_ml=True,
        ml_block_disabled=True,
        ml_veto_enabled=False,
        ml_shadow_enabled=False,
    ) == "SHADOW"

    main._log_entry_debug(
        "SOLUSDT",
        "LONG",
        price=100.0,
        tp=102.0,
        sl=99.0,
        rr=2.0,
        lev=10,
        size=1.0,
        ml_mode="SHADOW",
        proba=0.62,
        threshold=0.58,
        factor=1.0,
        band="full",
    )

    assert messages
    assert "ml_mode=SHADOW" in messages[-1]
    assert "proba=0.62" in messages[-1]
    assert "thr=0.58" in messages[-1]
