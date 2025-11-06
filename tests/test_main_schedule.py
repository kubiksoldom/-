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
    main.loss_streak = main.LOSS_STREAK_MAX
    main.last_loss_time = time.time()

    ok, reason = main.can_enter_now()
    assert not ok
    assert reason == "loss_cooldown"

    # reset globals for other tests
    main.loss_streak = 0
    main.last_loss_time = 0.0
