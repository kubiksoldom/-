import sys
import types

import pytest

import utils
from utils import pre_trade_check


def _install_stub(monkeypatch, *, min_qty=0.01, qty_step=0.01, min_notional=5.0, reliable=True):
    stub = types.SimpleNamespace()
    stub.get_min_order_filters = lambda symbol: (min_qty, qty_step, min_notional)
    stub.filters_reliable = lambda symbol: reliable
    monkeypatch.setitem(sys.modules, "bybit_api", stub)
    cfg_stub = types.SimpleNamespace(
        COMMISSION_PER_SIDE=0.0006,
        SPREAD_MAX_PCT=0.0008,
        MAX_IM_PERCENT=30.0,
    )
    monkeypatch.setattr(utils, "config", cfg_stub, raising=False)


def test_pre_trade_check_success(monkeypatch):
    _install_stub(monkeypatch)
    result = pre_trade_check(
        "BTCUSDT",
        price=100.0,
        qty=0.1,
        spread=0.0005,
        margin_state={"im_pct": 10.0, "frozen": False},
    )
    assert result["ok"] is True
    assert result["qty"] == pytest.approx(0.1)


def test_pre_trade_check_blocks_spread(monkeypatch):
    _install_stub(monkeypatch)
    result = pre_trade_check(
        "BTCUSDT",
        price=100.0,
        qty=0.1,
        spread=0.01,
        margin_state={"im_pct": 5.0, "frozen": False},
    )
    assert result["ok"] is False
    assert result["why"] == "spread"


def test_pre_trade_check_min_notional_with_fee(monkeypatch):
    _install_stub(monkeypatch, min_qty=0.1, qty_step=0.1, min_notional=5.0)
    result = pre_trade_check(
        "ETHUSDT",
        price=4.99,
        qty=1.0,
        spread=0.0001,
        margin_state={"im_pct": 5.0, "frozen": False},
    )
    assert result["ok"] is False
    assert result["why"] == "qty_adjust"


def test_pre_trade_check_margin_freeze(monkeypatch):
    _install_stub(monkeypatch)
    result = pre_trade_check(
        "SOLUSDT",
        price=100.0,
        qty=0.5,
        spread=0.0001,
        margin_state={"im_pct": 35.0, "frozen": True},
    )
    assert result["ok"] is False
    assert result["why"] == "margin"
