import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
import paper_engine  # noqa: E402


def _make_broker(monkeypatch, price_ref, logs):
    monkeypatch.setattr(config, "PAPER_SYNC_BALANCE", 0, raising=False)
    monkeypatch.setattr(config, "VIRTUAL_START_BALANCE", 1000.0, raising=False)
    monkeypatch.setattr(config, "TAKER_FEE", 0.0, raising=False)
    monkeypatch.setattr(config, "SLIPPAGE_BPS", 0.0, raising=False)
    monkeypatch.setattr(config, "STRATEGY_COOLDOWN", 0, raising=False)
    monkeypatch.setattr(paper_engine.real, "get_min_order_filters", lambda symbol: (0.001, 0.001, 0.0))
    monkeypatch.setattr(paper_engine.real, "get_current_price", lambda symbol: price_ref["price"])
    monkeypatch.setattr(paper_engine.real, "get_ticker_snapshot", lambda symbol: {"last_price": price_ref["price"]})
    monkeypatch.setattr(paper_engine, "append_trade_event", lambda event: None)
    monkeypatch.setattr(paper_engine, "log", lambda message, level="INFO": logs.append(message))
    return paper_engine.PaperBroker()


def test_paper_open_buy_logs_long(monkeypatch):
    logs = []
    broker = _make_broker(monkeypatch, {"price": 100.0}, logs)

    broker.place_market_order("BTCUSDT", "Buy", 1.0)

    assert any("[PAPER-OPEN]" in msg and "[LONG]" in msg for msg in logs)


def test_paper_open_sell_logs_short(monkeypatch):
    logs = []
    broker = _make_broker(monkeypatch, {"price": 100.0}, logs)

    broker.place_market_order("BTCUSDT", "Sell", 1.0)

    assert any("[PAPER-OPEN]" in msg and "[SHORT]" in msg for msg in logs)


def test_long_pnl_positive_when_price_rises(monkeypatch):
    price = {"price": 100.0}
    broker = _make_broker(monkeypatch, price, [])
    broker.place_market_order("BTCUSDT", "Buy", 1.0)

    price["price"] = 110.0
    broker.close_position_by_market("BTCUSDT")

    assert broker.get_equity() == pytest.approx(1010.0)


def test_short_pnl_positive_when_price_falls(monkeypatch):
    price = {"price": 100.0}
    broker = _make_broker(monkeypatch, price, [])
    broker.place_market_order("BTCUSDT", "Sell", 1.0)

    price["price"] = 90.0
    broker.close_position_by_market("BTCUSDT")

    assert broker.get_equity() == pytest.approx(1010.0)


def test_short_pnl_negative_when_price_rises(monkeypatch):
    price = {"price": 100.0}
    broker = _make_broker(monkeypatch, price, [])
    broker.place_market_order("BTCUSDT", "Sell", 1.0)

    price["price"] = 110.0
    broker.close_position_by_market("BTCUSDT")

    assert broker.get_equity() == pytest.approx(990.0)
