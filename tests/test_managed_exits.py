import os
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PAPER_MODE"] = "1"
os.environ["SAFE_MODE"] = "1"
os.environ["BYBIT_API_KEY"] = ""
os.environ["BYBIT_API_SECRET"] = ""
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import bybit_api  # noqa: E402
import config  # noqa: E402
import main  # noqa: E402
import paper_engine  # noqa: E402


class ManagedPaperBroker:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.close_calls = []

    def close_position_by_market(self, symbol, qty):
        self.close_calls.append((symbol, qty))
        return self.accepted

    def place_market_order(self, *_args, **_kwargs):
        raise AssertionError("Real trading endpoint called in PAPER_MODE")


class CloseReporter:
    def __init__(self):
        self.calls = []

    def close(self, **kwargs):
        self.calls.append(kwargs)


def _state(*, side="Buy", sl_price=95.0, tp_price=110.0):
    return {
        "price": 100.0,
        "side": side,
        "qty": 2.0,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "entry_fee": None,
        "max_upnl": None,
        "trade_id": "trade-123",
        "exploration": True,
        "position_id": None,
        "entry_ts": None,
        "session_id": None,
        "git_sha": None,
        "leverage": None,
        "strategy": None,
        "regime": None,
        "atr_entry": None,
        "ml_probability": None,
        "ml_threshold": None,
        "paper": None,
        "entry_slippage": 0.0,
        "funding": None,
        "exit_reason": None,
        "_closing": False,
    }


def test_managed_position_state_persists_router_levels_and_trade_id():
    position = main.managed_position_state(
        entry_price=100.0,
        side="Sell",
        qty=2.0,
        sl_price=105.0,
        tp_price=90.0,
        trade_id="trade-123",
        exploration=True,
    )

    assert position == {
        "price": 100.0,
        "side": "Sell",
        "qty": 2.0,
        "sl_price": 105.0,
        "tp_price": 90.0,
        "entry_fee": None,
        "max_upnl": None,
        "trade_id": "trade-123",
        "exploration": True,
        "position_id": None,
        "exit_reason": None,
        "_closing": False,
    }


def test_long_stop_loss_trigger():
    assert main.protective_exit_reason(_state(side="Buy"), 95.0) == "stop_loss"


def test_long_take_profit_trigger():
    assert main.protective_exit_reason(_state(side="Buy"), 110.0) == "take_profit"


def test_short_stop_loss_trigger():
    position = _state(side="Sell", sl_price=105.0, tp_price=90.0)

    assert main.protective_exit_reason(position, 105.0) == "stop_loss"


def test_short_take_profit_trigger():
    position = _state(side="Sell", sl_price=105.0, tp_price=90.0)

    assert main.protective_exit_reason(position, 90.0) == "take_profit"


@pytest.mark.parametrize(
    ("side", "current", "sl_price", "tp_price"),
    [
        ("Buy", 100.0, 95.0, 110.0),
        ("Sell", 100.0, 105.0, 90.0),
    ],
)
def test_protective_exit_reason_ignores_price_inside_levels(side, current, sl_price, tp_price):
    position = _state(side=side, sl_price=sl_price, tp_price=tp_price)

    assert main.protective_exit_reason(position, current) is None


def test_close_managed_long_position_is_accounted_and_cleared_once(monkeypatch):
    broker = ManagedPaperBroker()
    reporter = CloseReporter()
    entry = {"BTCUSDT": _state(side="Buy")}
    bot_logs = []
    cycle_logs = []
    registered = []
    debug_logs = []
    monkeypatch.setattr(main, "_log_bot_trade", lambda symbol, side, **kwargs: bot_logs.append((symbol, side, kwargs)) or kwargs.get("trade_id"))
    monkeypatch.setattr(main, "write_cycle_log", lambda payload: cycle_logs.append(payload))
    monkeypatch.setattr(main, "register_trade_result", lambda pnl, fees=0.0: registered.append((pnl, fees)))
    monkeypatch.setattr(main, "_log_close_debug", lambda symbol, **kwargs: debug_logs.append((symbol, kwargs)))

    closed = main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        110.0,
        "take_profit",
        do_trade=True,
        tg_reporter=reporter,
        commission_rate=0.001,
    )

    assert closed is True
    assert broker.close_calls == [("BTCUSDT", 2.0)]
    assert len(bot_logs) == len(cycle_logs) == len(registered) == len(debug_logs) == len(reporter.calls) == 1
    assert bot_logs[0][2]["trade_id"] == "trade-123"
    assert bot_logs[0][2]["meta"]["exit_reason"] == "take_profit"
    assert registered[0][0] == pytest.approx(19.58)
    assert registered[0][1] == pytest.approx(0.42)
    assert cycle_logs[0]["gross_pnl"] == pytest.approx(20.0)
    assert cycle_logs[0]["entry_fee"] == pytest.approx(0.2)
    assert cycle_logs[0]["exit_fee"] == pytest.approx(0.22)
    assert cycle_logs[0]["exit_reason"] == "take_profit"
    assert cycle_logs[0]["trade_id"] == "trade-123"
    assert reporter.calls[0]["reason"] == "take_profit"
    assert entry["BTCUSDT"]["price"] is None
    assert entry["BTCUSDT"]["trade_id"] is None
    assert entry["BTCUSDT"]["exit_reason"] == "take_profit"

    assert main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        110.0,
        "take_profit",
        do_trade=True,
        tg_reporter=reporter,
        commission_rate=0.001,
    ) is False
    assert broker.close_calls == [("BTCUSDT", 2.0)]
    assert len(registered) == 1


def test_close_managed_short_position_uses_correct_prices_and_pnl(monkeypatch):
    broker = ManagedPaperBroker()
    entry = {"BTCUSDT": _state(side="Sell", sl_price=105.0, tp_price=90.0)}
    cycle_logs = []
    registered = []
    monkeypatch.setattr(main, "_log_bot_trade", lambda *_args, **kwargs: kwargs.get("trade_id") or "trade")
    monkeypatch.setattr(main, "write_cycle_log", lambda payload: cycle_logs.append(payload))
    monkeypatch.setattr(main, "register_trade_result", lambda pnl, fees=0.0: registered.append((pnl, fees)))
    monkeypatch.setattr(main, "_log_close_debug", lambda *_args, **_kwargs: None)

    assert main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        90.0,
        "take_profit",
        do_trade=True,
        commission_rate=0.001,
    ) is True

    assert registered[0][0] == pytest.approx(19.62)
    assert registered[0][1] == pytest.approx(0.38)
    assert cycle_logs[0]["direction"] == "short"
    assert cycle_logs[0]["buy_price"] == pytest.approx(90.0)
    assert cycle_logs[0]["sell_price"] == pytest.approx(100.0)


def test_rejected_close_keeps_position_for_retry_without_accounting(monkeypatch):
    broker = ManagedPaperBroker(accepted=False)
    entry = {"BTCUSDT": _state()}
    monkeypatch.setattr(main, "_log_bot_trade", lambda *_args, **_kwargs: pytest.fail("close log must not be written"))
    monkeypatch.setattr(main, "write_cycle_log", lambda *_args, **_kwargs: pytest.fail("cycle log must not be written"))
    monkeypatch.setattr(main, "register_trade_result", lambda *_args, **_kwargs: pytest.fail("result must not be registered"))

    assert main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        95.0,
        "stop_loss",
        do_trade=True,
    ) is False

    assert broker.close_calls == [("BTCUSDT", 2.0)]
    assert entry["BTCUSDT"]["price"] == 100.0
    assert entry["BTCUSDT"]["_closing"] is False


@pytest.mark.parametrize(
    ("side", "current", "sl_price", "tp_price", "expected_reason"),
    [
        ("Buy", 95.0, 95.0, 110.0, "stop_loss"),
        ("Buy", 110.0, 95.0, 110.0, "take_profit"),
        ("Sell", 105.0, 105.0, 90.0, "stop_loss"),
        ("Sell", 90.0, 105.0, 90.0, "take_profit"),
    ],
)
def test_sl_tp_closes_actual_paper_position_without_real_order(
    monkeypatch,
    side,
    current,
    sl_price,
    tp_price,
    expected_reason,
):
    price = {"value": 100.0}
    monkeypatch.setattr(config, "PAPER_SYNC_BALANCE", 0, raising=False)
    monkeypatch.setattr(config, "VIRTUAL_START_BALANCE", 1000.0, raising=False)
    monkeypatch.setattr(config, "TAKER_FEE", 0.0, raising=False)
    monkeypatch.setattr(config, "SLIPPAGE_BPS", 0.0, raising=False)
    monkeypatch.setattr(config, "STRATEGY_COOLDOWN", 0, raising=False)
    monkeypatch.setattr(config, "HARD_NOTIONAL_CAP", 0.0, raising=False)
    monkeypatch.setattr(paper_engine.real, "get_min_order_filters", lambda _symbol: (0.001, 0.001, 0.0))
    monkeypatch.setattr(paper_engine.real, "get_current_price", lambda _symbol: price["value"])
    monkeypatch.setattr(paper_engine.real, "get_ticker_snapshot", lambda _symbol: {"last_price": price["value"]})
    monkeypatch.setattr(
        paper_engine.real,
        "place_market_order",
        lambda *_args, **_kwargs: pytest.fail("Real trading endpoint called in PAPER_MODE"),
    )
    monkeypatch.setattr(paper_engine, "append_trade_event", lambda _event: None)
    monkeypatch.setattr(main, "_log_bot_trade", lambda *_args, **kwargs: kwargs.get("trade_id") or "trade")
    monkeypatch.setattr(main, "write_cycle_log", lambda _payload: None)
    monkeypatch.setattr(main, "register_trade_result", lambda _pnl, _fees=0.0: None)
    monkeypatch.setattr(main, "_log_close_debug", lambda *_args, **_kwargs: None)
    broker = paper_engine.PaperBroker()
    assert broker.place_market_order("BTCUSDT", side, 2.0)
    entry = {
        "BTCUSDT": main.managed_position_state(
            entry_price=100.0,
            side=side,
            qty=2.0,
            sl_price=sl_price,
            tp_price=tp_price,
            trade_id="trade-123",
        ),
    }
    price["value"] = current

    reason = main.process_protective_exit(
        broker,
        entry,
        "BTCUSDT",
        current,
        do_trade=True,
        commission_rate=0.0,
    )
    assert reason == expected_reason
    assert broker.has_open_position("BTCUSDT") is False


def test_paper_close_uses_actual_fills_and_both_fees(monkeypatch):
    price = {"value": 100.0}
    monkeypatch.setattr(config, "PAPER_SYNC_BALANCE", 0, raising=False)
    monkeypatch.setattr(config, "VIRTUAL_START_BALANCE", 1000.0, raising=False)
    monkeypatch.setattr(config, "TAKER_FEE", 0.001, raising=False)
    monkeypatch.setattr(config, "SLIPPAGE_BPS", 10.0, raising=False)
    monkeypatch.setattr(config, "STRATEGY_COOLDOWN", 0, raising=False)
    monkeypatch.setattr(config, "HARD_NOTIONAL_CAP", 0.0, raising=False)
    monkeypatch.setattr(paper_engine.real, "get_min_order_filters", lambda _symbol: (0.001, 0.001, 0.0))
    monkeypatch.setattr(paper_engine.real, "get_current_price", lambda _symbol: price["value"])
    monkeypatch.setattr(paper_engine.real, "get_ticker_snapshot", lambda _symbol: {"last_price": price["value"]})
    monkeypatch.setattr(
        paper_engine.real,
        "place_market_order",
        lambda *_args, **_kwargs: pytest.fail("Real trading endpoint called in PAPER_MODE"),
    )
    monkeypatch.setattr(paper_engine, "append_trade_event", lambda _event: None)
    broker = paper_engine.PaperBroker()
    order_result = broker.place_market_order("BTCUSDT", "Buy", 2.0)
    entry = {
        "BTCUSDT": main.managed_position_state(
            entry_price=order_result["fill_price"],
            side="Buy",
            qty=order_result["qty"],
            sl_price=95.0,
            tp_price=110.0,
            trade_id="trade-fill",
            entry_fee=order_result["entry_fee"],
        ),
    }
    cycle_logs = []
    registered = []
    monkeypatch.setattr(main, "_log_bot_trade", lambda *_args, **kwargs: kwargs.get("trade_id") or "trade")
    monkeypatch.setattr(main, "write_cycle_log", lambda payload: cycle_logs.append(payload))
    monkeypatch.setattr(main, "register_trade_result", lambda pnl, fees=0.0: registered.append((pnl, fees)))
    monkeypatch.setattr(main, "_log_close_debug", lambda *_args, **_kwargs: None)
    price["value"] = 110.0

    assert main.process_protective_exit(
        broker,
        entry,
        "BTCUSDT",
        110.0,
        do_trade=True,
        commission_rate=0.001,
    ) == "take_profit"

    close_event = cycle_logs[0]
    assert close_event["entry_price"] == pytest.approx(100.1)
    assert close_event["exit_price"] == pytest.approx(109.89)
    assert close_event["entry_fee"] == pytest.approx(0.2002)
    assert close_event["exit_fee"] == pytest.approx(0.21978)
    assert close_event["pnl"] == pytest.approx(broker.get_equity() - 1000.0)
    assert len(registered) == 1
    assert registered[0][0] == pytest.approx(close_event["pnl"])
    assert registered[0][1] == pytest.approx(close_event["fees"])


@pytest.mark.parametrize("alias", ["trail", "dynamic_tp", "manual"])
def test_noncanonical_exit_reason_is_rejected(alias):
    with pytest.raises(ValueError):
        main.normalize_exit_reason(alias)


def test_bybit_close_returns_true_only_after_reduce_only_order(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bybit_api,
        "get_positions",
        lambda _symbol=None: {"result": {"list": [{"symbol": "BTCUSDT", "side": "Buy", "size": "2"}]}},
    )
    monkeypatch.setattr(
        bybit_api,
        "place_market_order",
        lambda symbol, side, qty, reduce_only=False: calls.append((symbol, side, qty, reduce_only)) or {"result": {"orderId": "1"}},
    )

    assert bybit_api.close_position_by_market("BTCUSDT", 2.0) is True
    assert calls == [("BTCUSDT", "Sell", 2.0, True)]


def test_bybit_close_returns_false_when_position_is_absent(monkeypatch):
    monkeypatch.setattr(bybit_api, "get_positions", lambda _symbol=None: {"result": {"list": []}})
    monkeypatch.setattr(
        bybit_api,
        "place_market_order",
        lambda *_args, **_kwargs: pytest.fail("order endpoint must not be called"),
    )

    assert bybit_api.close_position_by_market("BTCUSDT", 2.0) is False
