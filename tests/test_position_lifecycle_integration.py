import os
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PAPER_MODE"] = "1"
os.environ["SAFE_MODE"] = "1"
os.environ["TELEGRAM_ENABLED"] = "0"
os.environ["BYBIT_API_KEY"] = ""
os.environ["BYBIT_API_SECRET"] = ""
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import config  # noqa: E402
import main  # noqa: E402
import paper_engine  # noqa: E402


FORBIDDEN_REAL_CALL = "Real trading endpoint called in PAPER_MODE"


class PaperLifecycle:
    def __init__(self, broker, price, state_path, logs, cycle_logs, results):
        self.broker = broker
        self.price = price
        self.state_path = state_path
        self.logs = logs
        self.cycle_logs = cycle_logs
        self.results = results
        self.entry = {}

    def open_position(
        self,
        *,
        symbol="BTCUSDT",
        side="Buy",
        qty=2.0,
        entry_price=100.0,
        sl_price=95.0,
        tp_price=110.0,
        trade_id="trade-integration",
    ):
        self.price[symbol] = float(entry_price)
        result = self.broker.place_market_order(symbol, side, qty)
        assert result
        self.entry[symbol] = main.managed_position_state(
            entry_price=result["fill_price"],
            side=side,
            qty=result["qty"],
            sl_price=sl_price,
            tp_price=tp_price,
            trade_id=trade_id,
            entry_fee=result["entry_fee"],
            position_id=result["position_id"],
        )
        main.save_managed_positions(self.entry, self.state_path)
        return self.entry[symbol]

    def manage(self, symbol, current_price, *, commission_rate=0.0, trail_drop_pct=0.02):
        self.price[symbol] = float(current_price)
        return main.manage_open_position_at_price(
            self.broker,
            self.entry,
            symbol,
            current_price,
            do_trade=True,
            commission_rate=commission_rate,
            trail_drop_pct=trail_drop_pct,
            managed_positions_file=self.state_path,
        )


@pytest.fixture(autouse=True)
def reset_runtime_state():
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


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    price = {"BTCUSDT": 100.0, "ETHUSDT": 100.0}
    logs = []
    cycle_logs = []
    results = []

    monkeypatch.setattr(config, "PAPER_SYNC_BALANCE", 0, raising=False)
    monkeypatch.setattr(config, "VIRTUAL_START_BALANCE", 1000.0, raising=False)
    monkeypatch.setattr(config, "TAKER_FEE", 0.0, raising=False)
    monkeypatch.setattr(config, "SLIPPAGE_BPS", 0.0, raising=False)
    monkeypatch.setattr(config, "STRATEGY_COOLDOWN", 0, raising=False)
    monkeypatch.setattr(config, "HARD_NOTIONAL_CAP", 0.0, raising=False)
    monkeypatch.setattr(paper_engine.real, "get_min_order_filters", lambda _symbol: (0.001, 0.001, 0.0))
    monkeypatch.setattr(paper_engine.real, "get_current_price", lambda symbol: price[symbol])
    monkeypatch.setattr(
        paper_engine.real,
        "get_ticker_snapshot",
        lambda symbol: {"last_price": price[symbol]},
    )

    def forbidden_real_call(*_args, **_kwargs):
        raise AssertionError(FORBIDDEN_REAL_CALL)

    for method_name in (
        "place_market_order",
        "close_position_by_market",
        "set_leverage",
        "force_close_all_positions_absolute",
        "close_all_positions",
    ):
        monkeypatch.setattr(paper_engine.real, method_name, forbidden_real_call)

    monkeypatch.setattr(paper_engine, "append_trade_event", lambda _event: None)
    monkeypatch.setattr(main, "get_margin_state", lambda: {"im_pct": 0.0, "frozen": False})
    monkeypatch.setattr(main, "log", lambda message, level="INFO": logs.append(message))
    monkeypatch.setattr(
        main,
        "_log_bot_trade",
        lambda *_args, **kwargs: kwargs.get("trade_id") or "trade-integration",
    )
    monkeypatch.setattr(main, "write_cycle_log", lambda payload: cycle_logs.append(payload))
    monkeypatch.setattr(main, "register_trade_result", lambda pnl, fees=0.0: results.append((pnl, fees)))
    monkeypatch.setattr(main, "_log_close_debug", lambda *_args, **_kwargs: None)

    return PaperLifecycle(
        paper_engine.PaperBroker(),
        price,
        tmp_path / "managed_positions.json",
        logs,
        cycle_logs,
        results,
    )


def test_graceful_stop_manages_paper_position_until_strategy_close(lifecycle, monkeypatch):
    lifecycle.open_position(tp_price=105.0)
    monkeypatch.setattr(
        lifecycle.broker,
        "force_close_all_positions_absolute",
        lambda: pytest.fail("graceful stop must not force-close positions"),
    )

    assert main.request_graceful_shutdown() is True
    assert main.can_enter_now() == (False, "pause_entries")
    assert main.advance_graceful_shutdown(
        lifecycle.broker,
        ["BTCUSDT"],
        do_trade=True,
        force_close_on_exit=False,
        entry_state=lifecycle.entry,
    ) == "managing"
    assert lifecycle.broker.has_open_position("BTCUSDT") is True

    assert lifecycle.manage("BTCUSDT", 105.0) == "take_profit"
    assert lifecycle.broker.has_open_position("BTCUSDT") is False
    assert main.advance_graceful_shutdown(
        lifecycle.broker,
        ["BTCUSDT"],
        do_trade=True,
        force_close_on_exit=False,
        entry_state=lifecycle.entry,
    ) == "graceful_exit"

    assert lifecycle.logs == [
        "[SHUTDOWN] graceful shutdown requested",
        "[SHUTDOWN] entries disabled, managing open positions",
        "[SHUTDOWN] all positions closed, exiting",
    ]
    assert lifecycle.cycle_logs[-1]["exit_reason"] == "take_profit"
    assert main.load_managed_positions(lifecycle.state_path) == {}


def test_break_blocks_entries_while_trailing_stop_manages_position(lifecycle):
    lifecycle.open_position(sl_price=80.0, tp_price=130.0)

    assert main.set_session_entries_allowed(False, log_change=False) is False
    assert main.can_enter_now() == (False, "session_closed")
    assert lifecycle.manage("BTCUSDT", 110.0, trail_drop_pct=0.02) is None
    assert lifecycle.broker.has_open_position("BTCUSDT") is True

    assert lifecycle.manage("BTCUSDT", 107.0, trail_drop_pct=0.02) == "trailing_stop"
    assert lifecycle.broker.has_open_position("BTCUSDT") is False
    assert lifecycle.cycle_logs[-1]["exit_reason"] == "trailing_stop"


def test_no_profit_exit_still_runs_during_break(lifecycle, monkeypatch):
    monkeypatch.setattr(config, "TAKER_FEE", 0.001, raising=False)
    lifecycle.open_position(sl_price=80.0, tp_price=130.0)
    main.set_session_entries_allowed(False, log_change=False)

    assert lifecycle.manage(
        "BTCUSDT",
        105.0,
        commission_rate=0.001,
        trail_drop_pct=0.50,
    ) is None
    assert lifecycle.manage(
        "BTCUSDT",
        100.0,
        commission_rate=0.001,
        trail_drop_pct=0.50,
    ) == "no_profit"
    assert lifecycle.cycle_logs[-1]["exit_reason"] == "no_profit"


def test_pair_reselection_keeps_position_managed_until_take_profit(lifecycle):
    lifecycle.open_position(tp_price=110.0)
    new_entry_symbols = ["ETHUSDT"]
    broker_positions = lifecycle.broker.get_positions()

    management_symbols = main.build_management_symbols(
        new_entry_symbols,
        lifecycle.entry,
        broker_positions,
    )

    assert "BTCUSDT" not in new_entry_symbols
    assert management_symbols == ["ETHUSDT", "BTCUSDT"]
    assert lifecycle.manage("BTCUSDT", 110.0) == "take_profit"
    assert lifecycle.broker.has_open_position("BTCUSDT") is False


@pytest.mark.parametrize(
    ("side", "current_price", "sl_price", "tp_price", "expected_reason"),
    [
        ("Buy", 95.0, 95.0, 110.0, "stop_loss"),
        ("Buy", 110.0, 95.0, 110.0, "take_profit"),
        ("Sell", 105.0, 105.0, 90.0, "stop_loss"),
        ("Sell", 90.0, 105.0, 90.0, "take_profit"),
    ],
)
def test_management_pass_executes_long_and_short_sl_tp(
    lifecycle,
    side,
    current_price,
    sl_price,
    tp_price,
    expected_reason,
):
    lifecycle.open_position(side=side, sl_price=sl_price, tp_price=tp_price)

    assert lifecycle.manage("BTCUSDT", current_price) == expected_reason
    assert lifecycle.broker.has_open_position("BTCUSDT") is False
    assert lifecycle.cycle_logs[-1]["exit_reason"] == expected_reason
    assert lifecycle.cycle_logs[-1]["direction"] == ("long" if side == "Buy" else "short")
    assert len(lifecycle.results) == 1


def test_restart_recovers_runtime_state_and_resumes_management(lifecycle):
    original = lifecycle.open_position(tp_price=110.0, trade_id="trade-before-restart")
    lifecycle.entry = {}

    loaded = main.load_managed_positions(lifecycle.state_path)
    recovered, report = main.reconcile_managed_positions(
        lifecycle.broker.get_positions(),
        loaded,
    )
    lifecycle.entry = recovered

    assert report["broker_unavailable"] is False
    assert recovered["BTCUSDT"]["trade_id"] == original["trade_id"]
    assert recovered["BTCUSDT"]["position_id"] == original["position_id"]
    assert lifecycle.manage("BTCUSDT", 110.0) == "take_profit"
    assert lifecycle.broker.has_open_position("BTCUSDT") is False


def test_panic_is_an_explicit_forced_path_not_graceful_stop(lifecycle, monkeypatch):
    lifecycle.open_position()
    close_calls = []
    real_close_all = lifecycle.broker.close_all_positions

    def counted_close_all():
        close_calls.append(True)
        return real_close_all()

    monkeypatch.setattr(lifecycle.broker, "close_all_positions", counted_close_all)

    assert main.execute_panic_close(
        lifecycle.broker,
        lifecycle.entry,
        managed_positions_file=lifecycle.state_path,
    ) is True

    assert close_calls == [True]
    assert lifecycle.broker.has_open_position("BTCUSDT") is False
    assert lifecycle.entry["BTCUSDT"]["exit_reason"] == "panic"
    assert main.GRACEFUL_STOP_REQUESTED is False
    assert main.GRACEFUL_STOP_ACTIVE is False
    assert not any(message.startswith("[SHUTDOWN]") for message in lifecycle.logs)
    assert main.load_managed_positions(lifecycle.state_path) == {}


def test_paper_lifecycle_never_calls_real_trading_endpoints(lifecycle):
    lifecycle.open_position(symbol="ETHUSDT", side="Sell", sl_price=105.0, tp_price=90.0)

    assert lifecycle.manage("ETHUSDT", 90.0) == "take_profit"
    assert lifecycle.broker.has_open_position("ETHUSDT") is False
