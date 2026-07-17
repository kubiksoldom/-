import json
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

import config  # noqa: E402
import main  # noqa: E402
import paper_engine  # noqa: E402


def _state(
    *,
    symbol="BTCUSDT",
    side="Buy",
    qty=2.0,
    price=100.0,
    sl_price=95.0,
    tp_price=110.0,
    trade_id="trade-123",
    position_id="position-123",
):
    return {
        symbol: main.managed_position_state(
            entry_price=price,
            side=side,
            qty=qty,
            sl_price=sl_price,
            tp_price=tp_price,
            trade_id=trade_id,
            entry_fee=0.2,
            exploration=True,
            position_id=position_id,
        )
    }


def _broker_row(
    *,
    symbol="BTCUSDT",
    side="Buy",
    qty="2",
    price="100",
    sl="",
    tp="",
    position_id="position-123",
):
    return {
        "symbol": symbol,
        "side": side,
        "size": qty,
        "avgPrice": price,
        "stopLoss": sl,
        "takeProfit": tp,
        "positionId": position_id,
    }


def test_managed_positions_are_written_atomically_without_unknown_fields(tmp_path, monkeypatch):
    path = tmp_path / "managed_positions.json"
    entry = _state()
    entry["BTCUSDT"]["api_key"] = "must-not-be-persisted"
    entry["BTCUSDT"]["_closing"] = True
    replace_calls = []
    real_replace = os.replace

    def recording_replace(source, target):
        source_path = Path(source)
        replace_calls.append((source_path, Path(target), source_path.exists()))
        return real_replace(source, target)

    monkeypatch.setattr(main.os, "replace", recording_replace)

    assert main.save_managed_positions(entry, path) == path

    payload = json.loads(path.read_text(encoding="utf-8"))
    saved = payload["positions"]["BTCUSDT"]
    assert payload["version"] == 1
    assert saved["trade_id"] == "trade-123"
    assert saved["position_id"] == "position-123"
    assert "api_key" not in saved
    assert "_closing" not in saved
    assert replace_calls == [(replace_calls[0][0], path, True)]
    assert replace_calls[0][0] != path
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_replace_failure_keeps_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "managed_positions.json"
    main.save_managed_positions(_state(qty=1.0), path)
    original = path.read_text(encoding="utf-8")

    def rejected_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(main.os, "replace", rejected_replace)

    with pytest.raises(OSError, match="replace failed"):
        main.save_managed_positions(_state(qty=9.0), path)

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_load_ignores_non_allowlisted_secret_fields(tmp_path):
    path = tmp_path / "managed_positions.json"
    payload = main.managed_positions_payload(_state())
    payload["positions"]["BTCUSDT"]["api_secret"] = "must-not-load"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = main.load_managed_positions(path)

    assert loaded["BTCUSDT"]["trade_id"] == "trade-123"
    assert "api_secret" not in loaded["BTCUSDT"]


def test_reconcile_uses_broker_truth_and_removes_stale_local_position():
    local = _state(price=99.0, qty=1.0, sl_price=96.0, tp_price=108.0)
    local.update(_state(symbol="XRPUSDT", position_id="stale-position"))
    broker_rows = [_broker_row(qty="2", price="100", sl="94", tp="112")]

    recovered, report = main.reconcile_managed_positions(broker_rows, local)

    assert set(recovered) == {"BTCUSDT"}
    assert recovered["BTCUSDT"]["price"] == pytest.approx(100.0)
    assert recovered["BTCUSDT"]["qty"] == pytest.approx(2.0)
    assert recovered["BTCUSDT"]["sl_price"] == pytest.approx(94.0)
    assert recovered["BTCUSDT"]["tp_price"] == pytest.approx(112.0)
    assert recovered["BTCUSDT"]["trade_id"] == "trade-123"
    assert report["stale_removed"] == ["XRPUSDT"]


def test_reconcile_falls_back_to_saved_levels_when_broker_has_none():
    recovered, report = main.reconcile_managed_positions([_broker_row()], _state())

    assert recovered["BTCUSDT"]["sl_price"] == pytest.approx(95.0)
    assert recovered["BTCUSDT"]["tp_price"] == pytest.approx(110.0)
    assert report["missing_sl"] == []
    assert report["missing_tp"] == []


def test_reconcile_scales_fee_and_peak_after_external_partial_close():
    local = _state(qty=2.0)
    local["BTCUSDT"]["max_upnl"] = 4.0

    recovered, report = main.reconcile_managed_positions(
        [_broker_row(qty="1")],
        local,
    )

    assert recovered["BTCUSDT"]["entry_fee"] == pytest.approx(0.1)
    assert recovered["BTCUSDT"]["max_upnl"] == pytest.approx(2.0)
    assert report["qty_changed"] == ["BTCUSDT"]


def test_mismatched_position_id_does_not_reuse_old_levels_or_trade_id():
    broker_rows = [_broker_row(position_id="new-position")]

    recovered, report = main.reconcile_managed_positions(broker_rows, _state())

    state = recovered["BTCUSDT"]
    assert state["sl_price"] is None
    assert state["tp_price"] is None
    assert state["trade_id"].startswith("recovered-")
    assert state["trade_id"] != "trade-123"
    assert report["replaced"] == ["BTCUSDT"]
    assert report["missing_sl"] == ["BTCUSDT"]
    assert report["missing_tp"] == ["BTCUSDT"]


def test_broker_only_position_is_recovered_with_stable_identity():
    broker_rows = [_broker_row(position_id="broker-position")]

    first, report = main.reconcile_managed_positions(broker_rows, {})
    second, _ = main.reconcile_managed_positions(broker_rows, first)

    assert report["broker_only"] == ["BTCUSDT"]
    assert report["missing_sl"] == ["BTCUSDT"]
    assert report["missing_tp"] == ["BTCUSDT"]
    assert first["BTCUSDT"]["trade_id"] == second["BTCUSDT"]["trade_id"]
    assert first["BTCUSDT"]["position_id"] == "broker-position"


def test_unavailable_broker_snapshot_keeps_local_state():
    local = _state()

    recovered, report = main.reconcile_managed_positions(
        [],
        local,
        broker_snapshot_available=False,
    )

    assert recovered == local
    assert report["broker_unavailable"] is True
    assert report["stale_removed"] == []


def test_local_recovered_state_prevents_clean_exit_when_broker_is_unavailable():
    class UnavailableBroker:
        def get_positions(self):
            raise RuntimeError("offline")

        def has_open_position(self, _symbol):
            raise RuntimeError("offline")

    assert main.has_any_open_positions(
        UnavailableBroker(),
        ["BTCUSDT"],
        _state(),
    ) is True


def test_accepted_close_is_persisted_before_reporting_failure(tmp_path, monkeypatch):
    class Broker:
        def __init__(self):
            self.calls = 0

        def close_position_by_market(self, _symbol, _qty):
            self.calls += 1
            return True

    path = tmp_path / "managed_positions.json"
    entry = _state()
    broker = Broker()
    main.save_managed_positions(entry, path)

    def fail_report(*_args, **_kwargs):
        raise RuntimeError("report failed")

    monkeypatch.setattr(main, "_log_close_debug", fail_report)

    with pytest.raises(RuntimeError, match="report failed"):
        main.close_managed_position(
            broker,
            entry,
            "BTCUSDT",
            110.0,
            "take_profit",
            do_trade=True,
            managed_positions_file=path,
        )

    assert broker.calls == 1
    assert main.load_managed_positions(path) == {}
    assert main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        110.0,
        "take_profit",
        do_trade=True,
        managed_positions_file=path,
    ) is False
    assert broker.calls == 1


def test_restart_recovery_manages_and_closes_existing_paper_position(tmp_path, monkeypatch):
    price = {"value": 100.0}
    state_path = tmp_path / "managed_positions.json"
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
    order = broker.place_market_order("BTCUSDT", "Buy", 2.0)
    original_state = {
        "BTCUSDT": main.managed_position_state(
            entry_price=order["fill_price"],
            side="Buy",
            qty=order["qty"],
            sl_price=95.0,
            tp_price=105.0,
            trade_id="restart-trade",
            entry_fee=order["entry_fee"],
            position_id=order["position_id"],
        )
    }
    main.save_managed_positions(original_state, state_path)

    runtime_state = {}
    persisted = main.load_managed_positions(state_path)
    broker_rows = main._position_rows(broker.get_positions())
    recovered, report = main.reconcile_managed_positions(broker_rows, persisted)
    runtime_state.update(recovered)

    assert report["stale_removed"] == []
    assert runtime_state["BTCUSDT"]["trade_id"] == "restart-trade"
    assert runtime_state["BTCUSDT"]["position_id"] == order["position_id"]
    assert "BTCUSDT" in main.build_management_symbols(["ETHUSDT"], runtime_state, broker_rows)

    price["value"] = 105.0
    reason = main.process_protective_exit(
        broker,
        runtime_state,
        "BTCUSDT",
        105.0,
        do_trade=True,
        commission_rate=0.0,
        managed_positions_file=state_path,
    )

    assert reason == "take_profit"
    assert broker.has_open_position("BTCUSDT") is False
    assert main.load_managed_positions(state_path) == {}


def test_paper_positions_expose_recovery_fields(monkeypatch):
    monkeypatch.setattr(config, "PAPER_SYNC_BALANCE", 0, raising=False)
    monkeypatch.setattr(config, "STRATEGY_COOLDOWN", 0, raising=False)
    monkeypatch.setattr(config, "HARD_NOTIONAL_CAP", 0.0, raising=False)
    monkeypatch.setattr(paper_engine.real, "get_min_order_filters", lambda _symbol: (0.001, 0.001, 0.0))
    monkeypatch.setattr(paper_engine.real, "get_current_price", lambda _symbol: 100.0)
    monkeypatch.setattr(paper_engine.real, "get_ticker_snapshot", lambda _symbol: {"last_price": 100.0})
    monkeypatch.setattr(paper_engine, "append_trade_event", lambda _event: None)
    broker = paper_engine.PaperBroker()
    order = broker.place_market_order("BTCUSDT", "Sell", 1.0)

    row = broker.get_positions()["result"]["list"][0]

    assert row["symbol"] == "BTCUSDT"
    assert row["side"] == "Sell"
    assert float(row["avgPrice"]) == pytest.approx(order["fill_price"])
    assert row["positionId"] == order["position_id"]
    assert row["stopLoss"] == ""
    assert row["takeProfit"] == ""
