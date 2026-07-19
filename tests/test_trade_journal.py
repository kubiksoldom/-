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
os.environ["PAPER_SYNC_BALANCE"] = "0"
os.environ["BYBIT_API_KEY"] = ""
os.environ["BYBIT_API_SECRET"] = ""
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import config  # noqa: E402
import main  # noqa: E402
import paper_engine  # noqa: E402
import trade_journal  # noqa: E402


def _record(**overrides):
    record = {
        "trade_id": "trade-1",
        "session_id": "session-1",
        "git_sha": "a" * 40,
        "symbol": "btcusdt",
        "direction": "Buy",
        "entry_ts": "2026-07-19T10:00:00+03:00",
        "exit_ts": "2026-07-19T08:30:00+00:00",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "qty": 2.0,
        "leverage": 5,
        "entry_fee": 0.2,
        "exit_fee": 0.22,
        "funding": 0.0,
        "slippage": 0.4,
        "gross_pnl": 20.0,
        "net_pnl": 19.58,
        "exit_reason": "take_profit",
        "strategy": "trend",
        "regime": "high_volatility",
        "atr_entry": 1.25,
        "ml_probability": 0.74,
        "ml_threshold": 0.7,
        "paper": True,
    }
    record.update(overrides)
    return record


def test_normalized_trade_record_has_exact_schema_and_utc_timestamps():
    normalized = trade_journal.normalize_trade_record(_record())

    assert tuple(normalized) == trade_journal.TRADE_JOURNAL_FIELDS
    assert normalized["symbol"] == "BTCUSDT"
    assert normalized["direction"] == "long"
    assert normalized["entry_ts"] == "2026-07-19T07:00:00+00:00"
    assert normalized["exit_ts"] == "2026-07-19T08:30:00+00:00"


def test_trade_journal_append_is_idempotent_and_rejects_conflict(tmp_path):
    path = tmp_path / "trade_journal.jsonl"

    first, first_appended = trade_journal.append_trade_record(path, _record())
    second, second_appended = trade_journal.append_trade_record(path, _record())

    assert first_appended is True
    assert second_appended is False
    assert first == second
    assert trade_journal.read_trade_records(path) == [first]
    assert path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(trade_journal.TradeJournalConflictError):
        trade_journal.append_trade_record(path, _record(exit_price=111.0, gross_pnl=22.0, net_pnl=21.58))


@pytest.mark.parametrize(
    "changes",
    [
        {"exit_ts": "2026-07-19T06:00:00+00:00"},
        {"net_pnl": 999.0},
        {"ml_probability": 1.1},
        {"exit_reason": "dynamic_tp"},
        {"api_key": "must-not-be-written"},
    ],
)
def test_trade_journal_rejects_inconsistent_or_non_schema_data(changes):
    with pytest.raises(trade_journal.TradeJournalError):
        trade_journal.normalize_trade_record(_record(**changes))


@pytest.mark.parametrize(
    ("side", "current_price", "expected_direction", "expected_slippage"),
    [
        ("Buy", 110.0, "long", 0.42),
        ("Sell", 90.0, "short", 0.38),
    ],
)
def test_paper_close_writes_one_complete_trade_without_real_endpoint(
    tmp_path,
    monkeypatch,
    side,
    current_price,
    expected_direction,
    expected_slippage,
):
    price = {"value": 100.0}
    monkeypatch.setattr(config, "PAPER_SYNC_BALANCE", 0, raising=False)
    monkeypatch.setattr(config, "VIRTUAL_START_BALANCE", 1000.0, raising=False)
    monkeypatch.setattr(config, "TAKER_FEE", 0.001, raising=False)
    monkeypatch.setattr(config, "SLIPPAGE_BPS", 10.0, raising=False)
    monkeypatch.setattr(config, "STRATEGY_COOLDOWN", 0, raising=False)
    monkeypatch.setattr(config, "HARD_NOTIONAL_CAP", 0.0, raising=False)
    monkeypatch.setattr(
        paper_engine.real,
        "get_min_order_filters",
        lambda _symbol: (0.001, 0.001, 0.0),
    )
    monkeypatch.setattr(
        paper_engine.real,
        "get_current_price",
        lambda _symbol: price["value"],
    )
    monkeypatch.setattr(
        paper_engine.real,
        "get_ticker_snapshot",
        lambda _symbol: {"last_price": price["value"]},
    )

    def forbidden_real_call(*_args, **_kwargs):
        raise AssertionError("Real trading endpoint called in PAPER_MODE")

    for name in (
        "place_market_order",
        "close_position_by_market",
        "set_leverage",
        "force_close_all_positions_absolute",
        "close_all_positions",
    ):
        monkeypatch.setattr(paper_engine.real, name, forbidden_real_call)

    monkeypatch.setattr(paper_engine, "append_trade_event", lambda _event: None)
    monkeypatch.setattr(
        main,
        "_log_bot_trade",
        lambda *_args, **kwargs: kwargs.get("trade_id") or "trade-journal",
    )
    monkeypatch.setattr(main, "write_cycle_log", lambda _payload: None)
    monkeypatch.setattr(main, "register_trade_result", lambda _pnl, _fees=0.0: None)
    monkeypatch.setattr(main, "_log_close_debug", lambda *_args, **_kwargs: None)

    broker = paper_engine.PaperBroker()
    order = broker.place_market_order("BTCUSDT", side, 2.0)
    assert isinstance(order, dict)
    entry_reference = 100.0
    entry_fill = order["fill_price"]
    entry_slippage = abs(entry_fill - entry_reference) * order["qty"]
    entry = {
        "BTCUSDT": main.managed_position_state(
            entry_price=entry_fill,
            side=side,
            qty=order["qty"],
            sl_price=95.0 if side == "Buy" else 105.0,
            tp_price=110.0 if side == "Buy" else 90.0,
            trade_id="trade-journal",
            entry_fee=order["entry_fee"],
            position_id=order["position_id"],
            entry_ts="2025-01-01T00:00:00+00:00",
            session_id="session-journal",
            git_sha="b" * 40,
            leverage=7,
            strategy="router-test",
            regime="trend",
            atr_entry=1.5,
            ml_probability=0.73,
            ml_threshold=0.7,
            paper=True,
            entry_slippage=entry_slippage,
            funding=0.0,
        )
    }
    journal_path = tmp_path / "trade_journal.jsonl"
    price["value"] = current_price

    assert main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        current_price,
        "take_profit",
        do_trade=True,
        commission_rate=0.001,
        trade_journal_file=journal_path,
    ) is True
    assert main.close_managed_position(
        broker,
        entry,
        "BTCUSDT",
        current_price,
        "take_profit",
        do_trade=True,
        commission_rate=0.001,
        trade_journal_file=journal_path,
    ) is False

    rows = trade_journal.read_trade_records(journal_path)
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == trade_journal.TRADE_JOURNAL_FIELDS
    assert row["trade_id"] == "trade-journal"
    assert row["direction"] == expected_direction
    assert row["exit_reason"] == "take_profit"
    assert row["strategy"] == "router-test"
    assert row["regime"] == "trend"
    assert row["leverage"] == pytest.approx(7.0)
    assert row["slippage"] == pytest.approx(expected_slippage)
    assert row["paper"] is True
    assert row["net_pnl"] == pytest.approx(
        row["gross_pnl"] - row["entry_fee"] - row["exit_fee"]
    )
