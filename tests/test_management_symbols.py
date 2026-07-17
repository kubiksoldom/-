import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PAPER_MODE"] = "1"
os.environ["SAFE_MODE"] = "1"
os.environ["BYBIT_API_KEY"] = ""
os.environ["BYBIT_API_SECRET"] = ""
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import main  # noqa: E402


class PositionsBroker:
    def __init__(self, positions):
        self.positions = positions

    def get_positions(self):
        return {"result": {"list": list(self.positions)}}

    def has_open_position(self, symbol):
        return any(
            str(position.get("symbol") or "").upper() == str(symbol).upper()
            and abs(float(position.get("size") or 0.0)) > 1e-12
            for position in self.positions
        )

    def place_market_order(self, *_args, **_kwargs):
        raise AssertionError("Real trading endpoint called in PAPER_MODE")


def _position(symbol, *, size="1", side="Buy"):
    return {"symbol": symbol, "side": side, "size": size}


def test_get_open_position_symbols_reads_all_broker_positions():
    broker = PositionsBroker([
        _position("btcusdt", size="0.25"),
        _position("ETHUSDT", size="0"),
        _position("SOLUSDT", size="2", side="Sell"),
    ])

    assert main.get_open_position_symbols(broker) == {"BTCUSDT", "SOLUSDT"}


def test_management_symbols_union_entry_broker_and_local_state():
    entry_symbols = ["ETHUSDT", "SOLUSDT"]
    broker_positions = {"result": {"list": [_position("BTCUSDT")]}}
    entry_state = {
        "XRPUSDT": {"price": 0.5, "side": "Buy", "qty": 10},
        "OLDUSDT": {"price": None, "side": None, "qty": None},
    }

    assert main.build_management_symbols(entry_symbols, entry_state, broker_positions) == [
        "ETHUSDT",
        "SOLUSDT",
        "BTCUSDT",
        "XRPUSDT",
    ]


def test_pair_reselection_keeps_removed_open_position_managed():
    new_entry_symbols = ["ETHUSDT"]
    broker_positions = [_position("BTCUSDT")]

    management_symbols = main.build_management_symbols(
        new_entry_symbols,
        {},
        broker_positions,
    )

    assert management_symbols == ["ETHUSDT", "BTCUSDT"]
    assert "BTCUSDT" not in {symbol.upper() for symbol in new_entry_symbols}


def test_set_pairs_equivalent_does_not_remove_locally_tracked_position():
    updated_entry_symbols = ["SOLUSDT"]
    entry_state = {
        "BTCUSDT": {
            "price": 60_000.0,
            "side": "Sell",
            "qty": 0.01,
            "max_upnl": None,
        },
    }

    management_symbols = main.build_management_symbols(
        updated_entry_symbols,
        entry_state,
        {"result": {"list": []}},
    )

    assert management_symbols == ["SOLUSDT", "BTCUSDT"]


def test_stale_empty_local_state_is_not_kept_as_management_symbol():
    management_symbols = main.build_management_symbols(
        ["ETHUSDT"],
        {"BTCUSDT": {"price": None, "side": None, "qty": None}},
        {"result": {"list": []}},
    )

    assert management_symbols == ["ETHUSDT"]


def test_has_any_open_positions_is_broker_wide_not_entry_pair_scoped():
    broker = PositionsBroker([_position("BTCUSDT")])

    assert main.has_any_open_positions(broker, ["ETHUSDT"]) is True

