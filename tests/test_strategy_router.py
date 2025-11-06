import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import strategy  # noqa: E402


def test_decide_with_router_short_series_returns_hold():
    candles = [[1, 1, 1, 1, 1]] * 10
    result = strategy.decide_with_router("BTCUSDT", "1m", candles, ctx={})
    assert result["action"] == "hold"
    assert result["reason"] == "not_enough_data"


def test_decide_with_router_returns_expected_structure(monkeypatch):
    candles = []
    price = 100.0
    volume = 5.0
    for i in range(60):
        base = price + i * 0.1
        candles.append([base, base + 0.5, base - 0.5, base + 0.2, volume + i])

    class DummySig:
        def __init__(self):
            self.action = "buy"
            self.reason = "stub"
            self.sl = 0.95
            self.tp = 1.05
            self.meta = {"confidence": 0.4}

    class DummyRouter:
        def decide(self, symbol, timeframe, pack, ctx):
            assert symbol == "BTCUSDT"
            assert timeframe == "1m"
            for key in ("open", "high", "low", "close", "volume"):
                assert key in pack
                assert isinstance(pack[key], np.ndarray)
            return DummySig()

    monkeypatch.setattr(strategy, "_router", lambda: DummyRouter())
    monkeypatch.setattr(strategy, "detect_candle_patterns", lambda data: [
        {"name": "Bullish", "side": "buy", "confidence": 0.8, "bar_index": -1}
    ])
    strategy._recent_pattern_marks.clear()

    result = strategy.decide_with_router("BTCUSDT", "1m", candles, ctx={"mode": "test"})

    assert set(result.keys()) == {"action", "reason", "sl", "tp", "meta"}
    assert result["action"] == "buy"
    assert result["reason"] == "stub"
    assert result["sl"] == 0.95
    assert result["tp"] == 1.05
    assert "patterns" in result["meta"]
    assert result["meta"]["patterns"][0]["name"] == "Bullish"
