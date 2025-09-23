import importlib
import sys
from pathlib import Path
from typing import List

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _reload_strategy():
    """Reload config and strategy to pick up fresh env values."""
    if "config" not in sys.modules:
        import config  # noqa: F401  # ensure module is loaded
    else:
        importlib.reload(sys.modules["config"])

    if "strategy" not in sys.modules:
        import strategy  # noqa: F401  # ensure module is loaded
    else:
        importlib.reload(sys.modules["strategy"])

    return sys.modules["strategy"]


@pytest.fixture
def sample_candles() -> List[List[float]]:
    # generate 40 candles with a mild uptrend
    candles = []
    for i in range(40):
        base = 100 + i * 0.5
        candles.append([
            base,         # open
            base + 0.8,   # high
            base - 0.6,   # low
            base + 0.4,   # close
            1000 + i * 10 # volume
        ])
    return candles


def test_detect_impulse_test_buy(monkeypatch, sample_candles):
    monkeypatch.setenv("STRATEGY_MODE", "test")
    monkeypatch.setenv("TEST_SIGNAL_MODE", "buy")
    monkeypatch.setenv("DEBUG_TRADING", "0")

    strategy = _reload_strategy()
    assert strategy.detect_impulse(sample_candles) == "buy"


def test_detect_impulse_test_sell(monkeypatch, sample_candles):
    monkeypatch.setenv("STRATEGY_MODE", "test")
    monkeypatch.setenv("TEST_SIGNAL_MODE", "sell")
    monkeypatch.setenv("DEBUG_TRADING", "0")

    strategy = _reload_strategy()
    assert strategy.detect_impulse(sample_candles) == "sell"


def test_detect_impulse_test_alt_toggle(monkeypatch, sample_candles):
    monkeypatch.setenv("STRATEGY_MODE", "test")
    monkeypatch.setenv("TEST_SIGNAL_MODE", "alt")
    monkeypatch.setenv("DEBUG_TRADING", "0")

    strategy = _reload_strategy()
    first = strategy.detect_impulse(sample_candles)
    second = strategy.detect_impulse(sample_candles)

    assert {first, second} == {"buy", "sell"}


def test_detect_impulse_prod_mode(monkeypatch, sample_candles):
    monkeypatch.setenv("STRATEGY_MODE", "prod")
    monkeypatch.setenv("TEST_SIGNAL_MODE", "buy")  # should be ignored in prod
    monkeypatch.setenv("DEBUG_TRADING", "0")

    strategy = _reload_strategy()
    signal = strategy.detect_impulse(sample_candles)

    assert signal in {"buy", "sell", "hold"}
