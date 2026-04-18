import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import build_ml_dataset_from_fills as dataset  # noqa: E402
import ml_veto  # noqa: E402
import config  # noqa: E402


def test_label_trade_uses_high_low_indexes():
    price = 100.0
    future = [
        [0, 101.0, 99.0, 100.5, 1.0],
        [0, 105.0, 100.0, 104.0, 1.0],
    ]
    hit_tp = dataset.label_trade(price, future, tp_pct=0.02, sl_pct=0.02)
    assert hit_tp == 1

    future_sl = [
        [0, 100.5, 95.0, 96.0, 1.0],
    ]
    hit_sl = dataset.label_trade(price, future_sl, tp_pct=0.05, sl_pct=0.03)
    assert hit_sl == 0


def test_process_row_produces_feature_row(monkeypatch):
    dataset.LAST_OI_SNAPSHOT.clear()

    class DummyProvider:
        def get_kline_before(self, symbol, end_ms, minutes, interval):
            candles = []
            for i in range(minutes):
                ts = end_ms - (minutes - i) * 60_000
                base = 100 + i * 0.1
                candles.append([ts, base + 0.5, base - 0.5, base + 0.2, 10 + i])
            return candles, {}

        def fetch_snapshot_any(self, symbol):
            return {
                "last_price": 105.0,
                "index_price": 104.5,
                "high": 106.0,
                "low": 103.0,
                "vol_24h": 12000.0,
                "open_interest": 5000.0,
                "funding_rate": 0.0001,
            }

        def get_kline_forward(self, symbol, start_ms, minutes, interval):
            candles = []
            for i in range(minutes):
                ts = start_ms + (i + 1) * 60_000
                base = 105 + i * 0.2
                candles.append([ts, base + 0.4, base - 0.6, base + 0.1, 12 + i])
            return candles, {}

    row = pd.Series(
        {
            "symbol": "BTCUSDT",
            "ts": 1_700_000_000_000,
            "price": 105.0,
            "qty": 0.5,
            "side": "Buy",
        }
    )

    idx, result = dataset.process_row(0, row, DummyProvider(), source_name="synthetic")

    assert idx == 0
    assert result is not None
    assert result["target"] in (0, 1)
    assert result["feature_last_price"] > 0
    assert result["source"] == "synthetic"


def test_load_model_and_meta_with_stub(monkeypatch):
    dummy_model = object()
    meta = {
        "metrics": {"precision_week": 0.7},
        "thresholds": {"used": 0.6, "global": 0.55},
        "features": ["feature_last_price", "feature_qty", "feature_direction"],
        "atr_percentiles": {"p50": 0.01, "p90": 0.03},
    }

    monkeypatch.setattr(ml_veto, "get_model_and_meta_cached", lambda *_, **__: (dummy_model, meta))
    monkeypatch.setattr(ml_veto, "_update_ml_status", lambda *_, **__: None)

    model, loaded_meta = ml_veto.load_model_and_meta()

    assert model is dummy_model
    assert loaded_meta is meta


def test_load_model_and_meta_does_not_block_when_disabled_and_weekly_missing(monkeypatch):
    dummy_model = object()
    meta = {"metrics": {}, "thresholds": {}, "features": []}
    status_calls = []

    monkeypatch.setattr(ml_veto, "get_model_and_meta_cached", lambda *_, **__: (dummy_model, meta))
    monkeypatch.setattr(ml_veto, "_update_ml_status", lambda *args, **kwargs: status_calls.append((args, kwargs)))
    monkeypatch.setattr(ml_veto, "ML_IGNORE_WEEKLY", False, raising=False)
    monkeypatch.delenv("DISABLE_ML_BLOCK", raising=False)
    monkeypatch.setattr(config, "DISABLE_ML_BLOCK", True, raising=False)

    model, loaded_meta = ml_veto.load_model_and_meta()

    assert model is dummy_model
    assert loaded_meta is meta
    assert status_calls
    (args, kwargs) = status_calls[-1]
    assert args[0] == "degraded"
    assert args[1] is False
    assert kwargs["reason"] == "manual_override"


def test_load_model_and_meta_does_not_block_when_disabled_and_artifacts_missing(monkeypatch):
    status_calls = []
    monkeypatch.setattr(ml_veto, "get_model_and_meta_cached", lambda *_, **__: (None, None))
    monkeypatch.setattr(ml_veto, "_update_ml_status", lambda *args, **kwargs: status_calls.append((args, kwargs)))
    monkeypatch.delenv("DISABLE_ML_BLOCK", raising=False)
    monkeypatch.setattr(config, "DISABLE_ML_BLOCK", True, raising=False)

    model, loaded_meta = ml_veto.load_model_and_meta()

    assert model is None
    assert loaded_meta is None
    assert status_calls
    (args, kwargs) = status_calls[-1]
    assert args[0] == "unavailable"
    assert args[1] is False
    assert kwargs["reason"] == "manual_override"


def test_predict_ok_with_fixtures(monkeypatch):
    monkeypatch.setattr(ml_veto, "ML_PROBA_STRICT", 0.6, raising=False)
    monkeypatch.setattr(ml_veto, "ML_SHADOW_MODE", 0, raising=False)
    monkeypatch.setattr(ml_veto, "ML_ACCEPT_DELTA_EV", 0.0, raising=False)
    monkeypatch.setattr(config, "ML_CONF_HIGH", 0.8, raising=False)
    monkeypatch.setattr(config, "ML_CONF_MID", 0.6, raising=False)

    meta = {
        "metrics": {"precision_week": 0.8},
        "thresholds": {"used": 0.6, "global": 0.55},
        "features": ["feature_last_price", "feature_qty", "feature_direction"],
        "atr_percentiles": {"p50": 0.01, "p90": 0.03},
        "ev_params": {"fee": 0.001, "r_avg": 1.8},
    }

    class DummyModel:
        def predict_proba(self, X):
            assert X.shape[1] == len(meta["features"])
            return np.array([[0.15, 0.85]])

    candles = []
    for i in range(60):
        base = 100 + i * 0.2
        candles.append([base, base + 0.5, base - 0.4, base + 0.1, 10 + i])

    dummy_bybit = SimpleNamespace(
        get_ticker_snapshot=lambda symbol: {
            "last_price": 110.0,
            "index_price": 109.5,
            "high": 111.0,
            "low": 108.0,
            "vol_24h": 5000.0,
            "open_interest": 7000.0,
            "funding_rate": 0.0001,
        },
        get_orderbook_spread=lambda symbol, depth=1: 0.0005,
        orderbook_imbalance=lambda symbol, depth=5: 0.05,
    )
    monkeypatch.setitem(sys.modules, "bybit_api", dummy_bybit)

    outcome = ml_veto.predict_ok(
        DummyModel(),
        meta,
        symbol="BTCUSDT",
        direction="long",
        qty=1.0,
        price=110.0,
        atr=0.5,
        candles=candles,
    )

    assert outcome.ok is True
    assert pytest.approx(outcome.proba, rel=1e-6) == 0.85
    assert outcome.factor > 0.0
    assert outcome.features_ok is True
    assert outcome.band in {"full", "reduced", "shadow"}
