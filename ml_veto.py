# -*- coding: utf-8 -*-
# ml_veto.py
import os
import json
import math
import pickle
import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import config
from utils import log

MODEL_FILE = getattr(config, "MODEL_FILE", os.getenv("MODEL_FILE", "rf_model.pkl"))
MODEL_META = getattr(config, "MODEL_META", os.getenv("MODEL_META", "model_meta.json"))

_MODEL_CACHE: Dict[str, Any] = {
    "model_path": None,
    "model_mtime": None,
    "model": None,
    "meta_path": None,
    "meta_mtime": None,
    "meta": None,
}

def _rsi_from_closes(closes: List[float], period: int = 14) -> float:
    s = pd.Series(list(map(float, closes)), dtype=float)
    if len(s) < period + 1:
        return 50.0
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1] if not rsi.empty else 50.0
    return float(50.0 if (pd.isna(v) or math.isnan(v)) else v)

def atr_abs(ohlc_list: List[List[float]], period: int = 14) -> float:
    if len(ohlc_list) < 2:
        return 0.0
    trs = []
    for i in range(1, len(ohlc_list)):
        _, h, l, c, _ = ohlc_list[i]
        _, _, _, pc, _ = ohlc_list[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return 0.0
    last = trs[-period:] if len(trs) >= period else trs
    return float(np.mean(last))

def _momentum_and_volumes_from_candles(candles: List[List[float]]) -> Tuple[float, float, float, float]:
    closes = [float(x[3]) for x in candles] if candles else []
    vols   = [float(x[4]) for x in candles] if candles else []
    pd5  = (closes[-1] - closes[-6])  if len(closes) > 6  else 0.0
    vs5  = sum(vols[-6:])             if len(vols)   > 6  else 0.0
    pd15 = (closes[-1] - closes[-16]) if len(closes) > 16 else 0.0
    vs15 = sum(vols[-16:])            if len(vols)   > 16 else 0.0
    return pd5, vs5, pd15, vs15

def _vectorize_features(meta: Optional[Dict], feats_dict: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    order = None
    try:
        order = (meta or {}).get("features", None)
    except Exception:
        order = None
    if not order:
        order = [
            "index_price","last_price","high","low","vol_24h",
            "open_interest","funding_rate","rsi","atr_abs","atr_norm",
            "volatility","hour","weekday","pct_from_high","dist_to_index",
            "price_delta_5m","volume_sum_5m","price_delta_15m","volume_sum_15m",
            "spread_bps","fee_bps","qty","direction"
        ]
    row = [float(feats_dict.get(k, 0.0)) for k in order]
    return np.array([row], dtype=float), order

def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        path = path.strip()
    except Exception:
        pass
    return os.path.abspath(path) if path else None


def load_model_and_meta():
    global _MODEL_CACHE

    model_path = _resolve_path(MODEL_FILE)
    meta_path = _resolve_path(MODEL_META)

    cache = _MODEL_CACHE

    def _mtime(path: Optional[str]) -> Optional[float]:
        if not path:
            return None
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    model_mtime = _mtime(model_path)
    meta_mtime = _mtime(meta_path)

    cache_hit = (
        cache.get("model") is not None
        and cache.get("model_path") == model_path
        and cache.get("model_mtime") == model_mtime
        and cache.get("meta_path") == meta_path
        and cache.get("meta_mtime") == meta_mtime
    )

    if cache_hit:
        return cache.get("model"), cache.get("meta")

    try:
        if not model_path:
            raise FileNotFoundError("MODEL_FILE path is empty")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        log(f"[ML] Не удалось загрузить модель: {e}")
        if cache.get("model") is not None:
            return cache.get("model"), cache.get("meta")
        return None, None

    meta = None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            log(f"[ML] Не удалось загрузить мета-информацию: {e}")
            meta = cache.get("meta") if cache.get("meta_path") == meta_path else None

    _MODEL_CACHE = {
        "model_path": model_path,
        "model_mtime": model_mtime,
        "model": model,
        "meta_path": meta_path,
        "meta_mtime": meta_mtime,
        "meta": meta,
    }

    log_meta = meta_path if meta_path else "<none>"
    log(f"[ML] Модель: {MODEL_FILE}; мета: {log_meta} — загружены.")
    return model, meta

def predict_ok(
    model,
    meta: Optional[Dict],
    symbol: str,
    direction: str,
    qty: float,
    price: Optional[float] = None,
    atr: Optional[float] = None,
    candles: Optional[List[List[float]]] = None
) -> Tuple[bool, float, float]:
    default_thr = float(getattr(config, "ML_THRESHOLD", 0.58))
    if model is None or meta is None:
        return True, 0.5, default_thr

    try:
        if (candles is None) or (price is None):
            from bybit_api import fetch_price_history
        from bybit_api import get_ticker_snapshot, get_orderbook_spread

        if candles is None:
            candles = fetch_price_history(symbol, limit=60)
        closes = [float(x[3]) for x in candles] if candles else []

        snap = get_ticker_snapshot(symbol)
        last = float(snap.get("last_price", 0.0))
        idx  = float(snap.get("index_price", 0.0))
        high = float(snap.get("high", 0.0)) or (max(closes) if closes else 0.0)
        low  = float(snap.get("low", 0.0))  or (min(closes) if closes else 0.0)
        if price is None:
            price = last

        rsi_val = _rsi_from_closes(closes) if closes else 50.0
        ohlc = [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in candles] if candles else []
        atr_abs_val = float(atr) if (atr is not None) else (atr_abs(ohlc) if ohlc else 0.0)
        atr_norm = (atr_abs_val / price) if price and price > 0 else 0.0

        if len(closes) >= 24:
            last24 = pd.Series(closes[-24:])
            volatility = float((last24.std()) / (last24.mean() or 1.0))
        else:
            volatility = 0.0

        now = datetime.datetime.utcnow()
        hour, weekday = now.hour, now.weekday()
        pct_from_high = (price - high) / (high + 1e-9) if high > 0 else 0.0
        dist_to_index = (price - idx) / (idx + 1e-9) if idx > 0 else 0.0

        price_delta_5m, volume_sum_5m, price_delta_15m, volume_sum_15m = _momentum_and_volumes_from_candles(candles)

        spread = get_orderbook_spread(symbol, depth=1)
        spread_bps = float(spread * 10_000.0)
        fee_bps = float(getattr(config, "TAKER_FEE", 0.0006) * 10_000.0)

        feats = {
            "index_price": idx,
            "last_price": float(price),
            "high": high,
            "low": low,
            "vol_24h": float(snap.get("vol_24h", 0.0)),
            "open_interest": float(snap.get("open_interest", 0.0)),
            "funding_rate": float(snap.get("funding_rate", 0.0)),
            "rsi": rsi_val,
            "atr_abs": atr_abs_val,
            "atr_norm": atr_norm,
            "volatility": float(volatility),
            "hour": float(hour),
            "weekday": float(weekday),
            "pct_from_high": float(pct_from_high),
            "dist_to_index": float(dist_to_index),
            "price_delta_5m": float(price_delta_5m),
            "volume_sum_5m": float(volume_sum_5m),
            "price_delta_15m": float(price_delta_15m),
            "volume_sum_15m": float(volume_sum_15m),
            "spread_bps": spread_bps,
            "fee_bps": fee_bps,
            "qty": float(qty),
            "direction": float(1 if direction == "long" else -1),
        }
        X, _ = _vectorize_features(meta, feats)

        thr_block = (meta or {}).get("thresholds", {}) or {}
        thr = float(thr_block.get("used", thr_block.get("global", default_thr)))

        proba = float(np.clip(model.predict_proba(X)[0][1], 0.0, 1.0))
        return (proba >= thr), proba, thr

    except Exception as e:
        log(f"[ML] predict_err: {e}")
        return True, 0.5, default_thr
