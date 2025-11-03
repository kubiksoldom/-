# -*- coding: utf-8 -*-
# ml_veto.py
import os
import json
import math
import pickle
import datetime
from typing import Any, Dict, List, Tuple, Optional, Set, NamedTuple

import numpy as np
import pandas as pd

import config
from utils import log, set_ml_status, write_cycle_log, now_iso


class PredictOutcome(NamedTuple):
    ok: bool
    proba: float
    threshold: float
    factor: float
    band: str
    meta_threshold: float
    strict_threshold: float
    effective_threshold: float
    features_ok: bool
    reason: str

LAST_OI: Dict[str, float] = {}

MODEL_FILE = getattr(config, "MODEL_FILE", os.getenv("MODEL_FILE", "rf_model.pkl"))
MODEL_META = getattr(config, "MODEL_META", os.getenv("MODEL_META", "model_meta.json"))

ML_SHADOW_MODE = bool(int(os.getenv("ML_SHADOW_MODE", str(getattr(config, "ML_SHADOW_MODE", 0)))))
ML_PROBA_STRICT = float(os.getenv("ML_PROBA_STRICT", str(getattr(config, "ML_PROBA_STRICT", 0.72))))
ML_REVERT_DD_PCT = float(os.getenv("ML_REVERT_DD_PCT", str(getattr(config, "ML_REVERT_DD_PCT", 6.0))))
ML_ACCEPT_DELTA_EV = float(os.getenv("ML_ACCEPT_DELTA_EV", str(getattr(config, "ML_ACCEPT_DELTA_EV", 0.0))))

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_META_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOGGED: Set[Tuple[str, Optional[float], str, Optional[float]]] = set()
_ML_LAST_STATUS: Optional[Tuple[str, bool, str]] = None


def _log_shadow_event(symbol: str,
                      direction: str,
                      *,
                      proba: float,
                      th_meta: Optional[float],
                      th_strict: float,
                      th_effective: float,
                      features_ok: bool,
                      reason: str,
                      decision_ok: bool) -> None:
    try:
        side = "buy" if decision_ok and direction == "long" else (
            "sell" if decision_ok and direction == "short" else "skip"
        )
        payload: Dict[str, Any] = {
            "tag": "ML_SHADOW",
            "symbol": symbol,
            "side": side,
            "proba": float(proba),
            "th_meta": float(th_meta) if th_meta is not None else None,
            "th_strict": float(th_strict),
            "th_eff": float(th_effective),
            "features_ok": bool(features_ok),
        }
        if reason:
            payload["reason"] = str(reason)
        payload.setdefault("ts_utc", now_iso())
        write_cycle_log(payload)
    except Exception as exc:
        log(f"[ML][shadow-log] {symbol} error: {exc}")


def _extract_precision_week(meta: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
    try:
        value = metrics.get("precision_week")
        if value is None:
            return None
        val = float(value)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _update_ml_status(status: str,
                      paused: bool,
                      *,
                      reason: str = "",
                      precision: Optional[float] = None,
                      threshold: Optional[float] = None) -> None:
    global _ML_LAST_STATUS
    set_ml_status(status, paused, reason=reason, precision_week=precision, threshold=threshold)
    key = (status, paused, reason)
    if _ML_LAST_STATUS == key:
        return
    _ML_LAST_STATUS = key
    if paused:
        if status == "unavailable":
            log("[ML] Модель недоступна — торговля приостановлена", level="WARNING")
        elif status == "unsafe" and precision is not None and threshold is not None:
            log(
                f"[ML] Торговля приостановлена: weekly precision {precision:.3f} < порога {threshold:.3f}",
                level="WARNING",
            )
        elif status == "degraded":
            log("[ML] Торговля приостановлена: нет weekly precision в метаданных", level="WARNING")
        else:
            log(f"[ML] Торговля приостановлена: {reason or status}", level="WARNING")
    else:
        if precision is not None and threshold is not None:
            log(f"[ML] Активна: weekly precision {precision:.3f} (порог {threshold:.3f})")
        else:
            log("[ML] Активна: weekly precision в норме")

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


def _bb_width(closes: List[float], period: int = 20) -> float:
    if len(closes) < period:
        return 0.0
    window = np.array(closes[-period:], dtype=float)
    mid = float(window.mean())
    std = float(window.std(ddof=0))
    if mid == 0:
        return 0.0
    return (4.0 * std) / mid


def _zscore_latest(closes: List[float], period: int = 20) -> float:
    if len(closes) < period:
        return 0.0
    window = np.array(closes[-period:], dtype=float)
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    if std == 0:
        return 0.0
    return float((window[-1] - mean) / std)


def _adx_like(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 0.0
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)
    up = np.maximum(0.0, h[1:] - h[:-1])
    dn = np.maximum(0.0, l[:-1] - l[1:])
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    if len(up) < period or len(tr) < period:
        return 0.0
    up_n = up[-period:].mean()
    dn_n = dn[-period:].mean()
    tr_n = tr[-period:].mean()
    if tr_n <= 0:
        return 0.0
    plus_di = (up_n / tr_n) * 100.0
    minus_di = (dn_n / tr_n) * 100.0
    dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9) * 100.0
    return float(dx)


def triple_barrier_label(
    closes: List[float],
    atr: float,
    horizon: int = 10,
    up_mult: float = 1.5,
    down_mult: float = 1.5,
) -> int:
    """Возвращает 1/0/-1 по triple-barrier разметке."""
    if len(closes) < 2:
        return 0
    atr = max(float(atr), 1e-9)
    horizon = max(1, int(horizon))
    ref = float(closes[-1])
    up_barrier = ref * (1 + up_mult * atr / ref)
    dn_barrier = ref * (1 - down_mult * atr / ref)
    future = closes[-horizon:]
    for price in future:
        if price >= up_barrier:
            return 1
        if price <= dn_barrier:
            return -1
    return 0 if future[-1] == ref else (1 if future[-1] > ref else -1)

def _vectorize_features(meta: Optional[Dict], feats_dict: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    order = None
    try:
        order = (meta or {}).get("features", None)
    except Exception:
        order = None
    if not order:
        order = [
            "feature_index_price","feature_last_price","feature_high","feature_low","feature_vol_24h",
            "feature_open_interest","feature_funding_rate","feature_rsi","feature_atr_abs","feature_atr_norm",
            "feature_atr_pct","feature_volatility","feature_hour","feature_weekday","feature_pct_from_high",
            "feature_dist_to_index","feature_price_delta_5m","feature_volume_sum_5m","feature_price_delta_15m",
            "feature_volume_sum_15m","feature_spread_bps","feature_fee_bps","feature_ret_1","feature_ret_3",
            "feature_ret_5","feature_ret_10","feature_mom_k","feature_book_imbalance","feature_oi_change",
            "feature_qty","feature_direction","feature_tp_pct_used","feature_sl_pct_used"
        ]
        order += [
            "index_price","last_price","high","low","vol_24h",
            "open_interest","funding_rate","rsi","atr_abs","atr_norm",
            "volatility","hour","weekday","pct_from_high","dist_to_index",
            "price_delta_5m","volume_sum_5m","price_delta_15m","volume_sum_15m",
            "spread_bps","fee_bps","ret_1","ret_3","ret_5","ret_10",
            "mom_k","book_imbalance","oi_change","qty","direction","tp_pct_used","sl_pct_used"
        ]
    uniq_order: List[str] = []
    seen: Set[str] = set()
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        uniq_order.append(key)
    row = [float(feats_dict.get(k, 0.0)) for k in uniq_order]
    return np.array([row], dtype=float), uniq_order

def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        path = path.strip()
    except Exception:
        pass
    return os.path.abspath(path) if path else None


def _maybe_clear_cache() -> None:
    """Сбрасывает кэш, если выставлен ML_CACHE_BUST=1."""
    if os.getenv("ML_CACHE_BUST") == "1":
        _MODEL_CACHE.clear()
        _META_CACHE.clear()
        _CACHE_LOGGED.clear()
        os.environ["ML_CACHE_BUST"] = "0"
        log("[ML] cache bust requested; caches cleared")


def get_model_and_meta_cached(
    model_path: str = "rf_model.pkl",
    meta_path: str = "model_meta.json",
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Загружает модель и мету с диска с кэшированием."""

    _maybe_clear_cache()

    resolved_model_path = _resolve_path(model_path) or _resolve_path(MODEL_FILE)
    resolved_meta_path = _resolve_path(meta_path) or _resolve_path(MODEL_META)

    def _mtime(path: Optional[str]) -> Optional[float]:
        if not path:
            return None
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    model_mtime = _mtime(resolved_model_path)
    meta_mtime = _mtime(resolved_meta_path)

    model_obj: Optional[Any] = None
    meta_obj: Optional[Dict[str, Any]] = None

    model_key = resolved_model_path or "<none>"
    model_entry = _MODEL_CACHE.get(model_key)
    if (
        model_entry
        and model_entry.get("model") is not None
        and model_entry.get("mtime") == model_mtime
    ):
        model_obj = model_entry["model"]
    else:
        try:
            if not resolved_model_path:
                raise FileNotFoundError("MODEL_FILE path is empty")
            with open(resolved_model_path, "rb") as f:
                model_obj = pickle.load(f)
            _MODEL_CACHE[model_key] = {
                "model": model_obj,
                "mtime": model_mtime,
            }
        except Exception as e:
            log(f"[ML] Не удалось загрузить модель: {e}")
            if model_entry and model_entry.get("model") is not None:
                model_obj = model_entry.get("model")
            else:
                model_obj = None

    meta_key = resolved_meta_path or "<none>"
    meta_entry = _META_CACHE.get(meta_key)
    if (
        meta_entry
        and meta_entry.get("meta") is not None
        and meta_entry.get("mtime") == meta_mtime
    ):
        meta_obj = meta_entry["meta"]
    else:
        if resolved_meta_path and os.path.exists(resolved_meta_path):
            try:
                with open(resolved_meta_path, "r", encoding="utf-8") as f:
                    meta_obj = json.load(f)
                _META_CACHE[meta_key] = {
                    "meta": meta_obj,
                    "mtime": meta_mtime,
                }
            except Exception as e:
                log(f"[ML] Не удалось загрузить мета-информацию: {e}")
                if meta_entry and meta_entry.get("meta") is not None:
                    meta_obj = meta_entry.get("meta")
        else:
            meta_obj = meta_entry.get("meta") if meta_entry else None

    log_key = (
        resolved_model_path or "<none>",
        model_mtime,
        resolved_meta_path or "<none>",
        meta_mtime,
    )
    if log_key not in _CACHE_LOGGED:
        features_count = len((meta_obj or {}).get("features", []) or [])
        thresholds_info = (meta_obj or {}).get("thresholds", {}) or {}
        log(
            f"[ML] model loaded into cache (features={features_count}, thresholds={thresholds_info})"
        )
        _CACHE_LOGGED.add(log_key)

    return model_obj, meta_obj


def load_model_and_meta():
    """Загружает модель и метаданные, сигнализируя о недоступности."""

    model_obj, meta_obj = get_model_and_meta_cached(MODEL_FILE, MODEL_META)
    threshold = float(getattr(config, "ML_MIN_WEEKLY_PREC", 0.52))
    precision_week = _extract_precision_week(meta_obj)

    if model_obj is None or meta_obj is None:
        _update_ml_status(
            "unavailable",
            True,
            reason="artifacts_missing",
            precision=precision_week,
            threshold=threshold,
        )
        return None, None

    if precision_week is None:
        _update_ml_status(
            "degraded",
            True,
            reason="weekly_precision_missing",
            precision=None,
            threshold=threshold,
        )
        return None, meta_obj

    if precision_week < threshold:
        _update_ml_status(
            "unsafe",
            True,
            reason="precision_drop",
            precision=precision_week,
            threshold=threshold,
        )
        return None, meta_obj

    _update_ml_status("ok", False, precision=precision_week, threshold=threshold)
    return model_obj, meta_obj


def _confidence_to_factor(prob: float) -> Tuple[float, str]:
    """Преобразует вероятность в коэффициент размера позиции."""

    try:
        high_thr = float(getattr(config, "ML_CONF_HIGH", 0.80))
    except Exception:
        high_thr = 0.80
    try:
        mid_thr = float(getattr(config, "ML_CONF_MID", 0.65))
    except Exception:
        mid_thr = 0.65

    if not math.isfinite(prob):
        return 0.0, "invalid"
    if prob >= high_thr:
        return 1.0, "full"
    if prob >= mid_thr:
        return 0.5, "reduced"
    return 0.0, "blocked"

def predict_ok(
    model,
    meta: Optional[Dict],
    symbol: str,
    direction: str,
    qty: float,
    price: Optional[float] = None,
    atr: Optional[float] = None,
    candles: Optional[List[List[float]]] = None,
) -> PredictOutcome:
    default_thr = float(getattr(config, "ML_THRESHOLD", 0.58))
    if model is None or meta is None:
        return PredictOutcome(
            ok=False,
            proba=0.0,
            threshold=default_thr,
            factor=0.0,
            band="unavailable",
            meta_threshold=default_thr,
            strict_threshold=max(default_thr, float(ML_PROBA_STRICT)),
            effective_threshold=max(default_thr, float(ML_PROBA_STRICT)),
            features_ok=False,
            reason="unavailable",
        )

    try:
        if (candles is None) or (price is None):
            from bybit_api import fetch_price_history
        from bybit_api import get_ticker_snapshot, get_orderbook_spread, orderbook_imbalance

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

        now = datetime.datetime.now(datetime.UTC)
        hour, weekday = now.hour, now.weekday()
        pct_from_high = (price - high) / (high + 1e-9) if high > 0 else 0.0
        dist_to_index = (price - idx) / (idx + 1e-9) if idx > 0 else 0.0

        price_delta_5m, volume_sum_5m, price_delta_15m, volume_sum_15m = _momentum_and_volumes_from_candles(candles)

        spread = get_orderbook_spread(symbol, depth=1)
        spread_bps = float(spread * 10_000.0)
        fee_bps = float(getattr(config, "TAKER_FEE", 0.0006) * 10_000.0)
        imbalance = float(orderbook_imbalance(symbol, depth=5))

        ret1 = ret3 = ret5 = ret10 = 0.0
        mom_k = 0.0
        bb_w = _bb_width(closes)
        z_last = _zscore_latest(closes)
        highs = [float(x[1]) for x in candles] if candles else []
        lows = [float(x[2]) for x in candles] if candles else []
        adx_like = _adx_like(highs, lows, closes)
        if len(closes) >= 2:
            arr = np.array(closes, dtype=float)
            ret = np.diff(arr) / arr[:-1]
            ret1 = float(ret[-1])
            if len(arr) >= 4:
                ret3 = float(arr[-1] / arr[-4] - 1.0)
            if len(arr) >= 6:
                ret5 = float(arr[-1] / arr[-6] - 1.0)
            if len(arr) >= 11:
                ret10 = float(arr[-1] / arr[-11] - 1.0)
            mom_k = float(arr[-1] - arr[-min(len(arr), 15)])

        prev_oi = LAST_OI.get(symbol, float(snap.get("open_interest", 0.0)))
        cur_oi = float(snap.get("open_interest", 0.0))
        oi_change = cur_oi - prev_oi
        LAST_OI[symbol] = cur_oi

        feature_map = {
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
            "atr_pct": atr_norm * 100.0,
            "bb_width": bb_w,
            "zscore": z_last,
            "adx_like": adx_like,
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
            "ret_1": ret1,
            "ret_3": ret3,
            "ret_5": ret5,
            "ret_10": ret10,
            "mom_k": mom_k,
            "book_imbalance": imbalance,
            "oi_change": oi_change,
            "qty": float(qty),
            "direction": float(1 if direction == "long" else -1),
            "tp_pct_used": 0.0,
            "sl_pct_used": 0.0,
        }
        feats = {f"feature_{k}": float(v) for k, v in feature_map.items()}
        for k, v in feature_map.items():
            feats.setdefault(k, float(v))
        X, _ = _vectorize_features(meta, feats)

        thr_block = (meta or {}).get("thresholds", {}) or {}
        atr_percentiles = (meta or {}).get("atr_percentiles", {}) or {}

        def _flt(value, fallback=None):
            try:
                if value is None:
                    raise ValueError
                return float(value)
            except (TypeError, ValueError):
                return fallback

        thr_default = _flt(thr_block.get("used"))
        if thr_default is None:
            thr_default = _flt(thr_block.get("global"), default_thr)
        thr_global = _flt(thr_block.get("global"), thr_default)
        thr_ev_only = _flt(thr_block.get("ev_only"), thr_default)

        regime_thresholds = {}
        for key in ("regime_low", "regime_high", "regime_ultra"):
            regime_thresholds[key] = _flt(thr_block.get(key), thr_default)

        p50_val = _flt(atr_percentiles.get("p50"))
        p90_val = _flt(atr_percentiles.get("p90"))
        if p50_val is not None and p90_val is not None and p90_val < p50_val:
            p90_val = p50_val

        current_regime = None
        if p50_val is not None and p90_val is not None and math.isfinite(atr_norm):
            if atr_norm < p50_val:
                current_regime = "regime_low"
            elif atr_norm < p90_val:
                current_regime = "regime_high"
            else:
                current_regime = "regime_ultra"

        used_mode = str(thr_block.get("used_mode", "global"))
        thr_to_apply = thr_default if thr_default is not None else default_thr

        if used_mode == "global":
            thr_to_apply = thr_default if thr_default is not None else default_thr
        elif used_mode == "ev_only":
            thr_to_apply = thr_ev_only if thr_ev_only is not None else thr_to_apply
        else:
            preferred_regime = used_mode if used_mode in regime_thresholds else None
            regime_key = current_regime or preferred_regime
            regime_thr = regime_thresholds.get(regime_key) if regime_key else None
            if regime_thr is None and preferred_regime and preferred_regime in regime_thresholds:
                regime_thr = regime_thresholds.get(preferred_regime)
            if regime_thr is None:
                regime_thr = thr_default if thr_default is not None else thr_global
            thr_to_apply = regime_thr if regime_thr is not None else thr_to_apply

        thr = float(thr_to_apply if thr_to_apply is not None else default_thr)
        meta_thr = thr
        try:
            meta_thr = float((meta or {}).get("proba_threshold", thr))
        except Exception:
            meta_thr = thr
        strict_thr = max(float(meta_thr), float(ML_PROBA_STRICT))

        features_ok = bool(np.all(np.isfinite(X)))
        proba = 0.0
        predict_error = False
        predict_exc: Optional[Exception] = None
        if features_ok:
            try:
                proba = float(np.clip(model.predict_proba(X)[0][1], 0.0, 1.0))
            except Exception as exc:
                predict_error = True
                predict_exc = exc
        decision_reason = ""
        if not features_ok:
            decision_reason = "nan_features"
        if predict_error:
            decision_reason = "predict_error"

        effective_thr = strict_thr
        thr_used = strict_thr
        decision_ok = features_ok and not predict_error and (proba >= strict_thr)

        ev_params = (meta or {}).get("ev_params", {}) or {}
        try:
            ev_fee = float(ev_params.get("fee", 0.0014))
        except Exception:
            ev_fee = 0.0014
        try:
            ev_r_avg = float(ev_params.get("r_avg", 1.8))
        except Exception:
            ev_r_avg = 1.8
        ev_est = (proba * ev_r_avg) - ((1.0 - proba) * 1.0) - ev_fee
        if ML_ACCEPT_DELTA_EV > 0.0 and decision_ok and ev_est < ML_ACCEPT_DELTA_EV:
            decision_ok = False
            decision_reason = "low_ev"

        factor, band = _confidence_to_factor(proba)
        if decision_ok and factor <= 0.0:
            decision_ok = False
            decision_reason = "low_confidence"

        if not decision_ok and not decision_reason:
            decision_reason = "below_threshold"

        if predict_error and ML_SHADOW_MODE:
            _log_shadow_event(
                symbol,
                direction,
                proba=0.0,
                th_meta=meta_thr,
                th_strict=strict_thr,
                th_effective=strict_thr,
                features_ok=False,
                reason=decision_reason or "predict_error",
                decision_ok=False,
            )
            return PredictOutcome(
                ok=True,
                proba=0.0,
                threshold=thr_used,
                factor=1.0,
                band="shadow",
                meta_threshold=meta_thr,
                strict_threshold=strict_thr,
                effective_threshold=effective_thr,
                features_ok=False,
                reason=decision_reason or "predict_error",
            )
        if predict_error:
            log(f"[ML] predict_err: {predict_exc}")
            return PredictOutcome(
                ok=False,
                proba=0.0,
                threshold=thr_used,
                factor=0.0,
                band="error",
                meta_threshold=meta_thr,
                strict_threshold=strict_thr,
                effective_threshold=effective_thr,
                features_ok=features_ok,
                reason=decision_reason or "predict_error",
            )

        if ML_SHADOW_MODE:
            _log_shadow_event(
                symbol,
                direction,
                proba=proba,
                th_meta=meta_thr,
                th_strict=strict_thr,
                th_effective=effective_thr,
                features_ok=features_ok,
                reason=decision_reason if not decision_ok else "",
                decision_ok=decision_ok,
            )
            decision_label = "pass" if decision_ok else "block"
            log(
                f"[ML][SHADOW] {symbol} {direction}: prob={proba:.3f} thr={strict_thr:.3f} ev={ev_est:.4f} → {decision_label}"
            )
            return PredictOutcome(
                ok=True,
                proba=proba,
                threshold=thr_used,
                factor=1.0,
                band="shadow",
                meta_threshold=meta_thr,
                strict_threshold=strict_thr,
                effective_threshold=effective_thr,
                features_ok=features_ok,
                reason=decision_reason,
            )

        band_value = band if features_ok else "invalid"
        return PredictOutcome(
            ok=decision_ok and not predict_error,
            proba=proba,
            threshold=thr_used,
            factor=float(factor),
            band=band_value,
            meta_threshold=meta_thr,
            strict_threshold=strict_thr,
            effective_threshold=effective_thr,
            features_ok=features_ok,
            reason=decision_reason,
        )

    except Exception as e:
        log(f"[ML] predict_err: {e}")
        return PredictOutcome(
            ok=False,
            proba=0.0,
            threshold=default_thr,
            factor=0.0,
            band="error",
            meta_threshold=default_thr,
            strict_threshold=max(default_thr, float(ML_PROBA_STRICT)),
            effective_threshold=max(default_thr, float(ML_PROBA_STRICT)),
            features_ok=False,
            reason="error",
        )
