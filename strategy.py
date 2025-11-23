# -*- coding: utf-8 -*-
"""
strategy.py (универсальный, с роутером) — 2025-09-22
----------------------------------------------------
Совместим с твоей старой логикой:
  • detect_impulse(candles) -> "buy"|"sell"|"hold"  (старый API сохранён)
Добавлено:
  • decide_with_router(symbol, timeframe, candles, ctx) -> dict
    dict: {action, reason, sl, tp, meta{regime, strategy, entry_hint, confidence}}

Контент:
  - Помощники (EMA, ATR, …)
  - Стратегии: ImpulseBreakout (обновлённая версия твоей), EngulfingTrend, InsideNR4, PinBarLevel
  - Детектор режима рынка (trend / range / squeeze / chaos_lowvol)
  - Онлайн-бандит (по win-rate/avgR, Thompson-приближённый скор)
  - Роутер StrategyRouter
  - Внешние функции: detect_impulse, decide_with_router

Параметры берутся из config.py, но есть дефолты, чтобы всё работало из коробки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json, os, math, time, random

import numpy as np
import pandas as pd

from utils import fee_aware_r_min, log, write_cycle_log, now_iso

# === CONFIG (мягкие импорты с дефолтами) ===
try:
    import config
except Exception:
    class _C: pass
    config = _C()

def _cfg(name, default):
    return getattr(config, name, default)

# ===== helpers =====
def _ema_series(series: pd.Series, period: int) -> pd.Series:
    n = int(max(1, period))
    if len(series) < max(3, n):
        last = float(series.iloc[-1]) if len(series) else 0.0
        return pd.Series([last] * len(series), dtype=float, index=series.index)
    return series.ewm(span=n, adjust=False).mean()

def _momentum(series: pd.Series, lookback: int) -> float:
    lb = int(max(1, lookback))
    if len(series) <= lb:
        return 0.0
    a = float(series.iloc[-1]); b = float(series.iloc[-1 - lb])
    if b == 0: return 0.0
    return (a / b) - 1.0

def _atr_abs(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
    n = int(max(1, period))
    if len(closes) < n + 1:
        return 0.0
    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    val = float(atr.iloc[-1]) if len(atr) and pd.notna(atr.iloc[-1]) else 0.0
    return max(0.0, val)

def _downsample_closes_ohlc(candles: List[List[float]], m: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if m <= 1:
        s_close = pd.Series([float(c[3]) for c in candles], dtype=float)
        s_high  = pd.Series([float(c[1]) for c in candles], dtype=float)
        s_low   = pd.Series([float(c[2]) for c in candles], dtype=float)
        s_open  = pd.Series([float(c[0]) for c in candles], dtype=float)
        return s_open, s_high, s_low, s_close
    n = len(candles)
    k = n // m
    if k < 5:
        s_close = pd.Series([float(c[3]) for c in candles], dtype=float)
        s_high  = pd.Series([float(c[1]) for c in candles], dtype=float)
        s_low   = pd.Series([float(c[2]) for c in candles], dtype=float)
        s_open  = pd.Series([float(c[0]) for c in candles], dtype=float)
        return s_open, s_high, s_low, s_close
    ovals, hvals, lvals, cvals = [], [], [], []
    for i in range(k):
        chunk = candles[i*m:(i+1)*m]
        ovals.append(float(chunk[0][0]))
        hvals.append(max(float(x[1]) for x in chunk))
        lvals.append(min(float(x[2]) for x in chunk))
        cvals.append(float(chunk[-1][3]))
    return (
        pd.Series(ovals, dtype=float),
        pd.Series(hvals, dtype=float),
        pd.Series(lvals, dtype=float),
        pd.Series(cvals, dtype=float),
    )

# === numpy индикаторы для роутера ===
def ema_np(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    k = 2.0 / (n + 1.0)
    s = None
    for i, v in enumerate(x):
        if np.isnan(v): continue
        s = v if s is None else (v*k + s*(1-k))
        out[i] = s
    return out

def atr_np(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    h, l, c = map(lambda a: np.asarray(a, dtype=float), (h,l,c))
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    out = np.full_like(c, np.nan)
    if len(tr) >= n:
        roll = np.convolve(tr, np.ones(n)/n, mode='valid')
        out[n:] = roll
    return out

def slope_np(series: np.ndarray, n: int = 20) -> float:
    y = np.asarray(series[-n:], dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2 or np.any(np.isnan(y)): return 0.0
    x_mean, y_mean = x.mean(), y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return float(num/den) if den > 0 else 0.0

def adx_like_np(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> float:
    up = np.maximum(0.0, h[1:] - h[:-1])
    dn = np.maximum(0.0, l[:-1] - l[1:])
    if len(up) < n: return 0.0
    up_n = up[-n:].mean()
    dn_n = dn[-n:].mean()
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    tr_n = tr[-n:].mean() if len(tr) >= n else 1.0
    if tr_n <= 1e-12: return 0.0
    plus_di = (up_n / tr_n) * 100.0
    minus_di = (dn_n / tr_n) * 100.0
    dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9) * 100.0
    return float(dx)


def _rolling_mean_std(x: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    n = max(1, int(n))
    x = np.asarray(x, dtype=float)
    out_mean = np.full_like(x, np.nan)
    out_std = np.full_like(x, np.nan)
    if len(x) < n:
        return out_mean, out_std
    csum = np.cumsum(np.insert(x, 0, 0.0))
    csum2 = np.cumsum(np.insert(x * x, 0, 0.0))
    for i in range(n - 1, len(x)):
        total = csum[i + 1] - csum[i + 1 - n]
        total2 = csum2[i + 1] - csum2[i + 1 - n]
        mean = total / n
        var = max(total2 / n - mean * mean, 0.0)
        out_mean[i] = mean
        out_std[i] = math.sqrt(var)
    return out_mean, out_std


def bb_width_np(close: np.ndarray, n: int, k: float = 2.0) -> np.ndarray:
    """Относительная ширина полос Боллинджера (upper-lower)/middle."""
    close = np.asarray(close, dtype=float)
    mean, std = _rolling_mean_std(close, n)
    out = np.full_like(close, np.nan)
    valid = mean != 0.0
    width = 2.0 * k * std
    out[valid] = width[valid] / np.abs(mean[valid])
    out[~valid] = np.nan
    return out


def donchian_levels(high: np.ndarray, low: np.ndarray, n: int) -> Tuple[float, float]:
    """Возвращает максимум/минимум за последние n баров (включая текущий)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if len(high) < n or len(low) < n:
        return float(high[-1] if len(high) else 0.0), float(low[-1] if len(low) else 0.0)
    hi = float(np.max(high[-n:]))
    lo = float(np.min(low[-n:]))
    return hi, lo


def rolling_vol_np(ret: np.ndarray, n: int) -> np.ndarray:
    """Оценка скользящей волатильности (стд. отклонение) для массива доходностей."""
    ret = np.asarray(ret, dtype=float)
    _, std = _rolling_mean_std(ret, n)
    return std


def vol_percentile(vol: np.ndarray, n: int) -> float:
    """Позиция последнего значения волатильности в процентиле по окну n."""
    vol = np.asarray(vol, dtype=float)
    n = max(1, int(n))
    if len(vol) < n:
        return 50.0
    window = vol[-n:]
    last = window[-1]
    rank = float(np.sum(window <= last))
    return (rank / len(window)) * 100.0


def welford_mean_var(mean: float, m2: float, count: int, value: float) -> Tuple[float, float, int]:
    """Онлайн-обновление среднего и несмещённой дисперсии (алгоритм Вэлфорда)."""
    count_new = count + 1
    delta = value - mean
    mean_new = mean + delta / count_new
    delta2 = value - mean_new
    m2_new = m2 + delta * delta2
    return mean_new, m2_new, count_new


def mann_whitney_pvalue(sample_a: List[float], sample_b: List[float]) -> float:
    """Приближённый p-value критерия Манна–Уитни (нормальное приближение)."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.concatenate([a, b]))) + 1
    rank_a = ranks[: len(a)]
    u = float(rank_a.sum() - len(a) * (len(a) + 1) / 2.0)
    mean_u = len(a) * len(b) / 2.0
    var_u = len(a) * len(b) * (len(a) + len(b) + 1) / 12.0
    if var_u <= 0:
        return 1.0
    z = (u - mean_u) / math.sqrt(var_u)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def atr_r_targets(entry: float,
                  direction: str,
                  atr_value: float,
                  k_sl: float,
                  k_tp: float,
                  r_min: float) -> Tuple[float, float]:
    """Возвращает (SL, TP) по ATR-модели с минимальным R."""
    entry = float(entry)
    atr_value = max(float(atr_value), 1e-9)
    k_sl = max(float(k_sl), 0.0)
    k_tp = max(float(k_tp), 0.0)
    r_min = max(float(r_min), 0.0)
    if direction == "buy":
        sl = entry - k_sl * atr_value
        sl = min(sl, entry - 1e-9)
        r = entry - sl
        tp = entry + max(k_tp * atr_value, r_min * r)
    else:
        sl = entry + k_sl * atr_value
        sl = max(sl, entry + 1e-9)
        r = sl - entry
        tp = entry - max(k_tp * atr_value, r_min * r)
    return float(sl), float(tp)

# ===== Контракты =====
@dataclass
class Candidate:
    def __init__(self, side, entry, sl, tp, reason, confidence, strategy):
        self.side = side
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.reason = reason
        self.confidence = confidence
        self.strategy = strategy
        self.score = confidence   # 🔥 FIX


@dataclass
class Signal:
    action: str        # 'buy'|'sell'|'hold'
    reason: str
    sl: Optional[float]
    tp: Optional[float]
    meta: Dict[str, Any]

# ===== Стратегии =====
class StrategyBase:
    name = "base"
    def propose(self, candles: Dict[str, np.ndarray], ctx: Dict[str, Any]) -> Optional[Candidate]:
        raise NotImplementedError

class EngulfingTrend(StrategyBase):
    name = "engulfing_trend"
    def propose(self, candles, ctx):
        o,h,l,c = candles['open'], candles['high'], candles['low'], candles['close']
        min_bars = int(_cfg("MIN_BARS", getattr(config, "MIN_BARS", 210)))
        if len(c) < min_bars:
            return None
        a = atr_np(h,l,c,14); e50 = ema_np(c,50); e200 = ema_np(c,200)
        if math.isnan(a[-1]) or math.isnan(e50[-1]) or math.isnan(e200[-1]): return None
        bull = c[-1] > e50[-1] > e200[-1]
        bear = c[-1] < e50[-1] < e200[-1]
        body_last = abs(c[-1]-o[-1])
        engulf_bull = (c[-1]>o[-1]) and (o[-1]<=c[-2]) and (c[-1]>=o[-2]) and (body_last>=0.7*a[-1])
        engulf_bear = (c[-1]<o[-1]) and (o[-1]>=c[-2]) and (c[-1]<=o[-2]) and (body_last>=0.7*a[-1])
        if bull and engulf_bull:
            entry = float(c[-1])
            sl = float(l[-1]-0.5*a[-1])
            r = entry - sl
            tp = float(entry + 2.0*r)
            return Candidate("buy", entry, sl, tp, "engulfing_trend", 0.65, self.name)
        if bear and engulf_bear:
            entry = float(c[-1])
            sl = float(h[-1]+0.5*a[-1])
            r = sl - entry
            tp = float(entry - 2.0*r)
            return Candidate("sell", entry, sl, tp, "engulfing_trend", 0.65, self.name)
        return None

class InsideNR4(StrategyBase):
    name = "inside_nr4"
    def propose(self, candles, ctx):
        o,h,l,c = candles['open'], candles['high'], candles['low'], candles['close']
        if len(c) < 60: return None
        a = atr_np(h,l,c,14); e50 = ema_np(c,50); e200 = ema_np(c,200)
        if math.isnan(a[-1]) or math.isnan(e50[-1]) or math.isnan(e200[-1]): return None
        bull = c[-1] > e50[-1] > e200[-1]
        bear = c[-1] < e50[-1] < e200[-1]
        mother_hi, mother_lo = h[-2], l[-2]
        inside = (h[-1] <= mother_hi) and (l[-1] >= mother_lo)
        if not inside: return None
        tr_last = max(h[-1]-l[-1], abs(h[-1]-c[-2]), abs(l[-1]-c[-2]))
        trs = []
        for i in range(4):
            tr_i = max(h[-2-i]-l[-2-i], abs(h[-2-i]-c[-3-i]), abs(l[-2-i]-c[-3-i]))
            trs.append(tr_i)
        if tr_last >= min(trs): return None
        if bull:
            entry = float(mother_hi)
            sl = float(mother_lo - 0.25*a[-1])
            tp = float(entry + 1.5*(entry - sl))
            return Candidate("buy", entry, sl, tp, "inside_nr4_trend", 0.58, self.name)
        if bear:
            entry = float(mother_lo)
            sl = float(mother_hi + 0.25*a[-1])
            tp = float(entry - 1.5*(sl - entry))
            return Candidate("sell", entry, sl, tp, "inside_nr4_trend", 0.58, self.name)
        return None

class PinBarLevel(StrategyBase):
    name = "pinbar_level"
    def propose(self, candles, ctx):
        o,h,l,c = candles['open'], candles['high'], candles['low'], candles['close']
        if len(c) < 40: return None
        a = atr_np(h,l,c,14)
        if math.isnan(a[-1]): return None
        body = abs(c[-1]-o[-1])
        up_w = h[-1] - max(c[-1], o[-1])
        dn_w = min(c[-1], o[-1]) - l[-1]
        if dn_w >= 2*body and dn_w >= 0.8*a[-1]:
            entry = float(c[-1])
            sl = float(l[-1] - 0.25*a[-1])
            tp = float(entry + 2*(entry - sl))
            return Candidate("buy", entry, sl, tp, "bull_pinbar", 0.55, self.name)
        if up_w >= 2*body and up_w >= 0.8*a[-1]:
            entry = float(c[-1])
            sl = float(h[-1] + 0.25*a[-1])
            tp = float(entry - 2*(sl - entry))
            return Candidate("sell", entry, sl, tp, "bear_pinbar", 0.55, self.name)
        return None

class ImpulseBreakout(StrategyBase):
    """
    Обновлённая версия твоей импульс-стратегии.
    Даёт Candidate (с entry/sl/tp), а не только “buy/sell/hold”.
    """
    name = "impulse_breakout"
    def propose(self, candles, ctx):
        # candles ожидается как словарь numpy массивов (см. ниже convert)
        o,h,l,c,v = (candles[k] for k in ("open","high","low","close","volume"))
        # перекладка в pandas для твоей логики
        opens  = pd.Series(o, dtype=float)
        highs  = pd.Series(h, dtype=float)
        lows   = pd.Series(l, dtype=float)
        closes = pd.Series(c, dtype=float)
        vols   = pd.Series(v, dtype=float)

        # Параметры (с дефолтами)
        EMA_FAST     = int(_cfg("EMA_FAST", 9))
        EMA_SLOW     = int(_cfg("EMA_SLOW", 21))
        MOM_LOOKBACK = int(_cfg("MOM_LOOKBACK", 5))
        MOM_MIN_PCT  = float(_cfg("MOM_MIN_PCT", 0.001))
        BRK_LOOKBACK = int(_cfg("BREAKOUT_LOOKBACK", 20))
        BRK_PAD_PCT  = float(_cfg("BREAKOUT_PAD_PCT", 0.0005))
        ATR_PERIOD         = int(_cfg("STRAT_ATR_PERIOD", 14))
        MIN_BODY_ATR_FRAC  = float(_cfg("STRAT_MIN_BODY_ATR_FRAC", 0.15))
        VOL_LOOKBACK       = int(_cfg("STRAT_VOL_LOOKBACK", 30))
        VOL_BOOST_MIN      = float(_cfg("STRAT_VOL_BOOST_MIN", 1.10))
        EMA_SLOPE_LOOKBACK = int(_cfg("STRAT_EMA_SLOPE_LOOKBACK", 3))
        MIN_ATR_PCT_GLOBAL = float(_cfg("MIN_ATR_PCT", 0.0008))
        MOM_K_ATR          = float(_cfg("STRAT_MOM_K_ATR", 0.8))
        PAD_K_ATR          = float(_cfg("STRAT_PAD_K_ATR", 0.3))
        CONFIRM_BARS       = int(_cfg("STRAT_CONFIRM_BARS", 2))
        USE_PULLBACK       = bool(int(_cfg("STRAT_USE_PULLBACK", 1)))
        PULLBACK_LOOKBACK  = int(_cfg("STRAT_PULLBACK_LOOKBACK", 6))
        PULLBACK_BAND_ATR_K= float(_cfg("STRAT_PULLBACK_BAND_ATR_K", 0.5))
        HTF_ENABLE         = bool(int(_cfg("STRAT_HTF_ENABLE", 1)))
        HTF_MINUTES        = int(_cfg("STRAT_HTF_MINUTES", 5))
        HTF_EMA_FAST       = int(_cfg("STRAT_HTF_EMA_FAST", 9))
        HTF_EMA_SLOW       = int(_cfg("STRAT_HTF_EMA_SLOW", 21))
        SL_ATR_K           = float(_cfg("STRAT_SL_ATR_K", 1.2))
        TP_ATR_K           = float(_cfg("STRAT_TP_ATR_K", 3.0))
        fee_rate           = float(_cfg("TAKER_FEE", 0.0006))
        MIN_R              = fee_aware_r_min(float(_cfg("STRAT_MIN_R", 1.5)), fee_rate)

        need_len = max(
            EMA_SLOW + 2, BRK_LOOKBACK + CONFIRM_BARS + 2, MOM_LOOKBACK + 2,
            ATR_PERIOD + 2, VOL_LOOKBACK + 2, EMA_SLOPE_LOOKBACK + 2
        )
        if len(closes) < need_len:
            return None

        ema_fast_series = _ema_series(closes, EMA_FAST)
        ema_slow_series = _ema_series(closes, EMA_SLOW)
        ema_fast = float(ema_fast_series.iloc[-1])
        ema_slow = float(ema_slow_series.iloc[-1])
        mom      = _momentum(closes, MOM_LOOKBACK)

        atr_abs = _atr_abs(highs, lows, closes, ATR_PERIOD)
        last_px = float(closes.iloc[-1])
        atr_pct = (atr_abs / last_px) if last_px > 0 else 0.0

        mom_min_eff = max(MOM_MIN_PCT, MOM_K_ATR * atr_pct)
        pad_eff     = max(BRK_PAD_PCT, PAD_K_ATR * atr_pct)

        excl = max(1, CONFIRM_BARS)
        hist_high = h[:-excl] if excl > 0 else h
        hist_low = l[:-excl] if excl > 0 else l
        if len(hist_high) < BRK_LOOKBACK or len(hist_low) < BRK_LOOKBACK:
            return None
        prev_high, prev_low = donchian_levels(hist_high, hist_low, BRK_LOOKBACK)
        level_up  = prev_high * (1.0 + pad_eff) if prev_high > 0 else float("inf")
        level_dn  = prev_low  * (1.0 - pad_eff) if prev_low  > 0 else float("-inf")

        if len(closes) < CONFIRM_BARS + 1:
            return None
        last_seq = closes.iloc[-CONFIRM_BARS:]
        confirm_up   = (prev_high > 0) and bool((last_seq > level_up).all())
        confirm_down = (prev_low  > 0) and bool((last_seq < level_dn).all())

        body_abs = abs(float(closes.iloc[-1]) - float(opens.iloc[-1]))
        body_ok = (atr_abs > 0) and (body_abs >= MIN_BODY_ATR_FRAC * atr_abs)
        atr_ok  = (atr_pct >= MIN_ATR_PCT_GLOBAL)

        vol_ma = float(vols.iloc[-VOL_LOOKBACK:].ewm(span=VOL_LOOKBACK, adjust=False).mean().iloc[-1]) \
                 if VOL_LOOKBACK > 2 else float(vols.mean())
        last_vol = float(vols.iloc[-1])
        vol_boost = (last_vol / vol_ma) if vol_ma > 0 else 0.0
        vol_ok = (vol_ma > 0) and (vol_boost >= VOL_BOOST_MIN)

        ema_fast_prev = float(ema_fast_series.iloc[-1 - EMA_SLOPE_LOOKBACK])
        ema_slope = ema_fast - ema_fast_prev

        had_pullback_long = True
        had_pullback_short = True
        if USE_PULLBACK and PULLBACK_LOOKBACK > 0 and atr_abs > 0:
            lo = - (CONFIRM_BARS + PULLBACK_LOOKBACK)
            hi = - CONFIRM_BARS if CONFIRM_BARS > 0 else None
            window_cl = closes.iloc[lo:hi]
            window_ema = ema_fast_series.iloc[lo:hi]
            if len(window_cl) >= 2 and len(window_ema) == len(window_cl):
                band = PULLBACK_BAND_ATR_K * atr_abs
                had_pullback_long  = bool(((window_ema - window_cl) >= band).any())
                had_pullback_short = bool(((window_cl - window_ema) >= band).any())

        htf_ok_long, htf_ok_short = True, True
        if HTF_ENABLE and HTF_MINUTES > 1:
            _htf_open, _htf_high, _htf_low, htf_close = _downsample_closes_ohlc(
                [[float(oo), float(hh), float(ll), float(cc), 0.0] for oo,hh,ll,cc in zip(opens, highs, lows, closes)],
                HTF_MINUTES
            )
            if len(htf_close) >= max(HTF_EMA_SLOW + 2, 8):
                htf_fast = float(_ema_series(htf_close, HTF_EMA_FAST).iloc[-1])
                htf_slow = float(_ema_series(htf_close, HTF_EMA_SLOW).iloc[-1])
                htf_ok_long  = htf_fast > htf_slow
                htf_ok_short = htf_fast < htf_slow

        bull = (
            (ema_fast > ema_slow) and (ema_slope > 0) and
            (mom >= mom_min_eff) and confirm_up and body_ok and vol_ok and atr_ok and
            had_pullback_long and htf_ok_long
        )
        bear = (
            (ema_fast < ema_slow) and (ema_slope < 0) and
            (mom <= -mom_min_eff) and confirm_down and body_ok and vol_ok and atr_ok and
            had_pullback_short and htf_ok_short
        )

        if bull:
            entry = float(last_px)
            sl, tp = atr_r_targets(entry, "buy", atr_abs, SL_ATR_K, TP_ATR_K, MIN_R)
            return Candidate("buy", entry, sl, tp, "impulse_breakout", 0.62, self.name)
        if bear:
            entry = float(last_px)
            sl, tp = atr_r_targets(entry, "sell", atr_abs, SL_ATR_K, TP_ATR_K, MIN_R)
            return Candidate("sell", entry, sl, tp, "impulse_breakout", 0.62, self.name)
        return None

# ===== Роутер =====
class StrategyRouter:
    def __init__(self, store_path: str = "data/strategy_perf.json"):
        # порядок важен
        self.strats: List[StrategyBase] = [
            ImpulseBreakout(),
            EngulfingTrend(),
            InsideNR4(),
            PinBarLevel(),
        ]
        self.path = store_path
        self.stats: Dict[str, Dict[str, Any]] = self._load()

        # === Bandit (многорукий бандит для выбора лучшей стратегии) ===
        # Lazy init — если файла нет, создаём пустой
        self.bandit_path = "data/bandit_perf.json"
        if os.path.exists(self.bandit_path):
            try:
                with open(self.bandit_path, "r", encoding="utf-8") as f:
                    self.bandit_data = json.load(f)
            except:
                self.bandit_data = {}
        else:
            self.bandit_data = {}

        # простейший epsilon-greedy bandit
        self.bandit_eps = 0.10

        def _bandit_score(rec):
            wins = rec.get("wins", 0)
            losses = rec.get("losses", 0)
            total = wins + losses
            if total == 0:
                return 0.5
            return wins / total

        self.bandit_score = _bandit_score


    def detect_regime(self, candles: Dict[str,np.ndarray]) -> str:
        h,l,c = candles['high'], candles['low'], candles['close']
        if len(c) < 120:
            return "warmup"
        atr_series = atr_np(h,l,c,14)
        if np.isnan(atr_series[-1]):
            return "warmup"
        price = max(float(c[-1]), 1e-9)
        atr_pct = float(atr_series[-1] / price)
        min_atr_pct = float(_cfg("REGIME_MIN_ATR_PCT", 0.0015))
        if atr_pct < min_atr_pct:
            return "chaos_lowvol"

        e50 = ema_np(c,50)
        e200 = ema_np(c,200)
        bb = bb_width_np(c,20)
        bb_pct20 = vol_percentile(bb, 60)
        bb_pct40 = max(bb_pct20, vol_percentile(bb, 120))
        dx = adx_like_np(h,l,c,14)
        adx_trend = float(_cfg("REGIME_ADX_TREND", 22.0))
        adx_range = float(_cfg("REGIME_ADX_RANGE", 18.0))
        slope50 = slope_np(e50, 20)
        if (dx >= adx_trend) and (float(e50[-1]) > float(e200[-1])) and (abs(slope50) > 0):
            return "trend"

        returns = np.diff(np.log(np.clip(c, 1e-9, None)))
        vol_hist = rolling_vol_np(returns, 30)
        energy_rising = False
        if len(vol_hist) >= 5 and not np.isnan(vol_hist[-1]):
            energy_rising = vol_hist[-1] > np.nanmean(vol_hist[-5:])

        if bb_pct20 < 20.0 and energy_rising:
            return "squeeze"
        if (dx < adx_range) and (bb_pct40 < 40.0):
            return "range"
        return "trend" if dx >= adx_range else "range"

    def _key(self, symbol: str, tf: str, regime: str, strat: str) -> str:
        return f"{symbol}|{tf}|{regime}|{strat}"

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.path): return {}
        try:
            with open(self.path,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path,"w",encoding="utf-8") as f:
            json.dump(self.stats,f,ensure_ascii=False,indent=2)

    def _pick_by_bandit(self, symbol, tf, regime, cands):
        """
        Упрощённый стабильный выбор стратегии.
        Если bandit не инициализирован — используем max confidence.
        """

        # ----------------------------
        # 1. Надёжный fallback
        # ----------------------------
        try:
            return max(cands, key=lambda x: x.confidence)
        except Exception as e:
            log(f"[BANDIT-FALLBACK] {e}")
            return cands[0]


        for cand in cands:
            k = self._key(symbol, tf, regime, cand.strategy)
            st = self.stats.get(k, {
                "wins": 0,
                "losses": 0,
                "r_sum": 0.0,
                "n": 0,
                "mean_r": 0.0,
                "m2": 0.0,
                "count": 0.0,
                "ewma_wr": 0.5,
                "ewma_r": 0.0,
                "history": [],
                "alpha": 1.0,
                "beta": 1.0,
            })

            # aging для беты: притягиваем к 1.0, чтобы не уходить в ноль
            alpha = 1.0 + (float(st.get("alpha", 1.0)) - 1.0) * aging
            beta = 1.0 + (float(st.get("beta", 1.0)) - 1.0) * aging
            bandit_count = float(st.get("bandit_count", st.get("count", 0.0))) * aging

            st.update({"alpha": alpha, "beta": beta, "bandit_count": bandit_count})
            self.stats[k] = st

            try:
                sample = float(np.random.beta(alpha, beta))
            except Exception:
                sample = alpha / (alpha + beta)

            history = list(st.get("history", []))
            drift_penalty = 1.0
            drift_window = max(5, int(_cfg("ROUTER_DRIFT_WINDOW", 25)))
            if len(history) >= drift_window * 2:
                recent = history[-drift_window:]
                past = history[-2 * drift_window: -drift_window]
                p_val = mann_whitney_pvalue(recent, past)
                if p_val < 0.05:
                    drift_penalty = 0.5

            ewma_wr = float(st.get("ewma_wr", 0.5))
            ewma_r = float(st.get("ewma_r", 0.0))
            mean_r = float(st.get("mean_r", 0.0))
            m2 = float(st.get("m2", 0.0))
            var_r = (m2 / (max(bandit_count, 1.0) - 1.0)) if bandit_count > 1.0 else max(m2, 1e-6)
            kelly_cap = float(_cfg("ROUTER_KELLY_CAP", 0.03))
            kelly = max(0.0, min(mean_r / max(var_r, 1e-6), kelly_cap))

            ucb_bonus = math.sqrt(max(1e-9, math.log(total_plays) / (bandit_count + 1.0)))
            score = (sample + 0.5 * ewma_wr + ewma_r + 0.3 * cand.confidence + kelly) * drift_penalty + ucb_bonus

            if bandit_count < min_obs:
                explore_pool.append((cand, st))

            scored.append((cand, st, alpha, beta, sample, score, bandit_count))
            if score > best_score:
                best_score, best = score, cand

        forced_choice: Optional[Candidate] = None
        forced_flag = False
        if random.random() < forced_rate:
            forced_choice = random.choice(cands)
            forced_flag = True
        elif explore_pool:
            forced_choice = random.choice([c for c, _ in explore_pool])
            forced_flag = True

        chosen = forced_choice or best
        chosen_sample = 0.0
        chosen_alpha = chosen_beta = 1.0
        chosen_count = 0.0
        for cand, st, alpha, beta, sample, score, bandit_count in scored:
            if cand is chosen:
                chosen_sample = sample
                chosen_alpha = alpha
                chosen_beta = beta
                chosen_count = float(st.get("bandit_count", bandit_count))
                break

        log(
            f"[BANDIT] {symbol}/{tf}/{regime}: выбрана {chosen.strategy} "
            f"sample={chosen_sample:.3f} alpha={chosen_alpha:.2f} beta={chosen_beta:.2f} "
            f"count={chosen_count:.1f} forced={forced_flag} pool={len(cands)}"
        )

        return chosen

    def decide(self, symbol: str, tf: str, candles: Dict[str,np.ndarray], ctx: Dict[str,Any]) -> Signal:
        o,h,l,c = candles['open'], candles['high'], candles['low'], candles['close']
        min_bars = int(_cfg("MIN_BARS", getattr(config, "MIN_BARS", 210)))
        have_bars = len(c)
        if have_bars < min_bars:
            log(f"[ROUTER] {symbol}/{tf}: недостаточно баров ({have_bars}/{min_bars})")
            return Signal(
                "hold",
                f"warmup_bars:{have_bars}/{min_bars}",
                None,
                None,
                {"why": "insufficient_bars", "have": have_bars, "need": min_bars}
            )
        a = atr_np(h,l,c,14)
        if math.isnan(a[-1]):
            return Signal("hold","warming_atr",None,None,{})

        vola = a[-1] / max(c[-1], 1e-9)

        vola_min = float(_cfg("MIN_ATR_PCT", 0.0))
        vola_max = float(_cfg("MAX_ATR_PCT", 1.0))  # ← читаем из .env

        log(f"[DEBUG] vola_min={vola_min} vola_max={vola_max}")

        if not (vola_min <= vola <= vola_max):
            return Signal("hold", f"vola_out_of_range:{vola:4f}", None, None, {})

        regime = self.detect_regime(candles)

        whitelist = {
            "trend": {"impulse_breakout", "engulfing_trend", "inside_nr4", "pinbar_level"},
            "squeeze": {"inside_nr4", "pinbar_level"},
            "range": {"inside_nr4", "pinbar_level"},
            "chaos_lowvol": {"inside_nr4", "pinbar_level"},
            "warmup": set()
        }.get(regime, {"inside_nr4", "pinbar_level"})



        cands: List[Candidate] = []
        for strat in self.strats:
            if whitelist and strat.name not in whitelist:
                continue
            cand = strat.propose(candles, ctx)
            log(f"[DEBUG-CAND] {strat.name} → {cand}")
            if cand:
                cands.append(cand)


        if not cands:
            return Signal("hold", f"no_candidates_{regime}", None, None, {"regime":regime})

        best = self._pick_by_bandit(symbol, tf, regime, cands)

        # ML-veto (если прокинут через ctx или config)
        ml_prob = ctx.get("ml_prob", None)
        ml_thr  = ctx.get("ml_thr", float(_cfg("ML_VETO_THRESHOLD", 0.58)))
        ml_enabled = bool(int(_cfg("ML_VETO_ENABLE", 0)))
        if ml_enabled and (ml_prob is not None) and (ml_prob < ml_thr):
            return Signal("hold", f"ml_veto:{ml_prob:.2f}<{ml_thr:.2f}", None, None, {"regime":regime})

        if ctx.get("max_positions_reached"):
            return Signal("hold","max_positions_reached",None,None,{"regime":regime})

        return Signal(best.side, f"{best.reason}|{regime}", best.sl, best.tp, {
            "regime": regime,
            "strategy": best.strategy,
            "entry_hint": best.entry,
            "confidence": best.confidence
        })


# ====== Публичные функции (совместимость) ======
_alt_flip = False  # для TEST_SIGNAL_MODE="alt"

def detect_impulse(candles: List[List[float]]) -> str:
    """
    СТАРЫЙ API: принимает [[o,h,l,c,v], ...] и возвращает 'buy'/'sell'/'hold'.
    Внутри использует обновлённую импульс-стратегию, но без роутера и без SL/TP.
    """
    global _alt_flip
    if not candles or len(candles) < 20:
        return "hold"

    strategy_mode = str(_cfg("STRATEGY_MODE", "prod")).lower()
    if strategy_mode == "test":
        mode = str(_cfg("TEST_SIGNAL_MODE", "alt")).lower()
        if mode == "buy":
            sig = "buy"
        elif mode == "sell":
            sig = "sell"
        else:
            _alt_flip = not _alt_flip
            sig = "buy" if _alt_flip else "sell"
        if _cfg("DEBUG_TRADING", True):
            print(f"[SIG-TEST] mode={mode} → {sig}")
        return sig

    # Конвертируем вход в numpy dict
    o = np.array([float(c[0]) for c in candles], dtype=float)
    h = np.array([float(c[1]) for c in candles], dtype=float)
    l = np.array([float(c[2]) for c in candles], dtype=float)
    c = np.array([float(c[3]) for c in candles], dtype=float)
    v = np.array([float(c[4]) for c in candles], dtype=float)
    pack = {"open":o,"high":h,"low":l,"close":c,"volume":v}

    cand = ImpulseBreakout().propose(pack, ctx={})
    return cand.side if cand else "hold"

# Главная новая точка входа для “умного выбора стратегии”
_router_singleton: Optional[StrategyRouter] = None
_recent_pattern_marks: Dict[Tuple[str, str, str, int], float] = {}

def _router() -> StrategyRouter:
    global _router_singleton
    if _router_singleton is None:
        store = str(_cfg("STRAT_PERF_STORE", "data/strategy_perf.json"))
        _router_singleton = StrategyRouter(store_path=store)
    return _router_singleton

def _pattern_enabled() -> bool:
    try:
        return bool(int(_cfg("ENABLE_CANDLE_PATTERNS", 1)))
    except Exception:
        return True


def detect_candle_patterns(ohlcv: List[List[float]]) -> List[Dict[str, Any]]:
    """Проверяет последние бары на классические свечные паттерны."""
    if not ohlcv or len(ohlcv) < 5 or not _pattern_enabled():
        return []

    try:
        lookback = max(5, int(_cfg("CANDLE_LOOKBACK_BARS", 60)))
    except Exception:
        lookback = 60
    try:
        min_conf = float(_cfg("CANDLE_MIN_CONF", 0.55))
    except Exception:
        min_conf = 0.55
    try:
        vol_filter = bool(int(_cfg("PATTERN_VOL_FILTER", 1)))
    except Exception:
        vol_filter = True

    window = ohlcv[-lookback:]
    opens = np.array([float(x[0]) for x in window], dtype=float)
    highs = np.array([float(x[1]) for x in window], dtype=float)
    lows = np.array([float(x[2]) for x in window], dtype=float)
    closes = np.array([float(x[3]) for x in window], dtype=float)
    vols = np.array([float(x[4]) if len(x) > 4 else 0.0 for x in window], dtype=float)
    n = len(window)

    if n < 5:
        return []

    def body(i: int) -> float:
        return abs(closes[i] - opens[i])

    def direction(i: int) -> str:
        if closes[i] > opens[i]:
            return "buy"
        if closes[i] < opens[i]:
            return "sell"
        return "neutral"

    def range_size(i: int) -> float:
        return max(1e-9, highs[i] - lows[i])

    def upper_shadow(i: int) -> float:
        return highs[i] - max(opens[i], closes[i])

    def lower_shadow(i: int) -> float:
        return min(opens[i], closes[i]) - lows[i]

    if vol_filter:
        recent = vols[-min(20, n):]
        median_vol = float(np.median(recent[recent > 0])) if np.any(recent > 0) else 0.0
    else:
        median_vol = 0.0

    def apply_volume(conf: float, i: int) -> float:
        if not vol_filter or median_vol <= 0:
            return conf
        vol = float(vols[i])
        if vol <= 0:
            return conf - 0.05
        if vol < 0.6 * median_vol:
            return conf - 0.07
        if vol > 1.4 * median_vol:
            return conf + 0.05
        return conf

    patterns: List[Dict[str, Any]] = []
    start = max(2, n - 10)

    def add_pattern(name: str, side: str, conf: float, idx: int):
        conf = apply_volume(conf, idx)
        conf = max(0.0, min(1.0, conf))
        if conf < min_conf:
            return
        patterns.append({
            "name": name,
            "side": side,
            "confidence": round(conf, 3),
            "bar_index": idx - n
        })

    for i in range(start, n):
        rng = range_size(i)
        if rng <= 1e-9:
            continue

        # Doji
        if body(i) <= 0.1 * rng:
            conf = 0.58 + min(0.12, (0.1 * rng - body(i)) / rng)
            add_pattern("Doji", "neutral", conf, i)

        if i < 1:
            continue

        prev_body = body(i - 1)
        prev_dir = direction(i - 1)
        curr_dir = direction(i)

        # Engulfing
        if prev_body > 0:
            if curr_dir == "buy" and prev_dir == "sell":
                if closes[i] >= opens[i - 1] and opens[i] <= closes[i - 1]:
                    ratio = body(i) / max(prev_body, 1e-9)
                    conf = 0.64 + min(0.18, max(0.0, ratio - 1.0) * 0.15)
                    if i >= 3 and closes[i] < closes[i - 3]:
                        conf -= 0.05
                    add_pattern("Bullish Engulfing", "buy", conf, i)
            if curr_dir == "sell" and prev_dir == "buy":
                if closes[i] <= opens[i - 1] and opens[i] >= closes[i - 1]:
                    ratio = body(i) / max(prev_body, 1e-9)
                    conf = 0.64 + min(0.18, max(0.0, ratio - 1.0) * 0.15)
                    if i >= 3 and closes[i] > closes[i - 3]:
                        conf -= 0.05
                    add_pattern("Bearish Engulfing", "sell", conf, i)

        # Harami
        if curr_dir == "buy" and prev_dir == "sell":
            if highs[i] <= max(opens[i - 1], closes[i - 1]) and lows[i] >= min(opens[i - 1], closes[i - 1]):
                conf = 0.6 - min(0.1, (body(i) / max(prev_body, 1e-9)) * 0.1)
                add_pattern("Bullish Harami", "buy", conf, i)
        if curr_dir == "sell" and prev_dir == "buy":
            if highs[i] <= max(opens[i - 1], closes[i - 1]) and lows[i] >= min(opens[i - 1], closes[i - 1]):
                conf = 0.6 - min(0.1, (body(i) / max(prev_body, 1e-9)) * 0.1)
                add_pattern("Bearish Harami", "sell", conf, i)

        # Piercing Line / Dark Cloud Cover
        mid_prev = (opens[i - 1] + closes[i - 1]) / 2.0
        if prev_dir == "sell" and curr_dir == "buy":
            if opens[i] < lows[i - 1] and closes[i] > mid_prev:
                conf = 0.63 + min(0.12, max(0.0, closes[i] - mid_prev) / range_size(i - 1))
                add_pattern("Piercing Line", "buy", conf, i)
        if prev_dir == "buy" and curr_dir == "sell":
            if opens[i] > highs[i - 1] and closes[i] < mid_prev:
                conf = 0.63 + min(0.12, max(0.0, mid_prev - closes[i]) / range_size(i - 1))
                add_pattern("Dark Cloud Cover", "sell", conf, i)

        # Hammer family
        lower = lower_shadow(i)
        upper = upper_shadow(i)
        if lower >= 2.2 * body(i) and upper <= 0.4 * body(i):
            drift = closes[i - 1] - closes[max(0, i - 4)] if i >= 4 else closes[i - 1] - closes[0]
            conf = 0.62 + min(0.12, max(0.0, lower / rng) * 0.2)
            if drift < 0:
                conf += 0.05
            add_pattern("Hammer", "buy", conf, i)
        if upper >= 2.2 * body(i) and lower <= 0.4 * body(i):
            drift = closes[i - 1] - closes[max(0, i - 4)] if i >= 4 else closes[i - 1] - closes[0]
            conf = 0.62 + min(0.12, max(0.0, upper / rng) * 0.2)
            if drift < 0:
                add_pattern("Inverted Hammer", "buy", conf, i)
            else:
                add_pattern("Shooting Star", "sell", conf, i)

        # Morning/Evening Star
        if i >= 2:
            dir1 = direction(i - 2)
            body_first = body(i - 2)
            body_second = body(i - 1)
            mid_first = (opens[i - 2] + closes[i - 2]) / 2.0
            gap_down = opens[i - 1] < closes[i - 2] and opens[i] >= closes[i - 1]
            gap_up = opens[i - 1] > closes[i - 2] and opens[i] <= closes[i - 1]
            if dir1 == "sell" and curr_dir == "buy" and body_second <= body_first * 0.6 and gap_down:
                if closes[i] > mid_first:
                    conf = 0.66 + min(0.14, (closes[i] - mid_first) / range_size(i - 2))
                    add_pattern("Morning Star", "buy", conf, i)
            if dir1 == "buy" and curr_dir == "sell" and body_second <= body_first * 0.6 and gap_up:
                if closes[i] < mid_first:
                    conf = 0.66 + min(0.14, (mid_first - closes[i]) / range_size(i - 2))
                    add_pattern("Evening Star", "sell", conf, i)

        # Three Soldiers / Crows
        if i >= 2:
            dirs = [direction(i - j) for j in range(2, -1, -1)]
            if all(d == "buy" for d in dirs):
                if closes[i] > closes[i - 1] > closes[i - 2]:
                    conf = 0.68 + min(0.12, (closes[i] - opens[i - 2]) / max(1e-9, opens[i - 2]))
                    add_pattern("Three White Soldiers", "buy", conf, i)
            if all(d == "sell" for d in dirs):
                if closes[i] < closes[i - 1] < closes[i - 2]:
                    conf = 0.68 + min(0.12, (opens[i - 2] - closes[i]) / max(1e-9, opens[i - 2]))
                    add_pattern("Three Black Crows", "sell", conf, i)

    patterns.sort(key=lambda p: p["confidence"], reverse=True)
    return patterns


def decide_with_router(symbol: str, timeframe: str, candles_ohlcv: List[List[float]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Универсальный вызов из main:
      result = decide_with_router(symbol, "1m", candles, ctx)
      if result['action'] in ('buy','sell'): ... использовать sl/tp
    """
    if not candles_ohlcv or len(candles_ohlcv) < 30:
        return {"action":"hold","reason":"not_enough_data","sl":None,"tp":None,"meta":{}}

    # в numpy-словари
    o = np.array([float(x[0]) for x in candles_ohlcv], dtype=float)
    h = np.array([float(x[1]) for x in candles_ohlcv], dtype=float)
    l = np.array([float(x[2]) for x in candles_ohlcv], dtype=float)
    c = np.array([float(x[3]) for x in candles_ohlcv], dtype=float)
    v = np.array([float(x[4]) for x in candles_ohlcv], dtype=float)
    pack = {"open":o,"high":h,"low":l,"close":c,"volume":v}

    sig = _router().decide(symbol, timeframe, pack, ctx or {})

    meta = dict(sig.meta or {})
    patterns = detect_candle_patterns(candles_ohlcv)
    top_patterns = patterns[:3]
    meta["patterns"] = top_patterns

    if sig.action in {"buy", "sell"} and top_patterns:
        base_conf = float(meta.get("confidence", 0.0) or 0.0)
        aligned = [p for p in top_patterns if p["side"] == sig.action]
        try:
            min_conf = float(_cfg("CANDLE_MIN_CONF", 0.55))
        except Exception:
            min_conf = 0.55
        if aligned:
            boost = min(0.15, 0.05 * len(aligned) + 0.05 * max(p["confidence"] for p in aligned))
            meta["confidence"] = round(min(1.0, base_conf + boost), 4)
        elif base_conf < min_conf:
            strong = [p for p in top_patterns if p["side"] == sig.action and p["confidence"] >= 0.7]
            if strong:
                meta["confidence"] = round(max(min_conf, base_conf), 4)

    if top_patterns:
        for patt in top_patterns:
            if patt.get("bar_index") != -1:
                continue
            key = (symbol, timeframe, patt["name"], int(patt["bar_index"]))
            last_mark = _recent_pattern_marks.get(key)
            if last_mark is not None and time.time() - last_mark < 60:
                continue
            _recent_pattern_marks[key] = time.time()
            msg = f"[PATTERN] {symbol} {patt['name']} side={patt['side']} conf={patt['confidence']:.2f}"
            log(msg)
            try:
                write_cycle_log({
                    "tag": "pattern",
                    "timestamp": now_iso(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "name": patt["name"],
                    "side": patt["side"],
                    "confidence": patt["confidence"],
                    "bar_index": patt["bar_index"],
                })
            except Exception:
                pass

    return {
        "action": sig.action,
        "reason": sig.reason,
        "sl": sig.sl,
        "tp": sig.tp,
        "meta": meta,
    }
