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
from typing import Any, Dict, List, Optional
import json, os, math, time

import numpy as np
import pandas as pd

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

# ===== Контракты =====
@dataclass
class Candidate:
    side: str          # 'buy'|'sell'
    entry: float
    sl: float
    tp: float
    reason: str
    confidence: float  # 0..1
    strategy: str

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
        if len(c) < 210: return None
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
        prev_high = float(highs.iloc[-BRK_LOOKBACK - excl : -excl].max())
        prev_low  = float(lows.iloc[-BRK_LOOKBACK  - excl : -excl].min())
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
            htf_open, htf_high, htf_low, htf_close = _downsample_closes_ohlc(
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
            sl = float(last_px - max(atr_abs*1.0, (last_px - prev_low)))  # консервативно
            r = entry - sl
            tp = float(entry + max(1.5*r, atr_abs*1.5))
            return Candidate("buy", entry, sl, tp, "impulse_breakout", 0.62, self.name)
        if bear:
            entry = float(last_px)
            sl = float(last_px + max(atr_abs*1.0, (prev_high - last_px)))
            r = sl - entry
            tp = float(entry - max(1.5*r, atr_abs*1.5))
            return Candidate("sell", entry, sl, tp, "impulse_breakout", 0.62, self.name)
        return None

# ===== Роутер =====
class StrategyRouter:
    def __init__(self, store_path: str = "data/strategy_perf.json"):
        # порядок важен: сперва трендовая, потом компрессионные
        self.strats: List[StrategyBase] = [
            ImpulseBreakout(),
            EngulfingTrend(),
            InsideNR4(),
            PinBarLevel(),
        ]
        self.path = store_path
        self.stats: Dict[str, Dict[str, Any]] = self._load()

    def detect_regime(self, candles: Dict[str,np.ndarray]) -> str:
        h,l,c = candles['high'], candles['low'], candles['close']
        if len(c) < 60: return "warmup"
        a = atr_np(h,l,c,14)
        e50, e200 = ema_np(c,50), ema_np(c,200)
        dx = adx_like_np(h,l,c,14)
        vola = a[-1] / max(c[-1], 1e-9)
        sl50 = slope_np(e50, 20)
        if vola < 0.002: return "chaos_lowvol"
        if dx >= 20 and abs(sl50) > 0: return "trend"
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        if len(tr) >= 20 and np.percentile(tr[-7:], 70) < np.percentile(tr[-20:], 40):
            return "squeeze"
        return "range"

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

    def _pick_by_bandit(self, symbol: str, tf: str, regime: str, cands: List[Candidate]) -> Optional[Candidate]:
        best = None
        best_score = -1e9
        for cand in cands:
            k = self._key(symbol, tf, regime, cand.strategy)
            st = self.stats.get(k, {"wins":1, "losses":1, "r_sum":0.0, "n":0})
            a,b = st["wins"]+1, st["losses"]+1
            mean_wr = a/(a+b)
            avg_r = (st["r_sum"]/st["n"]) if st["n"]>0 else 0.0
            score = 2.0*mean_wr + 0.6*avg_r + 0.5*cand.confidence
            if score > best_score:
                best_score, best = score, cand
        return best

    def decide(self, symbol: str, tf: str, candles: Dict[str,np.ndarray], ctx: Dict[str,Any]) -> Signal:
        o,h,l,c = candles['open'], candles['high'], candles['low'], candles['close']
        if len(c) < 210:
            return Signal("hold","warmup",None,None,{"why":"insufficient_bars"})
        a = atr_np(h,l,c,14)
        if math.isnan(a[-1]):
            return Signal("hold","warming_atr",None,None,{})

        vola = a[-1]/max(c[-1],1e-9)
        vola_min = float(_cfg("ROUTER_MIN_ATR_PCT", 0.002))
        vola_max = float(_cfg("ROUTER_MAX_ATR_PCT", 0.06))
        if not (vola_min <= vola <= vola_max):
            return Signal("hold",f"vola_out_of_range:{vola:.4f}",None,None,{})

        regime = self.detect_regime(candles)

        whitelist = {
            "trend": {"impulse_breakout","engulfing_trend"},
            "squeeze": {"inside_nr4","pinbar_level"},
            "range": {"inside_nr4","pinbar_level"},
            "chaos_lowvol": set(),
            "warmup": set()
        }.get(regime, set())

        cands: List[Candidate] = []
        for strat in self.strats:
            if whitelist and strat.name not in whitelist: 
                continue
            cand = strat.propose(candles, ctx)
            if cand: cands.append(cand)

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

    def report_fill(self, symbol: str, tf: str, regime: str, strategy: str, r_result: float):
        k = self._key(symbol, tf, regime, strategy)
        st = self.stats.get(k, {"wins":0,"losses":0,"r_sum":0.0,"n":0,"updated":0})
        st["n"] += 1
        if r_result > 0:
            st["wins"] += 1
        else:
            st["losses"] += 1
        st["r_sum"] += float(r_result)
        st["updated"] = int(time.time())
        self.stats[k] = st
        self._save()

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

def _router() -> StrategyRouter:
    global _router_singleton
    if _router_singleton is None:
        store = str(_cfg("STRAT_PERF_STORE", "data/strategy_perf.json"))
        _router_singleton = StrategyRouter(store_path=store)
    return _router_singleton

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
    return {"action":sig.action, "reason":sig.reason, "sl":sig.sl, "tp":sig.tp, "meta":sig.meta}

# Вспомогательно: прокинуть результат сделки (в R) после закрытия
def report_trade_result(symbol: str, timeframe: str, regime: str, strategy: str, r_result: float):
    _router().report_fill(symbol, timeframe, regime, strategy, r_result)
