# -*- coding: utf-8 -*-
"""
build_ml_dataset_from_fills.py
--------------------------------
Генерит ml_dataset.csv из нормализованных сделок fills_all.csv.

ТОЛЬКО BYBIT (бинанс полностью вырезан).

Ускорения/надёжность:
• Параллельная обработка + глобальный rate-limit.
• LRU-кэш в памяти + ПЕРСИСТЕНТНЫЙ дисковый кеш свечей (история/горизонт).
• Экономичные индикаторы, жёсткая типизация.
• Возврат результатов в исходном порядке.
• Грациозная остановка (Ctrl+C): не подаются новые задачи, сохраняется partial.
• Фикс индексов после фильтров (без IndexError).

Env/параметры:
  FILLS_PATH (обязателен)  | fallback: BYBIT_CSV_PATH
  DATASET_SINCE=YYYY-MM-DD | отфильтровать сделки по дате (ts >= since)
  PARALLELISM, API_GUARD_RATE, API_GUARD_BURST, MIN_DELAY_BETWEEN_REQ, HTTP_RETRIES...
  ML_DATASET_PATH (по умолчанию ml_dataset.csv)
  BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET=(true|false)
  ENABLE_DISK_CACHE=1|0 (вкл/выкл дисковый кеш)
  DISK_CACHE_DIR (папка кеша, по умолчанию ./.dataset_cache)
"""

import os
import sys
import time
import math
import json
import gzip
import hashlib
import threading
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ====== ТЮНИНГ (из .env) ======
HISTORY_MINUTES       = int(os.getenv("HISTORY_MINUTES", "60"))
LABEL_HORIZON_MIN     = int(os.getenv("LABEL_HORIZON_MIN", "60"))

API_GUARD_RATE        = float(os.getenv("API_GUARD_RATE", "6.0"))
API_GUARD_BURST       = int(os.getenv("API_GUARD_BURST", "12"))

MIN_DELAY_BETWEEN_REQ = float(os.getenv("MIN_DELAY_BETWEEN_REQ", "0.06"))
HTTP_RETRIES          = int(os.getenv("HTTP_RETRIES", "6"))
MAX_BACKOFF           = float(os.getenv("MAX_BACKOFF", "8.0"))

PARALLELISM           = int(os.getenv("PARALLELISM", "12"))

ENABLE_LRU_CACHE      = bool(int(os.getenv("ENABLE_LRU_CACHE", "1")))
KLINE_CACHE_MAX       = int(os.getenv("KLINE_CACHE_MAX", "16384"))
SNAPSHOT_CACHE_TTL    = float(os.getenv("SNAPSHOT_CACHE_TTL", "20.0"))

LIMIT_ROWS_ENV        = os.getenv("LIMIT_ROWS", "").strip()
LIMIT_ROWS            = int(LIMIT_ROWS_ENV) if LIMIT_ROWS_ENV.isdigit() and int(LIMIT_ROWS_ENV) > 0 else None

TP_MODE               = os.getenv("TP_MODE", "adaptive")
TP_PCT_FIXED          = float(os.getenv("TP_PCT_FIXED", "0.0030"))
SL_PCT_FIXED          = float(os.getenv("SL_PCT_FIXED", "0.0025"))
TP_ATR_K              = float(os.getenv("TP_ATR_K", "5.0"))
SL_ATR_K              = float(os.getenv("SL_ATR_K", "4.0"))
TP_CLAMP              = (float(os.getenv("TP_CLAMP_LO", "0.0020")), float(os.getenv("TP_CLAMP_HI", "0.0060")))
SL_CLAMP              = (float(os.getenv("SL_CLAMP_LO", "0.0015")), float(os.getenv("SL_CLAMP_HI", "0.0040")))

MICRO_SLEEP           = float(os.getenv("MICRO_SLEEP", "0.0"))

# --- Дисковый кеш ---
ENABLE_DISK_CACHE     = bool(int(os.getenv("ENABLE_DISK_CACHE", "1")))
DISK_CACHE_DIR        = os.getenv("DISK_CACHE_DIR", ".dataset_cache").strip() or ".dataset_cache"
CACHE_VERSION         = "v1.2"  # менять при изменении формата ключей

# --- WINDOWS-SAFE PRINT -------------------------------------------------------
def _safe_str(s: str) -> str:
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        return str(s).encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return str(s)

def log(msg: str):
    sys.stdout.write(_safe_str(msg) + "\n")
    sys.stdout.flush()
# ------------------------------------------------------------------------------

_print_lock = threading.Lock()

def print_progress(done: int, total: int, prefix: str = "[build]"):
    if total <= 0:
        return
    done = min(done, total)
    pct = int(done * 100 / total) if total > 0 else 100
    rem = max(0, 100 - pct)
    with _print_lock:
        sys.stdout.write(f"\r{prefix} {pct}% | ostalos {rem}% ({done}/{total})")
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")

# ====== ИНДИКАТОРЫ ======
def calc_rsi_from_closes(closes, period=14) -> float:
    if len(closes) < period + 1:
        return 50.0
    s = pd.Series(list(map(float, closes)))
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if not s.empty else 50.0
    return float(val) if pd.notna(val) and not math.isnan(val) else 50.0

def calc_atr_from_ohlc(ohlc_list, period=14) -> float:
    if len(ohlc_list) < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(ohlc_list)):
        _, h, l, c, _ = ohlc_list[i]
        _, _, _, pc, _ = ohlc_list[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(sum(trs) / len(trs)) if trs else 0.0

def pct_volatility_from_closes(closes) -> float:
    if len(closes) < 5:
        return 0.0
    s = pd.Series(list(map(float, closes)))
    return float(s.pct_change().std(skipna=True) or 0.0)

# ====== HTTP / RATE-LIMIT ======
class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = float(rate_per_sec)
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: int = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

_BUCKET = TokenBucket(API_GUARD_RATE, API_GUARD_BURST)
_last_req_ts = 0.0
_last_req_lock = threading.Lock()

_SLOW_MODE_UNTIL = 0.0
_SLOW_LOCK = threading.Lock()

def _maybe_slow_mode_sleep():
    with _SLOW_LOCK:
        until = _SLOW_MODE_UNTIL
    if time.time() < until:
        time.sleep(min(0.5, MAX_BACKOFF))

def _arm_slow_mode():
    with _SLOW_LOCK:
        global _SLOW_MODE_UNTIL
        _SLOW_MODE_UNTIL = time.time() + 2.0

def _sleep_min_delay():
    global _last_req_ts
    if MIN_DELAY_BETWEEN_REQ <= 0:
        return
    with _last_req_lock:
        now = time.time()
        wait = max(0.0, MIN_DELAY_BETWEEN_REQ - (now - _last_req_ts))
        if wait > 0:
            time.sleep(wait)
        _last_req_ts = time.time()

# ====== КЭШИ ======
class LRU:
    def __init__(self, maxsize=1024):
        self.maxsize = maxsize
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        if not ENABLE_LRU_CACHE:
            return None
        with self.lock:
            if key in self.d:
                v = self.d.pop(key)
                self.d[key] = v
                return v
            return None

    def set(self, key, val):
        if not ENABLE_LRU_CACHE:
            return
        with self.lock:
            self.d[key] = val
            if len(self.d) > self.maxsize:
                self.d.popitem(last=False)

KLINE_CACHE = LRU(maxsize=KLINE_CACHE_MAX)
SNAP_CACHE: Dict[Tuple[str, str], Tuple[float, dict]] = {}
SNAP_LOCK = threading.Lock()

# -------- ПЕРСИСТЕНТНЫЙ дисковый кеш (JSON.GZ) --------
class DiskCache:
    def __init__(self, root: str):
        self.root = root
        try:
            os.makedirs(root, exist_ok=True)
        except Exception:
            pass

    def _hash(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

    def _path(self, namespace: str, h: str) -> str:
        d1, d2 = h[:2], h[2:4]
        p = os.path.join(self.root, namespace, d1, d2, h + ".json.gz")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def get(self, namespace: str, key: str):
        if not ENABLE_DISK_CACHE:
            return None
        h = self._hash(key)
        p = self._path(namespace, h)
        if not os.path.exists(p):
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, namespace: str, key: str, obj):
        if not ENABLE_DISK_CACHE:
            return
        h = self._hash(key)
        p = self._path(namespace, h)
        try:
            with gzip.open(p, "wt", encoding="utf-8") as f:
                json.dump(obj, f, separators=(",", ":"))
        except Exception:
            pass

DISK_CACHE = DiskCache(DISK_CACHE_DIR)

def _to_float_ohlcv_list(raw_list):
    out = []
    for k in raw_list:
        try:
            o = float(k[0]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
        except Exception:
            vals = [0.0, 0.0, 0.0, 0.0, 0.0]
            idxs = [0, 2, 3, 4, 5]
            for i, idx in enumerate(idxs):
                try:
                    vals[i] = float(k[idx])
                except Exception:
                    vals[i] = 0.0
            o, h, l, c, v = vals
        out.append([o, h, l, c, v])
    return out

# ====== ПРОВАЙДЕР МАРКЕТ-ДАННЫХ: BYBIT ======
class BybitProvider:
    def __init__(self, api_key: str, api_secret: str, testnet: bool):
        from pybit.unified_trading import HTTP
        self.HTTP = HTTP
        self.client = self.HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet, timeout=10, recv_window=20000)

    def _http_call(self, func, **kwargs):
        backoff = 0.40
        for attempt in range(1, HTTP_RETRIES + 1):
            while not _BUCKET.take():
                time.sleep(0.01)
            _sleep_min_delay()
            _maybe_slow_mode_sleep()
            try:
                return func(**kwargs)
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if ("10006" in msg) or ("rate limit" in low):
                    log(f"[RL] bybit 10006/limit -> backoff {backoff:.2f}s (try {attempt}/{HTTP_RETRIES})")
                    _arm_slow_mode()
                    time.sleep(backoff)
                    backoff = min(MAX_BACKOFF, backoff * 2.0)
                    continue
                soft = ("read timed out", "timeout", "temporary failure", "max retries exceeded",
                        "remotedisconnected", "bad gateway", "service temporarily unavailable", "gateway timeout")
                if any(s in low for s in soft):
                    log(f"[NET] bybit soft -> backoff {backoff:.2f}s")
                    time.sleep(backoff)
                    backoff = min(MAX_BACKOFF, backoff * 1.7)
                    continue
                if attempt < HTTP_RETRIES:
                    log(f"[WARN] bybit http: {msg[:120]} -> retry {attempt}/{HTTP_RETRIES}")
                    time.sleep(min(MAX_BACKOFF, backoff))
                    backoff = min(MAX_BACKOFF, backoff * 1.5)
                    continue
                raise

    def get_kline_block(self, category: str, symbol: str, interval="1",
                        start: Optional[int] = None, end: Optional[int] = None, limit: int = 60):
        """
        BYBIT get_kline, с in-memory LRU и дисковым кешем.
        Ключ кеша учитывает версию, категорию, символ, интервал, start/end, limit.
        """
        key_lru = ("bybit", category, symbol, interval, int(start or -1), int(end or -1), int(limit))
        cached = KLINE_CACHE.get(key_lru)
        if cached is not None:
            return cached

        ns = "kline_bybit"
        key_disk = f"{CACHE_VERSION}|{category}|{symbol}|{interval}|{int(start or -1)}|{int(end or -1)}|{int(limit)}"
        disk_hit = DISK_CACHE.get(ns, key_disk)
        if disk_hit is not None:
            out = _to_float_ohlcv_list(disk_hit)
            KLINE_CACHE.set(key_lru, out)
            return out

        kw = {"category": category, "symbol": symbol, "interval": interval, "limit": max(1, int(limit))}
        if start is not None: kw["start"] = int(start)
        if end   is not None: kw["end"]   = int(end)

        data = self._http_call(self.client.get_kline, **kw)
        raw = (data.get("result", {}) or {}).get("list", []) or []
        out = []
        # BYBIT: [[start, open, high, low, close, volume], ...]
        for k in raw:
            try:
                o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
            except Exception:
                o = h = l = c = v = 0.0
                try: o = float(k[1])
                except: pass
                try: h = float(k[2])
                except: pass
                try: l = float(k[3])
                except: pass
                try: c = float(k[4])
                except: pass
                try: v = float(k[5])
                except: pass
            out.append([o, h, l, c, v])

        # кешируем
        KLINE_CACHE.set(key_lru, out)
        try:
            DISK_CACHE.set(ns, key_disk, out)
        except Exception:
            pass
        return out

    def get_kline_before(self, symbol: str, end_ms: int, minutes: int, interval="1"):
        lim = max(1, int(minutes))
        kl = self.get_kline_block("linear", symbol, interval=interval, end=end_ms, limit=lim)
        if kl:
            return kl, "linear"
        kl = self.get_kline_block("spot", symbol, interval=interval, end=end_ms, limit=lim)
        return kl, ("spot" if kl else "linear")

    def get_kline_forward(self, symbol: str, start_ms: int, minutes: int, interval="1"):
        lim = max(1, int(minutes))
        kl = self.get_kline_block("linear", symbol, interval=interval, start=start_ms, limit=lim)
        if kl:
            return kl, "linear"
        kl = self.get_kline_block("spot", symbol, interval=interval, start=start_ms, limit=lim)
        return kl, ("spot" if kl else "linear")

    def fetch_snapshot_any(self, symbol: str):
        """
        Снапшот 24h/индекс/фандинг/OI.
        Только in-memory TTL (на диск не кладём).
        """
        now = time.time()
        k_lin = ("bybit", "linear", symbol)
        with SNAP_LOCK:
            if k_lin in SNAP_CACHE and (now - SNAP_CACHE[k_lin][0]) < SNAPSHOT_CACHE_TTL:
                return SNAP_CACHE[k_lin][1]
        # linear
        try:
            t = self._http_call(self.client.get_tickers, category="linear", symbol=symbol)["result"]["list"]
            if t:
                t = t[0]
                res = {
                    "index_price": float(t.get("indexPrice", 0) or 0.0),
                    "last_price":  float(t.get("lastPrice", 0) or 0.0),
                    "high":        float(t.get("highPrice24h", 0) or 0.0),
                    "low":         float(t.get("lowPrice24h", 0) or 0.0),
                    "vol_24h":     float(t.get("turnover24h", 0) or 0.0),
                    "open_interest": float(t.get("openInterest", 0) or 0.0),
                    "funding_rate":  float(t.get("fundingRate", 0) or 0.0),
                }
                with SNAP_LOCK:
                    SNAP_CACHE[k_lin] = (now, res)
                return res
        except Exception:
            pass
        # spot
        k_sp = ("bybit", "spot", symbol)
        with SNAP_LOCK:
            if k_sp in SNAP_CACHE and (now - SNAP_CACHE[k_sp][0]) < SNAPSHOT_CACHE_TTL:
                return SNAP_CACHE[k_sp][1]
        try:
            t = self._http_call(self.client.get_tickers, category="spot", symbol=symbol)["result"]["list"]
            if t:
                t = t[0]
                res = {
                    "index_price": 0.0,
                    "last_price":  float(t.get("lastPrice", 0) or 0.0),
                    "high":        float(t.get("highPrice24h", 0) or 0.0),
                    "low":         float(t.get("lowPrice24h", 0) or 0.0),
                    "vol_24h":     float(t.get("turnover24h", 0) or 0.0),
                    "open_interest": 0.0,
                    "funding_rate":  0.0,
                }
                with SNAP_LOCK:
                    SNAP_CACHE[k_sp] = (now, res)
                return res
        except Exception:
            pass
        return {
            "index_price": 0.0, "last_price": 0.0, "high": 0.0, "low": 0.0,
            "vol_24h": 0.0, "open_interest": 0.0, "funding_rate": 0.0
        }

# ====== TP/SL ======
def _clamp(x, lo, hi): return max(lo, min(hi, x))

def tp_sl_from_atr(atr_norm: float) -> Tuple[float, float]:
    if TP_MODE == "fixed":
        return float(TP_PCT_FIXED), float(SL_PCT_FIXED)
    atr_norm = float(atr_norm or 0.0)
    tp = _clamp(TP_ATR_K * atr_norm, *TP_CLAMP)
    sl = _clamp(SL_ATR_K * atr_norm, *SL_CLAMP)
    return float(tp), float(sl)

def label_trade(price: float, fwd_kl: List[List[float]], tp_pct: float, sl_pct: float) -> int:
    if price <= 0 or not fwd_kl:
        return 0
    top = price * (1.0 + float(tp_pct))
    bot = price * (1.0 - float(sl_pct))
    for k in fwd_kl:
        try:
            hi = float(k[2]); lo = float(k[3])  # [o,h,l,c,v]
        except Exception:
            hi = float(k[1]) if len(k) > 1 else 0.0
            lo = float(k[0]) if len(k) > 0 else 0.0
        if hi >= top:
            return 1
        if lo <= bot:
            return 0
    return 0

# ====== WORKER ======
def process_row(idx: int, row: pd.Series, provider, source_name: str) -> Tuple[int, Dict]:
    symbol = str(row["symbol"])
    ts_ms  = int(row["ts"])
    price  = float(row["price"])
    qty    = float(row["qty"])
    side_num = 1 if str(row["side"]).lower().startswith("b") else -1

    # История
    kl_before, _ = provider.get_kline_before(symbol, end_ms=ts_ms, minutes=HISTORY_MINUTES, interval="1")
    ohlc = [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in kl_before][-HISTORY_MINUTES:]
    closes = [float(x[3]) for x in ohlc]

    rsi = calc_rsi_from_closes(closes) if closes else 50.0
    atr_abs = calc_atr_from_ohlc(ohlc) if ohlc else 0.0
    last_px = price if price > 0 else (ohlc[-1][3] if ohlc else 0.0)
    atr_norm = (atr_abs / last_px) if last_px > 0 else 0.0
    vol = pct_volatility_from_closes(closes) if closes else 0.0

    snap = provider.fetch_snapshot_any(symbol)
    pct_from_high = (price - snap["high"]) / (snap["high"] + 1e-9) if snap["high"] else 0.0
    dist_to_index = (price - snap["index_price"]) / (snap["index_price"] + 1e-9) if snap["index_price"] else 0.0

    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    hour = dt.hour
    weekday = dt.weekday()

    # Горизонт (вперёд)
    fwd_kl, _ = provider.get_kline_forward(symbol, start_ms=ts_ms, minutes=LABEL_HORIZON_MIN, interval="1")

    tp_pct, sl_pct = tp_sl_from_atr(atr_norm)
    target = label_trade(price, fwd_kl, tp_pct, sl_pct)

    feat = {
        "ts": ts_ms, "symbol": symbol, "side": side_num, "price": price, "qty": qty,
        "index_price": snap["index_price"], "last_price": snap["last_price"],
        "high": snap["high"], "low": snap["low"], "vol_24h": snap["vol_24h"],
        "open_interest": snap["open_interest"], "funding_rate": snap["funding_rate"],
        "rsi": rsi, "atr_abs": atr_abs, "atr_norm": atr_norm, "volatility": vol,
        "hour": hour, "weekday": weekday,
        "pct_from_high": pct_from_high, "dist_to_index": dist_to_index,
        "tp_pct_used": tp_pct, "sl_pct_used": sl_pct,
        "target": int(target),
        "source": source_name,
    }
    if MICRO_SLEEP > 0:
        time.sleep(MICRO_SLEEP)
    return idx, feat

# ====== Грациозная остановка ======
_STOP = False
def _handle_stop(signum, frame):
    global _STOP
    if not _STOP:
        _STOP = True
        log("\n[STOP] Signal received — finishing queued tasks and saving partial dataset…")

signal.signal(signal.SIGINT, _handle_stop)
if hasattr(signal, "SIGTERM"):
    try:
        signal.signal(signal.SIGTERM, _handle_stop)
    except Exception:
        pass

# ====== MAIN ======
def main():
    load_dotenv()

    fills_path = os.getenv("FILLS_PATH") or os.getenv("BYBIT_CSV_PATH") or "fills_all.csv"
    if not os.path.exists(fills_path):
        raise FileNotFoundError(f"Ne naiden {fills_path}. Ukaži FILLS_PATH ili poloji fail ryadom.")

    # Bybit провайдер
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    testnet = str(os.getenv("BYBIT_TESTNET", "false")).lower() in ("1", "true", "yes")
    provider = BybitProvider(api_key, api_secret, testnet)
    source_name = "bybit"
    log(f"[cfg] DATA_SOURCE=bybit (testnet={testnet}) | disk_cache={'on' if ENABLE_DISK_CACHE else 'off'} → {DISK_CACHE_DIR}")

    df_fills = pd.read_csv(fills_path)
    need = {"ts", "symbol", "side", "price", "qty"}
    miss = [c for c in need if c not in df_fills.columns]
    if miss:
        raise RuntimeError(f"V {fills_path} net kolonok: {miss}. Peregeneriruy eksport.")

    # типизация/сортировка
    df_fills["ts"] = pd.to_numeric(df_fills["ts"], errors="coerce").fillna(0).astype(np.int64)
    df_fills["price"] = pd.to_numeric(df_fills["price"], errors="coerce").fillna(0.0).astype(float)
    df_fills["qty"] = pd.to_numeric(df_fills["qty"], errors="coerce").fillna(0.0).astype(float)
    df_fills["symbol"] = df_fills["symbol"].astype(str)
    df_fills["side"] = df_fills["side"].astype(str)
    df_fills = df_fills.sort_values("ts").reset_index(drop=True)

    # фильтр по since (если задан)
    since_str = os.getenv("DATASET_SINCE")
    if since_str:
        try:
            since_dt = datetime.strptime(since_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            since_ms = int(since_dt.timestamp() * 1000)
            df_fills = df_fills[df_fills["ts"] >= since_ms].copy().reset_index(drop=True)
            log(f"[filter] DATASET_SINCE={since_str} -> rows={len(df_fills)}")
        except Exception as e:
            log(f"[WARN] nekorrektnyi DATASET_SINCE={since_str}: {e}")

    if LIMIT_ROWS is not None:
        df_fills = df_fills.head(int(LIMIT_ROWS)).copy().reset_index(drop=True)

    total_all = int(len(df_fills))
    if total_all <= 0:
        log("[OK] nothing to do: input is empty.")
        return

    print_progress(0, total_all)

    results: List[Optional[Dict]] = [None] * total_all
    submitted = 0
    done = 0
    last_print = time.time()

    # Подаём задачи порциями: максимум PARALLELISM в полёте, чтобы CTRL+C быстро останавливал
    def submit_one(ex, n, row):
        return ex.submit(process_row, n, row, provider, source_name)

    with ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futures = set()
        n = 0
        it = df_fills.iterrows()

        # первичная загрузка очереди
        try:
            while len(futures) < PARALLELISM and n < total_all and not _STOP:
                _, row = next(it)
                futures.add(submit_one(ex, n, row))
                submitted += 1
                n += 1
        except StopIteration:
            pass

        while futures:
            fut = next(as_completed(futures))
            futures.remove(fut)
            try:
                idx, feat = fut.result()
                results[idx] = feat
            except Exception as e:
                log(f"[ERR] row failed: {str(e)[:160]}")
            done += 1

            # пополняем очередь, если можно и не нажали стоп
            if not _STOP and n < total_all:
                try:
                    _, row = next(it)
                    futures.add(submit_one(ex, n, row))
                    submitted += 1
                    n += 1
                except StopIteration:
                    pass

            now = time.time()
            if (now - last_print) > 0.1 or done == submitted:
                print_progress(done, submitted, prefix="[build]")
                last_print = now

            if _STOP and not futures:
                break

    # Собираем только готовые строки
    rows = [r for r in results[:submitted] if r is not None]
    df = pd.DataFrame(rows)

    # числовые колонки
    for c in df.columns:
        if c in ("symbol", "source"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    out_path = os.getenv("ML_DATASET_PATH", "ml_dataset.csv")

    if _STOP:
        part_path = out_path.replace(".csv", ".partial.csv")
        df.to_csv(part_path, index=False)
        pos = int(df["target"].sum()) if "target" in df else 0
        log(f"[STOP] Saved partial dataset: {part_path}, rows={len(df)} | +={pos} -={max(0,len(df)-pos)}")
        return

    df.to_csv(out_path, index=False)
    pos = int(df["target"].sum()) if "target" in df else 0
    neg = int(len(df) - pos)
    log(f"[OK] Saved dataset: {out_path}, rows={len(df)} | +={pos} -={neg} (pos_rate={pos/max(1,len(df)):.3f})")

if __name__ == "__main__":
    main()
