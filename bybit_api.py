# -*- coding: utf-8 -*-
"""
bybit_api.py — HTTP v5 wrapper с безопасными вызовами и утилитами для онлайна.
Совместим с существующим кодом проекта.

Ключевое:
- safe_request из api_guard — единая точка ретраев/бэкоффов.
- Использует только production-эндпоинт Bybit.
- Фолбек LINEAR→SPOT для снапшотов и свечей, чтобы не ловить 10001 на «нестандартных» символах.
"""

from __future__ import annotations

import os
import json
import time
import math
import copy
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterable
from threading import RLock

from env_loader import load_env
from pybit.unified_trading import HTTP

# утилиты проекта
from utils import log, tg_send, adjust_qty, make_order_link_id

# центр ретраев / лимитов — единая точка правды
from api_guard import safe_request

# =========================
# Инициализация клиента (уважаем .env и/или config.py)
# =========================
load_env()

try:
    import config as _cfg  # опционально
except Exception:
    _cfg = None

def _resolve_credentials() -> Tuple[str, str]:
    key = (os.getenv("BYBIT_API_KEY") or "").strip()
    secret = (os.getenv("BYBIT_API_SECRET") or "").strip()
    if _cfg is not None:
        try:
            if not key:
                key = str(getattr(_cfg, "BYBIT_API_KEY", "") or "").strip()
            if not secret:
                secret = str(getattr(_cfg, "BYBIT_API_SECRET", "") or "").strip()
        except Exception:
            pass
    return key, secret


def _paper_mode_active() -> bool:
    env_flag = os.getenv("PAPER_MODE")
    if env_flag is not None:
        s = env_flag.strip().lower()
        if s in {"1", "true", "yes", "y", "on", "t"}:
            return True
        if s in {"0", "false", "no", "n", "off", "f"}:
            return False
        try:
            return bool(int(float(env_flag)))
        except Exception:
            pass
    if _cfg is not None:
        try:
            val = getattr(_cfg, "PAPER_MODE", 0)
            return bool(int(val)) if isinstance(val, (int, float, str)) else bool(val)
        except Exception:
            pass
    return False


API_KEY, API_SECRET = _resolve_credentials()
BYBIT_ENDPOINT = "https://api.bybit.com"

try:
    _DEFAULT_KLINE_LIMIT = int(os.getenv("KLINE_HISTORY_LIMIT", "300"))
    if _cfg is not None and hasattr(_cfg, "KLINE_HISTORY_LIMIT"):
        _DEFAULT_KLINE_LIMIT = int(getattr(_cfg, "KLINE_HISTORY_LIMIT"))
except Exception:
    _DEFAULT_KLINE_LIMIT = 300

def _create_client(api_key: str, api_secret: str) -> HTTP:
    return HTTP(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        recv_window=20000,
        timeout=10,
    )


_CLIENT: Optional[HTTP] = None


def _ensure_client() -> HTTP:
    global _CLIENT, API_KEY, API_SECRET
    if _CLIENT is None:
        API_KEY, API_SECRET = _resolve_credentials()
        if not _paper_mode_active() and (not API_KEY or not API_SECRET):
            raise RuntimeError("API keys missing in REAL")
        _CLIENT = _create_client(API_KEY, API_SECRET)
    return _CLIENT


class _LazyHTTP:
    def __getattr__(self, item):
        return getattr(_ensure_client(), item)


client = _LazyHTTP()

_PARAMS_ERROR_LOG = Path("logs/errors/params_errors.jsonl")

_KLINE_CACHE: Dict[Tuple[str, str, str, int, Optional[int], Optional[int]], Tuple[float, Dict[str, Any]]] = {}
_KLINE_CACHE_TTL = 0.75  # секунды: повторные запросы в одном цикле не идут в сеть
_KLINE_CACHE_MAX = 256
_KLINE_CACHE_LOCK = RLock()

_TICKER_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}
_TICKER_CACHE_TTL = 1.5
_TICKER_CACHE_MAX = 256
_TICKER_CACHE_LOCK = RLock()

_INSTRUMENTS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_INSTRUMENTS_CACHE_TTL = 120.0
_INSTRUMENTS_CACHE_MAX = 256
_INSTRUMENTS_CACHE_LOCK = RLock()


def _cache_get(cache: Dict[Any, Tuple[float, Any]], lock: RLock, key: Any, ttl: float) -> Optional[Any]:
    now = time.time()
    with lock:
        entry = cache.get(key)
        if not entry:
            return None
        ts, value = entry
        if now - ts <= ttl:
            return copy.deepcopy(value)
        cache.pop(key, None)
        return None


def _cache_set(cache: Dict[Any, Tuple[float, Any]], lock: RLock, key: Any, value: Any, max_size: int) -> None:
    snapshot = copy.deepcopy(value)
    now = time.time()
    with lock:
        cache[key] = (now, snapshot)
        if max_size and len(cache) > max_size:
            # удаляем самые старые элементы, чтобы ограничить рост
            sorted_items = sorted(cache.items(), key=lambda kv: kv[1][0])
            for old_key, _ in sorted_items[:-max_size]:
                cache.pop(old_key, None)


def _cache_prune_expired(cache: Dict[Any, Tuple[float, Any]], lock: RLock, ttl: float) -> None:
    now = time.time()
    with lock:
        stale = [k for k, (ts, _) in cache.items() if now - ts > ttl * 4]
        for key in stale:
            cache.pop(key, None)


def _record_params_error(context: str, payload: Dict[str, Any]) -> None:
    try:
        _PARAMS_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        record = {"ts": time.time(), "context": context}
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(_PARAMS_ERROR_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:
        log(f"[params-error-log] {context}: {exc}")


def configure_client(api_key: str, api_secret: str) -> None:
    """Обновляет глобальные креды и HTTP клиент (используется после изменения ключей)."""
    global API_KEY, API_SECRET, _CLIENT
    API_KEY = (api_key or "").strip()
    API_SECRET = (api_secret or "").strip()
    _CLIENT = None
    with _KLINE_CACHE_LOCK:
        _KLINE_CACHE.clear()
    with _TICKER_CACHE_LOCK:
        _TICKER_CACHE.clear()
    with _INSTRUMENTS_CACHE_LOCK:
        _INSTRUMENTS_CACHE.clear()

_LEVERAGE_CACHE: Dict[str, Tuple[int, int, float]] = {}
_LEVERAGE_CACHE_TTL_SEC = 15 * 60  # 15 минут
_FILTER_META: Dict[str, Dict[str, Any]] = {}


def _fallback_max_leverage() -> int:
    """Возвращает максимально допустимое плечо из конфига либо дефолт."""
    for attr in ("ADAPTIVE_LEV_MAX", "DEFAULT_LEVERAGE"):
        try:
            if _cfg is not None:
                val = getattr(_cfg, attr, None)
                if val is not None:
                    iv = int(float(val))
                    if iv > 0:
                        return iv
        except Exception:
            continue
    return 100

# =========================
# Вспомогательное
# =========================
def _sorted_kline_list(resp_result_list) -> List[list]:
    """
    Безопасно достаём список свечей и сортируем по ts ASC.
    Bybit v5 часто отдаёт в обратном порядке.
    """
    lst = (resp_result_list or [])
    try:
        lst = sorted(lst, key=lambda x: int(x[0]))  # x[0] — start time (ms)
    except Exception:
        pass
    return lst

def _parse_ohlcv_rows(raw_rows: List[list]) -> List[List[float]]:
    """Превращает массив клайнов Bybit в [[o,h,l,c,v], ...] (float)."""
    out: List[List[float]] = []
    for k in raw_rows:
        # v5 формат: [start, open, high, low, close, volume, turnover]
        try:
            o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        except Exception:
            # иногда встречается строка-испорченная: пропустим
            continue
        out.append([o, h, l, c, v])
    return out

# =========================
# Базовые вызовы рынка/позиции/баланс
# =========================
def get_tickers_linear():
    return safe_request(client.get_tickers, category="linear")

def get_positions(symbol: Optional[str] = None, settleCoin: str = "USDT"):
    kw = {"category": "linear", "settleCoin": settleCoin}
    if symbol:
        kw["symbol"] = symbol
    return safe_request(client.get_positions, **kw)

def get_wallet_balance():
    return safe_request(client.get_wallet_balance, accountType="UNIFIED", coin="USDT")

def get_instruments_info(symbol: str):
    """
    В большинстве мест нам нужна линейная спецификация.
    Если биржа отвечает 10001/symbol invalid — тихо возвращаем пустой list.
    """
    sym = str(symbol or "").upper()
    if not sym:
        return {"result": {"list": []}}

    cached = _cache_get(_INSTRUMENTS_CACHE, _INSTRUMENTS_CACHE_LOCK, sym, _INSTRUMENTS_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        data = safe_request(client.get_instruments_info, category="linear", symbol=sym)
    except Exception as e:
        msg = str(e)
        if "10001" in msg or "symbol invalid" in msg:
            # нет linear-контракта → считаем, что фильтров нет
            log(f"[SKIP] instruments_info {sym}: linear недоступен")
            data = {"result": {"list": []}}
        else:
            raise

    _cache_set(_INSTRUMENTS_CACHE, _INSTRUMENTS_CACHE_LOCK, sym, data, _INSTRUMENTS_CACHE_MAX)
    _cache_prune_expired(_INSTRUMENTS_CACHE, _INSTRUMENTS_CACHE_LOCK, _INSTRUMENTS_CACHE_TTL)
    return data

def get_kline(symbol: str, interval: str = "1", limit: int = 30):
    return safe_request(client.get_kline, category="linear", symbol=symbol, interval=interval, limit=limit)

def get_server_time() -> int:
    """
    Утилита: помогает при расследовании 10002 (timestamp/recv_window).
    Возвращает серверное время (ms), если доступно, иначе 0.
    """
    try:
        info = safe_request(client.get_server_time)
        return int((info.get("time", {}) or {}).get("serverTime", 0))
    except Exception:
        return 0

def has_open_position(symbol: str) -> bool:
    try:
        data = get_positions(symbol)
        lst = (data.get("result", {}) or {}).get("list", []) or []
        for pos in lst:
            if abs(float(pos.get("size", 0))) > 0:
                return True
        return False
    except Exception as e:
        log(f"[❌] has_open_position({symbol}): {e}")
        return False

def _safe_total_equity(log_prefix: str = "[❌] Ошибка получения баланса") -> float:
    try:
        data = get_wallet_balance()
        lst = (data.get("result", {}) or {}).get("list", []) or []
        return float(lst[0].get("totalEquity", 0.0)) if lst else 0.0
    except Exception as e:
        log(f"{log_prefix}: {e}")
        return 0.0


def get_balance() -> float:
    """Совместимый интерфейс — отдаём totalEquity, если доступно."""
    return _safe_total_equity()


def get_equity() -> float:
    """Синоним get_balance для единообразия с paper_engine."""
    return _safe_total_equity("[❌] Ошибка получения equity")


def ping_credentials(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Dict[str, Any]:
    """Быстрая проверка ключей: лёгкий REST вызов и возврат баланса."""
    key = (api_key or API_KEY or "").strip()
    secret = (api_secret or API_SECRET or "").strip()
    if not key or not secret:
        return {"ok": False, "error": "Не заданы ключ и секрет."}
    local_client = _create_client(key, secret)
    try:
        data = safe_request(local_client.get_wallet_balance, accountType="UNIFIED", coin="USDT")
        lst = (data.get("result", {}) or {}).get("list", []) or []
        total = 0.0
        if lst:
            try:
                total = float(lst[0].get("totalEquity", 0.0))
            except Exception:
                total = 0.0
        return {"ok": True, "balance": total}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

def fetch_price_history(symbol: str, limit: Optional[int] = None, interval: str = "1"):
    """
    Возвращает список [[open, high, low, close, volume], ...] (ASC по времени).
    """
    eff_limit = int(limit or _DEFAULT_KLINE_LIMIT)
    try:
        data = get_kline(symbol, interval=interval, limit=eff_limit)
        rows = _sorted_kline_list(((data.get("result", {}) or {}).get("list", []) or []))
        return _parse_ohlcv_rows(rows)
    except Exception as e:
        log(f"[❌] История {symbol}: {e}")
        return []

def get_min_order_filters(symbol: str) -> Tuple[float, float, Optional[float]]:
    """
    Возвращает (min_qty, qty_step, min_notional) если доступно.
    На деривативах min_notional может отсутствовать — вернём None.
    Если linear недоступен → спокойно отдаём фолбек.
    """
    sym = str(symbol or "").upper()
    reliable = True
    try:
        data = get_instruments_info(sym)
        info_list = (data.get("result", {}) or {}).get("list", []) or []
        if not info_list:
            raise RuntimeError("empty instruments_info")
        info = info_list[0]
        lot = info.get("lotSizeFilter", {}) or {}
        min_qty = float(lot.get("minOrderQty", 0) or 0)
        qty_step = float(lot.get("qtyStep", 0) or 0)
        pr = info.get("priceFilter", {}) or {}
        min_notional = float(pr.get("minOrderAmt")) if pr.get("minOrderAmt") is not None else None
    except Exception as e:
        reliable = False
        log(f"[❌] Фильтры {sym}: {e}")
        try:
            min_qty = float(getattr(_cfg, "DEFAULT_MIN_QTY_FALLBACK", 0.001))
        except Exception:
            min_qty = 1.0
        try:
            qty_step = float(getattr(_cfg, "DEFAULT_QTY_STEP_FALLBACK", 0.001))
        except Exception:
            qty_step = min_qty
        try:
            min_notional = float(getattr(_cfg, "DEFAULT_MIN_NOTIONAL_FALLBACK", 5.0))
        except Exception:
            min_notional = 5.0
        prev = _FILTER_META.get(sym, {})
        if prev.get("reliable", True):
            log(f"[FILTER] skip {sym}: no reliable filters")
    else:
        if min_qty <= 0 or qty_step <= 0:
            reliable = False
            log(f"[FILTER] skip {sym}: received non-positive filters {min_qty}/{qty_step}")
            try:
                min_qty = float(getattr(_cfg, "DEFAULT_MIN_QTY_FALLBACK", 0.001))
            except Exception:
                min_qty = 1.0
            try:
                qty_step = float(getattr(_cfg, "DEFAULT_QTY_STEP_FALLBACK", 0.001))
            except Exception:
                qty_step = min_qty
        if min_notional is None or min_notional <= 0:
            try:
                min_notional = float(getattr(_cfg, "DEFAULT_MIN_NOTIONAL_FALLBACK", 5.0))
            except Exception:
                min_notional = 5.0

    _FILTER_META[sym] = {
        "min_qty": float(min_qty or 0.0),
        "qty_step": float(qty_step or 0.0),
        "min_notional": float(min_notional or 0.0) if min_notional is not None else None,
        "reliable": bool(reliable),
        "ts": time.time(),
    }
    return float(min_qty or 0.0), float(qty_step or 0.0), min_notional


def filters_reliable(symbol: str) -> bool:
    sym = str(symbol or "").upper()
    meta = _FILTER_META.get(sym)
    if not meta:
        return True
    return bool(meta.get("reliable", True))


def _cache_get_leverage_limits(symbol: str) -> Optional[Tuple[int, int]]:
    sym = str(symbol or "").upper()
    if not sym:
        return None
    cached = _LEVERAGE_CACHE.get(sym)
    if not cached:
        return None
    min_lev, max_lev, ts = cached
    if time.time() - ts > _LEVERAGE_CACHE_TTL_SEC:
        return None
    return int(min_lev), int(max_lev)


def _cache_set_leverage_limits(symbol: str, min_lev: int, max_lev: int):
    sym = str(symbol or "").upper()
    if not sym:
        return
    _LEVERAGE_CACHE[sym] = (int(min_lev), int(max_lev), time.time())


def get_symbol_leverage_limits(symbol: str) -> Tuple[int, int]:
    """
    Возвращает (min_leverage, max_leverage) для символа, учитывая реальные ограничения.
    """
    cached = _cache_get_leverage_limits(symbol)
    if cached:
        return cached

    sym = str(symbol or "").upper()
    min_lev = 1
    max_lev = _fallback_max_leverage()

    try:
        data = get_instruments_info(sym)
        info_list = (data.get("result", {}) or {}).get("list", []) or []
        if info_list:
            info = info_list[0] or {}
            lev_filter = info.get("leverageFilter", {}) or {}
            min_raw = lev_filter.get("minLeverage")
            max_raw = lev_filter.get("maxLeverage")
            if min_raw is not None:
                min_lev = max(1, int(float(min_raw)))
            if max_raw is not None:
                max_lev = max(min_lev, int(float(max_raw)))
        else:
            log(f"[ℹ️] leverage limits {sym}: пустой instruments_info")
    except Exception as e:
        log(f"[❌] leverage limits {sym}: {e}")

    min_lev = max(1, min_lev)
    max_lev = max(min_lev, max_lev)
    _cache_set_leverage_limits(sym, min_lev, max_lev)
    return min_lev, max_lev


def get_max_leverage(symbol: str) -> int:
    """Удобный шорткат для получения максимального плеча по символу."""
    return get_symbol_leverage_limits(symbol)[1]


def get_current_price(symbol: str) -> float:
    """
    Быстрый lastPrice. Если linear не отвечает — тихий фолбек на spot.
    """
    # linear сперва
    try:
        data = safe_request(client.get_tickers, category="linear", symbol=symbol)
        lst = (data.get("result", {}) or {}).get("list", []) or []
        if lst:
            return float(lst[0].get("lastPrice", 0.0) or 0.0)
    except Exception:
        pass
    # spot фолбек
    try:
        data = safe_request(client.get_tickers, category="spot", symbol=symbol)
        lst = (data.get("result", {}) or {}).get("list", []) or []
        if lst:
            return float(lst[0].get("lastPrice", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0

# =========================
# Расширенные утилиты рынка
# =========================
def get_ticker_snapshot(symbol: str) -> Dict[str, float]:
    """
    Возвращает last_price/high/low/vol и др. поля.
    Сначала пробуем linear, если пусто/ошибка — spot.
    Отсутствующие для spot поля заполняем 0.0.
    """
    sym = str(symbol or "").upper()
    if not sym:
        return {"index_price":0.0,"last_price":0.0,"high":0.0,"low":0.0,
                "vol_24h":0.0,"open_interest":0.0,"funding_rate":0.0}

    cached = _cache_get(_TICKER_CACHE, _TICKER_CACHE_LOCK, sym, _TICKER_CACHE_TTL)
    if cached is not None:
        return cached

    # --- linear
    payload: Dict[str, float]
    try:
        d = safe_request(client.get_tickers, category="linear", symbol=sym)
        lst = ((d or {}).get("result") or {}).get("list") or []
        if lst:
            t = lst[0]
            payload = {
                "index_price":   float(t.get("indexPrice", 0) or 0.0),
                "last_price":    float(t.get("lastPrice", 0) or 0.0),
                "high":          float(t.get("highPrice24h", 0) or 0.0),
                "low":           float(t.get("lowPrice24h", 0) or 0.0),
                "vol_24h":       float(t.get("turnover24h", 0) or 0.0),
                "open_interest": float(t.get("openInterest", 0) or 0.0),
                "funding_rate":  float(t.get("fundingRate", 0) or 0.0),
            }
            _cache_set(_TICKER_CACHE, _TICKER_CACHE_LOCK, sym, payload, _TICKER_CACHE_MAX)
            _cache_prune_expired(_TICKER_CACHE, _TICKER_CACHE_LOCK, _TICKER_CACHE_TTL)
            return payload
    except Exception:
        pass

    # --- spot fallback
    try:
        d = safe_request(client.get_tickers, category="spot", symbol=sym)
        lst = ((d or {}).get("result") or {}).get("list") or []
        if lst:
            t = lst[0]
            payload = {
                "index_price":   0.0,
                "last_price":    float(t.get("lastPrice", 0) or 0.0),
                "high":          float(t.get("highPrice24h", 0) or 0.0),
                "low":           float(t.get("lowPrice24h", 0) or 0.0),
                "vol_24h":       float(t.get("turnover24h", 0) or 0.0),
                "open_interest": 0.0,
                "funding_rate":  0.0,
            }
            _cache_set(_TICKER_CACHE, _TICKER_CACHE_LOCK, sym, payload, _TICKER_CACHE_MAX)
            _cache_prune_expired(_TICKER_CACHE, _TICKER_CACHE_LOCK, _TICKER_CACHE_TTL)
            return payload
    except Exception:
        pass

    # --- ничего не нашли
    empty = {"index_price":0.0,"last_price":0.0,"high":0.0,"low":0.0,
             "vol_24h":0.0,"open_interest":0.0,"funding_rate":0.0}
    _cache_set(_TICKER_CACHE, _TICKER_CACHE_LOCK, sym, empty, _TICKER_CACHE_MAX)
    _cache_prune_expired(_TICKER_CACHE, _TICKER_CACHE_LOCK, _TICKER_CACHE_TTL)
    return empty


def _sum_sizes(levels: Iterable[Iterable[Any]]) -> float:
    """Безопасно суммирует объёмы из стакана (Bybit: [price, size, ...])."""
    total = 0.0
    for lvl in levels or []:
        try:
            qty = float(lvl[1])
        except (TypeError, ValueError, IndexError):
            qty = 0.0
        total += max(qty, 0.0)
    return total


def orderbook_imbalance(symbol: str, depth: int = 5) -> float:
    """Возвращает нормализованный дисбаланс лимитных заявок (Σbid-Σask)/Σtotal."""
    try:
        depth = max(1, int(depth))
        data = safe_request(
            client.get_orderbook,
            category="linear",
            symbol=symbol,
            limit=depth,
        )
        result = data.get("result", {}) or {}
        bids = result.get("b", []) or result.get("bids", []) or []
        asks = result.get("a", []) or result.get("asks", []) or []
        bid_sum = _sum_sizes(bids[:depth])
        ask_sum = _sum_sizes(asks[:depth])
        denom = bid_sum + ask_sum
        if denom <= 0:
            return 0.0
        return float((bid_sum - ask_sum) / denom)
    except Exception as e:
        log(f"[orderbook_imbalance] {symbol}: {e}", level="ERROR")
        return 0.0


def get_orderbook_spread(symbol: str, depth: int = 1) -> float:
    """
    Грубая оценка спрэда по лучшим бид/аск (linear).
    """
    try:
        d = safe_request(client.get_orderbook, category="linear", symbol=symbol, limit=depth)
        r = (d.get("result", {}) or {})
        bids = r.get("b", []) or r.get("bids", [])
        asks = r.get("a", []) or r.get("asks", [])
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0
            return (best_ask - best_bid) / mid if mid > 0 else 0.0
    except Exception:
        pass
    return 0.0

def get_kline_block(symbol: str, category: str = "linear",
                    interval: str = "1", start: Optional[int] = None,
                    end: Optional[int] = None, limit: int = 200):
    """
    Обёртка над get_kline с поддержкой start/end (ms). Если биржа игнорит end — вернёт последние.
    """
    sym = str(symbol or "").upper()
    cat = str(category or "linear").lower()
    start_key = int(start) if start is not None else None
    end_key = int(end) if end is not None else None
    cache_key = (sym, cat, str(interval), int(limit), start_key, end_key)

    cached = _cache_get(_KLINE_CACHE, _KLINE_CACHE_LOCK, cache_key, _KLINE_CACHE_TTL)
    if cached is not None:
        return cached

    kw = {"category": cat, "symbol": sym, "interval": interval, "limit": limit}
    if start is not None: kw["start"] = int(start)
    if end   is not None: kw["end"]   = int(end)
    try:
        resp = safe_request(client.get_kline, **kw)
        resp["result"] = resp.get("result", {}) or {}
        lst = _sorted_kline_list(resp["result"].get("list", []) or [])
        resp["result"]["list"] = lst
        _cache_set(_KLINE_CACHE, _KLINE_CACHE_LOCK, cache_key, resp, _KLINE_CACHE_MAX)
        _cache_prune_expired(_KLINE_CACHE, _KLINE_CACHE_LOCK, _KLINE_CACHE_TTL)
        return resp
    except Exception as e:
        log(f"[❌] get_kline_block {sym} {cat}: {e}")
        empty = {"result": {"list": []}}
        _cache_set(_KLINE_CACHE, _KLINE_CACHE_LOCK, cache_key, empty, _KLINE_CACHE_MAX)
        return empty


def get_kline_any(symbol: str,
                  interval: str = "1",
                  limit: Optional[int] = None,
                  end_ms: Optional[int] = None):
    """
    Возвращает ([[o,h,l,c,v], ...], source) со строгой сортировкой по времени.
    Сначала LINEAR, если пусто/ошибка — SPOT.
    """
    eff_limit = int(limit or _DEFAULT_KLINE_LIMIT)
    sym = str(symbol or "").upper()
    # linear
    r_lin = get_kline_block(sym, "linear", interval=interval, end=end_ms, limit=eff_limit)
    lst_lin = (r_lin.get("result", {}) or {}).get("list", []) or []
    if lst_lin:
        return _parse_ohlcv_rows(lst_lin), "linear"

    # spot fallback
    r_spot = get_kline_block(sym, "spot", interval=interval, end=end_ms, limit=eff_limit)
    lst_spot = (r_spot.get("result", {}) or {}).get("list", []) or []
    return _parse_ohlcv_rows(lst_spot), ("spot" if lst_spot else "linear")


# =========================
# Ордеры и управление позициями
# =========================
def place_market_order(symbol: str, side: str, qty: float, reduce_only: bool = False):
    """
    Маркет по рынку. Для закрытия позиции задавайте reduce_only=True.
    """
    if _paper_mode_active():
        raise RuntimeError("Trading HTTP заблокирован в PAPER режиме")
    _ensure_client()
    try:
        res = safe_request(
            client.place_order,
            category="linear",
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=str(qty),
            time_in_force="IOC",
            reduce_only=reduce_only,
            orderLinkId=make_order_link_id("BOT")
        )
        log(f"[📤] Market {side} {qty} {symbol} (reduce_only={reduce_only}) → ID: {res.get('result',{}).get('orderId')}")
        return res
    except Exception as e:
        log(f"[❌] Ошибка маркет-ордера: {e}")
        min_qty = qty_step = min_notional = None
        try:
            min_qty, qty_step, min_notional = get_min_order_filters(symbol)
        except Exception:
            pass
        _record_params_error(
            "place_market_order",
            {
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "reduce_only": bool(reduce_only),
                "error": str(e),
                "min_qty": min_qty,
                "qty_step": qty_step,
                "min_notional": min_notional,
            },
        )
        return None

def close_position_by_market(symbol: str, qty: Optional[float] = None, max_attempts: int = 5):
    """
    Закрывает позицию рыночным ордером с reduce_only=True (без переворота).
    """
    try:
        positions = get_positions(symbol)
        data = (positions.get("result", {}) or {}).get("list", []) or []
        found = False
        for pos in data:
            size = float(pos.get("size", 0) or 0)
            if size == 0:
                continue
            found = True
            side = pos.get("side")
            close_side = "Sell" if side == "Buy" else "Buy"
            order_qty = abs(size) if qty is None else min(abs(size), abs(qty))
            attempts = 0
            while attempts < max_attempts:
                res = place_market_order(symbol, close_side, order_qty, reduce_only=True)
                if res is not None:
                    log(f"[📤] Закрытие {symbol} {close_side} {order_qty}")
                    break
                attempts += 1
                time.sleep(0.3 * attempts)
        if not found:
            log(f"[INFO] Нет позиции для закрытия по {symbol}")
    except Exception as e:
        log(f"[❌] close_position_by_market: {e}")

def set_leverage(symbol: str, leverage: int = 10):
    if _paper_mode_active():
        raise RuntimeError("Trading HTTP заблокирован в PAPER режиме")
    _ensure_client()
    lev_requested = max(1, int(float(leverage or 1)))
    lev_to_set = lev_requested

    try:
        min_lev, max_lev = get_symbol_leverage_limits(symbol)
        clamped = max(min_lev, min(max_lev, lev_to_set))
        if lev_to_set != clamped:
            log(f"[LEV] {symbol}: clamp {lev_requested}x → {clamped}x (bounds {min_lev}-{max_lev})")
        lev_to_set = clamped
    except Exception as e:
        log(f"[❓] set_leverage {symbol}: не удалось проверить лимиты ({e})")

    try:
        res = safe_request(
            client.set_leverage,
            category="linear",
            symbol=symbol,
            buyLeverage=str(lev_to_set),
            sellLeverage=str(lev_to_set)
        )
        if not isinstance(res, dict):
            log(f"[❌] set_leverage {symbol}: unexpected response {res!r}")
            return False

        ret = res.get("retCode")
        if ret == 0:
            log(f"[LEVERAGE] {symbol}: {lev_to_set}x (req {lev_requested}x)")
            return True

        msg = res.get("retMsg", "")
        if str(ret) == "110043":
            log(f"[ℹ️] set_leverage {symbol}: leverage not modified")
        else:
            log(f"[❌] set_leverage {symbol}: retCode={ret} {msg}")
            _record_params_error(
                "set_leverage",
                {
                    "symbol": symbol,
                    "leverage_requested": lev_requested,
                    "ret_code": ret,
                    "ret_msg": msg,
                },
            )
        return False
    except Exception as e:
        if "110043" in str(e):
            log(f"[ℹ️] set_leverage {symbol}: leverage not modified")
        else:
            log(f"[❌] set_leverage {symbol}: {e}")
        _record_params_error(
            "set_leverage",
            {
                "symbol": symbol,
                "leverage_requested": lev_requested,
                "error": str(e),
            },
        )
        return False

# =========================
# Индикативные расчёты
# =========================
def get_atr(symbol: str, interval: str = "15", period: int = 14) -> float:
    """
    Простой ATR по close/high/low (среднее TR). Возвращает абсолютную цену, не %.
    """
    try:
        data = get_kline(symbol, interval=interval, limit=period + 1)
        rows = _sorted_kline_list(((data.get("result", {}) or {}).get("list", []) or []))
        closes = [float(c[4]) for c in rows]
        highs  = [float(c[2]) for c in rows]
        lows   = [float(c[3]) for c in rows]
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        return round(sum(trs) / len(trs), 6) if trs else 0.0
    except Exception as e:
        log(f"[❌] ATR {symbol}: {e}")
        return 0.0

def force_close_all_positions_absolute():
    try:
        data = safe_request(client.get_positions, category="linear", settleCoin="USDT")
        for pos in (data.get("result", {}) or {}).get("list", []) or []:
            symbol = pos.get("symbol")
            size = float(pos.get("size", 0) or 0)
            if symbol and abs(size) > 0:
                side = pos.get("side")
                close_side = "Sell" if side == "Buy" else "Buy"
                place_market_order(symbol, close_side, abs(size), reduce_only=True)
                log(f"[FORCE CLOSE] {symbol} qty={size}")
        log("[🏁] Все доступные позиции обработаны.")
    except Exception as e:
        log(f"[❌] ABSOLUTE force close: {e}")


def close_all_positions():
    """Публичный алиас для совместимости с paper_engine."""
    return force_close_all_positions_absolute()

def fast_pick_top_pairs(count: int = 2, min_volume_usdt: float = 10_000_000,
                        min_atr_rel: float = 0.0005, top_volume: int = 20, min_price: float = 0.008):
    """
    Быстрый фильтр ликвидных пар с минимальной волатильностью (linear universe).
    """
    ALLOWED_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "ADAUSDT",
        "MATICUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT", "OPUSDT", "LTCUSDT"
    ]
    try:
        tickers = ((get_tickers_linear() or {}).get("result", {}) or {}).get("list", []) or []
    except Exception as e:
        log(f"[❌] Тикеры: {e}")
        return []

    usdt_pairs = []
    for t in tickers:
        try:
            symbol = t.get("symbol")
            if not symbol or not symbol.endswith("USDT"):
                continue
            if symbol not in ALLOWED_SYMBOLS:
                continue
            price = float(t.get("lastPrice", 0) or 0)
            vol_24h = float(t.get("turnover24h", 0) or 0)
            if price < min_price or vol_24h < min_volume_usdt:
                continue
            usdt_pairs.append((symbol, vol_24h, price))
        except Exception:
            continue

    usdt_pairs = sorted(usdt_pairs, key=lambda x: x[1], reverse=True)[:top_volume]

    stats = []
    for symbol, vol_24h, price in usdt_pairs:
        try:
            candles = ((get_kline(symbol, interval="15", limit=20) or {}).get("result", {}) or {}).get("list", []) or []
            rows = _sorted_kline_list(candles)
            closes = [float(x[4]) for x in rows]
            if len(closes) < 5:
                continue
            import numpy as np
            atr_rel = float(np.std(closes) / (np.mean(closes) or 1.0))
            if atr_rel < min_atr_rel:
                continue
            stats.append((symbol, atr_rel, vol_24h, price))
        except Exception as e:
            log(f"[PAIR SKIP] {symbol}: {e}")
            continue

    stats = sorted(stats, key=lambda x: (x[1], x[2]), reverse=True)
    top = [x[0] for x in stats[:count]]
    log(f"[FAST PICK] Топ-{count}: {top}")
    return top

def get_margin_info():
    """
    Простая сводка по марже. Возвращает проценты IM/MM и equity (грубо).
    """
    try:
        w = safe_request(client.get_wallet_balance, accountType="UNIFIED", coin="USDT")
        equity = float(((w.get("result", {}) or {}).get("list", []) or [{}])[0].get("totalEquity", 0.0))
    except Exception:
        equity = 0.0
    try:
        pos = ((safe_request(client.get_positions, category="linear", settleCoin="USDT") or {}).get("result", {}) or {}).get("list", []) or []
        im_used = 0.0
        for p in pos:
            im_used += float(p.get("positionIM", 0) or 0)
        im_pct = (im_used / equity * 100) if equity > 0 else 0.0
        mm_pct = im_pct * 0.5  # очень грубо
        return {"IM": round(im_pct, 2), "MM": round(mm_pct, 2), "equity": round(equity, 2)}
    except Exception as e:
        log(f"[❌] get_margin_info error: {e}")
        return {"IM": 0.0, "MM": 0.0, "equity": equity}

# =========================
# История сделок (fills) и ордеров — импорт из Bybit
# =========================
def get_executions_page(
    category: str,
    start_ms: int,
    end_ms: int,
    symbol: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = 100
) -> Dict:
    """
    Одна страница исполнений (fills). Окно запроса — до ~7 дней.
    """
    params = {
        "category": category,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    if symbol:
        params["symbol"] = symbol
    if cursor:
        params["cursor"] = cursor

    return safe_request(client.get_executions, **params)

def get_executions_all(
    category: str,
    start_ms: int,
    end_ms: int,
    symbol: Optional[str] = None,
    limit: int = 100,
    sleep_sec: float = 0.2
) -> List[Dict]:
    """
    Обходит cursor-пагинацию и собирает все fills за интервал (до 7 дней).
    """
    all_items: List[Dict] = []
    cursor = None
    while True:
        resp = get_executions_page(
            category=category,
            start_ms=start_ms,
            end_ms=end_ms,
            symbol=symbol,
            cursor=cursor,
            limit=limit
        )
        result = resp.get("result", {}) or {}
        items = result.get("list", []) or []
        all_items.extend(items)

        cursor = result.get("nextPageCursor")
        if not cursor or len(items) == 0:
            break
        time.sleep(sleep_sec)
    return all_items

def iter_time_windows(start_ms: int, end_ms: int, window_ms: int = 7 * 24 * 60 * 60 * 1000):
    """
    Бьёт период на окна (ограничение API ~7 дней на запрос).
    """
    cur = start_ms
    while cur < end_ms:
        nxt = min(cur + window_ms - 1, end_ms)
        yield cur, nxt
        cur = nxt + 1

def fetch_executions_range(
    category: str,
    start_ms: int,
    end_ms: int,
    symbol: Optional[str] = None
) -> List[Dict]:
    """
    Тянет все fills за длинный период (разбивка на окна по 7 дней).
    """
    out: List[Dict] = []
    for s, e in iter_time_windows(start_ms, end_ms):
        out.extend(get_executions_all(category=category, start_ms=s, end_ms=e, symbol=symbol))
    return out

def normalize_fill(rec: Dict) -> Dict:
    """
    Нормализация одного fill под плоский формат.
    """
    return {
        "ts": int(rec.get("execTime", 0)),                 # ms
        "symbol": rec.get("symbol"),
        "side": (rec.get("side") or "").lower(),          # buy/sell
        "price": float(rec.get("execPrice", 0) or 0),
        "qty": float(rec.get("execQty", 0) or 0),
        "fee": float(rec.get("execFee", 0) or 0),
        "feeCurrency": rec.get("feeCurrency") or "USDT",
        "isMaker": bool(rec.get("isMaker", False)),
        "orderId": rec.get("orderId"),
        "execId": rec.get("execId"),
        "value": float(rec.get("execValue", 0) or 0),
        "orderLinkId": rec.get("orderLinkId"),
    }

# =========================
# Удобные обёртки: datetime → выгрузка и CSV
# =========================