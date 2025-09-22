# -*- coding: utf-8 -*-
"""
bybit_api.py — HTTP v5 wrapper с безопасными вызовами и утилитами для онлайна.
Совместим с существующим кодом проекта.

Ключевое:
- safe_request из api_guard — единая точка ретраев/бэкоффов.
- Уважает .env / config.py для переключения testnet/real (BYBIT_TESTNET=1/0).
- Фолбек LINEAR→SPOT для снапшотов и свечей, чтобы не ловить 10001 на «нестандартных» символах.
"""

from __future__ import annotations

import os
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# утилиты проекта
from utils import log, tg_send, adjust_qty, make_order_link_id

# центр ретраев / лимитов — единая точка правды
from api_guard import safe_request

# =========================
# Инициализация клиента (уважаем .env и/или config.py)
# =========================
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# 1 = testnet, 0 = mainnet
ENV_TESTNET = str(os.getenv("BYBIT_TESTNET", "0")).strip().lower() in ("1", "true", "yes")

try:
    import config as _cfg  # опционально
    CFG_TESTNET = bool(getattr(_cfg, "BYBIT_TESTNET", ENV_TESTNET))
except Exception:
    _cfg = None
    CFG_TESTNET = ENV_TESTNET

client = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    testnet=CFG_TESTNET,
    recv_window=20000,
    timeout=10
)

_LEVERAGE_CACHE: Dict[str, Tuple[int, int, float]] = {}
_LEVERAGE_CACHE_TTL_SEC = 15 * 60  # 15 минут


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
    try:
        return safe_request(client.get_instruments_info, category="linear", symbol=symbol)
    except Exception as e:
        msg = str(e)
        if "10001" in msg or "symbol invalid" in msg:
            # нет linear-контракта → считаем, что фильтров нет
            log(f"[SKIP] instruments_info {symbol}: linear недоступен")
            return {"result": {"list": []}}
        raise

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

def get_balance() -> float:
    """
    Возвращает totalEquity, если доступно (UNIFIED аккаунт).
    """
    try:
        data = get_wallet_balance()
        lst = (data.get("result", {}) or {}).get("list", []) or []
        return float(lst[0].get("totalEquity", 0.0)) if lst else 0.0
    except Exception as e:
        log(f"[❌] Ошибка получения баланса: {e}")
        return 0.0

def fetch_price_history(symbol: str, limit: int = 30, interval: str = "1"):
    """
    Возвращает список [[open, high, low, close, volume], ...] (ASC по времени).
    """
    try:
        data = get_kline(symbol, interval=interval, limit=limit)
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
    try:
        data = get_instruments_info(symbol)
        info_list = (data.get("result", {}) or {}).get("list", []) or []
        if not info_list:
            raise RuntimeError("empty instruments_info")
        info = info_list[0]
        lot = info.get("lotSizeFilter", {}) or {}
        min_qty = float(lot.get("minOrderQty", 0) or 0)
        qty_step = float(lot.get("qtyStep", 0) or 0)
        pr = info.get("priceFilter", {}) or {}
        min_notional = float(pr.get("minOrderAmt")) if pr.get("minOrderAmt") is not None else None
        return min_qty, qty_step, min_notional
    except Exception as e:
        log(f"[❌] Фильтры {symbol}: {e}")
        # Фолбэки на популярные пары, чтобы не падать
        if symbol.startswith("BTC"): return 0.001, 0.001, None
        if symbol.startswith("ETH"): return 0.01, 0.01, None
        return 1.0, 1.0, None


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

def get_24h_volume(symbol: str) -> float:
    try:
        data = safe_request(client.get_tickers, category="linear", symbol=symbol)
        lst = (data.get("result", {}) or {}).get("list", []) or []
        return float(lst[0].get("turnover24h", 0.0)) if lst else 0.0
    except Exception as e:
        log(f"[❌] 24h объём {symbol}: {e}")
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
    # --- linear
    try:
        d = safe_request(client.get_tickers, category="linear", symbol=symbol)
        lst = ((d or {}).get("result") or {}).get("list") or []
        if lst:
            t = lst[0]
            return {
                "index_price":   float(t.get("indexPrice", 0) or 0.0),
                "last_price":    float(t.get("lastPrice", 0) or 0.0),
                "high":          float(t.get("highPrice24h", 0) or 0.0),
                "low":           float(t.get("lowPrice24h", 0) or 0.0),
                "vol_24h":       float(t.get("turnover24h", 0) or 0.0),
                "open_interest": float(t.get("openInterest", 0) or 0.0),
                "funding_rate":  float(t.get("fundingRate", 0) or 0.0),
            }
    except Exception:
        pass

    # --- spot fallback
    try:
        d = safe_request(client.get_tickers, category="spot", symbol=symbol)
        lst = ((d or {}).get("result") or {}).get("list") or []
        if lst:
            t = lst[0]
            return {
                "index_price":   0.0,
                "last_price":    float(t.get("lastPrice", 0) or 0.0),
                "high":          float(t.get("highPrice24h", 0) or 0.0),
                "low":           float(t.get("lowPrice24h", 0) or 0.0),
                "vol_24h":       float(t.get("turnover24h", 0) or 0.0),
                "open_interest": 0.0,
                "funding_rate":  0.0,
            }
    except Exception:
        pass

    # --- ничего не нашли
    return {"index_price":0.0,"last_price":0.0,"high":0.0,"low":0.0,
            "vol_24h":0.0,"open_interest":0.0,"funding_rate":0.0}

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
    kw = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
    if start is not None: kw["start"] = int(start)
    if end   is not None: kw["end"]   = int(end)
    try:
        resp = safe_request(client.get_kline, **kw)
        resp["result"] = resp.get("result", {}) or {}
        lst = _sorted_kline_list(resp["result"].get("list", []) or [])
        resp["result"]["list"] = lst
        return resp
    except Exception as e:
        log(f"[❌] get_kline_block {symbol} {category}: {e}")
        return {"result": {"list": []}}

def get_kline_any(symbol: str, interval: str = "1", limit: int = 60, end_ms: Optional[int] = None):
    """
    Возвращает ([[o,h,l,c,v], ...], source) со строгой сортировкой по времени.
    Сначала LINEAR, если пусто/ошибка — SPOT.
    """
    # linear
    r_lin = get_kline_block(symbol, "linear", interval=interval, end=end_ms, limit=limit)
    lst_lin = (r_lin.get("result", {}) or {}).get("list", []) or []
    if lst_lin:
        return _parse_ohlcv_rows(lst_lin), "linear"

    # spot fallback
    r_spot = get_kline_block(symbol, "spot", interval=interval, end=end_ms, limit=limit)
    lst_spot = (r_spot.get("result", {}) or {}).get("list", []) or []
    return _parse_ohlcv_rows(lst_spot), ("spot" if lst_spot else "linear")

# =========================
# Ордеры и управление позициями
# =========================
def place_market_order(symbol: str, side: str, qty: float, reduce_only: bool = False):
    """
    Маркет по рынку. Для закрытия позиции задавайте reduce_only=True.
    """
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
        return False
    except Exception as e:
        if "110043" in str(e):
            log(f"[ℹ️] set_leverage {symbol}: leverage not modified")
        else:
            log(f"[❌] set_leverage {symbol}: {e}")
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

def _align_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    import math
    return math.floor(x / step) * step

def can_trade_min(symbol: str,
                  avail_usdt: float,
                  leverage_effective: float = 10.0,
                  hard_cap_share: float = 0.25,
                  min_notional_usdt_cfg: float = 5.0) -> tuple[bool, float, float, str]:
    """
    Проверяем, пролезает ли минимальная сделка по символу при заданном балансе/плече.
    Возвращает: (ok, min_qty_aligned, notional_min, reason)
    """
    price = get_current_price(symbol)
    min_qty, step, min_notional = get_min_order_filters(symbol)
    min_qty_aligned = max(min_qty, _align_step(min_qty, step))
    notional_min = price * min_qty_aligned

    # Биржевой минимум по нотационалу
    min_notional_floor = max(min_notional or 0.0, float(min_notional_usdt_cfg or 0.0))

    # Наш «хард» лимит с учётом плеча
    hard_cap_notional = max(0.0, float(avail_usdt) * float(hard_cap_share) * max(1.0, float(leverage_effective)))

    if notional_min < min_notional_floor:
        return False, min_qty_aligned, notional_min, "min_notional_floor"

    if notional_min > hard_cap_notional:
        return False, min_qty_aligned, notional_min, "hard_cap"

    return True, min_qty_aligned, notional_min, ""

def pick_pairs_by_balance(count: int,
                          avail_usdt: float,
                          leverage_effective: float = 10.0,
                          hard_cap_share: float = 0.25,
                          min_notional_usdt_cfg: float = 5.0,
                          min_price: float = 0.008,
                          top_volume: int = 30) -> list[str]:
    """
    Выбирает пары, по которым можно открыть хотя бы минимальную позицию,
    затем ранжирует по простому скору (объём 24h + «ATR%»).
    """
    ALLOWED_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "ADAUSDT",
        "MATICUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT", "OPUSDT", "LTCUSDT"
    ]
    try:
        tickers = ((get_tickers_linear() or {}).get("result", {}) or {}).get("list", []) or []
    except Exception as e:
        log(f"[❌] pick_pairs_by_balance/tickers: {e}")
        return []

    # предварительный фильтр по списку/цене
    pool = []
    for t in tickers:
        try:
            sym = t.get("symbol")
            if sym not in ALLOWED_SYMBOLS:
                continue
            price = float(t.get("lastPrice", 0) or 0)
            vol24 = float(t.get("turnover24h", 0) or 0)
            if price < min_price:
                continue
            pool.append((sym, vol24, price))
        except Exception:
            continue

    # топ по объёму, затем проверка достижимости min_qty с учётом плеча/каппа
    pool = sorted(pool, key=lambda x: x[1], reverse=True)[:top_volume]

    candidates = []
    for sym, vol24, price in pool:
        ok, min_qty_aligned, notional_min, reason = can_trade_min(
            sym, avail_usdt, leverage_effective, hard_cap_share, min_notional_usdt_cfg
        )
        if not ok:
            continue

        # прикидываем «ATR%» как stdev(close)/mean(close) на 15m
        try:
            kl = ((get_kline(sym, interval="15", limit=20) or {}).get("result", {}) or {}).get("list", []) or []
            rows = _sorted_kline_list(kl)
            closes = [float(x[4]) for x in rows]
            if len(closes) < 5:
                continue
            import numpy as np
            atr_rel = float(np.std(closes) / (np.mean(closes) or 1.0))
        except Exception:
            atr_rel = 0.0

        # скор: ликвидность и волатильность
        import math
        liq = math.log10(max(vol24, 1.0))
        score = (atr_rel * 0.6) + (liq * 0.4)

        candidates.append((sym, score, notional_min, vol24, atr_rel))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [c[0] for c in candidates[:count]]
    log(f"[PICK-BAL] avail={avail_usdt:.2f} lev~{leverage_effective} → {top} (n={len(candidates)})")
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

def get_order_history_all(
    category: str,
    start_ms: int,
    end_ms: int,
    symbol: Optional[str] = None,
    limit: int = 100,
    sleep_sec: float = 0.2
) -> List[Dict]:
    """
    История ордеров с пагинацией (для статусов, связки orderId → execs).
    """
    all_items: List[Dict] = []
    cursor = None
    while True:
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

        resp = safe_request(client.get_order_history, **params)
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
def fetch_fills(
    category: str = "linear",
    symbol: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[Dict]:
    """
    Принимает datetime (UTC), внутри дергает fetch_executions_range().
    Обязательно укажи start и end.
    """
    if not start or not end:
        raise ValueError("fetch_fills: нужны start и end (datetime, UTC)")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    return fetch_executions_range(
        category=category,
        start_ms=start_ms,
        end_ms=end_ms,
        symbol=symbol
    )

def export_fills_to_csv(
    path: str,
    category: str = "linear",
    symbol: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> str:
    """
    Тянет сделки и сохраняет CSV в нормализованном виде (через normalize_fill()).
    """
    raw = fetch_fills(category=category, symbol=symbol, start=start, end=end)
    rows = [normalize_fill(r) for r in raw]

    fields = [
        "ts","symbol","side","price","qty","fee","feeCurrency",
        "isMaker","orderId","execId","value","orderLinkId"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    log(f"[CSV] {len(rows)} rows → {path}")
    return path
