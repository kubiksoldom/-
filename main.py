# -*- coding: utf-8 -*-
# main.py — 2025-09-22: интеграция StrategyRouter (router_reason/strategy/regime в логах)

import os
import sys
import time
import json
import math
import datetime
from datetime import timedelta
import importlib
import subprocess
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
# ⬇️ подключаем новый роутер стратегий
from strategy import decide_with_router
from utils import (
    log, tg_send, write_cycle_log, adjust_qty,
    SAFE_MODE as SAFE_MODE_FROM_ENV
)
from ml_veto import load_model_and_meta, predict_ok, atr_abs as _atr_abs

# --- timezone helper (stdlib zoneinfo с фоллбэком) ---
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # будем работать в локальном времени без tz

# =========================
# CLI-переключатель PAPER_MODE
# =========================
if len(sys.argv) > 1:
    arg = sys.argv[1].lower()
    if arg in ("paper", "p"):
        config.PAPER_MODE = 1
        print("🧪 PAPER_MODE включен через терминал")
    elif arg in ("real", "r"):
        if "--yes" not in sys.argv:
            resp = input("⚠️ Запуск в РЕАЛЬНОМ режиме. Продолжить? [y/N]: ").strip().lower()
            if resp != "y":
                print("Отмена запуска.")
                raise SystemExit(1)
        config.PAPER_MODE = 0
        print("💰 REAL_MODE включен через терминал")

# Флаг «опасного» запуска (разрешить отключать SAFE_MODE руками)
UNSAFE_FLAG = ("--unsafe" in sys.argv)

# =========================
# Режимы / Безопасность
# =========================
PAPER_MODE = bool(getattr(config, "PAPER_MODE", 0))
# Локальный SAFE_MODE = из .env/конфига; можно снять только если есть флаг --unsafe
SAFE_MODE = bool(SAFE_MODE_FROM_ENV or getattr(config, "PAPER_MODE", 0))
if UNSAFE_FLAG and not PAPER_MODE:
    SAFE_MODE = False
if PAPER_MODE:
    import paper_engine as broker
    log("🧪 PAPER_MODE=ON — сделки моделируются в paper_engine, маркет-данные — реальные.")
else:
    import bybit_api as broker
    if SAFE_MODE:
        log("🛡 SAFE_MODE=ON — ордера на биржу НЕ отправляются (dry-run).")
    else:
        log("🚀 SAFE_MODE=OFF — ордера будут отправляться на биржу.")

# =========================
# Анти-тильт
# =========================
loss_streak = 0
last_loss_time = 0.0
LOSS_STREAK_MAX = max(3, int(getattr(config, "MAX_CONSECUTIVE_LOSSES", 3)))
LOSS_COOLDOWN  = int(getattr(config, "LOSS_COOLDOWN_SEC", 900))  # 15 мин

# =========================
# Тайминги цикла
# =========================
WORK_DURATION_SEC   = int(getattr(config, "WORK_DURATION_SEC", 3600))   # 60 мин
BREAK_DURATION_SEC  = int(getattr(config, "BREAK_DURATION_SEC", 600))   # 10 мин
ENTRY_COOLDOWN_SEC  = int(getattr(config, "ENTRY_COOLDOWN_SEC", 45))

DATA_ROOT           = getattr(config, "DATA_ROOT", None) or os.getenv("DATA_ROOT", "").strip()
RECORD_MARKET_DATA  = bool(int(os.getenv("RECORD_MARKET_DATA", str(getattr(config, "RECORD_MARKET_DATA", 1)))))

# частоты опроса
KLINE_REFRESH_SEC    = 2.0
SNAPSHOT_REFRESH_SEC = 1.0
PERSIST_EVERY_SEC    = 60.0
DD_CHECK_EVERY_SEC   = 10.0

# --- управление из UI через control.json ---
CONTROL_POLL_SEC = float(getattr(config, "CONTROL_POLL_SEC", 1.5))
PAUSE_ENTRIES = False  # глобальный флаг «пауза входов»

# =========================
# (2) Правило автоподбора числа пар по балансу (баланс→кол-во)
# =========================
AUTO_PAIRS_RULE = str(getattr(
    config,
    "AUTO_PAIRS_RULE",
    "0:1,10:1,25:1,50:2,100:3"
))

def _parse_pairs_rule(rule: str) -> List[Tuple[float, int]]:
    out = []
    for chunk in rule.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        lv, ct = chunk.split(":", 1)
        try:
            bal = float(lv); n = int(ct)
            out.append((bal, n))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out

def _pairs_count_for_balance(balance: float, rule: str) -> int:
    tbl = _parse_pairs_rule(rule)
    if not tbl:
        return max(1, int(getattr(config, "PAIRS_COUNT", 1)))
    chosen = tbl[0][1]
    for bal, n in tbl:
        if balance >= bal:
            chosen = n
        else:
            break
    return max(1, chosen)

# =========================
# (3) Фильтр «надёжных» пар по нотационалу min_qty*price
# =========================
PAIR_FILTER_MIN_NOTIONAL = float(getattr(config, "PAIR_FILTER_MIN_NOTIONAL", 5.0))  # левый край окна
PAIR_FILTER_HCAP_FRAC    = float(getattr(config, "PAIR_FILTER_HCAP_FRAC", 1.0))     # правый край = balance * frac

# =========================
# Хелперы по свечам (понимаем оба формата: [ts,o,h,l,c,v,…] ИЛИ [o,h,l,c,v])
# =========================
def _row_has_ts(row: List[Any]) -> bool:
    return isinstance(row, (list, tuple)) and len(row) >= 6 and isinstance(row[0], (int, float))

def kl_to_ohlcv(kl: List[List[Any]]) -> List[List[float]]:
    """Нормализует список свечей к виду [[o,h,l,c,v], ...]"""
    out: List[List[float]] = []
    for r in kl or []:
        try:
            if _row_has_ts(r):
                o,h,l,c,v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            else:
                o,h,l,c = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                v = float(r[4]) if len(r) > 4 else 0.0
            out.append([o,h,l,c,v])
        except Exception:
            continue
    return out

def kl_closes(kl: List[List[Any]]) -> List[float]:
    """Возвращает массив close по любому формату входных свечей."""
    closes: List[float] = []
    for r in kl or []:
        try:
            if _row_has_ts(r):
                closes.append(float(r[4]))
            else:
                closes.append(float(r[3]))
        except Exception:
            continue
    return closes

# =========================
# Хелперы (время/расписание)
# =========================
def utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _get_tz():
    tzname = os.getenv("TIMEZONE") or getattr(config, "TIMEZONE", None)
    if tzname and ZoneInfo:
        try:
            return ZoneInfo(tzname)
        except Exception:
            pass
    return None

def _now_local() -> datetime.datetime:
    tz = _get_tz()
    return datetime.datetime.now(tz) if tz else datetime.datetime.now()

def _parse_trade_windows(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int,int]] = []
    if not s:
        return out
    for chunk in str(s).split(","):
        ch = chunk.strip()
        if not ch or "-" not in ch:
            continue
        a, b = ch.split("-", 1)
        try:
            h1, m1 = [int(x) for x in a.strip().split(":")]
            h2, m2 = [int(x) for x in b.strip().split(":")]
            t1 = h1*60 + m1
            t2 = h2*60 + m2
            t1 = max(0, min(1439, t1))
            t2 = max(0, min(1440, t2))
            out.append((t1, t2))
        except Exception:
            continue
    return out

def _in_any_window(mins: int, windows: List[Tuple[int,int]]) -> bool:
    for s,e in windows:
        if s <= e:
            if s <= mins < e:
                return True
        else:
            if mins >= s or mins < e:
                return True
    return False

def _next_window_start(now_local: datetime.datetime, windows: List[Tuple[int,int]],
                       exclude_weekends: bool) -> Optional[datetime.datetime]:
    if not windows:
        return None
    cur_mins = now_local.hour*60 + now_local.minute
    candidates: List[datetime.datetime] = []
    wd = now_local.weekday()
    is_weekend = wd >= 5
    def _mk(day_dt: datetime.datetime, mins: int) -> datetime.datetime:
        h, m = divmod(mins, 60)
        return day_dt.replace(hour=h, minute=m, second=0, microsecond=0)
    if not (exclude_weekends and is_weekend):
        for s, e in windows:
            if s <= e:
                if cur_mins < s:
                    candidates.append(_mk(now_local, s))
            else:
                if not (cur_mins >= s):
                    candidates.append(_mk(now_local, s))
    if candidates:
        return min(candidates)
    for d in range(1, 8):
        day = now_local + timedelta(days=d)
        if exclude_weekends and day.weekday() >= 5:
            continue
        starts = sorted(s for s,_ in windows)
        if not starts:
            continue
        return _mk(day, starts[0])
    return None

def schedule_status() -> Tuple[bool, str, Optional[float]]:
    exclude_weekends = bool(int(os.getenv("EXCLUDE_WEEKENDS", str(getattr(config, "EXCLUDE_WEEKENDS", 1)))))
    windows_str = os.getenv("TRADE_HOURS_LOCAL", str(getattr(config, "TRADE_HOURS_LOCAL", "22:00-08:00"))).strip()
    windows = _parse_trade_windows(windows_str)
    now_loc = _now_local()
    wd = now_loc.weekday()
    is_weekend = (wd >= 5)
    cur_mins = now_loc.hour*60 + now_loc.minute

    if exclude_weekends and is_weekend:
        nxt = _next_window_start(now_loc, windows, exclude_weekends=True)
        return False, "weekend", (nxt.timestamp() if nxt else None)

    if windows and not _in_any_window(cur_mins, windows):
        nxt = _next_window_start(now_loc, windows, exclude_weekends)
        return False, "off_hours", (nxt.timestamp() if nxt else None)

    return True, "", None

def _control_path() -> str:
    log_path = os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))
    folder = os.path.abspath(os.path.dirname(log_path) or ".")
    return os.path.join(folder, "control.json")

def register_trade_result(pnl: float):
    global loss_streak, last_loss_time
    if pnl is None:
        return
    if pnl < 0:
        loss_streak += 1
        last_loss_time = time.time()
    elif pnl > 0:
        loss_streak = 0

# глобальные флаги расписания (читаются в can_enter_now)
SCHEDULE_ALLOWED = True
SCHEDULE_REASON = ""

def can_enter_now() -> Tuple[bool, str]:
    global PAUSE_ENTRIES, loss_streak, last_loss_time, SCHEDULE_ALLOWED, SCHEDULE_REASON
    if not SCHEDULE_ALLOWED:
        return False, f"schedule:{SCHEDULE_REASON or 'off'}"
    if PAUSE_ENTRIES:
        return False, "pause_entries"
    if loss_streak >= LOSS_STREAK_MAX and (time.time() - last_loss_time) < LOSS_COOLDOWN:
        return False, "loss_cooldown"
    return True, ""

def persist_candles_if_needed(symbol: str, kl: List[List[Any]], last_persist: Dict[str, float]):
    if not RECORD_MARKET_DATA or not DATA_ROOT:
        return
    now = time.time()
    if now - last_persist.get(symbol, 0.0) < PERSIST_EVERY_SEC:
        return
    try:
        if not kl or not _row_has_ts(kl[0]):
            return
        os.makedirs(os.path.join(DATA_ROOT, "candles"), exist_ok=True)
        path = os.path.join(DATA_ROOT, "candles", f"{symbol}.csv")
        rows = []
        for x in kl:
            if not _row_has_ts(x):
                continue
            ts = int(x[0]); o=float(x[1]); h=float(x[2]); l=float(x[3]); c=float(x[4]); v=float(x[5])
            rows.append([ts,o,h,l,c,v])
        if not rows:
            return
        df_new = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts").sort_values("ts")
        if os.path.exists(path):
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates("ts").sort_values("ts")
        else:
            df = df_new
        df.to_csv(path, index=False)
        last_persist[symbol] = now
    except Exception as e:
        log(f"[DATA] {symbol}: {e}")

def score_symbol(symbol: str) -> Optional[float]:
    try:
        kl_raw, _ = broker.get_kline_any(symbol, interval="1", limit=120)
        if not kl_raw or len(kl_raw) < 40:
            return None

        closes = pd.Series(kl_closes(kl_raw), dtype=float)
        if len(closes) < 40:
            return None

        snap = broker.get_ticker_snapshot(symbol)
        last_price = float(snap.get("last_price", 0.0)) or float(closes.iloc[-1])

        ohlcv = kl_to_ohlcv(kl_raw)
        atr = _atr_abs(ohlcv)
        atr_pct = (atr / max(last_price, 1e-9)) * 100.0

        mom30  = (closes.iloc[-1] / (closes.iloc[-31] if len(closes) > 31 else closes.iloc[0]) - 1.0) * 100.0
        mom120 = (closes.iloc[-1] / (closes.iloc[-121] if len(closes) > 121 else closes.iloc[0]) - 1.0) * 100.0
        diffs = closes.diff().dropna()
        same_dir = (diffs[-30:] > 0).mean()
        smooth = abs((same_dir - 0.5) * 2)

        vol24 = float(snap.get("vol_24h", 0.0))
        liq = math.log10(max(vol24, 1.0))
        score = (atr_pct * 0.35) + (liq * 0.25) + (mom30 * 0.2) + (mom120 * 0.1) + (smooth * 0.1)
        if vol24 < 1e6 or atr_pct < 0.05:
            score -= 5.0
        return score
    except Exception as e:
        log(f"[SCORE] {symbol}: {e}")
        return None

def _safe_order_filters(symbol: str) -> Tuple[float, float, float]:
    try:
        min_qty, step, min_notional = broker.get_min_order_filters(symbol)
    except Exception as e:
        log(f"[FILTERS] {symbol}: ошибка чтения: {e}")
        min_qty, step, min_notional = None, None, None

    if min_qty is None or min_qty <= 0:
        min_qty = float(getattr(config, "DEFAULT_MIN_QTY_FALLBACK", 0.001))
    if step is None or step <= 0:
        step = float(getattr(config, "DEFAULT_QTY_STEP_FALLBACK", 0.001))
    if min_notional is None or min_notional <= 0:
        min_notional = float(getattr(config, "DEFAULT_MIN_NOTIONAL_FALLBACK", 5.0))
    return float(min_qty), float(step), float(min_notional)

def _align_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

def _filter_by_linear_availability(universe: List[str]) -> List[str]:
    try:
        tl = broker.get_tickers_linear()
        lst = ((tl or {}).get("result") or {}).get("list") or []
        linear_syms = {str(x.get("symbol")) for x in lst if x.get("symbol")}
        out = []
        for s in universe:
            if s in linear_syms:
                out.append(s)
            else:
                log(f"[FILTER] drop {s}: нет linear USDT-перпа на этой среде")
        return out or universe
    except Exception as e:
        log(f"[FILTER] linear availability check failed: {e}")
        return universe

def _filter_universe_by_notional(universe: List[str], balance: float) -> List[str]:
    right_cap = max(0.0, balance * PAIR_FILTER_HCAP_FRAC)
    out = []
    for s in universe:
        try:
            price = float(broker.get_current_price(s)) or 0.0
            min_qty, step, _ = _safe_order_filters(s)
            min_qty_aligned = _align_qty(min_qty, step)
            min_notional = price * min_qty_aligned

            if min_notional < PAIR_FILTER_MIN_NOTIONAL:
                log(f"[FILTER] drop {s}: min_notional={min_notional:.2f} < left={PAIR_FILTER_MIN_NOTIONAL:.2f}")
                continue

            try:
                lev_guess = int(getattr(config, "DEFAULT_LEVERAGE", 10))
                if hasattr(broker, "get_symbol_leverage_limits"):
                    lo_hi = broker.get_symbol_leverage_limits(s)
                    if lo_hi and isinstance(lo_hi, (list, tuple)) and len(lo_hi) >= 2:
                        lev_guess = int(lo_hi[1] or lev_guess)
            except Exception:
                pass
            lev_guess = max(1, lev_guess)

            req_margin = min_notional / lev_guess
            if req_margin > right_cap:
                log(f"[FILTER] drop {s}: req_margin={req_margin:.2f} > cap={right_cap:.2f} (lev≈{lev_guess}x)")
                continue

            out.append(s)

        except Exception as e:
            emsg = str(e)
            if "10001" in emsg or "symbol invalid" in emsg.lower():
                log(f"[FILTER] skip {s}: not linear / invalid on this endpoint")
                continue
            log(f"[FILTER] {s}: {e}")
    return out

# =========================
# Adaptive leverage helpers
# =========================
def _parse_tier_rule(rule: str) -> List[Tuple[float, int]]:
    out = []
    for chunk in str(rule or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        lvl, lev = chunk.split(":", 1)
        try:
            out.append((float(lvl), int(float(lev))))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out or [(0.0, int(getattr(config, "DEFAULT_LEVERAGE", 10)))]

def _tier_leverage_for_balance(balance: float, rule: str) -> int:
    tbl = _parse_tier_rule(rule)
    chosen = tbl[0][1]
    for bal, lev in tbl:
        if balance >= bal:
            chosen = lev
        else:
            break
    return int(max(1, chosen))

def _broker_max_leverage(symbol: str, fallback: int = None) -> int:
    fb = int(fallback if fallback is not None else getattr(config, "ADAPTIVE_LEV_MAX", 100))
    try:
        if hasattr(broker, "get_max_leverage"):
            mx = broker.get_max_leverage(symbol)
            if mx and mx > 0:
                return int(mx)
    except Exception:
        pass
    return fb

def _compute_adaptive_leverage(symbol: str,
                               balance_usdt: float,
                               price: float,
                               atr_abs: float,
                               spread_rel: float = 0.0,
                               min_qty: float = 0.0,
                               qty_step: float = 0.0,
                               min_notional: float = 0.0) -> int:
    base_lev = _tier_leverage_for_balance(
        balance_usdt, str(getattr(config, "ADAPTIVE_LEV_TIER_RULE", "0:85,25:75,50:60,100:45,250:30,500:20"))
    )
    lev_min = int(getattr(config, "ADAPTIVE_LEV_MIN", 5))
    lev_max_cfg = int(getattr(config, "ADAPTIVE_LEV_MAX", 100))
    lev_max_exch = _broker_max_leverage(symbol, lev_max_cfg)
    lev_cap = max(1, min(lev_max_cfg, lev_max_exch))

    atr_ref = float(getattr(config, "ADAPTIVE_LEV_ATR_REF_PCT", 0.001))  # 0.1%
    atr_pct = (atr_abs / max(price, 1e-9)) if price > 0 else 0.0
    atr_k = 1.0
    if atr_ref > 0:
        atr_k = max(0.6, min(1.4, atr_ref / max(atr_pct, 1e-9)))

    spread_penalty_mult = float(getattr(config, "ADAPTIVE_LEV_SPREAD_PENALTY", 1.5))
    spread_max = float(getattr(config, "SPREAD_MAX_PCT", 0.0008))
    spr_k = 0.8 if (spread_rel > spread_max * spread_penalty_mult) else 1.0

    lev = int(round(base_lev * atr_k * spr_k))
    lev = max(lev_min, min(lev_cap, lev))

    if int(getattr(config, "ADAPTIVE_LEV_REQUIRE_AFFORDABLE", 1)):
        try:
            from utils import affordable_min_order
            min_not_usdt = float(getattr(config, "MIN_NOTIONAL_USDT", 5.0))
            target_min_notional = max(float(min_notional or 0.0), min_not_usdt)
            mq = _align_qty(float(min_qty or 0.0), float(qty_step or 0.0))
            if mq > 0 and price > 0:
                target_min_notional = max(target_min_notional, price * mq)

            while lev < lev_cap:
                chk = affordable_min_order(
                    price=price,
                    min_qty=max(mq, 1e-12),
                    min_notional_usdt=target_min_notional,
                    balance_usdt=float(balance_usdt or 0.0),
                    max_balance_share=float(getattr(config, "MAX_BALANCE_SHARE", 0.08)),
                    hard_cap_share=float(getattr(config, "HARD_CAP_SHARE", 0.25)),
                    leverage=lev,
                    taker_fee=float(getattr(config, "TAKER_FEE", 0.0006)),
                )
                if chk.get("ok"):
                    break
                lev += 1
            lev = min(lev, lev_cap)
        except Exception as e:
            log(f"[ADAPT-LEV] affordable_min_order check failed for {symbol}: {e}")

    return int(max(lev_min, min(lev, lev_cap)))

def select_top_pairs(base_list, count=2):
    scored = []
    for s in base_list:
        sc = score_symbol(s)
        if sc is not None:
            scored.append((s, sc))
    if not scored:
        from bybit_api import fast_pick_top_pairs
        return fast_pick_top_pairs(count=count)
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in scored[:count]]
    log(f"[PAIR-SELECT] Рейтинг: {scored[:min(len(scored), 8)]}")
    return top

# =========================
# Основной цикл
# =========================
def main_trading_cycle():
    cfg = config
    last_cfg_reload = time.time()
    CFG_RELOAD_SEC = 900

    max_dd_frac = float(getattr(cfg, "MAX_DRAWDOWN_PCT", 8.0)) / 100.0
    last_dd_check = 0.0

    # стартовый баланс
    try:
        start_balance = float(broker.get_balance())
    except Exception as e:
        log(f"[BAL] ошибка чтения баланса: {e}")
        start_balance = 0.0

    # (2) решаем, сколько пар вести от текущего баланса
    auto_pairs_n = _pairs_count_for_balance(start_balance, AUTO_PAIRS_RULE)

    # стартовая вселенная из конфига → фильтр по linear → фильтр по нотационалу
    base_universe = getattr(cfg, "TOP_LIQUID_PAIRS", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    base_universe = _filter_by_linear_availability(base_universe)
    base_universe = _filter_universe_by_notional(base_universe, start_balance)

    # скоринг и выборка top-N
    top_pairs = select_top_pairs(base_universe, count=auto_pairs_n)
    log(f"[PAIRS] Работаем с: {top_pairs}")

    mode_label = "PAPER" if PAPER_MODE else "REAL"
    tg_send(f"🟢 Старт [{mode_label}] Пары: {top_pairs}\nБаланс: {start_balance:.2f} USDT\nSAFE_MODE={int(SAFE_MODE)}")

    # состояния по символам
    entry: Dict[str, Dict[str, Any]] = {s: {"price": None, "side": None, "qty": None, "max_upnl": None} for s in top_pairs}
    last_entry_time = {s: 0.0 for s in top_pairs}

    # кэши
    last_kl: Dict[str, Tuple[float, List[List[Any]]]] = {}
    last_snap: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    last_persist: Dict[str, float] = {}

    # === адаптивное плечо ===
    last_lev_set: Dict[str, int] = {}
    last_lev_check_ts: Dict[str, float] = {}

    # первичная установка плеча
    if PAPER_MODE or not SAFE_MODE:
        for s in top_pairs:
            try:
                kl_raw, _ = broker.get_kline_any(s, interval="1", limit=60)
                ohlcv0 = kl_to_ohlcv(kl_raw)
                px = float(broker.get_current_price(s)) or (float(ohlcv0[-1][3]) if ohlcv0 else 0.0)
                atr0 = _atr_abs(ohlcv0) if ohlcv0 else 0.0
                try:
                    spr = float(broker.get_orderbook_spread(s, depth=int(getattr(cfg, "SPREAD_DEPTH", 1))) or 0.0)
                except Exception:
                    spr = 0.0
                mq, stp, mnot = _safe_order_filters(s)

                if int(getattr(cfg, "ADAPTIVE_LEV_ENABLED", 1)):
                    lev = _compute_adaptive_leverage(s, float(broker.get_balance()), px, atr0, spr, mq, stp, mnot)
                else:
                    lev = int(getattr(cfg, "DEFAULT_LEVERAGE", 10))

                try:
                    ok = broker.set_leverage(s, int(lev))
                except Exception as e:
                    log(f"[LEV] {s}: set_leverage exception {e}")
                else:
                    if ok is True:
                        last_lev_set[s] = int(lev)
                        last_lev_check_ts[s] = time.time()
                        log(f"[LEV] {s}: {int(lev)}x")
                    elif ok is False:
                        log(f"[ℹ️] set_leverage {s}: leverage not modified or rejected")
                    elif ok is None:
                        log(f"[LEV] {s}: set_leverage returned None (unexpected)")
                    else:
                        log(f"[LEV] {s}: set_leverage returned non-bool {ok!r}")
            except Exception as e:
                log(f"[LEV] {s}: {e}")

    # грузим модель
    model, meta = load_model_and_meta()
    log("[ML] Модель: rf_model.pkl; мета: model_meta.json — загружены.")

    DO_TRADE = PAPER_MODE or (not SAFE_MODE)

    session_on = True
    session_started_at = time.time()
    break_started_at = None

    # риск-параметры
    MAX_BALANCE_SHARE   = float(getattr(cfg, "MAX_BALANCE_SHARE", 0.08))
    USEABLE_BAL_SHARE   = float(getattr(cfg, "USEABLE_BAL_SHARE", 0.95))
    HARD_CAP_SHARE      = float(getattr(cfg, "HARD_CAP_SHARE", 0.25))

    SPREAD_MAX_PCT      = float(getattr(cfg, "SPREAD_MAX_PCT", 0.0008))
    SPREAD_DEPTH        = int(getattr(cfg, "SPREAD_DEPTH", 1))
    RISK_PER_TRADE_FRAC = float(getattr(cfg, "RISK_PER_TRADE_FRAC", 0.005))
    ATR_STOP_K          = float(getattr(cfg, "ATR_STOP_K", 3.5))

    # --- управление через control.json ---
    control_path = _control_path()
    control_last_ts = ""
    last_ctrl_check = 0.0
    # --- расписание / уведомления ---
    force_on_schedule = False
    last_sched_state: Optional[bool] = None
    SCHED_MANAGE_OPEN = bool(int(os.getenv("SCHEDULE_MANAGE_OPEN_POSITIONS", str(getattr(cfg, "SCHEDULE_MANAGE_OPEN_POSITIONS",1)))))

    def _read_control() -> List[Dict[str, Any]]:
        try:
            if not os.path.exists(control_path):
                return []
            with open(control_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _process_control(cmds: List[Dict[str, Any]]):
        nonlocal top_pairs, entry, last_entry_time, last_lev_set, last_lev_check_ts, control_last_ts, force_on_schedule
        global PAUSE_ENTRIES
        for c in cmds:
            ts = str(c.get("ts") or "")
            if control_last_ts and ts <= control_last_ts:
                continue

            if "pause_entries" in c:
                PAUSE_ENTRIES = bool(c.get("pause_entries"))
                log(f"[CTRL] entries {'paused' if PAUSE_ENTRIES else 'resumed'}")

            if c.get("panic_close"):
                log("[CTRL] panic_close received")
                tg_send("🛑 CTRL: panic_close")
                try:
                    if DO_TRADE:
                        broker.force_close_all_positions_absolute()
                except Exception as e:
                    log(f"[CTRL] panic_close error: {e}")
                raise KeyboardInterrupt()

            if "set_pairs" in c:
                req_pairs = [str(s) for s in (c.get("set_pairs") or []) if s]
                if req_pairs:
                    try:
                        base = _filter_by_linear_availability(req_pairs)
                        cur_bal = float(broker.get_balance())
                        base = _filter_universe_by_notional(base, cur_bal)
                        if base:
                            top_pairs = select_top_pairs(base, count=len(base))
                            log(f"[PAIRS] Обновлено через control.json: {top_pairs}")
                            for s in top_pairs:
                                entry.setdefault(s, {"price": None, "side": None, "qty": None, "max_upnl": None})
                                last_entry_time.setdefault(s, 0.0)
                    except Exception as e:
                        log(f"[CTRL/set_pairs] {e}")

            if "default_lev" in c:
                lev = 0
                try:
                    lev = int(c.get("default_lev") or 0)
                except Exception:
                    lev = 0
                if lev > 0:
                    log(f"[CTRL] default leverage request: {lev}x")
                    if int(getattr(config, "ADAPTIVE_LEV_ENABLED", 1)) == 0 and (PAPER_MODE or not SAFE_MODE):
                        for s in top_pairs:
                            try:
                                ok = broker.set_leverage(s, int(lev))
                            except Exception as e:
                                log(f"[LEV] {s}: set_leverage exception {e}")
                            else:
                                if ok is True:
                                    last_lev_set[s] = int(lev)
                                    last_lev_check_ts[s] = time.time()
                                    log(f"[LEV] {s}: → {int(lev)}x (from CTRL)")
                                elif ok is False:
                                    log(f"[ℹ️] set_leverage {s}: leverage not modified or rejected")
                                elif ok is None:
                                    log(f"[LEV] {s}: set_leverage returned None (unexpected)")
                                else:
                                    log(f"[LEV] {s}: set_leverage returned non-bool {ok!r}")

            if "force_on" in c:
                force_on_schedule = bool(c.get("force_on"))
                log(f"[CTRL] force_on_schedule={int(force_on_schedule)}")

            if ts:
                control_last_ts = ts

    try:
        while True:
            now = time.time()

            # --- опрос control.json ---
            if now - last_ctrl_check > CONTROL_POLL_SEC:
                try:
                    _process_control(_read_control())
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log(f"[CTRL] {e}")
                finally:
                    last_ctrl_check = now

            # --- расписание торговли (глобальный гейт на входы) ---
            allowed, reason, next_ts = schedule_status()
            if force_on_schedule:
                allowed = True
                reason = ""
                next_ts = None

            global SCHEDULE_ALLOWED, SCHEDULE_REASON
            SCHEDULE_ALLOWED = bool(allowed)
            SCHEDULE_REASON = str(reason or "")

            if last_sched_state is None or allowed != last_sched_state:
                last_sched_state = allowed
                if allowed:
                    tg_send("▶️ Торговое окно открыто.")
                    log("[SCHEDULE] window=ON")
                else:
                    if next_ts:
                        try:
                            tz = _get_tz()
                            dt_local = datetime.datetime.fromtimestamp(next_ts, tz) if tz else datetime.datetime.fromtimestamp(next_ts)
                            nxt_str = dt_local.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            nxt_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(next_ts))
                        tg_send(f"⏸ Вне торгового окна ({reason}). Следующая активация: {nxt_str}.")
                        log(f"[SCHEDULE] window=OFF reason={reason} next={nxt_str}")
                    else:
                        tg_send("⏸ Вне торгового окна.")
                        log(f"[SCHEDULE] window=OFF reason={reason}")

            if not allowed and not any(broker.has_open_position(s) for s in top_pairs):
                if next_ts:
                    sleep_for = max(0.5, min(60.0, next_ts - time.time()))
                    time.sleep(sleep_for)

            # hot-reload
            if now - last_cfg_reload > CFG_RELOAD_SEC:
                importlib.reload(config)
                cfg = config
                model, meta = load_model_and_meta()
                last_cfg_reload = now
                log("[HOT-RELOAD] config и модель обновлены.")

            # глобальная проверка просадки
            if now - last_dd_check > DD_CHECK_EVERY_SEC:
                try:
                    cur_bal = float(broker.get_balance())
                    if start_balance > 0:
                        dd = (start_balance - cur_bal) / start_balance
                        if dd >= max_dd_frac:
                            msg = f"⛔️ Макс. просадка {dd*100:.2f}% (порог {max_dd_frac*100:.1f}%). Останавливаю бота."
                            log(msg); tg_send(msg)
                            raise KeyboardInterrupt()
                except Exception as e:
                    log(f"[DDCHK] {e}")
                finally:
                    last_dd_check = now

            # переключение работа/перерыв
            if session_on:
                if now - session_started_at >= WORK_DURATION_SEC:
                    session_on = False
                    break_started_at = now
                    tg_send(f"⏸ Перерыв на {BREAK_DURATION_SEC//60} мин. Режим: [{mode_label}]")
                    try:
                        cur_bal = float(broker.get_balance())
                        auto_pairs_n = _pairs_count_for_balance(cur_bal, AUTO_PAIRS_RULE)
                        base_universe = getattr(cfg, "TOP_LIQUID_PAIRS", base_universe)
                        base_universe = _filter_by_linear_availability(base_universe)
                        base_universe = _filter_universe_by_notional(base_universe, cur_bal)
                        top_pairs = select_top_pairs(base_universe, count=auto_pairs_n)
                        log(f"[PAIRS] Новый подбор на перерыве: {top_pairs}")
                        for s in top_pairs:
                            entry.setdefault(s, {"price": None, "side": None, "qty": None, "max_upnl": None})
                            last_entry_time.setdefault(s, 0.0)
                    except Exception as e:
                        log(f"[BREAK/SELECT] {e}")
                    time.sleep(1.0)
                    continue
            else:
                if now - break_started_at >= BREAK_DURATION_SEC:
                    session_on = True
                    session_started_at = now
                    tg_send(f"▶️ Новая сессия на {WORK_DURATION_SEC//60} мин. Пары: {top_pairs}. Режим: [{mode_label}]")
                time.sleep(1.0)
                continue

            # === рабочая сессия ===
            for symbol in list(top_pairs):
                # kline
                ts_kl, kl_cached = last_kl.get(symbol, (0.0, None))
                if now - ts_kl > KLINE_REFRESH_SEC or kl_cached is None:
                    kl_cached, _src = broker.get_kline_any(symbol, interval="1", limit=60)
                    last_kl[symbol] = (now, kl_cached)
                if not kl_cached or len(kl_cached) < 10:
                    continue

                # запись свечей
                persist_candles_if_needed(symbol, kl_cached, last_persist)

                # snapshot
                ts_sn, snap_cached = last_snap.get(symbol, (0.0, None))
                if now - ts_sn > SNAPSHOT_REFRESH_SEC or snap_cached is None:
                    snap_cached = broker.get_ticker_snapshot(symbol)
                    last_snap[symbol] = (now, snap_cached)

                # нормализуем свечи
                ohlcv = kl_to_ohlcv(kl_cached)
                if len(ohlcv) < 10:
                    continue

                price = float(snap_cached.get("last_price", 0.0) or 0.0)
                if price <= 0 and ohlcv:
                    price = float(ohlcv[-1][3])

                atr_val = _atr_abs(ohlcv)

                # Спред-гейт
                try:
                    spread_rel = float(broker.get_orderbook_spread(symbol, depth=SPREAD_DEPTH) or 0.0)
                except Exception:
                    spread_rel = 0.0

                # --- адаптивный пересчёт плеча периодически ---
                if int(getattr(cfg, "ADAPTIVE_LEV_ENABLED", 1)) and (PAPER_MODE or not SAFE_MODE):
                    ts_last = last_lev_check_ts.get(symbol, 0.0)
                    if (time.time() - ts_last) > int(getattr(cfg, "ADAPTIVE_LEV_REEVAL_SEC", 300)):
                        mq, stp, mnot = _safe_order_filters(symbol)
                        lev_new = _compute_adaptive_leverage(symbol, float(broker.get_balance()), price, atr_val, spread_rel, mq, stp, mnot)
                        lev_prev = last_lev_set.get(symbol)
                        if (lev_prev is None) or (abs(int(lev_prev) - int(lev_new)) >= 5):
                            try:
                                ok = broker.set_leverage(symbol, int(lev_new))
                            except Exception as e:
                                log(f"[LEV] {symbol}: set_leverage exception {e}")
                            else:
                                if ok is True:
                                    last_lev_set[symbol] = int(lev_new)
                                    last_lev_check_ts[symbol] = time.time()
                                    log(f"[LEV] {symbol}: {lev_prev or '-'} → {int(lev_new)}x (atr%={(atr_val/price) if price>0 else 0:.5f}, spread={spread_rel:.5f})")
                                elif ok is False:
                                    log(f"[ℹ️] set_leverage {symbol}: leverage not modified or rejected")
                                elif ok is None:
                                    log(f"[LEV] {symbol}: set_leverage returned None (unexpected)")
                                else:
                                    log(f"[LEV] {symbol}: set_leverage returned non-bool {ok!r}")
                        else:
                            last_lev_check_ts[symbol] = time.time()

                if spread_rel > SPREAD_MAX_PCT:
                    log(f"[SKIP] {symbol}: wide_spread {spread_rel:.5f} > {SPREAD_MAX_PCT:.5f}")
                    continue

                # вне окна — ведём позиции, но НОВЫЕ входы запрещаем
                if not SCHEDULE_ALLOWED and not broker.has_open_position(symbol):
                    continue

                # Управление открытой позицией (динамический выход как было)
                if broker.has_open_position(symbol):
                    ent = entry.get(symbol) or {"price": None, "side": None, "qty": None, "max_upnl": None}
                    if ent["price"] and ent["side"] and ent["qty"]:
                        current = price
                        qty = float(ent["qty"])
                        entry_price = float(ent["price"])
                        side = ent["side"]

                        commission_rate = float(getattr(cfg, "COMMISSION_PER_SIDE", 0.0005))
                        commission = entry_price * qty * commission_rate * 2.0
                        profit = (current - entry_price) if side == "Buy" else (entry_price - current)
                        pnl = profit * qty
                        ent["max_upnl"] = pnl if ent.get("max_upnl") is None else max(ent["max_upnl"], pnl)

                        TRAIL_DROP = entry_price * float(getattr(cfg, "TRAIL_DROP_PCT", 0.004))

                        # Trailing выход
                        if (ent["max_upnl"] is not None) and (ent["max_upnl"] - pnl > TRAIL_DROP) and (ent["max_upnl"] > commission):
                            if DO_TRADE:
                                broker.close_position_by_market(symbol, qty)
                            else:
                                write_cycle_log({
                                    "symbol": symbol,
                                    "direction": "long" if side == "Buy" else "short",
                                    "buy_price": entry_price if side == "Buy" else None,
                                    "sell_price": current,
                                    "qty": qty,
                                    "pnl": pnl - commission,
                                    "event": "dynamic_tp_exit",
                                    "closed_at": utcnow_iso(),
                                    "reason": "dynamic_tp",
                                    "dry": True
                                })
                            register_trade_result(pnl - commission)
                            entry[symbol] = {"price": None, "side": None, "qty": None, "max_upnl": None}
                            continue

                        # Если уже был профит > комиссий, а вернулись ниже комиссий — выходим
                        if pnl < commission and (ent["max_upnl"] is not None) and (ent["max_upnl"] > commission):
                            if DO_TRADE:
                                broker.close_position_by_market(symbol, qty)
                            else:
                                write_cycle_log({
                                    "symbol": symbol,
                                    "direction": "long" if side == "Buy" else "short",
                                    "buy_price": entry_price if side == "Buy" else None,
                                    "sell_price": current,
                                    "qty": qty,
                                    "pnl": pnl - commission,
                                    "event": "no_profit_exit",
                                    "closed_at": utcnow_iso(),
                                    "reason": "no_profit",
                                    "dry": True
                                })
                            register_trade_result(pnl - commission)
                            entry[symbol] = {"price": None, "side": None, "qty": None, "max_upnl": None}
                            continue

                # ===== РЕШЕНИЕ О ВХОДЕ (через роутер стратегий) =====
                ok_now, reason = can_enter_now()
                if not ok_now:
                    log(f"[SKIP] {symbol}: {reason}")
                    continue
                if now - float(last_entry_time.get(symbol, 0.0)) < ENTRY_COOLDOWN_SEC:
                    continue

                # Достаточная волатильность
                min_atr_pct = float(getattr(cfg, "MIN_ATR_PCT", 0.0015))
                if atr_val < (min_atr_pct * max(price, 1e-9)):
                    log(f"[SKIP] {symbol}: low_atr atr={atr_val:.6f} < {min_atr_pct*price:.6f}")
                    continue

                # Решение роутера
                router_ctx = {
                    # ML-вeto можно провести позже отдельно (как и раньше у тебя),
                    # но сюда можно передать prob, если уже есть. На этом этапе prob ещё нет.
                    "max_positions_reached": False,
                }
                router_res = decide_with_router(symbol, "1m", ohlcv, router_ctx)
                signal = router_res.get("action", "hold")
                router_reason = router_res.get("reason")
                router_meta = router_res.get("meta") or {}
                router_sl = router_res.get("sl")
                router_tp = router_res.get("tp")

                if signal not in ("buy", "sell"):
                    if getattr(config, "DEBUG_TRADING", False):
                        log(f"[ROUTER] {symbol}: hold ({router_reason})")
                    continue

                # Фильтры ордеров (с фоллбэками)
                min_qty, step, min_notional = _safe_order_filters(symbol)

                # Баланс/лимиты
                try:
                    avail = float(getattr(broker, "get_available_balance", broker.get_balance)())
                except Exception:
                    avail = float(broker.get_balance())
                if avail <= 0:
                    log(f"[SKIP] {symbol}: no_balance")
                    continue

                # ===== Размер позиции от риска/ATR =====
                RISK_PER_TRADE_FRAC = float(getattr(cfg, "RISK_PER_TRADE_FRAC", 0.005))
                ATR_STOP_K          = float(getattr(cfg, "ATR_STOP_K", 3.5))

                risk_cap = max(0.0, avail * max(0.0, min(1.0, RISK_PER_TRADE_FRAC)))
                stop_dist = max(atr_val * max(ATR_STOP_K, 0.1), 1e-9)
                qty_risk_atr = risk_cap / stop_dist

                max_notional_cap = avail * max(0.0, min(1.0, float(getattr(cfg, "MAX_BALANCE_SHARE", 0.08))))
                qty_cap_share = max_notional_cap / max(price, 1e-9)

                raw_qty = min(qty_risk_atr, qty_cap_share)
                qty = adjust_qty(price, raw_qty, min_qty=min_qty, qty_step=step, min_notional=min_notional)

                if qty <= 0:
                    min_qty_aligned = _align_qty(min_qty, step)
                    min_notional_usdt = float(getattr(cfg, "MIN_NOTIONAL_USDT", 5.0))
                    notional_min = price * min_qty_aligned
                    hard_cap_notional = avail * max(0.0, min(1.0, float(getattr(cfg, "HARD_CAP_SHARE", 0.25))))
                    if (notional_min >= max(min_notional, min_notional_usdt)) and (notional_min <= hard_cap_notional):
                        qty = min_qty_aligned
                        log(f"[ADJUST] {symbol}: поднял qty до min_qty={qty} (notional~{notional_min:.4f})")
                    else:
                        log(f"[SKIP] {symbol}: qty<=0 after adjust (raw={raw_qty}, min_qty={min_qty}, step={step})")
                        continue

                # Финальная проверка нотационала
                min_notional_usdt = float(getattr(cfg, "MIN_NOTIONAL_USDT", 5.0))
                notional = price * qty
                if notional < max(min_notional, min_notional_usdt):
                    log(f"[SKIP] {symbol}: notional too small ({notional:.4f} < {max(min_notional, min_notional_usdt):.4f})")
                    continue

                # Контроль доступного «рабочего» баланса
                useable_cap = avail * max(0.0, min(1.0, float(getattr(cfg, "USEABLE_BAL_SHARE", 0.95))))
                if notional > useable_cap:
                    log(f"[SKIP] {symbol}: notional {notional:.4f} > useable_cap {useable_cap:.4f}")
                    continue

                # ML-фильтр (как раньше): считаем proba уже зная направление
                direction = "long" if signal == "buy" else "short"
                ok_ml, proba, thr = predict_ok(
                    model, meta, symbol, direction, qty,
                    price=price, atr=atr_val, candles=ohlcv
                )
                if int(getattr(cfg, "ML_VETO_ENABLED", 1)):
                    veto_thr = float(getattr(cfg, "ML_VETO_THR", 0.35))
                    if proba < veto_thr:
                        if int(getattr(cfg, "ML_VETO_LOG", 1)):
                            log(f"[ML-VETO] {symbol}: veto (prob={proba:.3f} < veto_thr={veto_thr:.3f}); router={router_reason}")
                        continue
                if not ok_ml:
                    log(f"[ML] {symbol}: отказ (prob={proba:.3f} < thr={thr:.3f}); router={router_reason}")
                    continue

                side = "Buy" if direction == "long" else "Sell"

                # === ОТКРЫТИЕ СДЕЛКИ ===
                if DO_TRADE:
                    if broker.place_market_order(symbol, side, qty):
                        entry[symbol] = {"price": price, "side": side, "qty": qty, "max_upnl": None}
                        write_cycle_log({
                            "symbol": symbol,
                            "direction": direction,
                            "buy_price": price if side == "Buy" else None,
                            "sell_price": price if side == "Sell" else None,
                            "qty": qty,
                            "event": "open",
                            "opened_at": utcnow_iso(),
                            "proba": round(proba, 4),
                            "thr": round(thr, 4),
                            "paper": bool(PAPER_MODE),
                            # ⬇️ добавлено: причина и выбор роутера
                            "router_reason": router_reason,
                            "router_strategy": (router_meta or {}).get("strategy"),
                            "router_regime": (router_meta or {}).get("regime"),
                            "router_sl": router_sl,
                            "router_tp": router_tp,
                        })
                        last_entry_time[symbol] = now
                    else:
                        log(f"[FAIL] {symbol}: place_market_order вернул False (router={router_reason})")
                else:
                    log(f"[DRY-OPEN] {symbol} {side} qty={qty} @~{price:.6f} (prob={proba:.3f} thr={thr:.3f}) [{router_reason}]")
                    write_cycle_log({
                        "symbol": symbol,
                        "direction": direction,
                        "buy_price": price if side == "Buy" else None,
                        "sell_price": price if side == "Sell" else None,
                        "qty": qty,
                        "event": "dry_open",
                        "opened_at": utcnow_iso(),
                        "proba": round(proba, 4),
                        "thr": round(thr, 4),
                        "dry": True,
                        # ⬇️ ровно то, на чём ты сказал я «остановился»
                        "router_reason": router_reason,
                        "router_strategy": (router_meta or {}).get("strategy"),
                        "router_regime": (router_meta or {}).get("regime"),
                        "router_sl": router_sl,
                        "router_tp": router_tp,
                    })

            time.sleep(0.25)

    except KeyboardInterrupt:
        log("[🛑] Ctrl+C — закрываю все позиции…")
        tg_send("🛑 Ручная остановка. Закрываю все позиции.")
        try:
            if DO_TRADE:
                broker.force_close_all_positions_absolute()
        except Exception as e:
            log(f"[❌] force close: {e}")
    except Exception as e:
        log(f"[FATAL] main: {e}")
        tg_send(f"❌ Критическая ошибка: {e}")
        raise
    finally:
        # ===== Финальная сводка: баланс, PnL по сделкам (+/-), equity-график =====
        try:
            end_balance = float(broker.get_balance())
        except Exception as e:
            log(f"[BAL-END] {e}")
            end_balance = 0.0
        delta = round(end_balance - start_balance, 2)

        total_p = 0.0
        total_n = 0.0
        cnt_p = 0
        cnt_n = 0
        equity_points: List[Tuple[str, float]] = []

        try:
            log_path = os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))
            if os.path.exists(log_path):
                closure_events = {
                    "paper_close",
                    "close",
                    "dynamic_tp_exit",
                    "no_profit_exit",
                    "manual_close",
                    "stop_exit",
                    "take_profit_exit",
                    "stop_loss",
                    "take_profit",
                }
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if '"pnl"' not in line:
                            continue
                        js = json.loads(line)
                        event = js.get("event")
                        if event not in closure_events:
                            continue
                        pnl = float(js.get("pnl", 0.0))
                        if pnl >= 0:
                            total_p += pnl; cnt_p += 1
                        else:
                            total_n += pnl; cnt_n += 1
                        ts = js.get("closed_at") or js.get("timestamp") or js.get("ts_utc")
                        if ts:
                            equity_points.append((ts, pnl))
        except Exception as e:
            log(f"[LOG-READ] {e}")

        # График equity
        try:
            if equity_points:
                df = pd.DataFrame(equity_points, columns=["timestamp", "pnl"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                df["equity"] = df["pnl"].cumsum()
                plt.figure(figsize=(10, 5))
                plt.plot(df["timestamp"], df["equity"], marker="o")
                plt.grid(True)
                plt.title("Equity по времени")
                plt.tight_layout()
                plt.savefig("equity_plot.png")
                plt.close()
        except Exception as e:
            log(f"[GRAPH] {e}")

        # Автодообучение (опционально)
        try:
            if bool(getattr(config, "AUTO_RETRAIN_ON_EXIT", 0)):
                log("[AUTO-TRAIN] retrain_model_from_dataset.py …")
                subprocess.run(["python", "retrain_model_from_dataset.py"], check=True)
                log("[AUTO-TRAIN] Готово.")
        except Exception as e:
            log(f"[AUTO-TRAIN] {e}")

        msg = (
            f"🔴 Завершено [{mode_label}].\n"
            f"Δ Баланса: {delta:+.2f} USDT\n"
            f"Сделки: +{cnt_p} (сумма {total_p:+.2f}),  -{cnt_n} (сумма {total_n:+.2f})"
        )
        tg_send(msg)
        log(f"[STATS] Итог: {delta:+.2f} USDT; +{cnt_p} / -{cnt_n} ; sum+={total_p:+.2f} sum-={total_n:+.2f}")

if __name__ == "__main__":
    main_trading_cycle()
