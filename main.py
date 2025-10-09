# -*- coding: utf-8 -*-
# main.py — 2025-09-22: интеграция StrategyRouter (router_reason/strategy/regime в логах)

import os
import sys
import time
import json
import math
import shutil
import statistics
import datetime
from datetime import timedelta
from pathlib import Path
import importlib
import subprocess
import argparse
import signal
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import utils
# ⬇️ подключаем новый роутер стратегий
from strategy import decide_with_router, welford_mean_var
from utils import (
    log, tg_send, write_cycle_log, adjust_qty,
    spread_penalty, fee_aware_r_min, kelly_capped,
    SAFE_MODE as SAFE_MODE_FROM_ENV
)

DATA_ROOT = (getattr(config, "DATA_ROOT", None) or os.getenv("DATA_ROOT", "").strip() or "./data")
KLINE_HISTORY_LIMIT = int(os.getenv("KLINE_HISTORY_LIMIT", str(getattr(config, "KLINE_HISTORY_LIMIT", 300))))
LOG_RU = bool(int(os.getenv("LOG_RU", str(getattr(config, "LOG_RU", 1)))))
ROUTER_HEARTBEAT_SEC = int(os.getenv("ROUTER_HEARTBEAT_SEC", str(getattr(config, "ROUTER_HEARTBEAT_SEC", 60))))

_last_router: Dict[str, Tuple[str, str, float]] = {}
_last_min_qty: Dict[str, float] = {}


class _SessionTee:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data: str):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


SESSION_STATE: Dict[str, Any] = {
    "dir": None,
    "meta": {},
    "meta_path": None,
    "log_handle": None,
    "start_ts": None,
    "start_monotonic": None,
    "mode": None,
    "log_jsonl": None,
    "shutdown_requested": False,
    "stats_path": None,
}

_STDOUT_ORIG = sys.stdout
_STDERR_ORIG = sys.stderr


def utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _session_env_snapshot() -> Dict[str, str]:
    keys = {
        "PAPER_MODE",
        "SAFE_MODE",
        "LOG_JSONL",
        "LOG_RU",
        "KLINE_HISTORY_LIMIT",
        "ROUTER_HEARTBEAT_SEC",
        "AUTO_PAIRS_RULE",
    }
    snap = {}
    for k in sorted(keys):
        val = os.environ.get(k)
        if val is not None:
            snap[k] = val
    return snap


def _read_git_sha() -> Optional[str]:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        sha = res.decode("utf-8", errors="ignore").strip()
        return sha or None
    except Exception:
        return None


def _ensure_unique_session_dir(base: Path) -> Path:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    candidate = base / ts
    idx = 1
    while candidate.exists():
        candidate = base / f"{ts}_{idx:02d}"
        idx += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _session_bootstrap(mode_hint: str = "auto") -> None:
    if SESSION_STATE["dir"] is not None:
        return
    root = Path(DATA_ROOT or "./data").expanduser()
    sessions_root = root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    session_dir = _ensure_unique_session_dir(sessions_root)

    meta_path = session_dir / "meta.json"
    log_path = session_dir / "log.txt"
    stats_path = session_dir / "stats.json"

    log_handle = open(log_path, "a", encoding="utf-8")
    tee = _SessionTee(_STDOUT_ORIG, log_handle)
    sys.stdout = tee
    sys.stderr = tee

    start_iso = utcnow_iso()
    meta = {
        "start_ts": start_iso,
        "git_sha": _read_git_sha(),
        "mode": str(mode_hint or "auto"),
        "env": _session_env_snapshot(),
        "pairs": [],
        "kline_limit": KLINE_HISTORY_LIMIT,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    SESSION_STATE.update({
        "dir": session_dir,
        "meta": meta,
        "meta_path": meta_path,
        "log_handle": log_handle,
        "start_ts": start_iso,
        "start_monotonic": time.monotonic(),
        "mode": str(mode_hint or "auto"),
        "log_jsonl": os.path.abspath(os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))),
        "stats_path": stats_path,
    })


def _session_update_meta(update: Dict[str, Any]) -> None:
    if SESSION_STATE["dir"] is None:
        return
    meta = dict(SESSION_STATE.get("meta") or {})
    meta.update(update or {})
    SESSION_STATE["meta"] = meta
    meta_path = SESSION_STATE.get("meta_path")
    if meta_path:
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log(f"[SESSION] ошибка обновления meta.json: {exc}")


def _session_log_path(name: str) -> Optional[str]:
    if SESSION_STATE["dir"] is None:
        return None
    return str(Path(SESSION_STATE["dir"]) / name)


def _session_symlink_or_copy(target: str, link_name: str) -> None:
    if SESSION_STATE["dir"] is None:
        return
    if not target or not os.path.exists(target):
        return
    link_path = Path(SESSION_STATE["dir"]) / link_name
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        os.symlink(os.path.abspath(target), link_path)
        return
    except Exception:
        try:
            shutil.copy2(target, link_path)
        except Exception as exc:
            log(f"[SESSION] не удалось связать {target} → {link_path}: {exc}")


def _session_write_index(stats: Dict[str, Any]) -> None:
    if SESSION_STATE["dir"] is None:
        return
    index_path = Path(SESSION_STATE["dir"]).parent / "index.json"
    payload = {
        "path": str(SESSION_STATE["dir"]),
        "start_ts": SESSION_STATE.get("start_ts"),
        "end_ts": stats.get("end_ts"),
        "mode": stats.get("mode") or SESSION_STATE.get("mode"),
        "pnl": stats.get("pnl_total"),
        "trades": stats.get("trades"),
        "winrate": stats.get("winrate"),
        "duration_sec": stats.get("duration_sec"),
    }
    rows: List[Dict[str, Any]] = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows = [r for r in data if r.get("path") != payload["path"]]
        except Exception:
            rows = []
    rows.append(payload)
    rows.sort(key=lambda r: (r.get("start_ts") or ""), reverse=True)
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        log(f"[SESSION] ошибка записи index.json: {exc}")


def _session_finalize(stats: Dict[str, Any], trades_rows: List[Dict[str, Any]]) -> None:
    if SESSION_STATE["dir"] is None:
        return
    stats_path = SESSION_STATE.get("stats_path")
    if stats_path:
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log(f"[SESSION] не удалось записать stats.json: {exc}")
    _session_write_index(stats)
    log_jsonl_path = SESSION_STATE.get("log_jsonl")
    if log_jsonl_path:
        _session_symlink_or_copy(log_jsonl_path, "log.jsonl")
    if trades_rows:
        t_path = _session_log_path("trades.jsonl")
        if t_path:
            try:
                with open(t_path, "w", encoding="utf-8") as f:
                    for row in trades_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                log(f"[SESSION] не удалось записать trades.jsonl: {exc}")
    else:
        if log_jsonl_path:
            _session_symlink_or_copy(log_jsonl_path, "trades.jsonl")

    handle = SESSION_STATE.get("log_handle")
    if handle:
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass
    sys.stdout = _STDOUT_ORIG
    sys.stderr = _STDERR_ORIG


def msg(key: str, **kw) -> str:
    """Форматирование ключевых сообщений с поддержкой RU/EN."""

    RU = {
        "ROUTER_HOLD_VOLA":      "[ROUTER] {symbol}: удержание — волатильность вне диапазона (ATR%={atr:.4f})",
        "ROUTER_HOLD_EMPTY":     "[ROUTER] {symbol}: удержание — нет условий для входа",
        "ENTRY_SKIP_OPEN_POS":   "[PAPER] {symbol}: позиция уже открыта, пропускаю новый вход — ордер не отправлен",
        "ADJUST_MIN_QTY":        "[ADJUST] {symbol}: скорректировал размер до min_qty={min_qty} (нотационал≈{notional:.4f})",
        "FAIL_PLACE_ORDER":      "[FAIL] {symbol}: не удалось разместить ордер (router={router})",
        "PAIR_RATING":           "[PAIR-SELECT] Рейтинг: {rating}",
        "PAIRS_NEW":             "[PAIRS] Новый подбор на перерыве: {pairs}",
        "PAIRS_CURRENT":         "[PAIRS] Работаем с: {pairs}",
        "PAIRS_CTRL_UPDATE":     "[PAIRS] Обновлено через control.json: {pairs}",
    }
    EN = {
        "ROUTER_HOLD_VOLA":      "[ROUTER] {symbol}: hold — volatility out of range (ATR%={atr:.4f})",
        "ROUTER_HOLD_EMPTY":     "[ROUTER] {symbol}: hold — no candidates",
        "ENTRY_SKIP_OPEN_POS":   "[PAPER] {symbol}: position already open, skip new entry — order not sent",
        "ADJUST_MIN_QTY":        "[ADJUST] {symbol}: raised qty to min_qty={min_qty} (notional≈{notional:.4f})",
        "FAIL_PLACE_ORDER":      "[FAIL] {symbol}: place_market_order False (router={router})",
        "PAIR_RATING":           "[PAIR-SELECT] Rating: {rating}",
        "PAIRS_NEW":             "[PAIRS] New selection: {pairs}",
        "PAIRS_CURRENT":         "[PAIRS] Working with: {pairs}",
        "PAIRS_CTRL_UPDATE":     "[PAIRS] Updated from control.json: {pairs}",
    }
    templates = RU if LOG_RU else EN
    tpl = templates.get(key)
    if not tpl:
        return f"{key} {kw}" if kw else key
    try:
        return tpl.format(**kw)
    except Exception:
        return tpl


def log_router(symbol: str, key: str, *, reason: str = "", **kw) -> None:
    """Троттлинг повторяющихся сообщений роутера."""

    message = msg(key, symbol=symbol, **kw)
    if reason:
        message = f"{message} ({reason})"
    now = time.time()
    last = _last_router.get(symbol)
    state = (key, message)
    if (not last) or (state[0] != last[0]) or (state[1] != last[1]) or (now - last[2] >= ROUTER_HEARTBEAT_SEC):
        log(message)
        _last_router[symbol] = (state[0], state[1], now)


def log_adjust_if_changed(symbol: str, min_qty: float, notional: float) -> None:
    prev = _last_min_qty.get(symbol)
    value = float(min_qty)
    if prev is None or not math.isclose(prev, value, rel_tol=1e-9, abs_tol=1e-12):
        log(msg("ADJUST_MIN_QTY", symbol=symbol, min_qty=value, notional=float(notional)))
        _last_min_qty[symbol] = value

# --- управление из UI через control.json ---
CONTROL_POLL_SEC = float(getattr(config, "CONTROL_POLL_SEC", 1.5))
PAUSE_ENTRIES = False  # глобальный флаг «пауза входов»
LOOP_SLEEP_SEC = 0.25
from ml_veto import load_model_and_meta, predict_ok, atr_abs as _atr_abs

# --- timezone helper (stdlib zoneinfo с фоллбэком) ---
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # будем работать в локальном времени без tz

# =========================
# CLI и режимы запуска
# =========================
def _parse_cli(argv: Optional[List[str]], with_help: bool) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Trading bot runner",
        add_help=with_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["paper", "p", "real", "r"],
        help="режим запуска: paper (бумажный) или real",
    )
    parser.add_argument("--yes", action="store_true", help="подтвердить запуск в real-режиме без вопроса")
    parser.add_argument("--unsafe", action="store_true", help="разрешить отключение SAFE_MODE в real")
    parser.add_argument("--ci", action="store_true", help="CI-режим: форс SAFE_MODE/PAPER_MODE, тихая телега")
    parser.add_argument("--once", action="store_true", help="выполнить один цикл и завершиться")
    if argv is None:
        argv = []
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


if __name__ == "__main__":
    CLI_ARGS, CLI_UNKNOWN = _parse_cli(sys.argv[1:], with_help=True)
    if CLI_UNKNOWN:
        print(f"[CLI] Неизвестные аргументы (игнорирую): {' '.join(CLI_UNKNOWN)}")
else:
    CLI_ARGS, CLI_UNKNOWN = _parse_cli([], with_help=False)


mode_arg = (CLI_ARGS.mode or "").lower()
mode_is_paper = mode_arg in {"paper", "p"}
mode_is_real = mode_arg in {"real", "r"}

_session_bootstrap("paper" if mode_is_paper else "real" if mode_is_real else "auto")


def _handle_sigterm(signum, frame):
    global PAUSE_ENTRIES
    SESSION_STATE["shutdown_requested"] = True
    try:
        PAUSE_ENTRIES = True
    except Exception:
        pass
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

if mode_is_paper:
    config.PAPER_MODE = 1
    print("🧪 PAPER_MODE включен через терминал")
elif mode_is_real:
    if CLI_ARGS.ci:
        print("[CI] Флаг --ci принудительно включает PAPER_MODE")
        config.PAPER_MODE = 1
    else:
        if not CLI_ARGS.yes:
            resp = input("⚠️ Запуск в РЕАЛЬНОМ режиме. Продолжить? [y/N]: ").strip().lower()
            if resp != "y":
                print("Отмена запуска.")
                raise SystemExit(1)
        config.PAPER_MODE = 0
        print("💰 REAL_MODE включен через терминал")

RUN_ONCE = bool(CLI_ARGS.once)
CI_MODE = bool(CLI_ARGS.ci)

# Флаг «опасного» запуска (разрешить отключать SAFE_MODE руками)
UNSAFE_FLAG = bool(CLI_ARGS.unsafe)

# =========================
# Режимы / Безопасность
# =========================
if CI_MODE:
    config.PAPER_MODE = 1
    try:
        config.SAFE_MODE = 1
    except Exception:
        pass
    os.environ.setdefault("PAPER_MODE", "1")
    os.environ.setdefault("SAFE_MODE", "1")
    CONTROL_POLL_SEC = min(CONTROL_POLL_SEC, 0.5)
    LOOP_SLEEP_SEC = 0.05
    _ci_tg_notice = {"shown": False}

    def _tg_stub_ci(message: str) -> None:
        if not _ci_tg_notice["shown"]:
            log("[CI] Telegram отправка отключена (--ci).")
            _ci_tg_notice["shown"] = True

    utils.tg_send = _tg_stub_ci  # type: ignore[attr-defined]
    tg_send = utils.tg_send
    log("[CI] SAFE_MODE=1, PAPER_MODE=1, fast logging включён.")

PAPER_MODE = bool(getattr(config, "PAPER_MODE", 0))
# Локальный SAFE_MODE = из .env/конфига; можно снять только если есть флаг --unsafe
SAFE_MODE = bool(SAFE_MODE_FROM_ENV or getattr(config, "PAPER_MODE", 0))
if CI_MODE:
    SAFE_MODE = True
elif UNSAFE_FLAG and not PAPER_MODE:
    SAFE_MODE = False

SESSION_STATE["mode"] = "paper" if PAPER_MODE else "real"
_session_update_meta({"mode": SESSION_STATE["mode"]})
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

# Онлайн метрики
kelly_mean = 0.0
kelly_m2 = 0.0
kelly_count = 0
kelly_fraction = float(getattr(config, "RISK_PER_TRADE_FRAC", 0.005))
kelly_f_max = float(getattr(config, "KELLY_F_MAX", 0.03))
equity_total = 0.0
equity_peak = 0.0
max_drawdown_usdt = 0.0
ulcer_accum = 0.0
ulcer_count = 0
session_day = None
session_results: List[float] = []

# =========================
# Тайминги цикла
# =========================
WORK_DURATION_SEC   = int(getattr(config, "WORK_DURATION_SEC", 3600))   # 60 мин
BREAK_DURATION_SEC  = int(getattr(config, "BREAK_DURATION_SEC", 600))   # 10 мин
ENTRY_COOLDOWN_SEC  = int(getattr(config, "ENTRY_COOLDOWN_SEC", 45))

RECORD_MARKET_DATA  = bool(int(os.getenv("RECORD_MARKET_DATA", str(getattr(config, "RECORD_MARKET_DATA", 1)))))

# частоты опроса
KLINE_REFRESH_SEC    = 2.0
SNAPSHOT_REFRESH_SEC = 1.0
PERSIST_EVERY_SEC    = 60.0
DD_CHECK_EVERY_SEC   = 10.0

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


def _control_state_path() -> str:
    folder = os.path.abspath(os.path.dirname(_control_path()) or ".")
    return os.path.join(folder, "control_state.json")


def _read_pause_state_from_disk() -> Optional[bool]:
    path = _control_state_path()
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "pause_entries" in data:
            return bool(data.get("pause_entries"))
    except Exception as e:
        log(f"[CTRL] ошибка чтения состояния паузы: {e}")
    return None


def _persist_pause_state(source: str) -> None:
    path = _control_state_path()
    payload = {
        "pause_entries": bool(PAUSE_ENTRIES),
        "ts": utcnow_iso(),
        "source": source,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"[CTRL] ошибка сохранения состояния паузы: {e}")


def _restore_pause_state() -> None:
    state = _read_pause_state_from_disk()
    if state is None:
        return
    global PAUSE_ENTRIES
    PAUSE_ENTRIES = bool(state)
    log(f"[CTRL] entries {'paused' if PAUSE_ENTRIES else 'resumed'} (restored)")


def register_trade_result(pnl: float):
    global loss_streak, last_loss_time
    global kelly_mean, kelly_m2, kelly_count, kelly_fraction
    global equity_total, equity_peak, max_drawdown_usdt, ulcer_accum, ulcer_count
    global session_day, session_results
    if pnl is None:
        return
    pnl = float(pnl)
    if pnl < 0:
        loss_streak += 1
        last_loss_time = time.time()
    elif pnl > 0:
        loss_streak = 0

    equity_total += pnl
    equity_peak = max(equity_peak, equity_total)
    drawdown = max(0.0, equity_peak - equity_total)
    max_drawdown_usdt = max(max_drawdown_usdt, drawdown)
    if equity_peak > 0:
        ulcer_accum += (drawdown / max(equity_peak, 1e-9)) ** 2
        ulcer_count += 1

    kelly_mean, kelly_m2, kelly_count = welford_mean_var(kelly_mean, kelly_m2, kelly_count, pnl)
    variance = (kelly_m2 / (kelly_count - 1)) if kelly_count > 1 else max(kelly_m2, 1e-9)
    kelly_fraction = kelly_capped(kelly_mean, variance, kelly_f_max)

    today = datetime.datetime.utcnow().date()
    if session_day != today:
        session_day = today
        session_results = []
    session_results.append(pnl)
    winrate = (sum(1 for r in session_results if r > 0) / len(session_results)) if session_results else 0.0
    mean_r = statistics.mean(session_results) if session_results else 0.0
    median_r = statistics.median(session_results) if session_results else 0.0

    sharpe = 0.0
    if variance > 1e-12:
        sharpe = (kelly_mean / math.sqrt(variance)) * math.sqrt(max(kelly_count, 1))

    log_every = max(1, int(getattr(config, "METRICS_LOG_EVERY", 5)))
    if kelly_count % log_every == 0:
        ulcer_index = math.sqrt(ulcer_accum / max(ulcer_count, 1)) if ulcer_count else 0.0
        log(
            f"[METRICS] trades={kelly_count} winrate_session={winrate:.2%} "
            f"mean={mean_r:.4f} median={median_r:.4f} "
            f"kelly={kelly_fraction:.3f} sharpe={sharpe:.2f} "
            f"max_dd={max_drawdown_usdt:.4f} ulcer={ulcer_index:.4f}"
        )

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
        kl_raw, _ = broker.get_kline_any(symbol, interval="1", limit=KLINE_HISTORY_LIMIT)
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
    spr_k = spread_penalty(spread_rel, spread_max, alpha=spread_penalty_mult)

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
                    qty_step=float(qty_step or 0.0),
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
    rating = scored[:min(len(scored), 8)]
    log(msg("PAIR_RATING", rating=rating))
    return top

# =========================
# Основной цикл
# =========================
def main_trading_cycle():
    cfg = config
    last_cfg_reload = time.time()
    CFG_RELOAD_SEC = 900

    # --- управление через control.json ---
    control_path = _control_path()
    _restore_pause_state()
    _persist_pause_state("startup")
    control_last_ts = ""
    last_ctrl_check = 0.0

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
    log(msg("PAIRS_CURRENT", pairs=", ".join(top_pairs)))
    _session_update_meta({"pairs": top_pairs})

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
                kl_raw, _ = broker.get_kline_any(s, interval="1", limit=KLINE_HISTORY_LIMIT)
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

    def _fmt_meta_float(value: Any) -> str:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return "n/a"

    cal_info = (meta or {}).get("calibration", {}) or {}
    thr_block = (meta or {}).get("thresholds", {}) or {}
    atr_pct = (meta or {}).get("atr_percentiles", {}) or {}

    cal_method = str(cal_info.get("method", "n/a"))
    used_mode = str(thr_block.get("used_mode", "global"))
    used_thr = _fmt_meta_float(thr_block.get("used", thr_block.get("global")))
    p50 = _fmt_meta_float(atr_pct.get("p50"))
    p90 = _fmt_meta_float(atr_pct.get("p90"))
    cache_flag = str(os.getenv("ML_CACHE_BUST", "0")).strip()
    cache_state = "bust-next" if cache_flag == "1" else "cache-on"
    log(
        f"[ML] summary: calibration={cal_method} | threshold={used_mode}({used_thr}) | "
        f"atr_p50/p90={p50}/{p90} | cache={cache_state}"
    )

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
                new_state = bool(c.get("pause_entries"))
                changed = (PAUSE_ENTRIES != new_state)
                PAUSE_ENTRIES = new_state
                if changed:
                    log(f"[CTRL] entries {'paused' if PAUSE_ENTRIES else 'resumed'}")
                _persist_pause_state("pause_entries")
            elif c.get("toggle_pause"):
                PAUSE_ENTRIES = not PAUSE_ENTRIES
                log(f"[CONTROL] toggle_pause: {'ON' if PAUSE_ENTRIES else 'OFF'}")
                log(f"[CTRL] entries {'paused' if PAUSE_ENTRIES else 'resumed'}")
                _persist_pause_state("toggle")

            if c.get("panic_close"):
                log("[CTRL] panic_close received")
                tg_send("🛑 CTRL: panic_close")
                SESSION_STATE["shutdown_requested"] = True
                PAUSE_ENTRIES = True
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
                            log(msg("PAIRS_CTRL_UPDATE", pairs=", ".join(top_pairs)))
                            _session_update_meta({"pairs": top_pairs})
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

    loop_iter = 0
    try:
        while True:
            loop_iter += 1
            if RUN_ONCE and loop_iter > 1:
                log("[MAIN] --once: завершение после одного цикла.")
                break
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
                            dd_msg = f"⛔️ Макс. просадка {dd*100:.2f}% (порог {max_dd_frac*100:.1f}%). Останавливаю бота."
                            log(dd_msg)
                            tg_send(dd_msg)
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
                        log(msg("PAIRS_NEW", pairs=", ".join(top_pairs)))
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
            positions_side: Dict[str, str] = {}
            try:
                raw_positions = getattr(broker, "get_positions", None)
                if callable(raw_positions):
                    data_pos = raw_positions()
                    lst = (data_pos.get("result", {}) or {}).get("list", []) if isinstance(data_pos, dict) else []
                    for pos in lst or []:
                        try:
                            qty = float(pos.get("size") or pos.get("qty") or pos.get("positionQty") or 0.0)
                        except Exception:
                            qty = 0.0
                        if abs(qty) <= 1e-12:
                            continue
                        sym = str(pos.get("symbol") or pos.get("coin") or "").strip()
                        if not sym:
                            continue
                        raw_side = str(pos.get("side") or pos.get("positionSide") or pos.get("direction") or "").lower()
                        if raw_side.startswith("buy") or raw_side.startswith("long"):
                            positions_side[sym] = "Buy"
                        elif raw_side.startswith("sell") or raw_side.startswith("short"):
                            positions_side[sym] = "Sell"
            except Exception:
                positions_side = {}

            for symbol in list(top_pairs):
                # kline
                ts_kl, kl_cached = last_kl.get(symbol, (0.0, None))
                if now - ts_kl > KLINE_REFRESH_SEC or kl_cached is None:
                    kl_cached, _src = broker.get_kline_any(symbol, interval="1", limit=KLINE_HISTORY_LIMIT)
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

                atr_pct_now = (atr_val / max(price, 1e-9)) * 100.0 if price > 0 else 0.0
                if signal not in ("buy", "sell"):
                    if getattr(config, "DEBUG_TRADING", False):
                        if router_reason and "vola" in str(router_reason):
                            log_router(symbol, "ROUTER_HOLD_VOLA", atr=atr_pct_now, reason=str(router_reason))
                        else:
                            log_router(symbol, "ROUTER_HOLD_EMPTY", reason=str(router_reason or ""))
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

                dynamic_frac = max(0.0, min(RISK_PER_TRADE_FRAC, kelly_fraction))
                risk_cap = max(0.0, avail * max(1e-6, min(1.0, dynamic_frac)))
                stop_dist = max(atr_val * max(ATR_STOP_K, 0.1), 1e-9)
                qty_risk_atr = risk_cap / stop_dist

                max_notional_cap = avail * max(0.0, min(1.0, float(getattr(cfg, "MAX_BALANCE_SHARE", 0.08))))
                qty_cap_share = max_notional_cap / max(price, 1e-9)

                penalty_alpha = float(getattr(cfg, "SPREAD_PENALTY_ALPHA", 1.0))
                spread_adj = spread_penalty(spread_rel, SPREAD_MAX_PCT, alpha=penalty_alpha)
                raw_qty = min(qty_risk_atr, qty_cap_share) * spread_adj
                qty = adjust_qty(price, raw_qty, min_qty=min_qty, qty_step=step, min_notional=min_notional)

                if qty <= 0:
                    min_qty_aligned = _align_qty(min_qty, step)
                    min_notional_usdt = float(getattr(cfg, "MIN_NOTIONAL_USDT", 5.0))
                    notional_min = price * min_qty_aligned
                    hard_cap_notional = avail * max(0.0, min(1.0, float(getattr(cfg, "HARD_CAP_SHARE", 0.25))))
                    if (notional_min >= max(min_notional, min_notional_usdt)) and (notional_min <= hard_cap_notional):
                        qty = min_qty_aligned
                        log_adjust_if_changed(symbol, qty, notional_min)
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

                existing_side = positions_side.get(symbol)
                if existing_side and existing_side == side:
                    log(msg("ENTRY_SKIP_OPEN_POS", symbol=symbol))
                    last_entry_time[symbol] = now
                    continue

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
                        log(msg("FAIL_PLACE_ORDER", symbol=symbol, router=str(router_reason or "-")), level="WARNING")
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

            time.sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        SESSION_STATE["shutdown_requested"] = True
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
        trades_rows: List[Dict[str, Any]] = []
        pnl_list: List[float] = []
        equity_points: List[Tuple[str, float]] = []

        try:
            end_balance = float(broker.get_balance())
        except Exception as e:
            log(f"[BAL-END] {e}")
            end_balance = 0.0
        delta = round(end_balance - start_balance, 2)

        duration_sec = 0.0
        if SESSION_STATE.get("start_monotonic") is not None:
            try:
                duration_sec = max(0.0, time.monotonic() - float(SESSION_STATE.get("start_monotonic") or 0.0))
            except Exception:
                duration_sec = 0.0
        duration_human = str(timedelta(seconds=int(duration_sec)))

        total_p = 0.0
        total_n = 0.0
        cnt_p = 0
        cnt_n = 0

        trade_events = {
            "open",
            "dry_open",
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

        log_path = SESSION_STATE.get("log_jsonl") or os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))
        try:
            if log_path and os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            js = json.loads(line)
                        except Exception:
                            continue
                        event = str(js.get("event") or "")
                        if event in trade_events:
                            trades_rows.append(js)
                        if event not in closure_events:
                            continue
                        pnl = float(js.get("pnl", 0.0) or 0.0)
                        if pnl >= 0:
                            total_p += pnl
                            cnt_p += 1
                        else:
                            total_n += pnl
                            cnt_n += 1
                        pnl_list.append(pnl)
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

        trades_total = cnt_p + cnt_n
        winrate = (cnt_p / trades_total) * 100.0 if trades_total else 0.0
        pnl_avg = (total_p + total_n) / trades_total if trades_total else 0.0
        sharpe = 0.0
        if len(pnl_list) > 1:
            try:
                mean_p = statistics.mean(pnl_list)
                std_p = statistics.stdev(pnl_list)
                if std_p > 1e-12:
                    sharpe = (mean_p / std_p) * math.sqrt(len(pnl_list))
            except Exception:
                sharpe = 0.0

        end_ts_iso = utcnow_iso()
        stats_payload = {
            "start_ts": SESSION_STATE.get("start_ts"),
            "end_ts": end_ts_iso,
            "duration_sec": duration_sec,
            "duration_human": duration_human,
            "mode": mode_label,
            "shutdown_requested": bool(SESSION_STATE.get("shutdown_requested")),
            "balance_start": start_balance,
            "balance_end": end_balance,
            "pnl_total": delta,
            "pnl_positive": total_p,
            "pnl_negative": total_n,
            "wins": cnt_p,
            "losses": cnt_n,
            "trades": trades_total,
            "winrate": winrate,
            "pnl_per_trade": pnl_avg,
            "max_drawdown": max_drawdown_usdt,
            "max_dd": max_drawdown_usdt,
            "sharpe": sharpe,
            "kelly_trades": kelly_count,
            "log_path": log_path,
            "pairs": (SESSION_STATE.get("meta") or {}).get("pairs"),
        }

        summary_msg = (
            f"🔴 Завершено [{mode_label}].\n"
            f"Δ Баланса: {delta:+.2f} USDT\n"
            f"Сделки: {trades_total} (побед {cnt_p}, поражений {cnt_n}, winrate {winrate:.1f}%)\n"
            f"Длительность: {duration_human}"
        )
        tg_send(summary_msg)
        log(
            f"[STATS] Итог: {delta:+.2f} USDT; +{cnt_p} / -{cnt_n}; "
            f"duration={duration_human}; trades={trades_total}; maxDD={max_drawdown_usdt:+.2f}; "
            f"winrate={winrate:.1f}%"
        )

        _session_finalize(stats_payload, trades_rows)

if __name__ == "__main__":
    main_trading_cycle()
