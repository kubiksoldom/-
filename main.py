# -*- coding: utf-8 -*-
# main.py — 2025-09-22: интеграция StrategyRouter (router_reason/strategy/regime в логах)

import os
import sys
import time
import json
import math
import shutil
import statistics
import uuid
import datetime
from datetime import timedelta, timezone
from pathlib import Path
import importlib
import subprocess
import argparse
import signal
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env_loader import load_env

# Загружаем .env до любых локальных импортов
load_env()

import config
import utils
# ⬇️ подключаем новый роутер стратегий
from strategy import decide_with_router, welford_mean_var
from utils import (
    log, tg_send, write_cycle_log, adjust_qty,
    spread_penalty, fee_aware_r_min, kelly_capped,
    SAFE_MODE as SAFE_MODE_FROM_ENV,
    write_session_summary,
    clamp,
    pre_trade_check,
    set_margin_state,
    get_margin_state,
    apply_leverage_ramp,
    fallback_leverage,
)

DATA_ROOT = (getattr(config, "DATA_ROOT", None) or os.getenv("DATA_ROOT", "").strip() or "./data")
KLINE_HISTORY_LIMIT = int(os.getenv("KLINE_HISTORY_LIMIT", str(getattr(config, "KLINE_HISTORY_LIMIT", 300))))
LOG_RU = utils.env_bool("LOG_RU", bool(getattr(config, "LOG_RU", 1)))
ROUTER_HEARTBEAT_SEC = int(os.getenv("ROUTER_HEARTBEAT_SEC", str(getattr(config, "ROUTER_HEARTBEAT_SEC", 60))))
MARGIN_POLL_SEC = float(getattr(config, "MARGIN_POLL_SEC", 15))
MAX_IM_PERCENT = float(getattr(config, "MAX_IM_PERCENT", 30.0))
CRIT_IM_PERCENT = float(getattr(config, "CRIT_IM_PERCENT", 60.0))
LEV_STEP_MAX = float(getattr(config, "LEV_STEP_MAX", 2.0))

MAX_ACTIVE_PAIRS = int(os.getenv("MAX_ACTIVE_PAIRS", "2"))
_raw_allowed_pairs = [s.strip().upper() for s in os.getenv("ALLOWED_PAIRS", "ETHUSDT,SOLUSDT").split(",") if s.strip()]
ALLOWED_PAIRS = list(dict.fromkeys(_raw_allowed_pairs)) or ["ETHUSDT", "SOLUSDT"]

_last_router: Dict[str, Tuple[str, str, float]] = {}
_last_min_qty: Dict[str, float] = {}

_LAST_PAIR_SELECTION: Dict[str, List[str]] = {"core": [], "explore": [], "chosen": []}
_SYMBOL_SCORE_META: Dict[str, Dict[str, float]] = {}

_PRECHECK_REASONS = {
    "bad_inputs": "некорректные параметры ордера",
    "filters_unavailable": "нет доступа к биржевым фильтрам",
    "filters_error": "ошибка получения фильтров",
    "filters_unreliable": "фильтры недостоверны",
    "qty_adjust": "не удалось привести объём к шагу",
    "min_notional": "недостаточный нотационал",
    "spread": "спрэд превышает лимит",
    "margin": "маржинальные ограничения",
}


def _ensure_runtime_dirs() -> None:
    for rel in ("logs", "logs/errors", "logs/metrics", "logs/sessions"):
        try:
            Path(rel).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


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


_ensure_runtime_dirs()


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

# Runtime bookkeeping for guaranteed session summary
SESSION_ACCOUNT_SNAPSHOT: Dict[str, Any] = {}
SESSION_RUNTIME_STATS: Dict[str, Any] = {
    "trades_total": 0,
    "pnl_accum": 0.0,
    "fees_accum": 0.0,
}

_STDOUT_ORIG = sys.stdout
_STDERR_ORIG = sys.stderr


def utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log_bot_trade(symbol: str,
                   side: str,
                   *,
                   price: Optional[float],
                   qty: Optional[float],
                   pnl: float = 0.0,
                   trade_id: Optional[str] = None,
                   exploration: bool = False,
                   meta: Optional[Dict[str, Any]] = None) -> str:
    trade_ref = trade_id or uuid.uuid4().hex
    payload: Dict[str, Any] = {
        "tag": "BOT_TRADE",
        "symbol": symbol,
        "side": side,
        "pnl": float(pnl or 0.0),
        "trade_id": trade_ref,
        "meta": {"exploration": bool(exploration)},
    }
    if price is not None:
        try:
            payload["price"] = float(price)
        except Exception:
            payload["price"] = price
    if qty is not None:
        try:
            payload["qty"] = float(qty)
        except Exception:
            payload["qty"] = qty
    if meta:
        extra = payload.setdefault("meta", {})
        for key, value in meta.items():
            if key == "exploration":
                continue
            extra[key] = value
    write_cycle_log(payload)
    return trade_ref


def _log_ml_decision(
    symbol: str,
    *,
    direction: str,
    side: str,
    proba: float,
    factor: float,
    band: str,
    meta_threshold: float,
    strict_threshold: float,
    effective_threshold: float,
    features_ok: bool,
) -> None:
    payload: Dict[str, Any] = {
        "tag": "ML_DECISION",
        "symbol": symbol,
        "side": side,
        "direction": direction,
        "proba": float(proba),
        "factor": float(factor),
        "band": str(band),
        "th_meta": float(meta_threshold),
        "th_strict": float(strict_threshold),
        "th_eff": float(effective_threshold),
        "features_ok": bool(features_ok),
        # дублируем старые ключи на всякий случай совместимости с лог-ридерами
        "meta_thr": float(meta_threshold),
        "strict_thr": float(strict_threshold),
        "effective_thr": float(effective_threshold),
    }
    write_cycle_log(payload)



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
    ts = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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


def _fatal_exit(message: str) -> None:
    log(f"ERROR [FATAL] {message}", level="ERROR")
    raise SystemExit(1)


def _confirm_real_env_flag() -> bool:
    try:
        default = getattr(config, "CONFIRM_REAL", 0)
    except Exception:
        default = 0
    return utils.env_bool("CONFIRM_REAL", bool(default))


def _resolve_api_credential(name: str) -> str:
    raw = os.getenv(name)
    if raw:
        return raw.strip()
    try:
        val = getattr(config, name, "")
    except Exception:
        val = ""
    if val is None:
        return ""
    return str(val).strip()


def _enforce_real_mode_guard(cli_args: argparse.Namespace, *, explicit_real: bool) -> None:
    if os.getenv("PAPER_MODE") is not None:
        if utils.env_bool("PAPER_MODE"):
            return
    elif bool(getattr(config, "PAPER_MODE", 0)):
        return

    if getattr(cli_args, "unsafe", False) and not getattr(cli_args, "yes", False):
        _fatal_exit("Флаг --unsafe требует подтверждения --yes.")

    confirm_env = _confirm_real_env_flag()
    if not confirm_env:
        if not explicit_real:
            _fatal_exit("Запуск real-режима возможен только с флагом --real.")
        if not getattr(cli_args, "yes", False):
            _fatal_exit("Добавьте --yes для явного подтверждения запуска в real-режиме.")

    api_key = _resolve_api_credential("BYBIT_API_KEY")
    api_secret = _resolve_api_credential("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        _fatal_exit("BYBIT_API_KEY/BYBIT_API_SECRET не заданы — real-режим запрещён.")


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
        "PAIRS_CURRENT":         "[PAIRS] Работаем с: {pairs} (cap={cap})",
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
        "PAIRS_CURRENT":         "[PAIRS] Working with: {pairs} (cap={cap})",
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
CONTROL_PROCESSED_IDS: Set[str] = set()
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
    parser.add_argument("--paper", dest="flag_paper", action="store_true", help="принудительно paper-режим")
    parser.add_argument("--real", dest="flag_real", action="store_true", help="принудительно real-режим")
    parser.add_argument("--yes", action="store_true", help="подтвердить запуск в real-режиме без вопроса")
    parser.add_argument("--unsafe", action="store_true", help="разрешить отключение SAFE_MODE в real")
    parser.add_argument("--ci", action="store_true", help="CI-режим: форс SAFE_MODE/PAPER_MODE, тихая телега")
    parser.add_argument("--once", action="store_true", help="выполнить один цикл и завершиться")
    if argv is None:
        argv = []
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def _cli_retrain_if_drift() -> int:
    """Сбор датасета и переобучение модели с проверкой weekly precision."""

    log("[RETRAIN] Запуск retrain-if-drift: сбор датасета…")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        subprocess.run([sys.executable, "build_ml_dataset_from_fills.py"], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        log(f"[RETRAIN] Ошибка build_ml_dataset_from_fills.py (код {exc.returncode})", level="ERROR")
        return exc.returncode or 1

    model_path = Path(getattr(config, "MODEL_FILE", os.getenv("MODEL_FILE", "rf_model.pkl")))
    meta_path = Path(getattr(config, "MODEL_META", os.getenv("MODEL_META", "model_meta.json")))
    tmp_model = model_path.with_name(model_path.name + ".retrain")
    tmp_meta = meta_path.with_name(meta_path.name + ".retrain")

    for tmp in (tmp_model, tmp_meta):
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    train_env = env.copy()
    train_env["MODEL_FILE"] = str(tmp_model)
    train_env["MODEL_META"] = str(tmp_meta)

    log("[RETRAIN] Обучение модели…")
    try:
        subprocess.run([sys.executable, "retrain_model_from_dataset.py"], check=True, env=train_env)
    except subprocess.CalledProcessError as exc:
        log(f"[RETRAIN] Ошибка retrain_model_from_dataset.py (код {exc.returncode})", level="ERROR")
        return exc.returncode or 1

    try:
        with open(tmp_meta, "r", encoding="utf-8") as fh:
            new_meta = json.load(fh)
    except Exception as exc:
        log(f"[RETRAIN] Не удалось прочитать временный model_meta.json: {exc}", level="ERROR")
        return 1

    metrics = new_meta.get("metrics") if isinstance(new_meta, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    try:
        precision_week = metrics.get("precision_week")
        precision_week = float(precision_week) if precision_week is not None else None
        if precision_week is not None and not math.isfinite(precision_week):
            precision_week = None
    except Exception:
        precision_week = None

    threshold = float(getattr(config, "ML_MIN_WEEKLY_PREC", 0.52))
    if precision_week is None:
        log("[RETRAIN] Нет weekly precision в отчёте — отклоняю новую модель", level="ERROR")
        return 1
    if precision_week < threshold:
        log(
            f"[RETRAIN] Weekly precision {precision_week:.3f} < порога {threshold:.3f} — модель не обновлена",
            level="ERROR",
        )
        return 2

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_model, model_path)
        os.replace(tmp_meta, meta_path)
    except Exception as exc:
        log(f"[RETRAIN] Не удалось заменить файлы модели: {exc}", level="ERROR")
        return 1

    log(f"[RETRAIN] Модель обновлена (weekly precision {precision_week:.3f} ≥ {threshold:.3f})")
    os.environ["ML_CACHE_BUST"] = "1"
    try:
        load_model_and_meta()
    except Exception as exc:
        log(f"[RETRAIN] Предупреждение: не удалось перезагрузить модель: {exc}", level="WARNING")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "retrain-if-drift":
        sys.exit(_cli_retrain_if_drift())
    CLI_ARGS, CLI_UNKNOWN = _parse_cli(sys.argv[1:], with_help=True)
    if CLI_UNKNOWN:
        print(f"[CLI] Неизвестные аргументы (игнорирую): {' '.join(CLI_UNKNOWN)}")
else:
    CLI_ARGS, CLI_UNKNOWN = _parse_cli([], with_help=False)


mode_flag_paper = bool(getattr(CLI_ARGS, "flag_paper", False))
mode_flag_real = bool(getattr(CLI_ARGS, "flag_real", False))
if mode_flag_real and mode_flag_paper:
    _fatal_exit("Флаги --paper и --real нельзя использовать одновременно.")

if getattr(CLI_ARGS, "unsafe", False) and not getattr(CLI_ARGS, "yes", False):
    _fatal_exit("Флаг --unsafe требует подтверждения --yes.")

mode_arg = (CLI_ARGS.mode or "").lower()
if mode_flag_real:
    mode_arg = "real"
elif mode_flag_paper:
    mode_arg = "paper"

mode_is_paper = mode_arg in {"paper", "p"}
mode_is_real = mode_arg in {"real", "r"}
explicit_real_request = mode_is_real or mode_flag_real
explicit_paper_request = mode_is_paper or mode_flag_paper

_session_bootstrap("paper" if explicit_paper_request else "real" if explicit_real_request else "auto")


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

if explicit_paper_request:
    config.PAPER_MODE = 1
elif explicit_real_request:
    config.PAPER_MODE = 0

_enforce_real_mode_guard(CLI_ARGS, explicit_real=explicit_real_request)

if explicit_paper_request:
    print("🧪 PAPER_MODE включен через терминал")
elif explicit_real_request:
    if CLI_ARGS.ci:
        print("[CI] Флаг --ci принудительно включает PAPER_MODE")
        config.PAPER_MODE = 1
    else:
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


def _mask(s, keep=4):
    try:
        s = str(s or "")
        if len(s) <= keep * 2:
            return ("*" * len(s)) if s else "<empty>"
        return s[:keep] + "*" * (len(s) - keep * 2) + s[-keep:]
    except Exception:
        return "<err>"


def _env_probe():
    keys = ["BYBIT_API_KEY", "BYBIT_API_SECRET", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]
    vals = {k: _mask(os.getenv(k, "")) for k in keys}
    log(f"[APP] ENV probe: {', '.join(f'{k}={v}' for k, v in vals.items())}")


_env_probe()

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
kelly_fraction = float(getattr(config, "RISK_PER_TRADE_FRAC", 0.0065))
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

RECORD_MARKET_DATA  = utils.env_bool("RECORD_MARKET_DATA", bool(getattr(config, "RECORD_MARKET_DATA", 1)))

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
    "0:2,25:3,60:4,120:5"
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
PAIR_FILTER_MIN_NOTIONAL = float(getattr(config, "PAIR_FILTER_MIN_NOTIONAL", 0.50))  # левый край окна
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
    force_schedule_off = utils.env_bool("FORCE_SCHEDULE_OFF", bool(getattr(config, "FORCE_SCHEDULE_OFF", 0)))
    exclude_weekends = utils.env_bool("EXCLUDE_WEEKENDS", bool(getattr(config, "EXCLUDE_WEEKENDS", 1)))
    windows_str = os.getenv("TRADE_HOURS_LOCAL", str(getattr(config, "TRADE_HOURS_LOCAL", "22:00-08:00"))).strip()
    if force_schedule_off or (not exclude_weekends and windows_str == "00:00-24:00"):
        return True, "forced", None
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


CONTROL_TTL_SEC = int(os.getenv("CONTROL_TTL_SEC", "600"))


def _control_parse_ts(cmd_ts: Any) -> Optional[float]:
    if cmd_ts is None:
        return None
    ts_str = str(cmd_ts).strip()
    if not ts_str:
        return None
    try:
        if ts_str.isdigit():
            return float(ts_str)
        return float(ts_str)
    except Exception:
        pass
    try:
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _control_is_fresh(cmd_ts: Any) -> bool:
    if isinstance(cmd_ts, (int, float)):
        ts_val = float(cmd_ts)
    else:
        ts_val = _control_parse_ts(cmd_ts)
        if ts_val is None:
            return False
    try:
        return (time.time() - float(ts_val)) <= CONTROL_TTL_SEC
    except Exception:
        return False


def _control_state_path() -> str:
    folder = os.path.abspath(os.path.dirname(_control_path()) or ".")
    return os.path.join(folder, "control_state.json")


def read_pairs_from_control() -> List[str]:
    path = _control_path()
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    def _normalize(obj: Any) -> List[str]:
        if isinstance(obj, str):
            return [part.strip().upper() for part in obj.split(",") if part.strip()]
        if isinstance(obj, (list, tuple, set)):
            out: List[str] = []
            for part in obj:
                out.extend(_normalize(part))
            return out
        if isinstance(obj, dict):
            out: List[str] = []
            for key, value in obj.items():
                key_str = str(key)
                if key_str in {"set_pairs", "pairs"}:
                    out.extend(_normalize(value))
                    continue
                if isinstance(value, (list, tuple, set, dict, str)):
                    out.extend(_normalize(value))
                if isinstance(key, str):
                    key_up = key.strip().upper()
                    if key_up.endswith("USDT") and len(key_up) >= 6:
                        out.append(key_up)
            return out
        return []

    extracted: List[str] = []
    if isinstance(data, list):
        for item in reversed(data):
            normalized = _normalize(item)
            if normalized:
                extracted = normalized
                break
        if not extracted:
            extracted = _normalize(data)
    else:
        extracted = _normalize(data)

    seen: List[str] = []
    for sym in extracted:
        symbol = sym.strip().upper()
        if symbol and symbol not in seen:
            seen.append(symbol)
    return seen


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


def register_trade_result(pnl: float, fees: float = 0.0):
    global loss_streak, last_loss_time
    global kelly_mean, kelly_m2, kelly_count, kelly_fraction
    global equity_total, equity_peak, max_drawdown_usdt, ulcer_accum, ulcer_count
    global session_day, session_results
    SESSION_RUNTIME_STATS["trades_total"] = int(SESSION_RUNTIME_STATS.get("trades_total", 0))
    SESSION_RUNTIME_STATS["pnl_accum"] = float(SESSION_RUNTIME_STATS.get("pnl_accum", 0.0))
    SESSION_RUNTIME_STATS["fees_accum"] = float(SESSION_RUNTIME_STATS.get("fees_accum", 0.0))
    if pnl is None:
        return
    pnl = float(pnl)
    fees = float(fees or 0.0)
    SESSION_RUNTIME_STATS["trades_total"] += 1
    SESSION_RUNTIME_STATS["pnl_accum"] += pnl
    SESSION_RUNTIME_STATS["fees_accum"] += fees
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

    today = datetime.datetime.now(timezone.utc).date()
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
    margin_state = get_margin_state()
    im_pct = 0.0
    try:
        im_pct = float(margin_state.get("im_pct", 0.0) or 0.0)
    except Exception:
        im_pct = 0.0
    if margin_state.get("frozen") or im_pct >= MAX_IM_PERCENT:
        return False, f"margin:{im_pct:.2f}%"
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
    global _SYMBOL_SCORE_META
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
        _SYMBOL_SCORE_META[symbol] = {
            "atr_pct": float(atr_pct),
            "atr_norm": float(atr / max(last_price, 1e-9)) if last_price > 0 else 0.0,
        }
        return score
    except Exception as e:
        log(f"[SCORE] {symbol}: {e}")
        _SYMBOL_SCORE_META.pop(symbol, None)
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


def _filters_reliable(symbol: str) -> bool:
    fn = getattr(broker, "filters_reliable", None)
    if callable(fn):
        try:
            return bool(fn(symbol))
        except Exception as e:
            log(f"[FILTERS] {symbol}: reliability check error {e}")
            return True
    return True

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
            if not _filters_reliable(s):
                log(f"[FILTER] drop {s}: no reliable filters yet")
                continue
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
    spread_max = float(getattr(config, "SPREAD_MAX_PCT", 0.0012))
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


def _set_leverage_with_retry(symbol: str, leverage: int, prev: Optional[int] = None, attempts: int = 3) -> bool:
    delay = 0.3
    target = max(1, int(leverage))
    for attempt in range(1, max(1, attempts) + 1):
        try:
            ok = broker.set_leverage(symbol, int(target))
        except Exception as e:
            log(f"[LEV] {symbol}: set_leverage attempt {attempt} failed: {e}")
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)
            continue
        if ok is True:
            return True
        if ok is False:
            log(f"[LEV] {symbol}: leverage {target}x rejected")
            return False
        log(f"[LEV] {symbol}: unexpected response {ok!r}")
        return False
    return False

def select_top_pairs(base_list, count=2):
    """
    Детерминированные топы + контролируемое исследование (ε-exploration).
    Берём core = (1-EXPL_FRAC)*count, остальное — случайно из середины рейтинга.
    """
    global _LAST_PAIR_SELECTION
    try:
        count_limit = int(count)
    except Exception:
        count_limit = MAX_ACTIVE_PAIRS
    count_limit = max(0, min(MAX_ACTIVE_PAIRS, count_limit))
    if count_limit <= 0:
        return []

    EXPL_FRAC = float(getattr(config, "EXPLORATION_FRAC", 0.35))   # доля исследуемых
    EXPL_BONUS_ATR = float(getattr(config, "EXPLORATION_ATR_BONUS", 0.15))

    ranked_base: List[str] = []
    seen: Set[str] = set()
    for raw in base_list:
        sym = str(raw).upper()
        if sym in seen:
            continue
        seen.add(sym)
        if ALLOWED_PAIRS and sym not in ALLOWED_PAIRS:
            continue
        ranked_base.append(sym)

    if not ranked_base:
        ranked_base = [str(raw).upper() for raw in base_list]

    scored: List[Tuple[str, float]] = []
    for s in ranked_base:
        sc = score_symbol(s)
        if sc is not None:
            scored.append((s, sc))

    if not scored:
        from bybit_api import fast_pick_top_pairs
        fallback = fast_pick_top_pairs(count=count_limit or MAX_ACTIVE_PAIRS)
        fallback = [str(sym).upper() for sym in fallback]
        chosen = fallback[:count_limit]
        core = list(chosen)
        explored: List[str] = []
        _LAST_PAIR_SELECTION = {"core": core, "explore": explored, "chosen": list(chosen)}
        log(f"[PAIR-SELECT] fallback={chosen}")
        log(msg("PAIRS_CURRENT", pairs=", ".join(chosen), cap=MAX_ACTIVE_PAIRS))
        return chosen

    scored.sort(key=lambda x: x[1], reverse=True)
    rating = scored[:min(len(scored), 8)]
    log(msg("PAIR_RATING", rating=rating))

    frac = max(0.0, min(1.0, EXPL_FRAC))
    core_n = max(1, min(count_limit, int(math.ceil((1.0 - frac) * max(1, count_limit)))))
    core = [s for s, _ in scored[:core_n]]

    # кандидаты из «середины» (следующие ~30), лёгкий бонус за волу
    mid = scored[core_n: core_n + 30]
    candidates = []
    for s, sc in mid:
        meta = _SYMBOL_SCORE_META.get(s, {})
        atr_norm = 0.0
        if "atr_norm" in meta:
            atr_norm = float(meta.get("atr_norm", 0.0))
        elif "atr_pct" in meta:
            atr_norm = float(meta.get("atr_pct", 0.0)) / 100.0
        bonus = atr_norm * EXPL_BONUS_ATR
        candidates.append((s, sc + bonus))

    import random
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x[1], reverse=True)
    rest_n = max(0, count_limit - len(core))
    explored: List[str] = []
    for s, _ in candidates:
        if s in core:
            continue
        explored.append(s)
        if len(explored) >= rest_n:
            break

    chosen = (core + explored)[:count_limit]

    pairs_from_control = read_pairs_from_control()
    if pairs_from_control:
        pairs_ranked = [sym for sym, _ in scored]
        control_filtered = [p for p in pairs_from_control if p in pairs_ranked]
        if control_filtered:
            unique_control: List[str] = []
            for sym in control_filtered:
                if sym not in unique_control:
                    unique_control.append(sym)
            chosen = unique_control[:count_limit]

    _LAST_PAIR_SELECTION = {"core": list(core), "explore": list(explored), "chosen": list(chosen)}
    log(f"[PAIR-SELECT] Ядро={core} Исследование={explored} (count={count_limit})")
    log(msg("PAIRS_CURRENT", pairs=", ".join(chosen), cap=MAX_ACTIVE_PAIRS))
    return chosen

# =========================
# Основной цикл
# =========================
def main_trading_cycle():
    cfg = config
    last_cfg_reload = time.time()
    CFG_RELOAD_SEC = 900
    session_reason = "normal"
    should_exit = False
    panic_requested = False
    exploration_set: Set[str] = set()
    ml_skip_notice: Set[str] = set()
    ml_shadow_enabled = bool(int(os.getenv("ML_SHADOW_MODE", str(getattr(config, "ML_SHADOW_MODE", 0)))))

    # --- управление через control.json ---
    control_path = _control_path()
    _restore_pause_state()
    _persist_pause_state("startup")
    control_last_ts: Optional[float] = None
    last_ctrl_check = 0.0

    max_dd_frac = float(getattr(cfg, "MAX_DRAWDOWN_PCT", 8.0)) / 100.0
    last_dd_check = 0.0
    last_margin_poll = 0.0
    margin_freeze_active = False
    margin_crit_active = False

    def _mitigate_margin_pressure() -> None:
        nonlocal last_lev_set, last_lev_check_ts
        try:
            data = broker.get_positions()
        except Exception as exc:
            log(f"[MARGIN] mitigation failed: {exc}")
            return
        positions = ((data.get("result", {}) or {}).get("list", []) if isinstance(data, dict) else []) or []
        if not positions:
            log("[MARGIN] критический уровень, но открытых позиций нет")
            return
        safe_lev = int(getattr(config, "ADAPTIVE_LEV_MIN", 5) or getattr(config, "DEFAULT_LEVERAGE", 10) or 5)
        safe_lev = max(1, min(safe_lev, int(getattr(config, "DEFAULT_LEVERAGE", 10) or 10)))
        for pos in positions:
            try:
                sym = str(pos.get("symbol") or pos.get("coin") or "").upper()
            except Exception:
                sym = ""
            if not sym:
                continue
            try:
                size = abs(float(pos.get("size") or pos.get("qty") or 0.0))
            except Exception:
                size = 0.0
            if size <= 0:
                continue
            prev_lev = last_lev_set.get(sym)
            if prev_lev is None or prev_lev > safe_lev:
                applied = _set_leverage_with_retry(sym, safe_lev, prev_lev)
                if applied:
                    last_lev_set[sym] = safe_lev
                    last_lev_check_ts[sym] = time.time()
                    log(f"[MARGIN] {sym}: понижение плеча до {safe_lev}x для снижения нагрузки")
            try:
                qty_half = size * 0.5
                broker.close_position_by_market(sym, qty_half)
                _log_bot_trade(
                    sym,
                    "close",
                    price=None,
                    qty=qty_half,
                    pnl=0.0,
                    trade_id=None,
                    exploration=False,
                    meta={"reason": "margin_mitigation", "partial": True},
                )
                log(f"[MARGIN] {sym}: частичное закрытие {qty_half:.6f}")
            except Exception as close_exc:
                log(f"[MARGIN] {sym}: не удалось частично закрыть позицию ({close_exc})")

    # стартовый баланс
    try:
        start_balance = float(broker.get_balance())
    except Exception as e:
        log(f"[BAL] ошибка чтения баланса: {e}")
        start_balance = 0.0

    try:
        start_equity = float(getattr(broker, "get_equity", broker.get_balance)())
    except Exception as e:
        log(f"[BAL] ошибка чтения equity: {e}")
        start_equity = start_balance

    set_margin_state(0.0, 0.0, start_equity, False, "")

    SESSION_ACCOUNT_SNAPSHOT.clear()
    SESSION_ACCOUNT_SNAPSHOT.update({
        "ts_start": SESSION_STATE.get("start_ts") or utcnow_iso(),
        "mode": "paper" if PAPER_MODE else "real",
        "start_balance": start_balance,
        "start_equity": start_equity,
        "pairs": [],
    })
    SESSION_RUNTIME_STATS["trades_total"] = 0
    SESSION_RUNTIME_STATS["pnl_accum"] = 0.0
    SESSION_RUNTIME_STATS["fees_accum"] = 0.0

    # (2) решаем, сколько пар вести от текущего баланса
    auto_pairs_n = _pairs_count_for_balance(start_balance, AUTO_PAIRS_RULE)

    # стартовая вселенная из конфига → фильтр по linear → фильтр по нотационалу
    base_universe = getattr(cfg, "TOP_LIQUID_PAIRS", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    base_universe = _filter_by_linear_availability(base_universe)
    base_universe = _filter_universe_by_notional(base_universe, start_balance)

    # скоринг и выборка top-N
    top_pairs = select_top_pairs(base_universe, count=auto_pairs_n)
    exploration_set = set(_LAST_PAIR_SELECTION.get("explore", []))
    _session_update_meta({"pairs": top_pairs})
    SESSION_ACCOUNT_SNAPSHOT["pairs"] = list(top_pairs)

    mode_label = "PAPER" if PAPER_MODE else "REAL"
    tg_send(f"🟢 Старт [{mode_label}] Пары: {top_pairs}\nБаланс: {start_balance:.2f} USDT\nSAFE_MODE={int(SAFE_MODE)}")

    # состояния по символам
    entry: Dict[str, Dict[str, Any]] = {
        s: {"price": None, "side": None, "qty": None, "max_upnl": None, "trade_id": None, "exploration": False}
        for s in top_pairs
    }
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
                if not _filters_reliable(s):
                    log(f"[LEV] {s}: пропуск инициализации — нет достоверных фильтров")
                    continue

                if int(getattr(cfg, "ADAPTIVE_LEV_ENABLED", 1)):
                    lev = _compute_adaptive_leverage(s, float(broker.get_balance()), px, atr0, spr, mq, stp, mnot)
                else:
                    lev = int(getattr(cfg, "DEFAULT_LEVERAGE", 10))

                applied = _set_leverage_with_retry(s, int(lev), last_lev_set.get(s))
                if applied:
                    last_lev_set[s] = int(lev)
                    last_lev_check_ts[s] = time.time()
                    log(f"[LEV] {s}: {int(lev)}x")
                else:
                    log(f"[ℹ️] set_leverage {s}: не удалось применить {int(lev)}x")
            except Exception as e:
                log(f"[LEV] {s}: {e}")

    # грузим модель
    model, meta = load_model_and_meta()
    ml_trading_enabled = bool(model and meta)
    if not ml_trading_enabled:
        log("[ML] Торговые сигналы приостановлены (см. статус ML).", level="WARNING")

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

    SPREAD_MAX_PCT      = float(getattr(cfg, "SPREAD_MAX_PCT", 0.0012))
    SPREAD_DEPTH        = int(getattr(cfg, "SPREAD_DEPTH", 1))
    RISK_PER_TRADE_FRAC = float(getattr(cfg, "RISK_PER_TRADE_FRAC", 0.0065))
    ATR_STOP_K          = float(getattr(cfg, "ATR_STOP_K", 3.5))

    # --- расписание / уведомления ---
    force_on_schedule = False
    last_sched_state: Optional[bool] = None
    SCHED_MANAGE_OPEN = utils.env_bool(
        "SCHEDULE_MANAGE_OPEN_POSITIONS",
        bool(getattr(cfg, "SCHEDULE_MANAGE_OPEN_POSITIONS", 1)),
    )

    def _read_control(clear_after: bool = False) -> List[Dict[str, Any]]:
        cmds: List[Dict[str, Any]] = []
        data: Any = []
        try:
            if os.path.exists(control_path):
                with open(control_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    ts_epoch = _control_parse_ts(item.get("ts"))
                    if ts_epoch is None:
                        continue
                    if not _control_is_fresh(ts_epoch):
                        continue
                    cmd = dict(item)
                    cmd["_ts_epoch"] = ts_epoch
                    cmds.append(cmd)
        except Exception:
            cmds = []
        if clear_after:
            try:
                tmp = Path(control_path + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump([], f)
                os.replace(tmp, control_path)
            except Exception:
                try:
                    if 'tmp' in locals() and tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
        return cmds

    def _process_control(cmds: List[Dict[str, Any]]):
        nonlocal top_pairs, entry, last_entry_time, last_lev_set, last_lev_check_ts, control_last_ts, force_on_schedule
        nonlocal session_reason, should_exit, panic_requested
        global PAUSE_ENTRIES
        for c in cmds:
            ts_epoch = c.pop("_ts_epoch", None)
            ts_raw = c.get("ts")
            if ts_epoch is None:
                ts_epoch = _control_parse_ts(ts_raw)
            if ts_epoch is None:
                continue
            if not _control_is_fresh(ts_epoch):
                continue
            cmd_id = str(c.get("cmd_id") or "").strip()
            if not cmd_id:
                try:
                    fingerprint = json.dumps(c, sort_keys=True, ensure_ascii=False)
                except Exception:
                    fingerprint = str(c)
                cmd_id = f"legacy:{ts_epoch}:{hash(fingerprint)}"
            if cmd_id in CONTROL_PROCESSED_IDS:
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

            cmd_name = str(c.get("cmd") or "").strip()

            if cmd_name == "panic_stop" or c.get("panic_close") or c.get("panic"):
                log("[CTRL] panic request received")
                tg_send("🛑 CTRL: panic")
                SESSION_STATE["shutdown_requested"] = True
                session_reason = "panic"
                panic_requested = True
                PAUSE_ENTRIES = True
                try:
                    if bool(c.get("close_all", True)) and hasattr(broker, "close_all_positions"):
                        broker.close_all_positions()
                    elif bool(c.get("close_all", True)):
                        broker.force_close_all_positions_absolute()
                except Exception as e:
                    log(f"[CTRL] panic close error: {e}")
                should_exit = True

            if c.get("stop"):
                log("[CTRL] stop requested")
                SESSION_STATE["shutdown_requested"] = True
                session_reason = "stop"
                should_exit = True
                PAUSE_ENTRIES = True

            if c.get("close_all") and not (c.get("panic") or c.get("panic_close")):
                try:
                    if hasattr(broker, "close_all_positions"):
                        broker.close_all_positions()
                    else:
                        broker.force_close_all_positions_absolute()
                    log("[CTRL] close_all processed")
                except Exception as e:
                    log(f"[CTRL] close_all error: {e}")

            if "set_pairs" in c:
                req_pairs = [str(s) for s in (c.get("set_pairs") or []) if s]
                if req_pairs:
                    try:
                        base = _filter_by_linear_availability(req_pairs)
                        cur_bal = float(broker.get_balance())
                        base = _filter_universe_by_notional(base, cur_bal)
                        if base:
                            top_pairs = select_top_pairs(base, count=len(base))
                            exploration_set = set(_LAST_PAIR_SELECTION.get("explore", []))
                            log(msg("PAIRS_CTRL_UPDATE", pairs=", ".join(top_pairs)))
                            _session_update_meta({"pairs": top_pairs})
                            SESSION_ACCOUNT_SNAPSHOT["pairs"] = list(top_pairs)
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

            CONTROL_PROCESSED_IDS.add(cmd_id)
            control_last_ts = max(control_last_ts or ts_epoch, ts_epoch)

    try:
        initial_cmds = _read_control(clear_after=True)
        if initial_cmds:
            _process_control(initial_cmds)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log(f"[CTRL] {e}")

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
            if should_exit:
                break

            if now - last_margin_poll >= MARGIN_POLL_SEC:
                last_margin_poll = now
                info = {}
                try:
                    info = broker.get_margin_info()
                except Exception as margin_exc:
                    log(f"[MARGIN] info error: {margin_exc}")
                    info = {}
                im_pct = 0.0
                mm_pct = 0.0
                equity_snapshot = 0.0
                if isinstance(info, dict):
                    try:
                        im_pct = float(info.get("IM") or info.get("im_pct") or 0.0)
                    except Exception:
                        im_pct = 0.0
                    try:
                        mm_pct = float(info.get("MM") or info.get("mm_pct") or 0.0)
                    except Exception:
                        mm_pct = 0.0
                    try:
                        equity_snapshot = float(info.get("equity") or 0.0)
                    except Exception:
                        equity_snapshot = 0.0
                if equity_snapshot <= 0:
                    try:
                        equity_snapshot = float(getattr(broker, "get_equity", broker.get_balance)())
                    except Exception:
                        equity_snapshot = 0.0

                frozen = im_pct >= MAX_IM_PERCENT
                crit = im_pct >= CRIT_IM_PERCENT
                freeze_reason = "crit_margin" if crit else ("high_margin" if frozen else "")

                if frozen and not margin_freeze_active:
                    log(f"[MARGIN] FROZEN: high margin usage IM={im_pct:.2f}% (limit {MAX_IM_PERCENT:.2f}%)")
                if not frozen and margin_freeze_active:
                    log(f"[MARGIN] thaw: IM={im_pct:.2f}% < {MAX_IM_PERCENT:.2f}% — новые входы разрешены")

                if crit and not margin_crit_active:
                    log(f"[MARGIN] CRITICAL: IM={im_pct:.2f}% > {CRIT_IM_PERCENT:.2f}% — снижаю риск")
                if crit:
                    _mitigate_margin_pressure()
                elif margin_crit_active and not crit:
                    log(f"[MARGIN] критический уровень снят (IM={im_pct:.2f}%)")

                margin_freeze_active = frozen
                margin_crit_active = crit
                set_margin_state(im_pct, mm_pct, equity_snapshot, frozen, freeze_reason)

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
                        exploration_set = set(_LAST_PAIR_SELECTION.get("explore", []))
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
                        if not _filters_reliable(symbol):
                            log(f"[LEV] {symbol}: пропуск пересчёта — нет достоверных фильтров")
                            last_lev_check_ts[symbol] = time.time()
                            continue
                        lev_prev = last_lev_set.get(symbol)
                        bad_data = (atr_val <= 0) or (spread_rel > SPREAD_MAX_PCT * 4)
                        if bad_data:
                            fallback_lev = fallback_leverage(int(getattr(config, "DEFAULT_LEVERAGE", 10)), lev_prev)
                            if lev_prev != fallback_lev:
                                log(f"[LEV] {symbol}: fallback leverage → {fallback_lev}x (atr={atr_val:.6f}, spread={spread_rel:.5f})")
                                applied = _set_leverage_with_retry(symbol, fallback_lev, lev_prev)
                                if applied:
                                    last_lev_set[symbol] = fallback_lev
                                else:
                                    log(f"[ℹ️] set_leverage {symbol}: fallback {fallback_lev}x не применён")
                            last_lev_check_ts[symbol] = time.time()
                            continue

                        lev_new_raw = _compute_adaptive_leverage(symbol, float(broker.get_balance()), price, atr_val, spread_rel, mq, stp, mnot)
                        lev_new_raw = max(1, int(lev_new_raw))
                        lev_new, ramp_reason = apply_leverage_ramp(lev_prev, lev_new_raw, LEV_STEP_MAX)

                        if lev_prev is None or int(lev_prev) != lev_new:
                            applied = _set_leverage_with_retry(symbol, lev_new, lev_prev)
                            if applied:
                                last_lev_set[symbol] = int(lev_new)
                                last_lev_check_ts[symbol] = time.time()
                                extra = f", reason={ramp_reason}" if ramp_reason else ""
                                atr_pct_dbg = (atr_val / price) if price > 0 else 0.0
                                log(f"[LEV] {symbol}: {lev_prev or '-'} → {lev_new}x (atr%={atr_pct_dbg:.5f}, spread={spread_rel:.5f}{extra})")
                            else:
                                log(f"[ℹ️] set_leverage {symbol}: не удалось применить {lev_new}x")
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
                                _log_bot_trade(
                                    symbol,
                                    "close",
                                    price=current,
                                    qty=qty,
                                    pnl=pnl - commission,
                                    trade_id=ent.get("trade_id"),
                                    exploration=ent.get("exploration", False),
                                    meta={"reason": "dynamic_tp"},
                                )
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
                            register_trade_result(pnl - commission, commission)
                            entry[symbol] = {
                                "price": None,
                                "side": None,
                                "qty": None,
                                "max_upnl": None,
                                "trade_id": None,
                                "exploration": False,
                            }
                            continue

                        # Если уже был профит > комиссий, а вернулись ниже комиссий — выходим
                        if pnl < commission and (ent["max_upnl"] is not None) and (ent["max_upnl"] > commission):
                            if DO_TRADE:
                                broker.close_position_by_market(symbol, qty)
                                _log_bot_trade(
                                    symbol,
                                    "close",
                                    price=current,
                                    qty=qty,
                                    pnl=pnl - commission,
                                    trade_id=ent.get("trade_id"),
                                    exploration=ent.get("exploration", False),
                                    meta={"reason": "no_profit"},
                                )
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
                            register_trade_result(pnl - commission, commission)
                            entry[symbol] = {
                                "price": None,
                                "side": None,
                                "qty": None,
                                "max_upnl": None,
                                "trade_id": None,
                                "exploration": False,
                            }
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

                if not ml_trading_enabled:
                    continue

                # Баланс/лимиты
                try:
                    avail = float(getattr(broker, "get_available_balance", broker.get_balance)())
                except Exception:
                    avail = float(broker.get_balance())
                if avail <= 0:
                    log(f"[SKIP] {symbol}: no_balance")
                    continue

                min_qty, step, min_notional = _safe_order_filters(symbol)
                if not _filters_reliable(symbol):
                    log(f"[SKIP] {symbol}: нет достоверных фильтров, вход запрещён")
                    continue

                margin_snapshot = get_margin_state()
                try:
                    equity_snapshot = float(margin_snapshot.get("equity", 0.0) or 0.0)
                except Exception:
                    equity_snapshot = 0.0
                if equity_snapshot <= 0:
                    equity_snapshot = avail

                atr_rel = (atr_val / max(price, 1e-9)) if price > 0 else 0.0
                if atr_rel <= 0:
                    log(f"[SKIP] {symbol}: atr_invalid atr={atr_val:.6f}")
                    continue

                risk_frac_cfg = float(getattr(cfg, "RISK_PER_TRADE_FRAC", 0.0065))
                risk_frac = max(0.0, min(risk_frac_cfg, kelly_fraction))
                atr_stop_k = max(float(getattr(cfg, "ATR_STOP_K", 1.2)), 1e-6)
                min_share = float(getattr(cfg, "MIN_SHARE", 0.001))
                max_share = float(getattr(cfg, "MAX_SHARE", float(getattr(cfg, "MAX_BALANCE_SHARE", 0.1))))

                denom = atr_rel * atr_stop_k * max(price, 1e-9)
                if denom <= 0:
                    log(f"[SKIP] {symbol}: denom_invalid denom={denom}")
                    continue

                share_raw = (risk_frac * equity_snapshot) / denom
                share_clamped = clamp(share_raw, min_share, max_share)
                penalty_alpha = float(getattr(cfg, "SPREAD_PENALTY_ALPHA", 1.0))
                spread_adj = spread_penalty(spread_rel, SPREAD_MAX_PCT, alpha=penalty_alpha)
                share_effective = max(0.0, share_clamped * spread_adj)
                notional_target = share_effective * equity_snapshot
                if notional_target <= 0:
                    log(f"[SKIP] {symbol}: notional_target<=0")
                    continue

                qty_target = notional_target / max(price, 1e-9)
                if qty_target <= 0:
                    log(f"[SKIP] {symbol}: qty_target<=0")
                    continue

                precheck = pre_trade_check(symbol, price, qty_target, spread=spread_rel, margin_state=margin_snapshot)
                if not precheck.get("ok"):
                    why = str(precheck.get("why") or "")
                    code = why.split(":", 1)[0]
                    human = _PRECHECK_REASONS.get(code, why)
                    log(f"[SKIP] {symbol}: {human}")
                    continue

                qty = float(precheck.get("qty") or 0.0)
                if qty <= 0:
                    log(f"[SKIP] {symbol}: qty_adjusted<=0")
                    continue

                # ML-фильтр (как раньше): считаем proba уже зная направление
                direction = "long" if signal == "buy" else "short"
                ml_result = predict_ok(
                    model, meta, symbol, direction, qty,
                    price=price, atr=atr_val, candles=ohlcv
                )
                proba = ml_result.proba
                thr = ml_result.threshold
                size_factor = ml_result.factor
                conf_band = ml_result.band
                ml_mode_current = int(getattr(cfg, "ML_USE_NEW_ON", getattr(config, "ML_USE_NEW_ON", 0)))
                apply_new_ml = ml_trading_enabled and ml_mode_current > 0
                if apply_new_ml and ml_mode_current == 1 and symbol not in exploration_set:
                    apply_new_ml = False
                    if symbol not in ml_skip_notice:
                        log(
                            f"[ML] {symbol}: пропускаем применение новой модели (core-пара, ML_USE_NEW_ON=1)")
                        ml_skip_notice.add(symbol)
                elif ml_trading_enabled and ml_mode_current == 0 and symbol not in ml_skip_notice:
                    log(
                        f"[ML] {symbol}: ML_USE_NEW_ON=0 → новая модель в теневом режиме (решения не применяются)")
                    ml_skip_notice.add(symbol)

                ml_veto_enabled = apply_new_ml and int(getattr(cfg, "ML_VETO_ENABLED", 1))
                if ml_veto_enabled:
                    veto_thr = float(getattr(cfg, "ML_VETO_THR", 0.35))
                    if proba < veto_thr:
                        if apply_new_ml and not ml_shadow_enabled:
                                log_ml_decision(
                                symbol,
                                direction=direction,
                                side="skip",
                                proba=proba,
                                factor=size_factor,
                                band=conf_band,
                                meta_threshold=ml_result.meta_threshold,
                                strict_threshold=ml_result.strict_threshold,
                                effective_threshold=ml_result.effective_threshold,
                                features_ok=ml_result.features_ok,
                            )
                        main
                        if int(getattr(cfg, "ML_VETO_LOG", 1)):
                            log(f"[ML-VETO] {symbol}: veto (prob={proba:.3f} < veto_thr={veto_thr:.3f}); router={router_reason}")
                        continue

# >>> FIX: ML decision & logging (no-shadow)
                        if apply_new_ml and not ml_shadow_enabled:
                            decision_side = "buy" if direction == "long" else "sell"

                        if not ml_result.ok:
        # Логируем пропуск с причинами и выходим к следующей паре
                            _log_ml_decision(
                                symbol,
                                direction=direction,
                                side="skip",
                                proba=proba,
                                factor=size_factor,
                                band=conf_band,
            meta_threshold=ml_result.meta_threshold,
            strict_threshold=ml_result.strict_threshold,
            effective_threshold=ml_result.effective_threshold,
            features_ok=ml_result.features_ok,
        )

                        if conf_band == "blocked":
                            mid_thr = float(getattr(cfg, "ML_CONF_MID", getattr(config, "ML_CONF_MID", 0.65)))
                            log(f"[ML] {symbol}: veto (prob={proba:.3f} < min={mid_thr:.3f}); router={router_reason}")
                        elif conf_band == "unavailable":
                            log(f"[ML] {symbol}: пропуск — ML недоступна; router={router_reason}")
                        elif conf_band == "error":
                            log(f"[ML] {symbol}: predict_err; router={router_reason}")
                        else:
                            log(f"[ML] {symbol}: отказ (prob={proba:.3f} < thr={thr:.3f}); router={router_reason}")
                        continue

    # OK — логируем принятое решение и идём дальше по обычному потоку
                        _log_ml_decision(
                            symbol,
                            direction=direction,
                            side=decision_side,
                            proba=proba,
                            factor=size_factor,
                            band=conf_band,
                            meta_threshold=ml_result.meta_threshold,
                            strict_threshold=ml_result.strict_threshold,
                            effective_threshold=ml_result.effective_threshold,
                            features_ok=ml_result.features_ok,
                        )

# <<< FIX
                if not apply_new_ml:
                    size_factor = 1.0
                if apply_new_ml and size_factor < 1.0:
                    scaled_qty = qty * max(size_factor, 0.0)
                    if scaled_qty <= 0:
                        log(f"[ML] {symbol}: после масштабирования qty<=0")
                        continue
                    scaled_precheck = pre_trade_check(
                        symbol, price, scaled_qty,
                        spread=spread_rel,
                        margin_state=margin_snapshot,
                    )
                    if not scaled_precheck.get("ok"):
                        why = str(scaled_precheck.get("why") or "")
                        code = why.split(":", 1)[0]
                        human = _PRECHECK_REASONS.get(code, why)
                        log(f"[ML] {symbol}: отказ после уменьшения объёма ({human})")
                        continue
                    qty_scaled = float(scaled_precheck.get("qty") or 0.0)
                    if qty_scaled <= 0:
                        log(f"[ML] {symbol}: qty после масштабирования <= 0")
                        continue
                    qty = qty_scaled
                    log(f"[ML] {symbol}: confidence {proba:.3f} → размер {size_factor*100:.0f}% (qty={qty:.6f})")

                notional = price * qty
                min_notional_usdt = float(getattr(cfg, "MIN_NOTIONAL_USDT", 5.0))
                if notional < max(min_notional, min_notional_usdt):
                    log(
                        f"[SKIP] {symbol}: notional too small ({notional:.4f} < {max(min_notional, min_notional_usdt):.4f})"
                    )
                    continue

                max_balance_share = max(0.0, min(1.0, float(getattr(cfg, "MAX_BALANCE_SHARE", 0.08))))
                if notional > avail * max_balance_share:
                    log(
                        f"[SKIP] {symbol}: notional {notional:.4f} > balance_share_cap {(avail * max_balance_share):.4f}"
                    )
                    continue

                hard_cap_notional = avail * max(0.0, min(1.0, float(getattr(cfg, "HARD_CAP_SHARE", 0.25))))
                if notional > hard_cap_notional:
                    log(f"[SKIP] {symbol}: notional {notional:.4f} > hard_cap {hard_cap_notional:.4f}")
                    continue

                useable_cap = avail * max(0.0, min(1.0, float(getattr(cfg, "USEABLE_BAL_SHARE", 0.95))))
                if notional > useable_cap:
                    log(f"[SKIP] {symbol}: notional {notional:.4f} > useable_cap {useable_cap:.4f}")
                    continue

                side = "Buy" if direction == "long" else "Sell"
                exploration_flag = symbol in exploration_set

                existing_side = positions_side.get(symbol)
                if existing_side and existing_side == side:
                    log(msg("ENTRY_SKIP_OPEN_POS", symbol=symbol))
                    last_entry_time[symbol] = now
                    continue

                # === ОТКРЫТИЕ СДЕЛКИ ===
                if DO_TRADE:
                    if broker.place_market_order(symbol, side, qty):
                        trade_meta = {
                            "router_reason": router_reason,
                            "router_strategy": (router_meta or {}).get("strategy"),
                            "router_regime": (router_meta or {}).get("regime"),
                            "router_sl": router_sl,
                            "router_tp": router_tp,
                            "paper": bool(PAPER_MODE),
                        }
                        trade_id = _log_bot_trade(
                            symbol,
                            "buy" if side == "Buy" else "sell",
                            price=price,
                            qty=qty,
                            pnl=0.0,
                            exploration=exploration_flag,
                            meta=trade_meta,
                        )
                        entry[symbol] = {
                            "price": price,
                            "side": side,
                            "qty": qty,
                            "max_upnl": None,
                            "trade_id": trade_id,
                            "exploration": exploration_flag,
                        }
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
                            "ml_factor": round(size_factor, 3),
                            "trade_id": trade_id,
                            "exploration": exploration_flag,
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
                        "ml_factor": round(size_factor, 3),
                    })

            time.sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        SESSION_STATE["shutdown_requested"] = True
        if panic_requested:
            session_reason = "panic"
        elif session_reason == "normal":
            session_reason = "stop"
        log("[🛑] Ctrl+C — закрываю все позиции…")
        tg_send("🛑 Ручная остановка. Закрываю все позиции.")
        try:
            if DO_TRADE:
                broker.force_close_all_positions_absolute()
        except Exception as e:
            log(f"[❌] force close: {e}")
    except Exception as e:
        session_reason = "error"
        log(f"[FATAL] main: {e}")
        tg_send(f"❌ Критическая ошибка: {e}")
        raise
    finally:
        trades_rows: List[Dict[str, Any]] = []
        pnl_list: List[float] = []
        equity_points: List[Tuple[str, float]] = []
        summary_payload: Dict[str, Any] = {}
        stats_payload: Dict[str, Any] = {}

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

        try:
            trades_total_log = cnt_p + cnt_n
            winrate = (cnt_p / trades_total_log) * 100.0 if trades_total_log else 0.0
            pnl_avg = (total_p + total_n) / trades_total_log if trades_total_log else 0.0
            sharpe = 0.0
            if len(pnl_list) > 1:
                try:
                    mean_p = statistics.mean(pnl_list)
                    std_p = statistics.stdev(pnl_list)
                    if std_p > 1e-12:
                        sharpe = (mean_p / std_p) * math.sqrt(len(pnl_list))
                except Exception:
                    sharpe = 0.0

            runtime_trades = int(SESSION_RUNTIME_STATS.get("trades_total", 0))
            runtime_fees = float(SESSION_RUNTIME_STATS.get("fees_accum", 0.0))
            runtime_pnl = float(SESSION_RUNTIME_STATS.get("pnl_accum", 0.0))
            trades_total = runtime_trades or trades_total_log
            realized_pnl = runtime_pnl if runtime_trades else (total_p + total_n)
            realized_pnl = round(realized_pnl, 8)
            fees_total = runtime_fees if runtime_trades else None
            if fees_total is not None:
                fees_total = round(fees_total, 8)

            try:
                end_equity = float(getattr(broker, "get_equity", broker.get_balance)())
            except Exception as e:
                log(f"[BAL-END] equity: {e}")
                end_equity = end_balance

            delta_balance = round(end_balance - start_balance, 8)
            max_dd_pct = None
            if start_balance > 0:
                try:
                    max_dd_pct = round((max_drawdown_usdt / start_balance) * 100.0, 6)
                except Exception:
                    max_dd_pct = None

            end_ts_iso = utcnow_iso()
            uptime_sec = int(duration_sec)

            summary_payload = {
                "ts_start": SESSION_ACCOUNT_SNAPSHOT.get("ts_start"),
                "ts_end": end_ts_iso,
                "uptime_sec": uptime_sec,
                "mode": SESSION_ACCOUNT_SNAPSHOT.get("mode", ("paper" if PAPER_MODE else "real")),
                "reason": session_reason,
                "pairs": SESSION_ACCOUNT_SNAPSHOT.get("pairs") or (SESSION_STATE.get("meta") or {}).get("pairs"),
                "trades_total": trades_total,
                "pnl_total": realized_pnl,
                "start_balance": start_balance,
                "end_balance": end_balance,
                "delta_balance": delta_balance,
                "fees_total": fees_total,
                "max_drawdown_pct": max_dd_pct,
                "start_equity": SESSION_ACCOUNT_SNAPSHOT.get("start_equity"),
                "end_equity": end_equity,
                "safe_mode": 1 if SAFE_MODE else 0,
                "log_path": log_path,
            }

            try:
                write_cycle_log({"event": "session_end", **summary_payload})
            except Exception as summary_log_exc:
                log(f"[SUMMARY] cycle_log error: {summary_log_exc}")

            stats_payload = {
                "start_ts": summary_payload["ts_start"],
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
                "pairs": summary_payload.get("pairs"),
            }

            summary_lines = [
                f"🔴 Завершено [{mode_label}] — причина: {session_reason}",
                f"Δ Баланса: {delta_balance:+.4f} USDT",
                f"Сделки: {trades_total} (побед {cnt_p}, поражений {cnt_n}, winrate {winrate:.1f}%)",
                f"PnL (реализ.): {realized_pnl:+.4f} USDT",
                f"Комиссии: {fees_total:.6f} USDT" if fees_total is not None else "Комиссии: н/д",
                f"Длительность: {duration_human}",
            ]
            tg_send("\n".join(summary_lines))
            log(
                f"[STATS] Итог: {delta_balance:+.4f} USDT; +{cnt_p} / -{cnt_n}; "
                f"duration={duration_human}; trades={trades_total}; maxDD={max_drawdown_usdt:+.2f}; "
                f"winrate={winrate:.1f}%"
            )

            try:
                summary_dir = os.path.dirname(
                    SESSION_STATE.get("log_jsonl")
                    or os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))
                )
                summary_paths = write_session_summary(summary_dir or ".", summary_payload)
                if summary_paths:
                    log(f"[SUMMARY] saved: {summary_paths}")
            except Exception as summary_exc:
                log(f"[SUMMARY] persist error: {summary_exc}")

        except Exception as summary_exc:
            log(f"[SUMMARY] critical: {summary_exc}")
            delta_balance = round(end_balance - start_balance, 8)
            summary_payload = {
                "ts_start": SESSION_ACCOUNT_SNAPSHOT.get("ts_start"),
                "ts_end": utcnow_iso(),
                "uptime_sec": int(duration_sec),
                "mode": SESSION_ACCOUNT_SNAPSHOT.get("mode", ("paper" if PAPER_MODE else "real")),
                "reason": "error",
                "pairs": SESSION_ACCOUNT_SNAPSHOT.get("pairs") or [],
                "trades_total": int(SESSION_RUNTIME_STATS.get("trades_total", 0)),
                "pnl_total": float(SESSION_RUNTIME_STATS.get("pnl_accum", 0.0)),
                "start_balance": start_balance,
                "end_balance": end_balance,
                "delta_balance": delta_balance,
                "fees_total": SESSION_RUNTIME_STATS.get("fees_accum"),
                "start_equity": SESSION_ACCOUNT_SNAPSHOT.get("start_equity"),
                "end_equity": None,
                "safe_mode": 1 if SAFE_MODE else 0,
                "log_path": log_path,
            }
            try:
                write_cycle_log({"event": "session_end", **summary_payload})
            except Exception as inner_exc:
                log(f"[SUMMARY] fallback cycle_log error: {inner_exc}")
            try:
                summary_dir = os.path.dirname(
                    SESSION_STATE.get("log_jsonl")
                    or os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))
                )
                write_session_summary(summary_dir or ".", summary_payload)
            except Exception as fallback_exc:
                log(f"[SUMMARY] fallback persist error: {fallback_exc}")
            stats_payload = {
                "start_ts": summary_payload.get("ts_start"),
                "end_ts": summary_payload.get("ts_end"),
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
                "trades": summary_payload.get("trades_total"),
                "winrate": 0.0,
                "pnl_per_trade": 0.0,
                "max_drawdown": max_drawdown_usdt,
                "max_dd": max_drawdown_usdt,
                "sharpe": 0.0,
                "kelly_trades": kelly_count,
                "log_path": log_path,
                "pairs": summary_payload.get("pairs"),
            }

        _session_finalize(stats_payload, trades_rows)

if __name__ == "__main__":
    main_trading_cycle()