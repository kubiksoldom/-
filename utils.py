# -*- coding: utf-8 -*-
# utils.py
"""
Общие утилиты:
- логирование в консоль и jsonl (потокобезопасная запись)
- отправка уведомлений в Telegram (чанкинг, мягкие ретраи, без спама при отсутствии токена)
- генерация orderLinkId
- округления под шаги биржи (Decimal, без флоат-дрейфа)
- безопасная проверка min_qty / qty_step / min_notional (защита от None)
- хелперы для чисел/ноционала/клампов и доступности минимального ордера

Совместимо со старой сигнатурой; экспортирует SAFE_MODE как глобальный флаг.
"""

from __future__ import annotations

import os
import csv
import json
import math
import time
import random
import config
import threading
import hashlib
import hmac
import socket
from collections import deque
from decimal import Decimal, ROUND_DOWN, getcontext, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from contextlib import closing

# ---- опциональные зависимости: НЕ требуем, но используем если есть ----
try:  # pragma: no cover - optional dependency
    import netifaces as _netifaces  # type: ignore
except Exception:  # pragma: no cover
    _netifaces = None

try:  # pragma: no cover - optional dependency
    import psutil as _psutil  # type: ignore
except Exception:  # pragma: no cover
    _psutil = None

import requests
from requests.adapters import HTTPAdapter

from env_loader import load_env

# ---------- CONFIG/ENV WITH FALLBACK ----------
load_env()

def _cfg_get(name: str, env_key: str, default: str) -> str:
    """Пытаемся взять из config.py, иначе из ENV, иначе default."""
    try:
        import config as _cfg  # type: ignore
    except Exception:
        _cfg = None
    if _cfg is not None and hasattr(_cfg, name):
        try:
            v = getattr(_cfg, name)
            return str(v)
        except Exception:
            pass
    return os.getenv(env_key, default)

LOG_JSONL      = _cfg_get("LOG_JSONL", "LOG_JSONL", "bot_cycle_log.jsonl")
LOG_ENABLED    = bool(int(_cfg_get("LOG_ENABLED", "LOG_ENABLED", "1")))
SAFE_MODE      = bool(int(_cfg_get("SAFE_MODE", "SAFE_MODE", "0")))

_TRUE_STRINGS = {"1", "true", "yes", "y", "on", "t"}
_FALSE_STRINGS = {"0", "false", "no", "n", "off", "f"}

TELEGRAM_TOKEN   = _cfg_get("TELEGRAM_TOKEN", "TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = _cfg_get("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID", "")

# сразу под TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
try:
    from . import config as _cfg  # type: ignore  # если есть локальный config
except Exception:
    _cfg = None


def _mask(s, keep=3):
    s = str(s or "")
    return (s[:keep] + "***" + s[-keep:]) if len(s) > keep * 2 else (("*" * len(s)) if s else "<empty>")


# Для Decimal — достаточно 28 знаков, чтобы уверенно резать шаги количества/цены
getcontext().prec = 28

# Общие локи
_LOG_LOCK = threading.Lock()
_LOG_DIR_LOCK = threading.Lock()
_LOG_PATH = Path(LOG_JSONL).expanduser()
_LOG_DIR_READY = False
_TG_WARN_LOCK = threading.Lock()
_TG_WARNED_NO_CREDS = False  # чтобы не спамить предупреждением на каждый вызов
_TG_SESSION: Optional[requests.Session] = None
_TG_SESSION_LOCK = threading.Lock()

_MARGIN_STATE: Dict[str, Any] = {
    "im_pct": 0.0,
    "mm_pct": 0.0,
    "equity": 0.0,
    "frozen": False,
    "reason": "",
    "updated": 0.0,
}
_MARGIN_STATUS_PATH = Path("logs/metrics/margin_status.json")

_ML_STATE: Dict[str, Any] = {
    "status": "unknown",
    "paused": True,
    "reason": "",
    "precision_week": None,
    "threshold": None,
    "updated": 0.0,
}
_ML_STATUS_PATH = Path("logs/metrics/ml_status.json")


# ---------- HELPERS ----------
def _utc_iso() -> str:
    """Возвращает UTC ISO8601 с 'Z' в конце (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def now_iso(tz_aware: bool = False) -> str:
    """Возвращает время в UTC ISO8601 с "Z"."""
    # Параметр tz_aware сохранён для обратной совместимости, но больше не влияет на результат.
    if tz_aware:
        pass
    try:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except Exception:
        return _utc_iso()

def _ensure_utf8(text: str) -> str:
    try:
        return str(text).encode("utf-8", "replace").decode("utf-8")
    except Exception:
        return str(text)


def _resolve_log_path() -> Path:
    candidate = str(LOG_JSONL or "").strip()
    if not candidate:
        candidate = "bot_cycle_log.jsonl"
    try:
        return Path(candidate).expanduser()
    except Exception:
        return Path("bot_cycle_log.jsonl")


def _ensure_log_directory() -> Path:
    global _LOG_PATH, _LOG_DIR_READY
    desired = _resolve_log_path()
    if desired != _LOG_PATH:
        with _LOG_DIR_LOCK:
            if desired != _LOG_PATH:
                _LOG_PATH = desired
                _LOG_DIR_READY = False
    if _LOG_DIR_READY:
        return _LOG_PATH
    with _LOG_DIR_LOCK:
        desired = _resolve_log_path()
        if desired != _LOG_PATH:
            _LOG_PATH = desired
            _LOG_DIR_READY = False
        if not _LOG_DIR_READY:
            try:
                _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                _LOG_DIR_READY = True
            except Exception as e:
                log(f"[LOG] mkdir: {e}", level="ERROR")
        return _LOG_PATH


def log(message: str, level: str = "INFO"):
    """
    Простое консольное логирование. Без падения.
    Используется по всему проекту и в api_guard (level="ERROR"/"WARNING"/"INFO").
    """
    t = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    payload = _ensure_utf8(message)
    try:
        print(f"[{t}] [{level}] {payload}", flush=True)
    except Exception:
        # в крайних случаях, когда stdout сломан
        pass

# при импорте utils один раз отметим, что видим
if TELEGRAM_TOKEN or TELEGRAM_CHAT_ID:
    log(f"[TG] creds present: token={_mask(TELEGRAM_TOKEN)} chat={_mask(TELEGRAM_CHAT_ID)}")


def write_cycle_log(data: Dict[str, Any]):
    """
    Пишем одну строку JSONL. Создаём папку при необходимости.
    Всегда проставляем UTC timestamp (ts_utc) поверх входных данных.
    """
    if not LOG_ENABLED:
        return
    path = _ensure_log_directory()

    try:
        try:
            line = dict(data)
        except Exception:
            line = {"payload": data}
        line["ts_utc"] = _utc_iso()
        serialized = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        log(f"[LOG] подготовка строки: {e}", level="ERROR")
        return

    try:
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(serialized + "\n")
                fh.flush()
    except Exception as e:
        log(f"[LOG] ошибка записи: {e}", level="ERROR")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        log(f"[MARGIN-STATE] write error: {exc}", level="ERROR")


def set_margin_state(im_pct: float,
                     mm_pct: float,
                     equity: float = 0.0,
                     frozen: bool = False,
                     reason: str = "") -> None:
    state = {
        "im_pct": float(im_pct or 0.0),
        "mm_pct": float(mm_pct or 0.0),
        "equity": float(equity or 0.0),
        "frozen": bool(frozen),
        "reason": str(reason or ""),
        "updated": time.time(),
    }
    _MARGIN_STATE.update(state)
    try:
        _atomic_write_json(_MARGIN_STATUS_PATH, state)
    except Exception:
        pass


def get_margin_state() -> Dict[str, Any]:
    return dict(_MARGIN_STATE)


def set_ml_status(status: str,
                  paused: bool,
                  *,
                  reason: str = "",
                  precision_week: Optional[float] = None,
                  threshold: Optional[float] = None) -> None:
    state = {
        "status": str(status or "unknown"),
        "paused": bool(paused),
        "reason": str(reason or ""),
        "precision_week": (float(precision_week) if precision_week is not None else None),
        "threshold": (float(threshold) if threshold is not None else None),
        "updated": time.time(),
    }
    _ML_STATE.update(state)
    try:
        _atomic_write_json(_ML_STATUS_PATH, state)
    except Exception:
        pass


def _normalize_trade_side(raw: Any) -> str:
    side = str(raw or "").strip().lower()
    if side in {"buy", "long", "longs"}:
        return "long"
    if side in {"sell", "short", "shorts"}:
        return "short"
    raise ValueError(f"unknown trade side: {raw!r}")


def _normalize_trade_status(raw: Any) -> str:
    status = str(raw or "").strip().lower()
    mapping = {
        "open": "open",
        "opened": "open",
        "partial": "partial",
        "partially_filled": "partial",
        "close": "closed",
        "closed": "closed",
        "filled": "closed",
        "cancel": "canceled",
        "cancelled": "canceled",
        "canceled": "canceled",
        "error": "error",
        "failed": "error",
    }
    if status in mapping:
        return mapping[status]
    raise ValueError(f"unknown trade status: {raw!r}")


def append_trade_event(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalises and writes a trade event to the cycle log."""

    if not isinstance(event_dict, dict):
        raise TypeError("event_dict must be a dict")

    payload = dict(event_dict)
    payload["event"] = "trade"
    payload.setdefault("ts", _utc_iso())

    payload["side"] = _normalize_trade_side(payload.get("side"))
    payload["status"] = _normalize_trade_status(payload.get("status"))

    def _float_or_none(key: str) -> Optional[float]:
        value = payload.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"trade event field '{key}' must be numeric") from None

    for numeric_key in ("qty", "price", "fee", "realized_pnl"):
        val = _float_or_none(numeric_key)
        if val is not None:
            payload[numeric_key] = val
        else:
            payload.pop(numeric_key, None)

    for text_key in ("symbol", "order_id", "client_id", "source", "note", "position_id"):
        if payload.get(text_key) is not None:
            payload[text_key] = _ensure_utf8(str(payload[text_key]))

    if not payload.get("symbol"):
        raise ValueError("trade event requires 'symbol'")
    if not payload.get("order_id"):
        raise ValueError("trade event requires 'order_id'")
    payload.setdefault("client_id", "")
    payload.setdefault("source", "MANUAL")
    payload.setdefault("note", "")

    write_cycle_log(payload)
    return payload


def _format_uptime(uptime_sec: int) -> str:
    uptime_sec = max(int(uptime_sec or 0), 0)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_session_summary(dir_path: os.PathLike | str, summary: Dict[str, Any]) -> Dict[str, Path]:
    """
    Сохраняет JSON и TXT-сводку сессии. Возвращает пути к файлам.
    В случае ошибок пытается сохранить хотя бы JSON с минимальными данными.
    """

    base_dir = Path(dir_path or ".").expanduser()
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log(f"[SUMMARY] mkdir failed: {exc}", level="ERROR")

    ts_start = str(summary.get("ts_start") or _utc_iso())
    ts_end = str(summary.get("ts_end") or _utc_iso())

    def _stamp_from_iso(value: str) -> str:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%Y%m%d_%H%M%S")
        except Exception:
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    stamp = _stamp_from_iso(ts_start)
    json_path = base_dir / f"session_{stamp}.json"
    txt_path = base_dir / f"session_{stamp}.txt"

    safe_summary = dict(summary)
    safe_summary.setdefault("ts_start", ts_start)
    safe_summary.setdefault("ts_end", ts_end)

    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(safe_summary, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        log(f"[SUMMARY] json write failed: {exc}", level="ERROR")
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({"ts_start": ts_start, "ts_end": ts_end}, fh, ensure_ascii=False)
        except Exception as inner:
            log(f"[SUMMARY] fallback json failed: {inner}", level="ERROR")

    try:
        uptime_sec = int(safe_summary.get("uptime_sec") or 0)
        uptime_str = _format_uptime(uptime_sec)
        pairs = safe_summary.get("pairs") or []
        if isinstance(pairs, (list, tuple)):
            pairs_str = ", ".join(str(p) for p in pairs) or "—"
        else:
            pairs_str = str(pairs)
        trades_total = safe_summary.get("trades_total")
        delta_balance = safe_summary.get("delta_balance")
        pnl_total = safe_summary.get("pnl_total")
        fees_total = safe_summary.get("fees_total")
        max_dd = safe_summary.get("max_drawdown_pct")
        reason = safe_summary.get("reason") or "unknown"
        mode = safe_summary.get("mode") or "paper"
        start_balance = safe_summary.get("start_balance")
        end_balance = safe_summary.get("end_balance")

        lines = [
            f"Сессия: {ts_start} — {ts_end} (uptime {uptime_str})",
            f"Режим: {mode}, Пары: {pairs_str}",
            f"Сделок: {trades_total if trades_total is not None else '—'}",
            f"Баланс: {start_balance} → {end_balance}  (Δ = {delta_balance})",
            f"PnL (реализ.): {pnl_total}, Комиссии: {fees_total}",
            f"Макс. просадка: {max_dd if max_dd is not None else '—'}",
            f"Причина завершения: {reason}",
            "",
        ]
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_ensure_utf8(line) for line in lines))
    except Exception as exc:
        log(f"[SUMMARY] txt write failed: {exc}", level="ERROR")

    return {"json": json_path, "txt": txt_path}

def safe_read_jsonl(path: str, limit: int = 1000) -> list[dict]:
    """
    Безопасно читает .jsonl-файл (построчный JSON).
    Возвращает список словарей, максимум `limit` последних записей.
    Если файл не существует или повреждён — возвращает [].
    """
    try:
        limit_val = int(limit)
    except Exception:
        limit_val = 0
    maxlen = limit_val if limit_val > 0 else None
    items: deque[dict] = deque(maxlen=maxlen)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except Exception:
                    continue
        return list(items)
    except Exception:
        return []


def _get_tg_session() -> requests.Session:
    global _TG_SESSION
    session = _TG_SESSION
    if session is not None:
        return session
    with _TG_SESSION_LOCK:
        session = _TG_SESSION
        if session is None:
            sess = requests.Session()
            try:
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
                sess.mount("https://", adapter)
                sess.mount("http://", adapter)
            except Exception:
                pass
            _TG_SESSION = sess
            session = sess
    return session


def tg_send(text: str):
    """
    Отправка в Telegram. Если сообщение длинное — бьём на чанки.
    Возвращаемся молча, если токены не заданы (одно предупреждение при первом вызове).
    Делает мягкие ретраи при временных ошибках сети/5xx.
    """
    global _TG_WARNED_NO_CREDS
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        with _TG_WARN_LOCK:
            if not _TG_WARNED_NO_CREDS:
                log("[TG] Не задан TELEGRAM_TOKEN/TELEGRAM_CHAT_ID — пропускаю отправку.", level="WARNING")
                _TG_WARNED_NO_CREDS = True
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    max_len = 3800  # запас под HTML
    if not isinstance(text, str):
        text = str(text)
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]
    session = _get_tg_session()

    for part in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": part,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        tries = 0
        while True:
            tries += 1
            try:
                r = session.post(url, data=payload, timeout=12)
                if r.ok:
                    break
                # HTTP 429/5xx — мягкий бэкофф
                if r.status_code in (429, 500, 502, 503, 504):
                    log(f"[TG] HTTP {r.status_code}: {r.text[:200]} … попытка {tries}", level="WARNING")
                    time.sleep(min(8.0, 0.5 * (2 ** (tries - 1))))
                    if tries < 3:
                        continue
                else:
                    log(f"[TG] ошибка {r.status_code}: {r.text[:200]}", level="ERROR")
                break
            except Exception as e:
                if tries < 3:
                    log(f"[TG] исключение: {e} … попытка {tries}", level="WARNING")
                    time.sleep(min(8.0, 0.5 * (2 ** (tries - 1))))
                    continue
                log(f"[TG] исключение: {e}", level="ERROR")
                break


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return bool(default)
    s = raw.strip().lower()
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    return bool(s)


def verify_pin_hash(pin: str, pin_hash: str) -> bool:
    if not pin_hash:
        return False
    if pin is None:
        return False
    digest = hashlib.sha256(pin.encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, pin_hash.strip().lower())


def compute_sha256(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_port_free(port: int, host: str = "0.0.0.0") -> bool:
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def get_local_ips(include_loopback: bool = True) -> List[str]:
    """
    Возвращает список локальных IPv4.
    Приоритет источников: netifaces -> psutil -> socket fallback.
    Ничего критичного от внешних пакетов не зависит.
    """
    ips: set[str] = set()

    # 1) netifaces (если установлен)
    if _netifaces is not None:  # pragma: no cover - depends on environment
        try:
            for iface in _netifaces.interfaces():
                addrs = _netifaces.ifaddresses(iface).get(_netifaces.AF_INET, [])
                for entry in addrs:
                    addr = entry.get("addr")
                    if addr and not addr.startswith("127."):
                        ips.add(addr)
        except Exception:
            pass

    # 2) psutil (если установлен и ещё нет адресов)
    if not ips and _psutil is not None:  # pragma: no cover - depends on environment
        try:
            for _name, addrs in _psutil.net_if_addrs().items():
                for a in addrs:
                    # у psutil family может быть int/enum; берём через getattr
                    fam = getattr(a, "family", None)
                    if fam == socket.AddressFamily.AF_INET or fam == getattr(socket, "AF_INET", None):
                        ip = getattr(a, "address", None)
                        if ip and isinstance(ip, str) and not ip.startswith("127."):
                            ips.add(ip)
        except Exception:
            pass

    # 3) socket fallback (на случай отсутствия обоих пакетов)
    if not ips:
        try:
            host = socket.gethostname()
            addr = socket.gethostbyname(host)
            if addr and not addr.startswith("127."):
                ips.add(addr)
        except Exception:
            pass

    if include_loopback:
        ips.add("127.0.0.1")
    else:
        ips.discard("127.0.0.1")

    return sorted(ips)


def make_order_link_id(prefix: str = "BOT") -> str:
    """
    Генерим компактный orderLinkId для Bybit.
    Формат: <prefix>-<ms>-<rnd>, обрезаем до 32 символов.
    """
    base = f"{prefix}-{int(time.time()*1000)}-{random.randint(1000, 9999)}"
    return base[:32]

def ts_to_epoch(ts: str) -> float:
    """
    Преобразует строку timestamp (UTC ISO или "%Y-%m-%d %H:%M:%S") в UNIX-epoch (float).
    Возвращает 0.0 при ошибке.
    """
    if not ts:
        return 0.0
    try:
        # ISO-формат: 2025-10-12T12:34:56Z
        if "T" in ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        # fallback: обычный формат
        dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return 0.0


# ---------- ЧИСЛОВЫЕ ХЕЛПЕРЫ ----------
def _dec(value, default: str = "0"):
    """
    Безопасно конвертирует значение в Decimal через str(...).
    Если value None/NaN/Inf/пусто — вернёт Decimal(default).
    """
    try:
        if value is None:
            return Decimal(default)
        s = str(value)
        if s.lower() in ("nan", "inf", "-inf", "infinity", "-infinity", ""):
            return Decimal(default)
        return Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


def safe_float(value, default: float = 0.0) -> float:
    """Безопасная конвертация к float (None/NaN/Inf -> default)."""
    try:
        if value is None:
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# ---------- ОКРУГЛЕНИЯ ПОД ШАГИ БИРЖИ ----------
def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Округляет вниз к ближайшему шагу: floor(value/step) * step"""
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step


def adjust_qty(price, qty, min_qty=0.0, qty_step=0.0, min_notional=0.0) -> float:
    """
    Корректирует количество под шаг qty_step (Decimal).
    - floor к шагу qty_step
    - проверка min_qty (если >0)
    - проверка min_notional (цена*кол-во), если >0
    Возвращает скорректированное количество как float. Если нельзя — 0.0.
    """
    try:
        price_d = _dec(price)
        qty_d = _dec(qty)
        step_d = _dec(qty_step)
        min_qty_d = _dec(min_qty)
        min_notional_d = _dec(min_notional)

        if price_d <= 0 or qty_d <= 0:
            return 0.0

        q = qty_d
        if step_d > 0:
            q = _floor_to_step(q, step_d)

        if min_qty_d > 0 and q < min_qty_d:
            return 0.0

        if min_notional_d > 0 and (price_d * q) < min_notional_d:
            return 0.0

        return float(q)
    except Exception as e:
        log(f"[adjust_qty] ошибка: {e}", level="ERROR")
        return 0.0


def pre_trade_check(symbol: str,
                    price: float,
                    qty: float,
                    *,
                    spread: Optional[float] = None,
                    margin_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Проверки перед отправкой ордера: фильтры, спрэд, маржинальные лимиты."""

    result = {"ok": False, "qty": 0.0, "why": ""}

    try:
        sym = str(symbol or "").upper()
    except Exception:
        sym = str(symbol)

    price_f = safe_float(price, 0.0)
    qty_f = safe_float(qty, 0.0)
    if price_f <= 0 or qty_f <= 0:
        result["why"] = "bad_inputs"
        return result

    try:
        from bybit_api import get_min_order_filters, filters_reliable  # type: ignore
    except Exception as exc:
        log(f"[pre_trade_check] import error: {exc}", level="ERROR")
        result["why"] = "filters_unavailable"
        return result

    try:
        min_qty, qty_step, min_notional = get_min_order_filters(sym)
    except Exception as exc:
        result["why"] = f"filters_error:{exc}"[:120]
        return result

    reliable = True
    try:
        reliable = bool(filters_reliable(sym))
    except Exception:
        reliable = True
    if not reliable:
        result["why"] = "filters_unreliable"
        return result

    adj_qty = adjust_qty(price_f, qty_f, min_qty=min_qty, qty_step=qty_step, min_notional=(min_notional or 0.0))
    if adj_qty <= 0:
        result["why"] = "qty_adjust"
        return result

    commission_rate = safe_float(getattr(config, "COMMISSION_PER_SIDE", 0.0006), 0.0006)
    notional = price_f * adj_qty
    min_notional_val = safe_float(min_notional, 0.0)
    if min_notional_val > 0:
        effective_notional = notional * (1.0 + abs(commission_rate) * 2.0)
        if effective_notional < min_notional_val:
            result["why"] = "min_notional"
            return result

    spread_max = safe_float(getattr(config, "SPREAD_MAX_PCT", 0.0008), 0.0008)
    if spread is not None and safe_float(spread, 0.0) > spread_max:
        result["why"] = "spread"
        return result

    state = margin_state or get_margin_state()
    im_pct = safe_float((state or {}).get("im_pct", 0.0), 0.0)
    frozen = bool((state or {}).get("frozen", False))
    max_im_pct = safe_float(getattr(config, "MAX_IM_PERCENT", 30.0), 30.0)
    if frozen or im_pct >= max_im_pct:
        result["why"] = "margin"
        return result

    result.update({"ok": True, "qty": float(adj_qty), "why": ""})
    return result


# ===== Доступность минимального ордера (min qty / min notional / cap) =====
def affordable_min_order(price: float,
                         min_qty: float,
                         min_notional_usdt: float,
                         balance_usdt: float,
                         max_balance_share: float,
                         hard_cap_share: float,
                         leverage: float,
                         qty_step: float = 0.0,
                         taker_fee: float = 0.0006) -> dict:
    """
    Возвращает словарь с оценкой доступности минимального ордера.
    Проверяет:
      - лот с min_qty (и его notional) >= min_notional_usdt,
      - требуемая маржа <= баланс * min(MAX_BALANCE_SHARE, HARD_CAP_SHARE),
      - с буфером на комиссию.
    """
    price = float(price or 0.0)
    min_qty = float(min_qty or 0.0)
    min_notional_usdt = float(min_notional_usdt or 0.0)
    balance_usdt = float(balance_usdt or 0.0)
    leverage = max(1.0, float(leverage or 1.0))
    cap_share = max(0.0, min(float(hard_cap_share), float(max_balance_share)))

    if price <= 0 or min_qty <= 0 or balance_usdt <= 0 or cap_share <= 0:
        return {"ok": False, "reason": "bad_inputs", "qty": 0.0}

    step_d = Decimal(str(qty_step)) if qty_step else Decimal("0")
    qty_d = Decimal(str(min_qty))
    if step_d > 0:
        qty_d = _floor_to_step(qty_d, step_d)
        if qty_d <= 0:
            qty_d = step_d
    min_qty_adj = float(qty_d)
    notional_minlot = price * min_qty_adj
    notional = max(notional_minlot, min_notional_usdt)
    fee_buffer = 1.0 + abs(float(taker_fee)) * 2.5  # небольшой буфер на слип/комиссию
    margin_required = (notional * fee_buffer) / leverage
    margin_cap = balance_usdt * cap_share

    return {
        "ok": margin_required <= margin_cap,
        "notional": notional,
        "min_qty": min_qty_adj,
        "qty": min_qty_adj if margin_required <= margin_cap else 0.0,
        "margin_required": margin_required,
        "margin_cap": margin_cap,
    }


# ---------- ПЛЕЧО / РАМПЫ ----------
def apply_leverage_ramp(previous: Optional[float],
                        candidate: float,
                        step_max: float) -> Tuple[int, str]:
    """Ограничивает изменение плеча по максимуму step_max и возвращает причину."""
    try:
        target = int(max(1, round(float(candidate))))
    except Exception:
        return 1, "invalid_candidate"

    if previous is None or float(previous) <= 0:
        return target, ""

    try:
        prev_val = int(max(1, round(float(previous))))
    except Exception:
        prev_val = target

    cap = max(1.0, float(step_max or 1.0))
    up_cap = max(1, int(math.ceil(prev_val * cap)))
    down_cap = max(1, int(math.floor(prev_val / cap)))

    reasons = []
    if target > up_cap:
        target = up_cap
        reasons.append("ramp_limit_up")
    if target < down_cap:
        target = down_cap
        reasons.append("ramp_limit_down")

    return target, ",".join(reasons)


def fallback_leverage(default_leverage: int, previous: Optional[int]) -> int:
    """Возвращает безопасное плечо при плохих данных (не выше последнего валидного)."""
    try:
        base = int(max(1, round(float(default_leverage))))
    except Exception:
        base = 1

    if previous is None:
        return base

    try:
        prev_val = int(max(1, round(float(previous))))
    except Exception:
        prev_val = base

    return max(1, min(base, prev_val))


def clamp(v, lo, hi):
    """Обрезает v в диапазон [lo, hi]. Все аргументы безопасно приводятся к float."""
    v = safe_float(v, 0.0)
    lo = safe_float(lo, v)
    hi = safe_float(hi, v)
    if lo > hi:
        lo, hi = hi, lo
    return min(max(v, lo), hi)


def spread_penalty(spread: float, spread_max: float, alpha: float = 1.0) -> float:
    """Коэффициент уменьшения размера позиции при широком спреде."""
    if spread_max <= 0:
        return 1.0
    ratio = safe_float(spread, 0.0) / max(spread_max, 1e-9)
    penalty = 1.0 - min(1.0, max(0.0, ratio * max(alpha, 0.0)))
    return clamp(penalty, 0.0, 1.0)


def fee_aware_r_min(r_min: float, fee_rate: float, sides: int = 2) -> float:
    """Поднимает целевой R_min с учётом комиссий (по умолчанию туда-обратно)."""
    fee_rate = abs(float(fee_rate))
    base = float(r_min)
    return max(base, fee_rate * float(max(1, sides)) * 2.0)


def kelly_capped(edge: float, variance: float, f_max: float = 0.03) -> float:
    """Капнутый Келли: edge/variance, ограниченный [0, f_max]."""
    variance = max(float(variance), 1e-9)
    edge = float(edge)
    f_star = edge / variance
    return clamp(f_star, 0.0, max(float(f_max), 0.0))


# ---------- СЕССИИ / ДИРЕКТОРИИ ДАННЫХ ----------
def get_data_root() -> Path:
    """Возвращает корень data, учитывая config.DATA_ROOT и переменную окружения."""
    raw = os.getenv("DATA_ROOT")
    if not raw:
        try:
            import config as _cfg  # type: ignore
        except Exception:  # pragma: no cover - конфиг опционален
            _cfg = None
        if _cfg is not None:
            raw = getattr(_cfg, "DATA_ROOT", None)
    if not raw:
        raw = "./data"
    return Path(str(raw)).expanduser()


def get_sessions_root(create: bool = True) -> Path:
    """Путь к data/sessions. При необходимости создаёт директорию."""
    root = get_data_root() / "sessions"
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return root


def list_session_directories() -> List[Path]:
    """Возвращает список каталогов сессий (сортировка по убыванию времени)."""
    root = get_sessions_root(create=False)
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return dirs

