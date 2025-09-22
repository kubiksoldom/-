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
import json
import math
import time
import random
import threading
from decimal import Decimal, ROUND_DOWN, getcontext, InvalidOperation
from datetime import datetime, timezone
from typing import Any, Dict

import requests
from dotenv import load_dotenv

# ---------- CONFIG/ENV WITH FALLBACK ----------
load_dotenv()

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

TELEGRAM_TOKEN   = _cfg_get("TELEGRAM_TOKEN", "TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = _cfg_get("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID", "")

# Для Decimal — достаточно 28 знаков, чтобы уверенно резать шаги количества/цены
getcontext().prec = 28

# Общие локи
_LOG_LOCK = threading.Lock()
_TG_WARN_LOCK = threading.Lock()
_TG_WARNED_NO_CREDS = False  # чтобы не спамить предупреждением на каждый вызов


# ---------- HELPERS ----------
def _utc_iso() -> str:
    """Возвращает UTC ISO8601 с 'Z' в конце (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(message: str, level: str = "INFO"):
    """
    Простое консольное логирование. Без падения.
    Используется по всему проекту и в api_guard (level="ERROR"/"WARNING"/"INFO").
    """
    t = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"[{t}] [{level}] {message}", flush=True)
    except Exception:
        # в крайних случаях, когда stdout сломан
        pass


def write_cycle_log(data: Dict[str, Any]):
    """
    Пишем одну строку JSONL. Создаём папку при необходимости.
    Всегда проставляем UTC timestamp (ts_utc) поверх входных данных.
    """
    if not LOG_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(LOG_JSONL) or ".", exist_ok=True)
    except Exception as e:
        log(f"[LOG] mkdir: {e}", level="ERROR")

    try:
        line = dict(data)
        line["ts_utc"] = _utc_iso()
        s = json.dumps(line, ensure_ascii=False)
        with _LOG_LOCK:
            with open(LOG_JSONL, "a", encoding="utf-8") as f:
                f.write(s + "\n")
    except Exception as e:
        log(f"[LOG] ошибка записи: {e}", level="ERROR")


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
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]

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
                r = requests.post(url, data=payload, timeout=12)
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


def make_order_link_id(prefix: str = "BOT") -> str:
    """
    Генерим компактный orderLinkId для Bybit.
    Формат: <prefix>-<ms>-<rnd>, обрезаем до 32 символов.
    """
    base = f"{prefix}-{int(time.time()*1000)}-{random.randint(1000, 9999)}"
    return base[:32]


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


def coalesce(*values, default=0.0) -> float:
    """Возвращает первый валидный float из списка (не None/не NaN/не Inf), иначе default."""
    for v in values:
        try:
            f = float(v)
            if not (math.isnan(f) or math.isinf(f)):
                return f
        except Exception:
            pass
    return float(default)


# ---------- ОКРУГЛЕНИЯ ПОД ШАГИ БИРЖИ ----------
def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Округляет вниз к ближайшему шагу: floor(value/step) * step"""
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step


def round_price_down(price, tick_size) -> float:
    """Округляет цену вниз под биржевой шаг (tick_size)."""
    p = _dec(price)
    t = _dec(tick_size)
    if p <= 0 or t <= 0:
        return 0.0
    return float(_floor_to_step(p, t))


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


# ===== Доступность минимального ордера (min qty / min notional / cap) =====
def affordable_min_order(price: float,
                         min_qty: float,
                         min_notional_usdt: float,
                         balance_usdt: float,
                         max_balance_share: float,
                         hard_cap_share: float,
                         leverage: float,
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
        return {"ok": False, "reason": "bad_inputs"}

    notional_minlot = price * min_qty
    notional = max(notional_minlot, min_notional_usdt)
    fee_buffer = 1.0 + taker_fee * 2.0  # туда-обратно
    margin_required = (notional * fee_buffer) / leverage
    margin_cap = balance_usdt * cap_share

    return {
        "ok": margin_required <= margin_cap,
        "notional": notional,
        "min_qty": min_qty,
        "margin_required": margin_required,
        "margin_cap": margin_cap,
    }


# ---------- ДОП. ХЕЛПЕРЫ ДЛЯ НОЦИОНАЛА/РИСКА ----------
def calc_notional(price, qty) -> float:
    """price * qty с защитой от None/NaN/Inf."""
    p = safe_float(price, 0.0)
    q = safe_float(qty, 0.0)
    return p * q


def fits_minimums(price, qty, min_qty=0.0, min_notional=0.0) -> bool:
    """
    Быстрая проверка: удовлетворяет ли (price, qty) биржевым минимумам.
    """
    p = _dec(price)
    q = _dec(qty)
    min_q = _dec(min_qty)
    min_not = _dec(min_notional)
    if p <= 0 or q <= 0:
        return False
    if min_q > 0 and q < min_q:
        return False
    if min_not > 0 and (p * q) < min_not:
        return False
    return True


def clamp(v, lo, hi):
    """Обрезает v в диапазон [lo, hi]. Все аргументы безопасно приводятся к float."""
    v = safe_float(v, 0.0)
    lo = safe_float(lo, v)
    hi = safe_float(hi, v)
    if lo > hi:
        lo, hi = hi, lo
    return min(max(v, lo), hi)


# ---------- СОВМЕСТИМОСТЬ ----------
def set_log_enabled():
    """Оставлено для совместимости (ничего не делает)."""
    pass
