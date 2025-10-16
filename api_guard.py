# -*- coding: utf-8 -*-
# api_guard.py
"""
Обёртка над вызовами pybit/HTTP с:
- локальным rate-limit (token bucket)
- минимальным интервалом между запросами поверх токен-бакета
- экспоненциальным бэкоффом с джиттером и учётом Retry-After
- распознаванием частых ошибок Bybit/HTTP/сети
- мягкой обработкой retCode 110043 (leverage not modified)
- эскалацией 10001 (params error / symbol invalid / precision/qty) наверх

Параметры берутся из config.py, а при его отсутствии — из .env:
  API_GUARD_RATE         (float, по умолчанию 4.8)
  API_GUARD_BURST        (int,   по умолчанию 10)
  HTTP_RETRIES           (int,   по умолчанию 6)
  MAX_BACKOFF            (float, по умолчанию 8.0)
  MIN_DELAY_BETWEEN_REQ  (float, по умолчанию 0.08)
"""

from __future__ import annotations

import json
import os
import time
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from utils import log

__all__ = [
    "safe_request",
    "TokenBucket",
    "set_rate",
    "set_min_delay",
]

# =========================
# Конфиг с фолбэком на env
# =========================
def _get_cfg(name: str, env_key: str, default: str) -> str:
    try:
        import config as _cfg  # type: ignore
    except Exception:
        _cfg = None
    if _cfg is not None and hasattr(_cfg, name):
        val = getattr(_cfg, name)
        return str(val)
    return os.getenv(env_key, default)

_API_RATE   = float(_get_cfg("API_GUARD_RATE", "API_GUARD_RATE", "4.8"))
_API_BURST  = int(float(_get_cfg("API_GUARD_BURST", "API_GUARD_BURST", "10")))
_HTTP_RETRY = int(float(_get_cfg("HTTP_RETRIES", "HTTP_RETRIES", "6")))
_MAX_BACKOFF= float(_get_cfg("MAX_BACKOFF", "MAX_BACKOFF", "8.0"))
_MIN_DELAY  = float(_get_cfg("MIN_DELAY_BETWEEN_REQ", "MIN_DELAY_BETWEEN_REQ", "0.08"))
_METRICS_INTERVAL = max(1.0, float(_get_cfg("API_GUARD_METRICS_SEC", "API_GUARD_METRICS_SEC", "30.0")))
_METRICS_PATH = Path("logs/metrics.jsonl")

# =========================
# Rate limit: Token Bucket
# =========================
class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = float(rate_per_sec)
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def take(self, n: int = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            # пополнение
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

_bucket = TokenBucket(rate_per_sec=_API_RATE, burst=_API_BURST)

# Минимальный интервал между запросами (дополнительно к бакету)
_min_delay_lock = threading.Lock()
_last_call_ts = 0.0

_metrics_lock = threading.Lock()
_metrics_state = {
    "bucket_miss": 0,
    "bucket_sleep_sec": 0.0,
    "min_delay_sleeps": 0,
    "min_delay_sleep_sec": 0.0,
    "last_flush": time.monotonic(),
}

def _respect_min_delay():
    global _last_call_ts
    if _MIN_DELAY <= 0:
        return
    with _min_delay_lock:
        now = time.monotonic()
        wait = _MIN_DELAY - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
            _bump_metric("min_delay_sleep", wait)
            now = time.monotonic()
        _last_call_ts = now


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flush_metrics_locked(now: Optional[float] = None) -> None:
    if now is None:
        now = time.monotonic()
    payload = {
        "ts": _utc_iso(),
        "bucket_miss": int(_metrics_state.get("bucket_miss", 0)),
        "bucket_sleep_sec": round(float(_metrics_state.get("bucket_sleep_sec", 0.0)), 6),
        "min_delay_sleeps": int(_metrics_state.get("min_delay_sleeps", 0)),
        "min_delay_sleep_sec": round(float(_metrics_state.get("min_delay_sleep_sec", 0.0)), 6),
        "rate_per_sec": _API_RATE,
        "burst": _API_BURST,
    }
    try:
        _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_METRICS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"[API-GUARD] metrics write failed: {exc}", level="ERROR")

    if payload["bucket_miss"] or payload["min_delay_sleeps"]:
        log(
            "[API-GUARD] metrics: bucket_miss=%d (sleep=%.3fs) | min_delay=%d (sleep=%.3fs)"
            % (
                payload["bucket_miss"],
                payload["bucket_sleep_sec"],
                payload["min_delay_sleeps"],
                payload["min_delay_sleep_sec"],
            )
        )

    _metrics_state.update({
        "bucket_miss": 0,
        "bucket_sleep_sec": 0.0,
        "min_delay_sleeps": 0,
        "min_delay_sleep_sec": 0.0,
        "last_flush": now,
    })


def _bump_metric(kind: str, value: float = 0.0) -> None:
    now = time.monotonic()
    with _metrics_lock:
        if kind == "bucket_miss":
            _metrics_state["bucket_miss"] += int(value or 1)
        elif kind == "bucket_sleep":
            _metrics_state["bucket_sleep_sec"] += float(value)
        elif kind == "min_delay_sleep":
            _metrics_state["min_delay_sleeps"] += 1
            _metrics_state["min_delay_sleep_sec"] += float(value)
        else:
            return

        if now - _metrics_state.get("last_flush", now) >= _METRICS_INTERVAL:
            _flush_metrics_locked(now)

# =========================
# Бэкофф/утилиты
# =========================
def _sleep_backoff(tries: int, base: float = 0.35, cap: float = 8.0, retry_after: Optional[float] = None):
    """
    Экспоненциальная задержка с джиттером.
    cap ограничивается глобальным _MAX_BACKOFF.
    """
    if retry_after is not None and retry_after > 0:
        t = float(retry_after)
    else:
        cap_eff = min(float(cap), float(_MAX_BACKOFF))
        t = min(base * (2 ** max(0, tries - 1)), cap_eff)
        t += random.uniform(0, 0.15 * t)
    time.sleep(t)

def _extract_http_info(e: Exception) -> Tuple[Optional[int], Optional[float]]:
    status_code = None
    retry_after = None
    resp = getattr(e, "response", None)
    if resp is not None:
        # requests-like
        try:
            status_code = int(getattr(resp, "status_code", None))
        except Exception:
            pass
        try:
            ra = getattr(resp, "headers", {}).get("Retry-After")
            if ra:
                retry_after = float(ra)
        except Exception:
            pass
    # попытка выковырять Retry-After из текста
    msg = str(e)
    if retry_after is None:
        low = msg.lower()
        if "retry-after" in low:
            try:
                # находим первое число после вхождения
                part = low.split("retry-after", 1)[1]
                num = "".join(ch for ch in part if (ch.isdigit() or ch == "."))[:12]
                if num:
                    retry_after = float(num)
            except Exception:
                pass
    return status_code, retry_after

def _is_transient_network(msg: str) -> bool:
    patterns = (
        "read timed out", "connection aborted", "temporary failure",
        "max retries exceeded", "remotedisconnected", "timeout",
        "timed out", "tlsv1_alert_internal_error", "connection reset by peer",
        "service temporarily unavailable", "bad gateway", "gateway timeout",
        "eof occurred in violation of protocol",
        "dns lookup failed", "proxy error", "remote end closed connection"
    )
    low = (msg or "")
    return any(p in low.lower() for p in patterns)

# =========================
# Public API
# =========================
def set_rate(rate_per_sec: float, burst: int | None = None):
    """Позволяет на лету поменять лимиты бакета."""
    global _bucket
    with _bucket.lock:
        _bucket.rate = float(rate_per_sec)
        if burst is not None:
            _bucket.capacity = max(1, int(burst))
            _bucket.tokens = min(_bucket.tokens, float(_bucket.capacity))

def set_min_delay(seconds: float):
    """Меняет минимальную паузу между запросами."""
    global _MIN_DELAY
    _MIN_DELAY = max(0.0, float(seconds))

def safe_request(fn: Callable, *args, max_tries: Optional[int] = None, **kwargs) -> Any:
    """
    Безопасный вызов к pybit.HTTP.<method>.
    Ретраит сетевые/временные ошибки и 429/5xx/часть retCode.
    Возвращает dict-ответ, либо поднимает исключение после исчерпания попыток.

    Особые случаи:
      - retCode 110043 (leverage not modified) → НЕ ошибка (возвращаем resp)
      - retCode 10002 (timestamp/recv_window) → мягкий бэкофф и ретрай
      - retCode 10001 (params error / symbol invalid / precision/qty) → поднимаем исключение сразу
    """
    tries = 0
    method_name = getattr(fn, "__name__", "call")

    if max_tries is None:
        max_tries = int(_HTTP_RETRY)

    while True:
        # доп. интервал между запросами
        _respect_min_delay()

        # токен-бакет
        if not _bucket.take():
            # ожидание следующего токена
            time.sleep(0.08)
            _bump_metric("bucket_miss", 1)
            _bump_metric("bucket_sleep", 0.08)
            continue

        tries += 1
        try:
            resp = fn(*args, **kwargs)

            # Bybit v5: { retCode, retMsg, result, ... }
            if isinstance(resp, dict) and "retCode" in resp:
                rc = int(resp.get("retCode", 0) or 0)
                if rc != 0:
                    msg = str(resp.get("retMsg", ""))

                    # --- нормальные «не ошибки» ---
                    if rc == 110043:  # leverage not modified
                        log(f"[API] {method_name}: 110043 leverage not modified — ок")
                        return resp

                    # --- известные ретраимые retCode ---
                    # 10006/10005/10003/10016 — rate/limit/cooldown
                    if rc in (10006, 10005, 10003, 10016):
                        log(f"[RL/API] {method_name}: retCode={rc} ({msg}) … попытка {tries}")
                        _sleep_backoff(tries, base=0.35, cap=_MAX_BACKOFF)
                        if tries < max_tries:
                            continue

                    # 10002 — timestamp/recv_window
                    if rc == 10002:
                        log(f"[TS] {method_name}: retCode=10002 ({msg}) … попытка {tries}")
                        _sleep_backoff(tries, base=0.25, cap=min(5.0, _MAX_BACKOFF))
                        if tries < max_tries:
                            continue

                    # 10001 — params error / symbol invalid / precision/qty → эскалируем
                    if rc == 10001:
                        raise RuntimeError(f"Bybit 10001: {msg}")

                    # прочие retCode — считаем ошибкой
                    raise RuntimeError(f"Bybit retCode {rc}: {msg}")

            # успех
            return resp

        except Exception as e:
            msg = str(e)
            status_code, retry_after = _extract_http_info(e)

            # --- фатальные авторизационные ---
            if status_code in (401, 403):
                log(f"[AUTH] {method_name}: HTTP {status_code}: {msg[:200]}")
                raise

            # --- 429 Too Many Requests ---
            if status_code == 429:
                log(f"[RL] {method_name}: HTTP 429: {msg[:200]} … попытка {tries}")
                _sleep_backoff(tries, base=0.35, cap=_MAX_BACKOFF, retry_after=retry_after)
                if tries < max_tries:
                    continue

            # --- 5xx ---
            if status_code in (500, 502, 503, 504):
                log(f"[NET] {method_name}: HTTP {status_code}: {msg[:200]} … попытка {tries}")
                _sleep_backoff(tries, base=0.5, cap=_MAX_BACKOFF, retry_after=retry_after)
                if tries < max_tries:
                    continue

            # --- сетевые временные ---
            if _is_transient_network(msg) and tries < max_tries:
                log(f"[NET] {method_name}: временный сбой сети: {msg[:200]} … попытка {tries}")
                _sleep_backoff(tries, base=0.40, cap=_MAX_BACKOFF)
                continue

            # --- «Qty/precision» — отдаём наверх без лишних попыток ---
            low = msg.lower()
            if ("10001" in low and "qty" in low) or ("10001" in low and "precision" in low) or ("symbol invalid" in low):
                # важно: это сигнальная ошибка для вызывающего кода
                raise

            # --- исчерпали попытки ---
            if tries >= max_tries:
                log(f"[API] сдаюсь после {tries} попыток ({method_name}): {msg[:220]}", level="ERROR")
                raise

            # --- дефолтный бэкофф и ещё попытка ---
            _sleep_backoff(tries, base=0.35, cap=_MAX_BACKOFF, retry_after=retry_after)
            continue
