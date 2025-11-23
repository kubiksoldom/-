# -*- coding: utf-8 -*-
"""
bybit_full_history_export.py
--------------------------------
Скачивает ПОЛНУЮ историю сделок (fills) из Bybit Unified Trading API (V5)
по категориям "linear" и "spot", разбивая период на окна (по умолчанию 7 дней).
Сделано быстро:
• Параллельная выборка окон (ThreadPoolExecutor)
• Глобальный token-bucket + минимальная межзапросная задержка только при необходимости
• Адаптивный бэкофф на 10006 / сетевые ошибки
• Дедуп по execId
• ENV-переменные для тонкой настройки скорости

Выходы:
  - fills_linear.jsonl
  - fills_spot.jsonl
  - fills_all.csv

ENV (пример .env):
  BYBIT_API_KEY=...
  BYBIT_API_SECRET=...
  EXPORT_WINDOW_DAYS=7
  EXPORT_START=2024-01-01
  EXPORT_END=now
  EXPORT_SYMBOL=              # опционально: сузить до одного символа
  EXPORT_WORKERS=8            # параллельные окна (на категорию)
  EXPORT_REQ_RATE=8.0         # токенов/сек на ВСЕ потоки (глобально)
  EXPORT_BURST=16             # размер ведра (разовый всплеск)
  EXPORT_MIN_DELAY=0.02       # минимальная пауза между любыми двумя запросами
  EXPORT_RETRIES=6
  EXPORT_BACKOFF_MAX=6.0
  EXPORT_REQ_LIMIT=100        # лимит элементов на страницу (макс 100 у Bybit)
"""

import os, time, json, csv
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pybit.unified_trading import HTTP
import config

# ================ ЛОГ ================
_print_lock = threading.Lock()
def log(msg: str):
    with _print_lock:
        print(msg, flush=True)

# ================ ENV / ТЮНИНГ ================
API_KEY    = getattr(config, "BYBIT_API_KEY", "")
API_SECRET = getattr(config, "BYBIT_API_SECRET", "")

WINDOW_DAYS   = int(getattr(config, "EXPORT_WINDOW_DAYS", 7))
EXPORT_SYMBOL = (getattr(config, "EXPORT_SYMBOL", "") or "").strip() or None
WORKERS       = int(getattr(config, "EXPORT_WORKERS", 8))

REQ_RATE      = float(getattr(config, "EXPORT_REQ_RATE", 8.0))     # токенов/сек на все потоки
REQ_BURST     = int(getattr(config, "EXPORT_BURST", 16))
MIN_DELAY     = float(getattr(config, "EXPORT_MIN_DELAY", 0.02))
RETRIES       = int(getattr(config, "EXPORT_RETRIES", 6))
BACKOFF_MAX   = float(getattr(config, "EXPORT_BACKOFF_MAX", 6.0))
PAGE_LIMIT    = int(getattr(config, "EXPORT_REQ_LIMIT", 100))

# Временные границы
def _parse_dt(s: str) -> datetime:
    if s.lower() in ("now", "today"):
        return datetime.now(timezone.utc)
    # ожидаем YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

START_STR = getattr(config, "EXPORT_START", "2024-01-01")
END_STR   = getattr(config, "EXPORT_END", "now")
START_MS  = int(_parse_dt(START_STR).timestamp() * 1000)
END_MS    = int(_parse_dt(END_STR).timestamp() * 1000)

# ================ TokenBucket (глобальный) ================
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

_BUCKET = TokenBucket(REQ_RATE, REQ_BURST)
_last_req_ts = 0.0
_last_req_lock = threading.Lock()

def _sleep_min_delay():
    global _last_req_ts
    if MIN_DELAY <= 0:
        return
    with _last_req_lock:
        now = time.time()
        wait = max(0.0, MIN_DELAY - (now - _last_req_ts))
        if wait > 0:
            time.sleep(wait)
        _last_req_ts = time.time()

# ================ HTTP SAFE CALL ================
SOFT_PATTERNS = (
    "Read timed out","Timeout","Temporary failure","Max retries exceeded",
    "RemoteDisconnected","Bad Gateway","Service Temporarily Unavailable","Gateway Timeout",
)

def safe_http_call(client: HTTP, fn, **kwargs):
    backoff = 0.25
    for attempt in range(1, RETRIES + 1):
        # глобальный токен-бакет
        while not _BUCKET.take():
            time.sleep(0.005)
        _sleep_min_delay()
        try:
            return fn(**kwargs)
        except Exception as e:
            msg = str(e)
            # 10006 / лимиты — ускоренный, но адаптивный бэкофф
            if "10006" in msg or "rate limit" in msg.lower():
                if attempt < RETRIES:
                    sleep = min(BACKOFF_MAX, backoff)
                    log(f"[RL] 10006 → backoff {sleep:.2f}s (try {attempt}/{RETRIES})")
                    time.sleep(sleep)
                    backoff *= 1.8
                    continue
                else:
                    raise
            if any(s.lower() in msg.lower() for s in SOFT_PATTERNS):
                if attempt < RETRIES:
                    sleep = min(BACKOFF_MAX, backoff)
                    log(f"[NET] {msg[:110]} → backoff {sleep:.2f}s")
                    time.sleep(sleep)
                    backoff *= 1.7
                    continue
                else:
                    raise
            # Прочие ошибки — пара попыток
            if attempt < RETRIES:
                sleep = min(BACKOFF_MAX, backoff)
                log(f"[WARN] {msg[:110]} → retry {attempt}/{RETRIES} (sleep {sleep:.2f}s)")
                time.sleep(sleep)
                backoff *= 1.5
                continue
            raise

# ================ Утилиты/нормализация ================
def normalize_fill(rec: Dict) -> Dict:
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

def iter_time_windows(start_ms: int, end_ms: int, window_ms: int) -> List[Tuple[int,int]]:
    out = []
    cur = start_ms
    while cur <= end_ms:
        nxt = min(cur + window_ms - 1, end_ms)
        out.append((cur, nxt))
        cur = nxt + 1
    return out

# ================ Пагинация executions ================
def get_executions_page(client: HTTP, category: str, start_ms: int, end_ms: int,
                        symbol: Optional[str]=None, cursor: Optional[str]=None,
                        limit: int=100) -> Dict:
    params = {"category": category, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    if symbol: params["symbol"] = symbol
    if cursor: params["cursor"] = cursor
    return safe_http_call(client, client.get_executions, **params)

def get_executions_all(client: HTTP, category: str, start_ms: int, end_ms: int,
                       symbol: Optional[str]=None, limit: int=100) -> List[Dict]:
    all_items, cursor = [], None
    while True:
        resp = get_executions_page(client, category, start_ms, end_ms, symbol=symbol, cursor=cursor, limit=limit)
        if resp.get("retCode", 0) != 0:
            raise RuntimeError(f"Bybit error: {resp.get('retCode')} {resp.get('retMsg')}")
        result = resp.get("result", {}) or {}
        items = result.get("list", []) or []
        all_items.extend(items)
        cursor = result.get("nextPageCursor")
        if not cursor or not items:
            break
        # минимальная пауза уже обеспечена глобально; доп. sleep не нужен
    return all_items

# ================ Выборка по окну (для пула потоков) ================
def fetch_window(args) -> Tuple[Tuple[int,int], List[Dict]]:
    (start_ms, end_ms, category, symbol, limit, api_key, api_secret) = args
    client = HTTP(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        recv_window=20000,
        timeout=10,
    )
    data = get_executions_all(client, category, start_ms, end_ms, symbol=symbol, limit=limit)
    return (start_ms, end_ms), data

# ================ Основной процесс категории ================
def fetch_category(category: str, start_ms: int, end_ms: int, symbol: Optional[str], out_jsonl: str) -> List[Dict]:
    log(f"\n=== Fetching category={category}, range=[{start_ms}..{end_ms}] ===")
    window_ms = WINDOW_DAYS * 24 * 60 * 60 * 1000
    windows = iter_time_windows(start_ms, end_ms, window_ms)

    args = [
        (s, e, category, symbol, PAGE_LIMIT, API_KEY, API_SECRET)
        for (s, e) in windows
    ]

    # Параллельная выборка окон
    merged: List[Dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch_window, a) for a in args]
        for fut in as_completed(futs):
            (s,e), chunk = fut.result()
            log(f"[{category}] window {datetime.fromtimestamp(s/1000, tz=timezone.utc).date()}.."
                f"{datetime.fromtimestamp(e/1000, tz=timezone.utc).date()} → {len(chunk)} fills")
            merged.extend(chunk)

    # Дедуп по execId
    seen, unique = set(), []
    for r in merged:
        xid = r.get("execId")
        if xid and xid in seen:
            continue
        if xid:
            seen.add(xid)
        unique.append(r)

    log(f"[{category}] total={len(merged)}, unique={len(unique)}")

    # Сохраняем JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return unique

# ================ MAIN ================
def main():
    if not API_KEY or not API_SECRET:
        raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET must be set in environment (.env)")

    # Можно параллелить и категории тоже
    cat_jobs = [("linear", "fills_linear.jsonl"), ("spot", "fills_spot.jsonl")]

    results: Dict[str, List[Dict]] = {}
    with ThreadPoolExecutor(max_workers=min(2, len(cat_jobs))) as ex:  # 2 категории максимум
        futs = {ex.submit(fetch_category, cat, START_MS, END_MS, EXPORT_SYMBOL, out): (cat, out)
                for (cat, out) in cat_jobs}
        for fut in as_completed(futs):
            cat, out = futs[fut]
            results[cat] = fut.result()

    # Объединяем и сохраняем нормализованный CSV
    all_norm = []
    for category, arr in results.items():
        for r in arr:
            n = normalize_fill(r)
            n["category"] = category
            all_norm.append(n)

    fields = ["ts","symbol","side","price","qty","fee","feeCurrency","isMaker",
              "orderId","execId","value","orderLinkId","category"]
    with open("fills_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_norm:
            w.writerow(r)

    log("\nDone! Files created: fills_linear.jsonl, fills_spot.jsonl, fills_all.csv")

if __name__ == "__main__":
    main()
