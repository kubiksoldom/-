# -*- coding: utf-8 -*-
"""
binance_fills_export.py — быстрый параллельный экспорт + безопасный мердж в общий лог.

Единый формат строк (совместим с build_ml_dataset_from_fills.py):
  ts,symbol,side,price,qty,fee,feeCurrency,isMaker,orderId,execId,value,orderLinkId

Что нового и почему быстрее:
  • Параллельность по символам (ThreadPoolExecutor) — каждый символ собирается отдельным воркером.
  • Крупные окна (дефолт 30 дней, настраивается --window-days) — меньше запросов.
  • Resume от текущего лога (--resume по умолчанию): стартуем от max(ts)+1 для каждого symbol в твоём fills_all.csv.
  • Мердж «по символам»: каждый символ завершился → сразу объединяем в общий лог (есть .bak) → прогресс не теряется при прерывании.
  • Дедуп ключом (symbol, execId) или fallback-комбинацией, унион заголовков сохраняется.
  • Разумный rate-limit (общий токен-бакет), бэкофф, sync времени (-1021).

ENV:
  BINANCE_API_KEY, BINANCE_API_SECRET

Примеры:
  # Собрать всё быстро и слить в fills_all.csv (фьючерсы USDT-M)
  python binance_fills_export.py --market futures --symbol BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT \
    --start 2020-08-01 --end 2025-09-01 --merge-to fills_all.csv

  # Только экспорт (без мерджа), 8 воркеров, окна 21 день:
  python binance_fills_export.py --market futures --symbol BTCUSDT,ETHUSDT \
    --start 2024-01-01 --end 2025-09-01 --out export.csv --workers 8 --window-days 21 --no-resume
"""

import os
import sys
import hmac
import csv
import time
import argparse
import hashlib
import shutil
import tempfile
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Iterable, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============== УТИЛИТЫ ВРЕМЕНИ/ЛОГА ==============

def _dt_parse(d: str) -> datetime:
    d = d.strip()
    if "T" in d:
        return datetime.fromisoformat(d).replace(tzinfo=timezone.utc) if "Z" not in d else datetime.fromisoformat(d.replace("Z", "+00:00"))
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _utcms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============== ПРОСТОЙ РЕЙТ-ЛИМИТЕР (общий на все потоки) ==============

class TokenBucket:
    def __init__(self, rate_per_sec: float = 8.0, burst: int = 16):
        self.rate = float(rate_per_sec)
        self.capacity = int(burst)
        self.tokens = float(burst)
        self.last = time.time()
        self.lock = threading.Lock()

    def take(self, n: int = 1) -> None:
        # блокирующий: ждём, пока можно взять n токенов
        while True:
            with self.lock:
                now = time.time()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
            time.sleep(0.03)

_BUCKET = TokenBucket(rate_per_sec=9.0, burst=20)  # чуть бодрее, но без фанатизма


# ============== КЛИЕНТ BINANCE REST (по экземпляру на воркер) ==============

def _make_session() -> requests.Session:
    s = requests.Session()
    # keep-alive, ретраи на 429/5xx, пул коннектов побольше
    retries = Retry(total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    s.headers.update({"Connection": "keep-alive"})
    return s

class BinanceREST:
    def __init__(self, market: str, api_key: str, api_secret: str, recv_window_ms: int = 60000):
        market = (market or "futures").strip().lower()
        if market not in ("spot", "futures"):
            raise ValueError("market должен быть 'spot' или 'futures'")
        self.market = market
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip().encode()
        self.recv_window_ms = int(recv_window_ms)

        self.base = "https://api.binance.com" if self.market == "spot" else "https://fapi.binance.com"
        self._time_offset_ms = 0  # серверное время - локальное

        self.sess = _make_session()

        self.sync_time()

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key, "User-Agent": "fills-export/1.2-fast-merge"}

    def _sign(self, qs: str) -> str:
        return hmac.new(self.api_secret, qs.encode(), hashlib.sha256).hexdigest()

    def _now_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _request(self, method: str, path: str, params: Dict = None, signed: bool = False) -> Dict:
        if params is None:
            params = {}

        # общий рейт-лимит
        _BUCKET.take(1)

        url = f"{self.base}{path}"
        if signed:
            params["timestamp"] = self._now_ms()
            params["recvWindow"] = self.recv_window_ms

        qs = urlencode(sorted(params.items()), doseq=True)
        if signed:
            sig = self._sign(qs)
            qs = f"{qs}&signature={sig}" if qs else f"signature={sig}"
        full_url = f"{url}?{qs}" if qs else url

        backoff = 0.4
        for attempt in range(7):
            try:
                if method == "GET":
                    r = self.sess.get(full_url, headers=self._headers(), timeout=20)
                else:
                    raise NotImplementedError("Только GET реализован")
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (418, 429):
                    _log(f"[RL] {r.status_code}: {r.text[:160]} → sleep {backoff:.2f}s")
                    time.sleep(backoff); backoff = min(6.0, backoff * 1.7)
                    continue
                try:
                    payload = r.json()
                except Exception:
                    payload = {"code": r.status_code, "msg": r.text[:200]}
                code = str(payload.get("code"))
                if code == "-1021":
                    _log("[TIME] -1021 → повторная синхронизация времени")
                    self.sync_time(force=True)
                    time.sleep(min(1.0, backoff))
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {payload}")
            except Exception as e:
                if attempt >= 6:
                    raise
                _log(f"[NET] {e} → retry in {backoff:.2f}s")
                time.sleep(backoff)
                backoff = min(6.0, backoff * 1.7)
        raise RuntimeError("Unreachable")

    def server_time(self) -> int:
        path = "/api/v3/time" if self.market == "spot" else "/fapi/v1/time"
        data = self._request("GET", path, signed=False)
        return int(data.get("serverTime", 0))

    def sync_time(self, force: bool = False):
        try:
            st = self.server_time()
            if st > 0:
                local = int(time.time() * 1000)
                self._time_offset_ms = st - local
                if force:
                    _log(f"[TIME] sync: offset={self._time_offset_ms} ms")
        except Exception as e:
            _log(f"[TIME] sync failed: {e}")

    # ---- сделки ----
    def user_trades(self, symbol: str, start_ms: int, end_ms: int, from_id: Optional[int] = None,
                    limit: int = 1000) -> List[Dict]:
        symbol = symbol.upper().strip()
        if self.market == "spot":
            path = "/api/v3/myTrades"
            params = {"symbol": symbol, "limit": min(int(limit), 1000)}
            if from_id is not None:
                params["fromId"] = int(from_id)
            else:
                params["startTime"] = int(start_ms)
                params["endTime"] = int(end_ms)
        else:
            path = "/fapi/v1/userTrades"
            params = {"symbol": symbol, "limit": min(int(limit), 1000)}
            if from_id is not None:
                params["fromId"] = int(from_id)
            else:
                params["startTime"] = int(start_ms)
                params["endTime"] = int(end_ms)
        data = self._request("GET", path, params=params, signed=True)
        if isinstance(data, dict) and "msg" in data and "code" in data:
            raise RuntimeError(f"API error: {data}")
        return list(data or [])

    # ---- нормализация ----
    @staticmethod
    def normalize_trade_spot(t: Dict) -> Dict:
        price = float(t.get("price", 0) or 0.0)
        qty = float(t.get("qty", 0) or 0.0)
        fee_raw = float(t.get("commission", 0) or 0.0)
        return {
            "ts": int(t.get("time", 0) or 0),
            "symbol": str(t.get("symbol") or "").upper(),
            "side": "buy" if bool(t.get("isBuyer", False)) else "sell",
            "price": price,
            "qty": qty,
            "fee": abs(fee_raw),
            "feeCurrency": str(t.get("commissionAsset") or ""),
            "isMaker": bool(t.get("isMaker", False)),
            "orderId": str(t.get("orderId")),
            "execId": str(t.get("id")),
            "value": price * qty,
            "orderLinkId": "",
        }

    @staticmethod
    def normalize_trade_futures(t: Dict) -> Dict:
        price = float(t.get("price", 0) or 0.0)
        qty = float(t.get("qty", 0) or 0.0)
        fee_raw = float(t.get("commission", 0) or 0.0)
        side = str(t.get("side", "") or "").upper()
        if side in ("BUY", "SELL"):
            side_str = "buy" if side == "BUY" else "sell"
        else:
            side_str = "buy" if bool(t.get("buyer", False)) else "sell"
        return {
            "ts": int(t.get("time", 0) or 0),
            "symbol": str(t.get("symbol") or "").upper(),
            "side": side_str,
            "price": price,
            "qty": qty,
            "fee": abs(fee_raw),
            "feeCurrency": str(t.get("commissionAsset") or "USDT"),
            "isMaker": bool(t.get("maker", False)),
            "orderId": str(t.get("orderId")),
            "execId": str(t.get("id")),
            "value": price * qty,
            "orderLinkId": str(t.get("clientOrderId") or ""),
        }


# ============== ВСПОМОГАТЕЛЬНОЕ ==============

FIELDS = ["ts","symbol","side","price","qty","fee","feeCurrency","isMaker","orderId","execId","value","orderLinkId"]

def iter_time_windows(start_ms: int, end_ms: int, window_ms: int) -> Iterable[Tuple[int,int]]:
    cur = int(start_ms)
    while cur <= end_ms:
        nxt = min(cur + window_ms - 1, end_ms)
        yield cur, nxt
        cur = nxt + 1

def dedup_key(row: Dict) -> str:
    sym = str(row.get("symbol","")).upper()
    exec_id = str(row.get("execId","")).strip()
    if exec_id:
        return f"{sym}|{exec_id}"
    return f"{sym}|{row.get('orderId','')}|{row.get('ts','')}|{row.get('price','')}|{row.get('qty','')}"

def read_csv_rows(path: str) -> (List[str], List[Dict]):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        rows = [dict(x) for x in r]
        return fields, rows

def write_csv_rows(path: str, fields: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

def merge_csv(existing_path: str, new_path: str, backup: bool = True, sort_ts: bool = True):
    # читаем существующий лог (если есть)
    if os.path.exists(existing_path):
        ex_fields, ex_rows = read_csv_rows(existing_path)
        _log(f"[MERGE] База: {len(ex_rows)} строк из {existing_path}")
    else:
        ex_fields, ex_rows = [], []
        _log(f"[MERGE] База не найдена, создадим новую: {existing_path}")

    # читаем новый экспорт
    new_fields, new_rows = read_csv_rows(new_path)
    _log(f"[MERGE] Добавляем: {len(new_rows)} строк из {new_path}")

    # объединённые поля
    fields = list(ex_fields) if ex_fields else list(FIELDS)
    for f in new_fields:
        if f not in fields:
            fields.append(f)
    for f in FIELDS:
        if f not in fields:
            fields.append(f)

    seen = set()
    out_rows = []

    def _push(row: Dict):
        k = dedup_key(row)
        if k in seen:
            return
        seen.add(k)
        out_rows.append(row)

    for r in ex_rows:
        _push(r)
    for r in new_rows:
        r["symbol"] = str(r.get("symbol","")).upper()
        if isinstance(r.get("isMaker"), str):
            v = r["isMaker"].strip().lower()
            r["isMaker"] = (v in ("true","1","yes","y"))
        _push(r)

    if sort_ts:
        try:
            out_rows.sort(key=lambda x: int(x.get("ts") or 0))
        except Exception:
            pass

    # бэкап
    if backup and os.path.exists(existing_path):
        bak = existing_path + ".bak"
        shutil.copy2(existing_path, bak)
        _log(f"[MERGE] Бэкап: {bak}")

    # атомарная запись
    tmpdir = tempfile.mkdtemp(prefix="fills_merge_")
    tmp_out = os.path.join(tmpdir, "merged.csv")
    write_csv_rows(tmp_out, fields, out_rows)
    shutil.move(tmp_out, existing_path)
    shutil.rmtree(tmpdir, ignore_errors=True)

    _log(f"[MERGE] Готово: {len(out_rows)} строк → {existing_path}")

def get_resume_ts_per_symbol(merge_to: str) -> Dict[str, int]:
    """Читаем существующий лог и берём max(ts) по каждому symbol (для резюма)."""
    res: Dict[str, int] = {}
    if not merge_to or not os.path.exists(merge_to):
        return res
    try:
        with open(merge_to, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                sym = (row.get("symbol") or "").upper()
                try:
                    ts = int(float(row.get("ts") or 0))
                except Exception:
                    continue
                if sym:
                    if sym not in res or ts > res[sym]:
                        res[sym] = ts
    except Exception as e:
        _log(f"[RESUME] read fail: {e}")
    return res


# ============== ЭКСПОРТ: ВОРКЕР ДЛЯ ОДНОГО СИМВОЛА ==============

def export_symbol_to_temp_csv(
    market: str, api_key: str, api_secret: str, symbol: str,
    start_ms: int, end_ms: int, window_ms: int
) -> str:
    """Возвращает путь к временному CSV с экспортом по symbol."""
    client = BinanceREST(market=market, api_key=api_key, api_secret=api_secret, recv_window_ms=60000)

    tmpdir = tempfile.mkdtemp(prefix=f"fills_{symbol}_")
    out_path = os.path.join(tmpdir, f"{symbol}.csv")
    total_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        for s, e in iter_time_windows(start_ms, end_ms, window_ms):
            s_h = datetime.fromtimestamp(s/1000, tz=timezone.utc).isoformat()
            e_h = datetime.fromtimestamp(e/1000, tz=timezone.utc).isoformat()
            _log(f"[{symbol}] окно: {s_h} → {e_h}")
            last_id = None
            while True:
                rows = client.user_trades(symbol=symbol, start_ms=s, end_ms=e, from_id=last_id, limit=1000)
                if not rows:
                    break
                for t in rows:
                    rec = (client.normalize_trade_spot(t) if client.market == "spot"
                           else client.normalize_trade_futures(t))
                    if rec["ts"] < s or rec["ts"] > e:
                        continue
                    w.writerow(rec)
                    total_written += 1
                last_id = int(rows[-1].get("id")) + 1 if rows and "id" in rows[-1] else None
                if len(rows) < 1000 or last_id is None:
                    break
    _log(f"[CSV] {symbol}: экспортировано {total_written} строк → {out_path}")
    return out_path


# ============== CLI ==============

def main():
    ap = argparse.ArgumentParser(description="Быстрый экспорт сделок Binance и мердж в общий лог.")
    ap.add_argument("--market", choices=["spot","futures"], default="futures", help="Рынок Binance: spot или futures (USDT-M)")
    ap.add_argument("--symbol", required=True, help="Символ или несколько через запятую, напр. BTCUSDT или BTCUSDT,ETHUSDT")
    ap.add_argument("--start", required=True, help="Начало периода (UTC): YYYY-MM-DD или YYYY-MM-DDTHH:MM")
    ap.add_argument("--end", required=True, help="Конец периода (UTC): YYYY-MM-DD или YYYY-MM-DDTHH:MM")
    ap.add_argument("--out", default="fills_binance_export.csv", help="Если без --merge-to: общий экспорт всех символов сюда")
    ap.add_argument("--merge-to", default="", help="Путь к общему логу для мерджа (напр. fills_all.csv). Если указан — мердж по мере готовности каждого символа.")
    ap.add_argument("--recvwindow", type=int, default=60000, help="recvWindow в миллисекундах (дефолт 60000)")
    ap.add_argument("--window-days", type=int, default=30, help="Размер окна в днях (дефолт 30)")
    ap.add_argument("--workers", type=int, default=0, help="Кол-во потоков (0=авто: 2*кол-во символов, но не более 8)")
    ap.add_argument("--no-backup", action="store_true", help="Не создавать .bak при мердже")
    ap.add_argument("--no-sort", action="store_true", help="Не сортировать merged по ts")
    ap.add_argument("--no-resume", action="store_true", help="Не пытаться резюмировать от текущего лога")
    args = ap.parse_args()

    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("❌ Нужны BINANCE_API_KEY и BINANCE_API_SECRET в переменных окружения.", file=sys.stderr)
        sys.exit(2)

    start_dt = _dt_parse(args.start)
    end_dt = _dt_parse(args.end)
    if "T" not in args.end:
        end_dt = end_dt + timedelta(days=1)
    start_ms_all = _utcms(start_dt)
    end_ms_all = _utcms(end_dt) - 1

    symbols = [s.strip().upper() for s in (args.symbol or "").split(",") if s.strip()]
    if not symbols:
        print("❌ Укажите хотя бы один символ через --symbol", file=sys.stderr)
        sys.exit(3)

    # резюмирование от существующего лога
    resume_map = {}
    if args.merge_to and not args.no_resume:
        resume_map = get_resume_ts_per_symbol(args.merge_to)
        if resume_map:
            _log(f"[RESUME] найдено max(ts) для {len(resume_map)} символов (используем ts+1).")

    # параллельность
    workers = int(args.workers or 0)
    if workers <= 0:
        workers = min(8, max(2, len(symbols) * 2))

    window_ms = int(args.window_days) * 24 * 60 * 60 * 1000
    if window_ms <= 0:
        window_ms = 30 * 24 * 60 * 60 * 1000

    # режим «мердж по мере готовности символа» или общий экспорт
    if args.merge_to:
        # собираем по символам в параллели → по завершении каждого символа делаем merge_csv
        futs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for sym in symbols:
                start_ms = start_ms_all
                if sym in resume_map:
                    start_ms = max(start_ms_all, resume_map[sym] + 1)  # резюмируемся
                    if start_ms > end_ms_all:
                        _log(f"[SKIP] {sym}: уже покрыт (resume ts >= end).")
                        continue
                _log(f"[RUN] {sym}: {start_ms}..{end_ms_all} (UTC ms), окна ~{args.window_days}d")
                futs.append(ex.submit(
                    export_symbol_to_temp_csv, args.market, api_key, api_secret, sym, start_ms, end_ms_all, window_ms
                ))

            for fut in as_completed(futs):
                try:
                    temp_csv = fut.result()
                    # мерджим сразу, безопасно (с .bak), чтобы закрепить прогресс
                    merge_csv(existing_path=args.merge_to, new_path=temp_csv,
                              backup=(not args.no_backup), sort_ts=(not args.no_sort))
                    # чистим временное
                    try:
                        shutil.rmtree(os.path.dirname(temp_csv), ignore_errors=True)
                    except Exception:
                        pass
                except Exception as e:
                    _log(f"[WORKER] ошибка: {e}")

        _log("[DONE] Все символы обработаны (merge-on-complete).")

    else:
        # без мерджа: делаем общий export в args.out
        tmpdir = tempfile.mkdtemp(prefix="fills_export_all_")
        # собираем параллельно во временные файлы, затем конкатенируем и дедупим в один CSV
        tmp_files = []
        futs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for sym in symbols:
                _log(f"[RUN] {sym}: {start_ms_all}..{end_ms_all} (UTC ms), окна ~{args.window_days}d")
                futs.append(ex.submit(
                    export_symbol_to_temp_csv, args.market, api_key, api_secret, sym, start_ms_all, end_ms_all, window_ms
                ))
            for fut in as_completed(futs):
                try:
                    tmp_files.append(fut.result())
                except Exception as e:
                    _log(f"[WORKER] ошибка: {e}")

        # сводим и дедупим
        if tmp_files:
            _log("[MERGE] объединяем все символы в единый экспорт …")
            # читаем все и пишем в единый tmp → дедупим → переносим на out
            all_rows = []
            all_fields = list(FIELDS)
            for p in tmp_files:
                flds, rows = read_csv_rows(p)
                for f in flds:
                    if f not in all_fields:
                        all_fields.append(f)
                all_rows.extend(rows)
            # дедуп
            seen = set()
            out_rows = []
            for r in all_rows:
                k = dedup_key(r)
                if k in seen:
                    continue
                seen.add(k)
                out_rows.append(r)
            # сортировка по ts
            try:
                out_rows.sort(key=lambda x: int(x.get("ts") or 0))
            except Exception:
                pass
            # запись
            write_csv_rows(args.out, all_fields, out_rows)
            _log(f"[CSV] общий экспорт готов: {len(out_rows)} строк → {args.out}")

        for p in tmp_files:
            try:
                shutil.rmtree(os.path.dirname(p), ignore_errors=True)
            except Exception:
                pass

        _log("[DONE] Экспорт без merge завершён.")


if __name__ == "__main__":
    main()
