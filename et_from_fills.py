# -*- coding: utf-8 -*-
"""
et_from_fills.py — чистый, рабочий парсер под формат твоего fills_all.csv
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_ts(ts_raw: str) -> datetime:
    """Парсер Bybit timestamps (миллисекунды или секунды)."""
    ts_raw = ts_raw.strip()
    if not ts_raw:
        raise ValueError("empty ts")

    # если чистое число
    if ts_raw.isdigit():
        v = float(ts_raw)
        # миллисекунды: 1e11+
        if abs(v) > 1e11:
            v /= 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc)

    # ISO формат на случай JSONL
    try:
        return datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except:
        pass

    raise ValueError(f"Unknown ts format: {ts_raw}")


def load_fills_csv(path: str) -> List[Dict[str, Any]]:
    """Рабочий парсер Bybit CSV — читает ВСЕ строки."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            # безопасный float
            def ffloat(x):
                try:
                    return float(x)
                except:
                    return 0.0

            try:
                ts_main = row.get("ts") or ""
                ts_val = parse_ts(ts_main)
            except:
                # строка битая → пропускаем
                continue

            out.append({
                "ts": ts_val,
                "symbol": row.get("symbol", "").upper(),
                "side": row.get("side", "").lower(),
                "price": ffloat(row.get("price")),
                "qty": ffloat(row.get("qty")),
                "fee": ffloat(row.get("fee")),
                "feeCurrency": row.get("feeCurrency", ""),
                "isMaker": row.get("isMaker", ""),
                "orderId": row.get("orderId", ""),
                "execId": row.get("execId", ""),
                "value": ffloat(row.get("value")),
                "orderLinkId": row.get("orderLinkId", ""),
                "category": row.get("category", ""),
            })

    return out


def build_equity(fills: List[Dict[str, Any]], start_equity: float = 0.0):
    """Строим equity на основе fees (realized_pnl у тебя отсутствует)."""
    eq = start_equity
    out = []

    # сортировка по времени
    fills_sorted = sorted(fills, key=lambda x: x["ts"])

    for r in fills_sorted:
        # PnL нет → делаем только комиссионные
        pnl = 0.0
        fee = abs(r["fee"])

        eq -= fee  # equity уменьшается на комиссию

        out.append({
            "ts": r["ts"],
            "symbol": r["symbol"],
            "side": r["side"],
            "qty": r["qty"],
            "price": r["price"],
            "pnl": pnl,
            "fee": fee,
            "equity": eq
        })

    return out


def save_equity_json(path: str, table: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in table:
            obj = r.copy()
            obj["ts"] = obj["ts"].isoformat()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python et_from_fills.py fills_all.csv")
        return

    src = sys.argv[1]
    fills = load_fills_csv(src)
    print(f"[OK] Загружено строк: {len(fills)}")

    eq = build_equity(fills)

    save_equity_json("equity.jsonl", eq)

    print("[OK] Сохранено: equity.jsonl")


if __name__ == "__main__":
    main()
