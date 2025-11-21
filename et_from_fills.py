# -*- coding: utf-8 -*-
"""
Рабочий парсер Bybit fills_all.csv на основе ИНДЕКСОВ, а не DictReader.
Поддерживает BOM, дубликаты ts, пустые строки, spot/linear.
"""

import csv
import json
from datetime import datetime, timezone
from typing import List, Dict, Any


def parse_ts(ts_raw: str) -> datetime:
    ts_raw = ts_raw.strip()
    if not ts_raw:
        raise ValueError("empty ts")

    if ts_raw.isdigit():
        v = float(ts_raw)
        if v > 1e11:  # миллисекунды
            v /= 1000
        return datetime.fromtimestamp(v, tz=timezone.utc)

    return datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))


def load_fills_csv(path: str) -> List[Dict[str, Any]]:
    out = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        header = next(reader, None)
        if not header:
            print("[ERR] Empty CSV.")
            return []

        for row in reader:
            # пустая строка
            if not row or len(row) < 5:
                continue

            # строка может быть короче (из-за пустого последнего ts)
            row = row + [""] * (14 - len(row))

            # Извлекаем по ИНДЕКСАМ:
            ts_raw      = row[0]
            symbol      = row[1]
            side        = row[2]
            price       = row[3]
            qty         = row[4]
            fee         = row[5]
            feeCurrency = row[6]
            isMaker     = row[7]
            orderId     = row[8]
            execId      = row[9]
            value       = row[10]
            orderLinkId = row[11]
            category    = row[12]

            if not ts_raw.strip():
                continue

            try:
                ts_val = parse_ts(ts_raw)
            except Exception:
                continue

            def ffloat(x):
                try:
                    return float(x)
                except:
                    return 0.0

            out.append({
                "ts": ts_val,
                "symbol": symbol.upper(),
                "side": side.lower(),
                "price": ffloat(price),
                "qty": ffloat(qty),
                "fee": ffloat(fee),
                "feeCurrency": feeCurrency,
                "isMaker": isMaker,
                "orderId": orderId,
                "execId": execId,
                "value": ffloat(value),
                "orderLinkId": orderLinkId,
                "category": category,
            })

    return out


def build_equity(fills, start=0.0):
    eq = start
    out = []

    for r in sorted(fills, key=lambda x: x["ts"]):
        fee = abs(r["fee"])
        pnl = 0.0
        eq -= fee
        out.append({
            "ts": r["ts"].isoformat(),
            "equity": eq,
            "fee": fee,
            "pnl": pnl,
            "symbol": r["symbol"],
            "side": r["side"],
            "qty": r["qty"],
            "price": r["price"],
        })

    return out


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python et_from_fills.py fills_all.csv")
        return

    src = sys.argv[1]
    fills = load_fills_csv(src)
    print(f"[OK] Загружено строк: {len(fills)}")

    eq = build_equity(fills)
    with open("equity.jsonl", "w", encoding="utf-8") as f:
        for row in eq:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[OK] Сохранено в equity.jsonl")


if __name__ == "__main__":
    main()
