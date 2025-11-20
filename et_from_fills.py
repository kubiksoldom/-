# -*- coding: utf-8 -*-
"""
Utility helpers for turning fills into an equity/effectiveness report.

The module is intentionally dependency-light so that it can be reused from
scripts, notebooks or sanity_check.py without pulling pandas.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "Fill",
    "load_fills",
    "generate_equity_table",
    "build_daily_summary",
    "save_equity_csv",
    "main",
]


@dataclass(order=True)
class Fill:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    realized_pnl: float = 0.0


def _parse_ts(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass
    raise ValueError(f"Unsupported timestamp value: {value!r}")


def _load_jsonl(path: Path) -> List[Fill]:
    fills: List[Fill] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            fills.append(_normalize_row(payload))
    return fills


def _load_csv(path: Path) -> List[Fill]:
    fills: List[Fill] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fills.append(_normalize_row(row))
    return fills


def _normalize_row(row: Dict[str, object]) -> Fill:
    ts = _parse_ts(row.get("ts") or row.get("timestamp") or row.get("time"))
    return Fill(
        ts=ts,
        symbol=str(row.get("symbol") or row.get("sym") or "").upper(),
        side=str(row.get("side") or row.get("direction") or "").lower(),
        qty=float(row.get("qty") or row.get("size") or 0.0),
        price=float(row.get("price") or row.get("executionPrice") or 0.0),
        fee=float(row.get("fee") or 0.0),
        realized_pnl=float(row.get("realized_pnl") or row.get("realizedPnl") or 0.0),
    )


def load_fills(path: str | Path) -> List[Fill]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported fills format: {path}")


def generate_equity_table(
    fills: Sequence[Fill],
    *,
    starting_equity: float = 0.0,
) -> List[Dict[str, object]]:
    equity = float(starting_equity)
    table: List[Dict[str, object]] = []
    for fill in sorted(fills):
        pnl = float(fill.realized_pnl or 0.0)
        fee = float(fill.fee or 0.0)
        equity += pnl - abs(fee)
        table.append(
            {
                "ts": fill.ts,
                "symbol": fill.symbol,
                "side": fill.side,
                "qty": float(fill.qty),
                "price": float(fill.price),
                "pnl": pnl,
                "fee": fee,
                "equity": equity,
            }
        )
    return table


def build_daily_summary(equity_table: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    current_date: Optional[str] = None
    for row in equity_table:
        ts = row.get("ts")
        if isinstance(ts, datetime):
            day = ts.date().isoformat()
        else:
            day = _parse_ts(ts).date().isoformat()
        if current_date != day:
            current_date = day
            current = {
                "date": day,
                "start_equity": row.get("equity"),
                "end_equity": row.get("equity"),
                "pnl": 0.0,
                "trades": 0,
            }
            summary.append(current)
        if current is None:
            continue
        current["end_equity"] = row.get("equity")
        current["trades"] = int(current.get("trades", 0)) + 1
        current["pnl"] = float(current["end_equity"]) - float(current["start_equity"])
    return summary


def save_equity_csv(equity_table: Sequence[Dict[str, object]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ts", "symbol", "side", "qty", "price", "pnl", "fee", "equity"]
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in equity_table:
            payload = {name: row.get(name) for name in fieldnames}
            ts = payload.get("ts")
            if isinstance(ts, datetime):
                payload["ts"] = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            writer.writerow(payload)
    return target


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an equity curve from fills.")
    parser.add_argument("source", help="Path to fills file (jsonl or csv)")
    parser.add_argument("--start", type=float, default=0.0, help="Starting equity")
    parser.add_argument(
        "--output",
        type=str,
        default="equity_report.csv",
        help="CSV file to store the generated equity table",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    fills = load_fills(args.source)
    equity_table = generate_equity_table(fills, starting_equity=args.start)
    save_equity_csv(equity_table, args.output)
    daily = build_daily_summary(equity_table)
    print(f"rows={len(equity_table)} daily={len(daily)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

