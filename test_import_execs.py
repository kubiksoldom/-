"""Utility script for importing execution data from Bybit."""

import json
import time
from typing import Optional

from bybit_api import fetch_executions_range, normalize_fill


def import_execs(symbol: Optional[str] = None, days: int = 14) -> int:
    """Download fills for the requested period and dump them to JSONL.

    Parameters
    ----------
    symbol:
        Optional symbol filter passed to the exchange API.
    days:
        Number of trailing days to fetch. Defaults to two weeks.

    Returns
    -------
    int
        Number of records written to ``bybit_import_fills.jsonl``.
    """

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    fills = fetch_executions_range(
        category="linear", start_ms=start_ms, end_ms=end_ms, symbol=symbol
    )
    with open("bybit_import_fills.jsonl", "w", encoding="utf-8") as f:
        for record in fills:
            f.write(json.dumps(normalize_fill(record), ensure_ascii=False) + "\n")
    return len(fills)


def main() -> None:
    rows = import_execs()
    print("rows:", rows)
    print("saved: bybit_import_fills.jsonl")


if __name__ == "__main__":
    main()
