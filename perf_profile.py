#!/usr/bin/env python3
"""Lightweight performance profiler for main loop and router."""
from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List
import functools

os.environ.setdefault("PAPER_MODE", "1")
os.environ.setdefault("SAFE_MODE", "1")
os.environ.setdefault("PERF_PROFILE_LOOPS", "300")

import main  # noqa: E402  # load after env tweaks
import strategy  # noqa: E402
import ml_veto  # noqa: E402

stats: Dict[str, List[float]] = {
    "fetch_kline": [],
    "analyze": [],
    "ml_predict": [],
    "route": [],
    "paper_engine": [],
}


def _wrap(target: Any, name: str, bucket: str) -> None:
    if not hasattr(target, name):
        return
    original: Callable[..., Any] = getattr(target, name)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            stats[bucket].append(elapsed)

    setattr(target, name, wrapper)


_wrap(main.broker, "get_kline_any", "fetch_kline")
_wrap(strategy, "detect_candle_patterns", "analyze")
_wrap(strategy, "decide_with_router", "route")
_wrap(ml_veto, "predict_ok", "ml_predict")
_wrap(main.broker, "place_market_order", "paper_engine")


def _write_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["module", "avg_ms", "p95_ms", "calls"])
        for module, values in stats.items():
            if values:
                avg = statistics.fmean(values)
                p95 = statistics.quantiles(values, n=100)[94]
            else:
                avg = p95 = 0.0
            writer.writerow([module, f"{avg:.3f}", f"{p95:.3f}", len(values)])


def main_entry() -> None:
    start = time.time()
    try:
        main.main_trading_cycle()
    finally:
        _write_report(Path("perf_report.csv"))
        duration = time.time() - start
        print(f"[perf-profile] finished in {duration:.2f}s", flush=True)


if __name__ == "__main__":
    main_entry()
