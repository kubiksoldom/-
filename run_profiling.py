#!/usr/bin/env python3
"""Static and runtime profiling helpers for the trading bot."""

from __future__ import annotations

import ast
import builtins
import cProfile
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "profiling" / "artifacts"


def _ensure_artifacts_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path


@dataclass
class ModuleInfo:
    name: str
    path: Path


def _collect_modules() -> Dict[str, ModuleInfo]:
    modules: Dict[str, ModuleInfo] = {}
    for path in _iter_python_files(ROOT):
        rel = path.relative_to(ROOT)
        if path.name == "__init__.py":
            if rel.parent == Path('.'):
                name = ""
            else:
                name = ".".join(rel.parent.parts)
        else:
            name = ".".join(rel.with_suffix("").parts)
        if not name:
            name = path.stem
        modules[name] = ModuleInfo(name=name, path=path)
    return modules


def build_dependency_map() -> Dict[str, List[str]]:
    modules = _collect_modules()
    available = set(modules.keys())
    dep_map: Dict[str, set[str]] = {name: set() for name in modules}

    for mod_name, info in modules.items():
        try:
            source = info.path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(source, filename=str(info.path))
        except Exception:
            continue

        for node in ast.walk(tree):
            target_name: Optional[str] = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target_name = alias.name
                    if target_name:
                        resolved = _resolve_import(target_name, available)
                        if resolved:
                            dep_map[mod_name].add(resolved)
            elif isinstance(node, ast.ImportFrom):
                if node.level and mod_name:
                    parent_parts = mod_name.split('.')
                    if node.level <= len(parent_parts):
                        base = parent_parts[:-node.level]
                    else:
                        base = []
                    module_parts = node.module.split('.') if node.module else []
                    candidate = ".".join(part for part in (*base, *module_parts) if part)
                    if candidate:
                        resolved = _resolve_import(candidate, available)
                        if resolved:
                            dep_map[mod_name].add(resolved)
                else:
                    target_name = node.module or ""
                    if target_name:
                        resolved = _resolve_import(target_name, available)
                        if resolved:
                            dep_map[mod_name].add(resolved)
    return {mod: sorted(deps) for mod, deps in sorted(dep_map.items()) if mod}


def _resolve_import(name: str, available: set[str]) -> Optional[str]:
    if name in available:
        return name
    parts = name.split('.')
    while parts:
        candidate = ".".join(parts)
        if candidate in available:
            return candidate
        parts.pop()
    root = name.split('.')[0]
    if root in available:
        return root
    return None


class FileIOCounter:
    """Track file IO by wrapping builtins.open."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._orig_open: Optional[Callable[..., Any]] = None

    def __enter__(self) -> "FileIOCounter":
        self._orig_open = builtins.open

        def tracked_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
            fh = self._orig_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]
            record = {
                "path": str(file),
                "mode": mode,
                "bytes_read": 0,
                "bytes_written": 0,
            }
            wrapper = _FileWrapper(fh, record)
            self.records.append(record)
            return wrapper

        builtins.open = tracked_open  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._orig_open is not None:
            builtins.open = self._orig_open  # type: ignore[assignment]
        self._orig_open = None


class _FileWrapper:
    def __init__(self, fh: Any, record: Dict[str, Any]) -> None:
        self._fh = fh
        self._record = record

    def write(self, data: Any) -> Any:
        size = _estimate_size(data)
        self._record["bytes_written"] += size
        return self._fh.write(data)

    def writelines(self, lines: Iterable[Any]) -> Any:
        total = 0
        for line in lines:
            total += _estimate_size(line)
        self._record["bytes_written"] += total
        return self._fh.writelines(lines)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        data = self._fh.read(*args, **kwargs)
        self._record["bytes_read"] += _estimate_size(data)
        return data

    def readline(self, *args: Any, **kwargs: Any) -> Any:
        data = self._fh.readline(*args, **kwargs)
        self._record["bytes_read"] += _estimate_size(data)
        return data

    def readlines(self, *args: Any, **kwargs: Any) -> Any:
        data = self._fh.readlines(*args, **kwargs)
        total = sum(_estimate_size(line) for line in data)
        self._record["bytes_read"] += total
        return data

    def __iter__(self):
        for line in self._fh:
            self._record["bytes_read"] += _estimate_size(line)
            yield line

    def __getattr__(self, item: str) -> Any:
        return getattr(self._fh, item)

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._fh.__exit__(exc_type, exc, tb)


def _estimate_size(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, bytes):
        return len(data)
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    if isinstance(data, Iterable):
        try:
            return sum(_estimate_size(x) for x in data)
        except TypeError:
            return 0
    return 0


def _format_func(func: Tuple[str, int, str]) -> str:
    file, line, name = func
    try:
        rel = Path(file).resolve().relative_to(ROOT)
    except Exception:
        rel = Path(file).name
    return f"{rel}:{line}:{name}"


def _profile_callable(label: str, runner: Callable[[], Any]) -> Dict[str, Any]:
    import tracemalloc
    profiler = cProfile.Profile()
    cpu_top: List[Dict[str, Any]] = []
    mem_top: List[Dict[str, Any]] = []
    io_records: List[Dict[str, Any]] = []
    error: Optional[str] = None

    start_time = time.perf_counter()
    with FileIOCounter() as counter:
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        try:
            profiler.enable()
            runner()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            profiler.disable()
            snap_after = tracemalloc.take_snapshot()
            tracemalloc.stop()
    duration = time.perf_counter() - start_time
    io_records = counter.records

    stats = profiler.getstats()
    ps = None
    if stats:
        import pstats
        ps = pstats.Stats(profiler)
        ps.sort_stats("cumulative")
        cpu_top = _extract_cpu(ps)
    mem_stats = snap_after.compare_to(snap_before, "lineno")
    mem_top = _extract_mem(mem_stats)

    return {
        "label": label,
        "duration_sec": duration,
        "cpu_top": cpu_top,
        "mem_top": mem_top,
        "io": io_records,
        "error": error,
    }


def _extract_cpu(stats_obj: "pstats.Stats", limit: int = 15) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for func, stat in stats_obj.stats.items():
        cc, nc, tt, ct, _callers = stat
        rows.append(
            {
                "func": _format_func(func),
                "ncalls": nc,
                "tottime": tt,
                "cumtime": ct,
                "percall": ct / nc if nc else 0.0,
            }
        )
    rows.sort(key=lambda row: row["cumtime"], reverse=True)
    return rows[:limit]


def _extract_mem(mem_stats: List[Any], limit: int = 15) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stat in mem_stats[:limit]:
        frame = stat.traceback[0]
        try:
            rel = Path(frame.filename).resolve().relative_to(ROOT)
        except Exception:
            rel = Path(frame.filename).name
        rows.append(
            {
                "location": f"{rel}:{frame.lineno}",
                "size_kb": round(stat.size_diff / 1024.0, 4),
                "count": stat.count_diff,
            }
        )
    return rows


# ===== Profiling runners =====

def _run_main_cycle_stub() -> None:
    import importlib
    from unittest import mock

    os.environ.setdefault("PAPER_MODE", "1")
    os.environ.setdefault("SAFE_MODE", "1")

    class StubBroker:
        def __init__(self) -> None:
            self._positions: Dict[str, Dict[str, Any]] = {}

        def get_balance(self) -> float:
            return 10_000.0

        def get_equity(self) -> float:
            return 10_000.0

        def get_kline_any(self, symbol: str, interval: str = "1", limit: int = 300):
            candles = []
            base = 100.0 + random.random() * 0.1
            for idx in range(limit):
                open_px = base + math.sin(idx / 10.0) * 0.2
                close_px = open_px + math.sin(idx / 5.0) * 0.1
                high_px = max(open_px, close_px) + 0.15
                low_px = min(open_px, close_px) - 0.15
                volume = 1000 + idx * 3
                candles.append([open_px, high_px, low_px, close_px, volume])
            return candles, {"symbol": symbol}

        def get_ticker_snapshot(self, symbol: str) -> Dict[str, Any]:
            return {
                "symbol": symbol,
                "last_price": 100.0,
                "index_price": 100.0,
                "bid1_price": 99.9,
                "ask1_price": 100.1,
                "vol_24h": 5_000_000,
            }

        def get_min_order_filters(self, symbol: str) -> Tuple[float, float, float]:
            return 0.001, 0.0001, 5.0

        def filters_reliable(self, symbol: str) -> bool:
            return True

        def get_tickers_linear(self) -> Dict[str, Any]:
            return {"result": {"list": []}}

        def get_current_price(self, symbol: str) -> float:
            return 100.0

        def get_symbol_leverage_limits(self, symbol: str) -> Tuple[int, int]:
            return 1, 50

        def set_leverage(self, symbol: str, leverage: int) -> bool:
            return True

        def get_instruments_info(self) -> Dict[str, Any]:
            return {"result": {"list": []}}

        def get_positions(self) -> Dict[str, Any]:
            return {"result": {"list": []}}

        def close_position_by_market(self, symbol: str, qty: float, **kwargs: Any) -> Dict[str, Any]:
            return {"symbol": symbol, "closed": qty}

        def get_margin_info(self) -> Dict[str, Any]:
            return {"IM": 5.0, "MM": 2.0, "equity": 10_000.0}

        def has_open_position(self, symbol: str) -> bool:
            return False

        def place_market_order(self, symbol: str, side: str, qty: float, **kwargs: Any) -> Dict[str, Any]:
            self._positions[symbol] = {"side": side, "qty": qty}
            return {"orderId": f"stub-{symbol}-{side}"}

    stub_broker = StubBroker()

    dummy_decision = {"action": "hold", "reason": "profiling", "sl": None, "tp": None, "meta": {}}

    def _stub_decide(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return dict(dummy_decision)

    with tempfile.TemporaryDirectory() as tmpdir:
        prev_data_root = os.environ.get("DATA_ROOT")
        os.environ["DATA_ROOT"] = tmpdir

        try:
            config_mod = importlib.import_module("config")
            try:
                config_mod.PAPER_MODE = 1
            except Exception:
                pass
            try:
                config_mod.SAFE_MODE = 1
            except Exception:
                pass
            try:
                config_mod.PAPER_SYNC_BALANCE = 0
            except Exception:
                pass
            try:
                config_mod.TOP_LIQUID_PAIRS = ["ETHUSDT"]
            except Exception:
                pass

            main = importlib.import_module("main")

            tmp_control = Path(tmpdir) / "control.json"
            tmp_control.write_text("[]", encoding="utf-8")

            patches = [
                mock.patch.object(main, "broker", stub_broker),
                mock.patch.object(main, "RUN_ONCE", True),
                mock.patch.object(main, "PAPER_MODE", True),
                mock.patch.object(main, "SAFE_MODE", True),
                mock.patch.object(main, "DD_CHECK_EVERY_SEC", 1e9),
                mock.patch.object(main, "MARGIN_POLL_SEC", 1e9),
                mock.patch.object(main, "LOOP_SLEEP_SEC", 0.0),
                mock.patch.object(main, "KLINE_HISTORY_LIMIT", 60),
                mock.patch.object(main, "AUTO_PAIRS_RULE", "balance:1"),
                mock.patch("main._pairs_count_for_balance", return_value=1),
                mock.patch("main._filter_by_linear_availability", side_effect=lambda pairs: list(pairs)),
                mock.patch("main._filter_universe_by_notional", side_effect=lambda pairs, _bal: list(pairs)),
                mock.patch(
                    "main.select_top_pairs",
                    side_effect=lambda pairs, count=1: list(pairs)[: max(1, count or 1)],
                ),
                mock.patch("main.score_symbol", return_value=1.0),
                mock.patch("main.schedule_status", return_value=(True, "", None)),
                mock.patch("main.load_model_and_meta", return_value=(None, {})),
                mock.patch("main.decide_with_router", side_effect=_stub_decide),
                mock.patch("main.write_cycle_log", side_effect=lambda data: None),
                mock.patch("main.tg_send", side_effect=lambda *_args, **_kwargs: None),
                mock.patch("main.time.sleep", side_effect=lambda _sec: None),
                mock.patch("main._restore_pause_state", side_effect=lambda *a, **k: None),
                mock.patch("main._persist_pause_state", side_effect=lambda *a, **k: None),
                mock.patch("main._control_path", return_value=tmp_control),
            ]
            with _patches(patches):
                main.main_trading_cycle()
        finally:
            if prev_data_root is None:
                os.environ.pop("DATA_ROOT", None)
            else:
                os.environ["DATA_ROOT"] = prev_data_root


def _run_safe_request_profile() -> None:
    import api_guard
    from unittest import mock

    attempt = {"n": 0}

    def flaky_call() -> Dict[str, Any]:
        attempt["n"] += 1
        if attempt["n"] % 3:
            raise RuntimeError("read timed out")
        return {"retCode": 0, "result": {"ok": True}}

    patches = [
        mock.patch("api_guard._bucket.take", return_value=True),
        mock.patch("api_guard._respect_min_delay", side_effect=lambda: None),
        mock.patch("api_guard._sleep_backoff", side_effect=lambda *a, **k: None),
        mock.patch("api_guard.time.sleep", side_effect=lambda _sec: None),
    ]

    def runner() -> None:
        for _ in range(20):
            attempt["n"] = 0
            api_guard.safe_request(flaky_call, max_tries=5)

    with _patches(patches):
        runner()


def _run_strategy_decide_profile() -> None:
    import strategy
    from unittest import mock

    random.seed(42)
    candles: List[List[float]] = []
    base = 1500.0
    for idx in range(240):
        open_px = base + math.sin(idx / 15.0) * 5
        close_px = open_px + math.sin(idx / 7.5) * 3
        high_px = max(open_px, close_px) + 2.5
        low_px = min(open_px, close_px) - 2.5
        volume = 1000 + idx * 2
        candles.append([open_px, high_px, low_px, close_px, volume])

    strategy._router_singleton = None  # reset cached router

    patches = [
        mock.patch("strategy.StrategyRouter._load", return_value={}),
        mock.patch("strategy.StrategyRouter._save", side_effect=lambda self: None),
        mock.patch("strategy.write_cycle_log", side_effect=lambda data: None),
        mock.patch("strategy.log", side_effect=lambda *_a, **_k: None),
    ]

    with _patches(patches):
        for _ in range(20):
            strategy.decide_with_router("ETHUSDT", "1m", candles, {"latest_price": candles[-1][3]})


def _run_tg_send_profile() -> None:
    import utils
    from unittest import mock

    class DummyResponse:
        def __init__(self, ok: bool = True, status_code: int = 200, text: str = "OK") -> None:
            self.ok = ok
            self.status_code = status_code
            self.text = text

    class DummySession:
        def post(self, url: str, data: Dict[str, Any], timeout: float = 12) -> DummyResponse:
            return DummyResponse()

    utils.TELEGRAM_TOKEN = "token"
    utils.TELEGRAM_CHAT_ID = "chat"

    with mock.patch("utils._get_tg_session", return_value=DummySession()):
        for _ in range(20):
            utils.tg_send("Профилирование Telegram отправки" * 3)


def _run_write_cycle_log_profile() -> None:
    import utils
    from unittest import mock

    utils.LOG_ENABLED = True

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cycle.jsonl"
        with mock.patch("utils._ensure_log_directory", return_value=path):
            for i in range(100):
                utils.write_cycle_log({"tag": "profiling", "index": i, "message": "log line"})


class _patches:
    def __init__(self, patches: List[Any]) -> None:
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        for p in reversed(self._patches):
            try:
                p.stop()
            except Exception:
                pass
        return False


def _aggregate_hotspots(reports: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    cpu_entries: List[Dict[str, Any]] = []
    mem_entries: List[Dict[str, Any]] = []
    io_summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"bytes_read": 0, "bytes_written": 0, "count": 0})

    for report in reports:
        label = report["label"]
        for entry in report.get("cpu_top", []):
            cpu_entries.append({"label": label, **entry})
        for entry in report.get("mem_top", []):
            mem_entries.append({"label": label, **entry})
        for rec in report.get("io", []):
            key = f"{rec['path']}|{rec['mode']}"
            acc = io_summary[key]
            acc["bytes_read"] += rec.get("bytes_read", 0)
            acc["bytes_written"] += rec.get("bytes_written", 0)
            acc["count"] += 1

    cpu_entries.sort(key=lambda row: row["cumtime"], reverse=True)
    mem_entries.sort(key=lambda row: abs(row["size_kb"]), reverse=True)

    io_rows = []
    for key, acc in io_summary.items():
        path, mode = key.split("|")
        io_rows.append({
            "path": path,
            "mode": mode,
            "bytes_read": acc["bytes_read"],
            "bytes_written": acc["bytes_written"],
            "count": acc["count"],
        })
    io_rows.sort(key=lambda row: row["bytes_written"] + row["bytes_read"], reverse=True)

    return {
        "cpu": cpu_entries[:10],
        "memory": mem_entries[:10],
        "io": io_rows[:10],
    }


def main() -> None:
    _ensure_artifacts_dir()

    dep_map = build_dependency_map()
    (ARTIFACT_DIR / "dependency_map.json").write_text(
        json.dumps(dep_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    profiles: List[Dict[str, Any]] = []

    profile_plan: List[Tuple[str, Callable[[], None]]] = [
        ("main.main_trading_cycle", _run_main_cycle_stub),
        ("api_guard.safe_request", _run_safe_request_profile),
        ("strategy.decide_with_router", _run_strategy_decide_profile),
        ("utils.tg_send", _run_tg_send_profile),
        ("utils.write_cycle_log", _run_write_cycle_log_profile),
    ]

    for label, runner in profile_plan:
        report = _profile_callable(label, runner)
        profiles.append(report)
        out_path = ARTIFACT_DIR / f"{label.replace('.', '_')}_profile.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    hotspots = _aggregate_hotspots(profiles)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profiles": profiles,
        "hotspots": hotspots,
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["Dependency map saved to profiling/artifacts/dependency_map.json", "Hotspots:"]
    lines.append("CPU Top 10:")
    for entry in hotspots["cpu"]:
        lines.append(
            f"  {entry['label']} → {entry['func']} | cum={entry['cumtime']:.6f}s tottime={entry['tottime']:.6f}s ncalls={entry['ncalls']}"
        )
    lines.append("Memory Top 10:")
    for entry in hotspots["memory"]:
        lines.append(
            f"  {entry['label']} → {entry['location']} | Δ={entry['size_kb']:.3f} KiB count={entry['count']}"
        )
    lines.append("IO Top 10:")
    for entry in hotspots["io"]:
        lines.append(
            f"  {entry['path']} ({entry['mode']}) | writes={entry['bytes_written']}B reads={entry['bytes_read']}B events={entry['count']}"
        )
    (ARTIFACT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
