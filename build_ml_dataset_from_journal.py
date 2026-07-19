"""Сборка ML-выборки v2 из журнала сделок и снимков момента входа.

Модуль полностью автономный: он не запрашивает текущие котировки и не может
подмешать в старую сделку информацию из будущего.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ml_entry_snapshot import (
    ML_ENTRY_FEATURES,
    ML_ENTRY_SNAPSHOT_VERSION,
    normalize_ml_entry_snapshot,
    read_ml_entry_snapshots,
)
from trade_journal import normalize_trade_record, read_trade_records


ML_DATASET_VERSION = 2
ML_DATASET_META_FIELDS = (
    "dataset_version",
    "feature_schema_version",
    "trade_id",
    "session_id",
    "git_sha",
    "symbol",
    "direction",
    "entry_ts",
    "exit_ts",
    "duration_sec",
    "entry_price",
    "exit_price",
    "qty",
    "leverage",
    "entry_fee",
    "exit_fee",
    "funding",
    "slippage",
    "gross_pnl",
    "net_pnl",
    "net_return",
    "target",
    "exit_reason",
    "strategy",
    "regime",
    "ml_probability",
    "ml_threshold",
    "model_applied",
    "paper",
)
ML_DATASET_FIELDS = ML_DATASET_META_FIELDS + tuple(
    f"feature_{name}" for name in ML_ENTRY_FEATURES
)


class MLDatasetBuildError(ValueError):
    """Исходные записи нельзя безопасно превратить в выборку."""


def _parse_utc(value: Any, key: str) -> dt.datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        raise MLDatasetBuildError(f"поле {key!r} содержит неверное время") from None
    if parsed.tzinfo is None:
        raise MLDatasetBuildError(f"поле {key!r} должно содержать часовой пояс")
    return parsed.astimezone(dt.timezone.utc)


def _finite(value: Any, key: str, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise MLDatasetBuildError(f"поле {key!r} должно быть числом") from None
    if not math.isfinite(parsed):
        raise MLDatasetBuildError(f"поле {key!r} должно быть конечным числом")
    return parsed


def _deduplicate_by_trade_id(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    unique: Dict[str, Dict[str, Any]] = {}
    duplicate_rows = 0
    for raw in rows:
        row = dict(raw)
        trade_id = str(row.get("trade_id") or "").strip()
        if not trade_id:
            raise MLDatasetBuildError(f"{label}: отсутствует trade_id")
        existing = unique.get(trade_id)
        if existing is None:
            unique[trade_id] = row
            continue
        if existing != row:
            raise MLDatasetBuildError(
                f"{label}: trade_id {trade_id!r} содержит конфликтующие записи"
            )
        duplicate_rows += 1
    return unique, duplicate_rows


def _planned_distance(snapshot: Mapping[str, Any], key: str) -> float:
    value = _finite(snapshot.get(key), key, allow_none=True)
    if value is None or value <= 0.0:
        raise MLDatasetBuildError(f"{key} отсутствует или не больше нуля")
    return value


def build_ml_dataset_rows(
    journal_records: Sequence[Mapping[str, Any]],
    entry_snapshots: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Соединить две сущности один-к-одному по trade_id и вернуть строки v2."""
    try:
        normalized_journal = [normalize_trade_record(row) for row in journal_records]
    except Exception as exc:
        raise MLDatasetBuildError(f"журнал сделок повреждён: {exc}") from exc
    try:
        normalized_snapshots = [
            normalize_ml_entry_snapshot(row) for row in entry_snapshots
        ]
    except Exception as exc:
        raise MLDatasetBuildError(f"снимки входа повреждены: {exc}") from exc

    journal_by_id, duplicate_journal_rows = _deduplicate_by_trade_id(
        normalized_journal, label="журнал сделок"
    )
    snapshots_by_id, duplicate_snapshot_rows = _deduplicate_by_trade_id(
        normalized_snapshots, label="снимки входа"
    )

    report: Dict[str, Any] = {
        "dataset_version": ML_DATASET_VERSION,
        "journal_rows": len(journal_records),
        "snapshot_rows": len(entry_snapshots),
        "unique_journal_trades": len(journal_by_id),
        "unique_snapshots": len(snapshots_by_id),
        "duplicate_journal_rows": duplicate_journal_rows,
        "duplicate_snapshot_rows": duplicate_snapshot_rows,
        "missing_snapshot": [],
        "orphan_snapshot": [],
        "invalid_pairs": [],
        "duplicate_feature_vectors": 0,
        "conflicting_feature_vectors": 0,
        "valid_join_pairs": 0,
        "rows_output": 0,
        "target_positive": 0,
        "target_negative": 0,
    }

    report["orphan_snapshot"] = sorted(set(snapshots_by_id) - set(journal_by_id))
    prepared: List[Tuple[dt.datetime, Dict[str, Any]]] = []

    for trade_id, trade in journal_by_id.items():
        snapshot = snapshots_by_id.get(trade_id)
        if snapshot is None:
            report["missing_snapshot"].append(trade_id)
            continue
        try:
            symbol = str(trade.get("symbol") or "").strip().upper()
            direction = str(trade.get("direction") or "").strip().lower()
            if symbol != str(snapshot.get("symbol") or "").strip().upper():
                raise MLDatasetBuildError("symbol не совпадает")
            if direction != str(snapshot.get("direction") or "").strip().lower():
                raise MLDatasetBuildError("direction не совпадает")
            if bool(trade.get("paper")) != bool(snapshot.get("paper")):
                raise MLDatasetBuildError("paper не совпадает")
            if str(trade.get("session_id") or "") != str(snapshot.get("session_id") or ""):
                raise MLDatasetBuildError("session_id не совпадает")
            trade_git_sha = str(trade.get("git_sha") or "").strip()
            snapshot_git_sha = str(snapshot.get("git_sha") or "").strip()
            if trade_git_sha and snapshot_git_sha and trade_git_sha != snapshot_git_sha:
                raise MLDatasetBuildError("git_sha не совпадает")

            entry_dt = _parse_utc(trade.get("entry_ts"), "entry_ts")
            exit_dt = _parse_utc(trade.get("exit_ts"), "exit_ts")
            captured_dt = _parse_utc(snapshot.get("captured_at"), "captured_at")
            if exit_dt < entry_dt:
                raise MLDatasetBuildError("выход раньше входа")
            if captured_dt > exit_dt:
                raise MLDatasetBuildError("снимок признаков создан после выхода")
            if abs((captured_dt - entry_dt).total_seconds()) > 300.0:
                raise MLDatasetBuildError("снимок признаков дальше пяти минут от входа")

            planned_tp_pct = _planned_distance(snapshot, "planned_tp_pct")
            planned_sl_pct = _planned_distance(snapshot, "planned_sl_pct")
            raw_features = snapshot.get("features")
            if not isinstance(raw_features, Mapping):
                raise MLDatasetBuildError("features отсутствуют")
            features = {
                name: float(_finite(raw_features.get(name), f"features.{name}"))
                for name in ML_ENTRY_FEATURES
            }

            # Направление, время и план берём из проверенных сущностей, а не из
            # произвольных значений в файле признаков.
            features["direction"] = 1.0 if direction == "long" else -1.0
            features["hour"] = float(entry_dt.hour)
            features["weekday"] = float(entry_dt.weekday())
            features["tp_pct_used"] = planned_tp_pct
            features["sl_pct_used"] = planned_sl_pct

            entry_price = float(_finite(trade.get("entry_price"), "entry_price"))
            exit_price = float(_finite(trade.get("exit_price"), "exit_price"))
            qty = float(_finite(trade.get("qty"), "qty"))
            gross_pnl = float(_finite(trade.get("gross_pnl"), "gross_pnl"))
            net_pnl = float(_finite(trade.get("net_pnl"), "net_pnl"))
            if entry_price <= 0.0 or exit_price <= 0.0 or qty <= 0.0:
                raise MLDatasetBuildError("цены и количество должны быть больше нуля")
            notional = entry_price * qty
            net_return = net_pnl / notional if notional > 0.0 else 0.0

            row: Dict[str, Any] = {
                "dataset_version": ML_DATASET_VERSION,
                "feature_schema_version": int(
                    snapshot.get("schema_version") or ML_ENTRY_SNAPSHOT_VERSION
                ),
                "trade_id": trade_id,
                "session_id": trade.get("session_id"),
                "git_sha": trade.get("git_sha"),
                "symbol": symbol,
                "direction": direction,
                "entry_ts": entry_dt.isoformat(),
                "exit_ts": exit_dt.isoformat(),
                "duration_sec": (exit_dt - entry_dt).total_seconds(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "leverage": _finite(trade.get("leverage"), "leverage", allow_none=True),
                "entry_fee": float(_finite(trade.get("entry_fee"), "entry_fee")),
                "exit_fee": float(_finite(trade.get("exit_fee"), "exit_fee")),
                "funding": _finite(trade.get("funding"), "funding", allow_none=True),
                "slippage": float(_finite(trade.get("slippage"), "slippage")),
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "target": 1 if net_pnl > 0.0 else 0,
                "exit_reason": trade.get("exit_reason"),
                "strategy": trade.get("strategy"),
                "regime": trade.get("regime"),
                "ml_probability": _finite(
                    trade.get("ml_probability"), "ml_probability", allow_none=True
                ),
                "ml_threshold": _finite(
                    trade.get("ml_threshold"), "ml_threshold", allow_none=True
                ),
                "model_applied": bool(snapshot.get("model_applied")),
                "paper": bool(trade.get("paper")),
            }
            for name in ML_ENTRY_FEATURES:
                row[f"feature_{name}"] = features[name]
            prepared.append((entry_dt, row))
        except Exception as exc:
            report["invalid_pairs"].append({"trade_id": trade_id, "reason": str(exc)})

    prepared.sort(key=lambda item: (item[0], str(item[1]["trade_id"])))
    report["valid_join_pairs"] = len(prepared)

    # Точная копия вектора не должна оказаться одновременно в прошлом и
    # будущем временном окне. Оставляем только самое раннее наблюдение.
    seen_vectors: Dict[Tuple[float, ...], int] = {}
    rows: List[Dict[str, Any]] = []
    for _entry_dt, row in prepared:
        vector = tuple(float(row[f"feature_{name}"]) for name in ML_ENTRY_FEATURES)
        previous_target = seen_vectors.get(vector)
        if previous_target is not None:
            report["duplicate_feature_vectors"] += 1
            if previous_target != int(row["target"]):
                report["conflicting_feature_vectors"] += 1
            continue
        seen_vectors[vector] = int(row["target"])
        rows.append({field: row.get(field) for field in ML_DATASET_FIELDS})

    report["missing_snapshot"] = sorted(report["missing_snapshot"])
    report["rows_output"] = len(rows)
    report["target_positive"] = sum(int(row["target"]) for row in rows)
    report["target_negative"] = len(rows) - int(report["target_positive"])
    report["join_coverage"] = len(prepared) / max(1, len(journal_by_id))
    report["retention_after_vector_dedup"] = len(rows) / max(1, len(prepared))
    if rows:
        report["entry_start"] = rows[0]["entry_ts"]
        report["entry_end"] = rows[-1]["entry_ts"]
    else:
        report["entry_start"] = None
        report["entry_end"] = None
    return rows, report


def _atomic_text_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_ml_dataset(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path).expanduser()

    def _write(handle) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=ML_DATASET_FIELDS)
        csv_writer.writeheader()
        for row in rows:
            unexpected = sorted(set(row) - set(ML_DATASET_FIELDS))
            if unexpected:
                raise MLDatasetBuildError(f"неожиданные поля выборки: {unexpected}")
            csv_writer.writerow({field: row.get(field) for field in ML_DATASET_FIELDS})

    _atomic_text_write(target, _write)


def write_build_report(path: os.PathLike[str] | str, report: Mapping[str, Any]) -> None:
    target = Path(path).expanduser()

    def _write(handle) -> None:
        json.dump(dict(report), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    _atomic_text_write(target, _write)


def build_from_files(
    journal_path: os.PathLike[str] | str,
    snapshot_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    report_path: os.PathLike[str] | str,
) -> Dict[str, Any]:
    journal_records = read_trade_records(journal_path)
    snapshots = read_ml_entry_snapshots(snapshot_path)
    rows, report = build_ml_dataset_rows(journal_records, snapshots)
    write_ml_dataset(output_path, rows)
    write_build_report(report_path, report)
    return report


def main() -> int:
    data_root = Path(os.getenv("DATA_ROOT", "./data")).expanduser()
    parser = argparse.ArgumentParser(
        description="Собрать автономную ML-выборку v2 из закрытых сделок"
    )
    parser.add_argument(
        "--journal", default=str(data_root / "trade_journal.jsonl")
    )
    parser.add_argument(
        "--snapshots", default=str(data_root / "ml_entry_snapshots.jsonl")
    )
    parser.add_argument(
        "--out", default=str(data_root / "ml_dataset_v2.csv")
    )
    parser.add_argument(
        "--report", default=str(data_root / "ml_dataset_v2_report.json")
    )
    args = parser.parse_args()

    report = build_from_files(args.journal, args.snapshots, args.out, args.report)
    print(
        "[ML-DATASET] "
        f"rows={report['rows_output']} "
        f"coverage={report['join_coverage']:.3f} "
        f"missing_snapshots={len(report['missing_snapshot'])} "
        f"duplicates={report['duplicate_feature_vectors']}"
    )
    if report["invalid_pairs"]:
        print(f"[ML-DATASET] invalid_pairs={len(report['invalid_pairs'])}")
    return 0 if report["rows_output"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
