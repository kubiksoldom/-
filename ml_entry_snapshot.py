"""Безопасное сохранение признаков, известных строго в момент входа."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


ML_ENTRY_SNAPSHOT_VERSION = 2

# Это ровно те признаки, которые текущий runtime умеет вычислить до отправки
# заявки. Поля результата сделки сюда намеренно не входят.
ML_ENTRY_FEATURES = (
    "index_price",
    "last_price",
    "high",
    "low",
    "vol_24h",
    "open_interest",
    "funding_rate",
    "rsi",
    "atr_abs",
    "atr_norm",
    "atr_pct",
    "bb_width",
    "zscore",
    "adx_like",
    "volatility",
    "hour",
    "weekday",
    "pct_from_high",
    "dist_to_index",
    "price_delta_5m",
    "volume_sum_5m",
    "price_delta_15m",
    "volume_sum_15m",
    "spread_bps",
    "fee_bps",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "mom_k",
    "book_imbalance",
    "oi_change",
    "qty",
    "direction",
    "tp_pct_used",
    "sl_pct_used",
)

ML_ENTRY_SNAPSHOT_FIELDS = (
    "schema_version",
    "trade_id",
    "session_id",
    "git_sha",
    "symbol",
    "direction",
    "captured_at",
    "paper",
    "strategy",
    "regime",
    "planned_tp_pct",
    "planned_sl_pct",
    "model_probability",
    "model_threshold",
    "model_applied",
    "features",
)

_SNAPSHOT_LOCK = threading.Lock()


class MLEntrySnapshotError(ValueError):
    """Снимок признаков не соответствует безопасной схеме."""


class MLEntrySnapshotConflictError(MLEntrySnapshotError):
    """Один trade_id связан с разными снимками входа."""


def _text(record: Mapping[str, Any], key: str, *, required: bool) -> Optional[str]:
    value = record.get(key)
    normalized = str(value or "").strip()
    if required and not normalized:
        raise MLEntrySnapshotError(f"поле {key!r} обязательно")
    return normalized or None


def _number(
    value: Any,
    key: str,
    *,
    required: bool = True,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        if required:
            raise MLEntrySnapshotError(f"поле {key!r} обязательно")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise MLEntrySnapshotError(f"поле {key!r} должно быть числом") from None
    if not math.isfinite(parsed):
        raise MLEntrySnapshotError(f"поле {key!r} должно быть конечным числом")
    if minimum is not None and parsed < minimum:
        raise MLEntrySnapshotError(f"поле {key!r} должно быть не меньше {minimum}")
    if maximum is not None and parsed > maximum:
        raise MLEntrySnapshotError(f"поле {key!r} должно быть не больше {maximum}")
    return parsed


def _utc_timestamp(value: Any, key: str) -> str:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            raise MLEntrySnapshotError(f"поле {key!r} содержит неверное время") from None
    if parsed.tzinfo is None:
        raise MLEntrySnapshotError(f"поле {key!r} должно содержать часовой пояс")
    return parsed.astimezone(dt.timezone.utc).isoformat()


def _direction(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {"buy": "long", "long": "long", "sell": "short", "short": "short"}
    if raw not in aliases:
        raise MLEntrySnapshotError(f"неподдерживаемое направление: {raw!r}")
    return aliases[raw]


def normalize_ml_entry_snapshot(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Проверить снимок и вернуть поля в каноническом порядке."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    missing = [field for field in ML_ENTRY_SNAPSHOT_FIELDS if field not in record]
    if missing:
        raise MLEntrySnapshotError(f"отсутствуют поля снимка: {missing}")
    unexpected = sorted(set(record) - set(ML_ENTRY_SNAPSHOT_FIELDS))
    if unexpected:
        raise MLEntrySnapshotError(f"неожиданные поля снимка: {unexpected}")

    try:
        schema_version = int(record.get("schema_version"))
    except (TypeError, ValueError):
        raise MLEntrySnapshotError("schema_version должен быть целым числом") from None
    if schema_version != ML_ENTRY_SNAPSHOT_VERSION:
        raise MLEntrySnapshotError(
            f"поддерживается только schema_version={ML_ENTRY_SNAPSHOT_VERSION}"
        )

    direction = _direction(record.get("direction"))
    paper = record.get("paper")
    model_applied = record.get("model_applied")
    if not isinstance(paper, bool):
        raise MLEntrySnapshotError("поле 'paper' должно быть логическим")
    if not isinstance(model_applied, bool):
        raise MLEntrySnapshotError("поле 'model_applied' должно быть логическим")

    raw_features = record.get("features")
    if not isinstance(raw_features, Mapping):
        raise MLEntrySnapshotError("поле 'features' должно быть объектом")
    missing_features = [name for name in ML_ENTRY_FEATURES if name not in raw_features]
    unexpected_features = sorted(set(raw_features) - set(ML_ENTRY_FEATURES))
    if missing_features:
        raise MLEntrySnapshotError(f"отсутствуют признаки: {missing_features}")
    if unexpected_features:
        raise MLEntrySnapshotError(f"неожиданные признаки: {unexpected_features}")

    features = {
        name: _number(raw_features.get(name), f"features.{name}")
        for name in ML_ENTRY_FEATURES
    }
    if float(features["last_price"]) <= 0.0:
        raise MLEntrySnapshotError("features.last_price должен быть больше нуля")
    if float(features["qty"]) <= 0.0:
        raise MLEntrySnapshotError("features.qty должен быть больше нуля")
    if not 0.0 <= float(features["hour"]) <= 23.0:
        raise MLEntrySnapshotError("features.hour должен быть в диапазоне 0..23")
    if not 0.0 <= float(features["weekday"]) <= 6.0:
        raise MLEntrySnapshotError("features.weekday должен быть в диапазоне 0..6")
    expected_direction = 1.0 if direction == "long" else -1.0
    if not math.isclose(float(features["direction"]), expected_direction, abs_tol=1e-12):
        raise MLEntrySnapshotError("features.direction не совпадает с направлением сделки")

    planned_tp_pct = _number(
        record.get("planned_tp_pct"), "planned_tp_pct", required=False, minimum=0.0
    )
    planned_sl_pct = _number(
        record.get("planned_sl_pct"), "planned_sl_pct", required=False, minimum=0.0
    )
    model_probability = _number(
        record.get("model_probability"),
        "model_probability",
        required=False,
        minimum=0.0,
        maximum=1.0,
    )
    model_threshold = _number(
        record.get("model_threshold"),
        "model_threshold",
        required=False,
        minimum=0.0,
        maximum=1.0,
    )

    normalized: Dict[str, Any] = {
        "schema_version": schema_version,
        "trade_id": _text(record, "trade_id", required=True),
        "session_id": _text(record, "session_id", required=True),
        "git_sha": _text(record, "git_sha", required=False),
        "symbol": str(_text(record, "symbol", required=True)).upper(),
        "direction": direction,
        "captured_at": _utc_timestamp(record.get("captured_at"), "captured_at"),
        "paper": paper,
        "strategy": _text(record, "strategy", required=False),
        "regime": _text(record, "regime", required=False),
        "planned_tp_pct": planned_tp_pct,
        "planned_sl_pct": planned_sl_pct,
        "model_probability": model_probability,
        "model_threshold": model_threshold,
        "model_applied": model_applied,
        "features": {name: float(features[name]) for name in ML_ENTRY_FEATURES},
    }
    return {field: normalized[field] for field in ML_ENTRY_SNAPSHOT_FIELDS}


def read_ml_entry_snapshots(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    """Прочитать и проверить все непустые строки файла снимков."""
    target = Path(path).expanduser()
    if not target.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(normalize_ml_entry_snapshot(json.loads(line)))
            except Exception as exc:
                raise MLEntrySnapshotError(
                    f"неверная строка снимков {line_number}: {exc}"
                ) from exc
    return rows


def append_ml_entry_snapshot(
    path: os.PathLike[str] | str,
    record: Mapping[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Идемпотентно дописать один снимок и синхронизировать его на диск."""
    normalized = normalize_ml_entry_snapshot(record)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    with _SNAPSHOT_LOCK:
        for existing in read_ml_entry_snapshots(target):
            if existing["trade_id"] != normalized["trade_id"]:
                continue
            if existing != normalized:
                raise MLEntrySnapshotConflictError(
                    f"trade_id {normalized['trade_id']!r} уже записан с другими признаками"
                )
            return normalized, False

        encoded = (
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("не удалось дописать снимок признаков")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    return normalized, True
