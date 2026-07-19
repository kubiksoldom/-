"""Append-only normalized journal for fully closed trading positions."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


TRADE_JOURNAL_FIELDS = (
    "trade_id",
    "session_id",
    "git_sha",
    "symbol",
    "direction",
    "entry_ts",
    "exit_ts",
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
    "exit_reason",
    "strategy",
    "regime",
    "atr_entry",
    "ml_probability",
    "ml_threshold",
    "paper",
)

TRADE_JOURNAL_EXIT_REASONS = frozenset(
    {
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "no_profit",
        "panic",
        "manual_force_close",
    }
)

_JOURNAL_LOCK = threading.Lock()


class TradeJournalError(ValueError):
    """Raised when a journal row or journal file is invalid."""


class TradeJournalConflictError(TradeJournalError):
    """Raised when one trade_id points to two different completed trades."""


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise TradeJournalError(f"поле {key!r} обязательно")
    return value


def _optional_text(record: Mapping[str, Any], key: str) -> Optional[str]:
    value = record.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _number(
    record: Mapping[str, Any],
    key: str,
    *,
    required: bool = True,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> Optional[float]:
    value = record.get(key)
    if value is None:
        if required:
            raise TradeJournalError(f"поле {key!r} обязательно")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise TradeJournalError(f"поле {key!r} должно быть числом") from None
    if not math.isfinite(parsed):
        raise TradeJournalError(f"поле {key!r} должно быть конечным числом")
    if strictly_positive and parsed <= 0:
        raise TradeJournalError(f"поле {key!r} должно быть больше нуля")
    if minimum is not None and parsed < minimum:
        raise TradeJournalError(f"поле {key!r} должно быть не меньше {minimum}")
    return parsed


def _timestamp(value: Any, key: str, *, required: bool) -> Optional[str]:
    if value in (None, ""):
        if required:
            raise TradeJournalError(f"поле {key!r} обязательно")
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = float(value)
        if not math.isfinite(raw):
            raise TradeJournalError(f"поле {key!r} содержит неверное время")
        if abs(raw) > 10_000_000_000:
            raw /= 1000.0
        parsed = dt.datetime.fromtimestamp(raw, tz=dt.timezone.utc)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            raise TradeJournalError(f"поле {key!r} содержит неверное время") from None
    if parsed.tzinfo is None:
        raise TradeJournalError(f"поле {key!r} должно содержать часовой пояс")
    return parsed.astimezone(dt.timezone.utc).isoformat()


def normalize_trade_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one completed trade and return the canonical field order."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    missing = [field for field in TRADE_JOURNAL_FIELDS if field not in record]
    if missing:
        raise TradeJournalError(f"отсутствуют поля журнала: {missing}")
    unexpected = sorted(set(record) - set(TRADE_JOURNAL_FIELDS))
    if unexpected:
        raise TradeJournalError(f"неожиданные поля журнала: {unexpected}")

    direction_raw = _required_text(record, "direction").lower()
    direction_aliases = {
        "buy": "long",
        "long": "long",
        "sell": "short",
        "short": "short",
    }
    if direction_raw not in direction_aliases:
        raise TradeJournalError(f"неподдерживаемое направление: {direction_raw!r}")
    direction = direction_aliases[direction_raw]

    exit_reason = _required_text(record, "exit_reason").lower()
    if exit_reason not in TRADE_JOURNAL_EXIT_REASONS:
        raise TradeJournalError(f"неподдерживаемая причина выхода: {exit_reason!r}")

    entry_ts = _timestamp(record.get("entry_ts"), "entry_ts", required=False)
    exit_ts = _timestamp(record.get("exit_ts"), "exit_ts", required=True)
    if entry_ts and exit_ts:
        if dt.datetime.fromisoformat(exit_ts) < dt.datetime.fromisoformat(entry_ts):
            raise TradeJournalError(
                "время выхода не может быть раньше времени входа"
            )

    paper_raw = record.get("paper")
    if not isinstance(paper_raw, bool):
        raise TradeJournalError("поле 'paper' должно быть логическим")

    normalized: Dict[str, Any] = {
        "trade_id": _required_text(record, "trade_id"),
        "session_id": _required_text(record, "session_id"),
        "git_sha": _optional_text(record, "git_sha"),
        "symbol": _required_text(record, "symbol").upper(),
        "direction": direction,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": _number(record, "entry_price", strictly_positive=True),
        "exit_price": _number(record, "exit_price", strictly_positive=True),
        "qty": _number(record, "qty", strictly_positive=True),
        "leverage": _number(record, "leverage", required=False, strictly_positive=True),
        "entry_fee": _number(record, "entry_fee", minimum=0.0),
        "exit_fee": _number(record, "exit_fee", minimum=0.0),
        "funding": _number(record, "funding", required=False),
        "slippage": _number(record, "slippage", minimum=0.0),
        "gross_pnl": _number(record, "gross_pnl"),
        "net_pnl": _number(record, "net_pnl"),
        "exit_reason": exit_reason,
        "strategy": _optional_text(record, "strategy"),
        "regime": _optional_text(record, "regime"),
        "atr_entry": _number(record, "atr_entry", required=False, minimum=0.0),
        "ml_probability": _number(record, "ml_probability", required=False, minimum=0.0),
        "ml_threshold": _number(record, "ml_threshold", required=False, minimum=0.0),
        "paper": paper_raw,
    }

    for key in ("ml_probability", "ml_threshold"):
        value = normalized[key]
        if value is not None and value > 1.0:
            raise TradeJournalError(f"поле {key!r} должно быть в диапазоне 0..1")

    funding = normalized["funding"]
    if funding is not None:
        expected_net = (
            float(normalized["gross_pnl"])
            - float(normalized["entry_fee"])
            - float(normalized["exit_fee"])
            + float(funding)
        )
        tolerance = max(1e-8, abs(expected_net) * 1e-8)
        if not math.isclose(float(normalized["net_pnl"]), expected_net, abs_tol=tolerance):
            raise TradeJournalError(
                "net_pnl не совпадает с gross_pnl, комиссиями и funding"
            )

    return {field: normalized[field] for field in TRADE_JOURNAL_FIELDS}


def read_trade_records(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    """Read and validate every non-empty JSON line from a journal."""
    target = Path(path).expanduser()
    if not target.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(normalize_trade_record(raw))
            except Exception as exc:
                raise TradeJournalError(
                    f"неверная строка журнала {line_number}: {exc}"
                ) from exc
    return rows


def _append_bytes(target: Path, chunks: Iterable[bytes]) -> None:
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        for chunk in chunks:
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("не удалось дописать строку журнала сделок")
                view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass


def append_trade_record(
    path: os.PathLike[str] | str,
    record: Mapping[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Append one idempotent completed-trade row and fsync it to disk."""
    normalized = normalize_trade_record(record)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    with _JOURNAL_LOCK:
        existing_rows = read_trade_records(target)
        for existing in existing_rows:
            if existing["trade_id"] != normalized["trade_id"]:
                continue
            if existing != normalized:
                raise TradeJournalConflictError(
                    f"trade_id {normalized['trade_id']!r} уже записан с другими данными"
                )
            return normalized, False

        encoded = (
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _append_bytes(target, [encoded])
    return normalized, True
