import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PAPER_MODE"] = "1"
os.environ["SAFE_MODE"] = "1"
os.environ["TELEGRAM_ENABLED"] = "0"
os.environ["PAPER_SYNC_BALANCE"] = "0"
os.environ["BYBIT_API_KEY"] = ""
os.environ["BYBIT_API_SECRET"] = ""
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import build_ml_dataset_from_journal as dataset_v2  # noqa: E402
import evaluate_ml_temporal as temporal  # noqa: E402
import main  # noqa: E402
import ml_entry_snapshot  # noqa: E402
import trade_journal  # noqa: E402


def _features(
    *,
    direction: str = "long",
    captured_at: str = "2026-07-19T08:00:00+00:00",
    seed: float = 1.0,
):
    timestamp = datetime.fromisoformat(captured_at).astimezone(timezone.utc)
    values = {
        name: seed + index / 1000.0
        for index, name in enumerate(ml_entry_snapshot.ML_ENTRY_FEATURES)
    }
    values.update(
        {
            "last_price": 100.0 + seed,
            "qty": 1.0 + seed / 100.0,
            "hour": float(timestamp.hour),
            "weekday": float(timestamp.weekday()),
            "direction": 1.0 if direction == "long" else -1.0,
            "tp_pct_used": 0.0,
            "sl_pct_used": 0.0,
        }
    )
    return values


def _snapshot(
    trade_id: str = "trade-1",
    *,
    direction: str = "long",
    symbol: str = "BTCUSDT",
    captured_at: str = "2026-07-19T08:00:00+00:00",
    seed: float = 1.0,
):
    return {
        "schema_version": ml_entry_snapshot.ML_ENTRY_SNAPSHOT_VERSION,
        "trade_id": trade_id,
        "session_id": "session-1",
        "git_sha": "a" * 40,
        "symbol": symbol,
        "direction": direction,
        "captured_at": captured_at,
        "paper": True,
        "strategy": "trend",
        "regime": "normal",
        "planned_tp_pct": 0.04,
        "planned_sl_pct": 0.02,
        "model_probability": 0.7,
        "model_threshold": 0.6,
        "model_applied": False,
        "features": _features(
            direction=direction,
            captured_at=captured_at,
            seed=seed,
        ),
    }


def _trade(
    trade_id: str = "trade-1",
    *,
    direction: str = "long",
    symbol: str = "BTCUSDT",
    entry_ts: str = "2026-07-19T08:00:00+00:00",
    exit_ts: str = "2026-07-19T08:10:00+00:00",
    gross_pnl: float = 5.0,
):
    entry_price = 100.0
    qty = 1.0
    exit_price = 105.0 if direction == "long" else 95.0
    entry_fee = 0.1
    exit_fee = 0.1
    return {
        "trade_id": trade_id,
        "session_id": "session-1",
        "git_sha": "a" * 40,
        "symbol": symbol,
        "direction": direction,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "qty": qty,
        "leverage": 2.0,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "funding": 0.0,
        "slippage": 0.0,
        "gross_pnl": gross_pnl,
        "net_pnl": gross_pnl - entry_fee - exit_fee,
        "exit_reason": "take_profit" if gross_pnl > 0 else "stop_loss",
        "strategy": "trend",
        "regime": "normal",
        "atr_entry": 1.0,
        "ml_probability": 0.7,
        "ml_threshold": 0.6,
        "paper": True,
    }


def test_entry_snapshot_has_exact_schema_is_idempotent_and_rejects_secrets(tmp_path):
    path = tmp_path / "ml_entry_snapshots.jsonl"
    normalized, appended = ml_entry_snapshot.append_ml_entry_snapshot(
        path, _snapshot()
    )
    repeated, repeated_appended = ml_entry_snapshot.append_ml_entry_snapshot(
        path, _snapshot()
    )

    assert appended is True
    assert repeated_appended is False
    assert repeated == normalized
    assert tuple(normalized) == ml_entry_snapshot.ML_ENTRY_SNAPSHOT_FIELDS
    assert tuple(normalized["features"]) == ml_entry_snapshot.ML_ENTRY_FEATURES
    assert normalized["captured_at"].endswith("+00:00")
    assert path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(ml_entry_snapshot.MLEntrySnapshotConflictError):
        ml_entry_snapshot.append_ml_entry_snapshot(path, _snapshot(seed=2.0))

    unsafe = _snapshot()
    unsafe["api_key"] = "must-not-be-written"
    with pytest.raises(ml_entry_snapshot.MLEntrySnapshotError):
        ml_entry_snapshot.normalize_ml_entry_snapshot(unsafe)
    assert "must-not-be-written" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("direction", "router_tp", "router_sl", "expected_direction"),
    [
        ("long", 110.0, 95.0, 1.0),
        ("short", 90.0, 105.0, -1.0),
    ],
)
def test_runtime_records_long_and_short_entry_plan(
    tmp_path, direction, router_tp, router_sl, expected_direction
):
    path = tmp_path / "entry.jsonl"
    features = _features(direction=direction)

    appended = main.record_ml_entry_snapshot(
        trade_id=f"trade-{direction}",
        session_id="session-1",
        git_sha="b" * 40,
        symbol="BTCUSDT",
        direction=direction,
        captured_at="2026-07-19T08:00:00+00:00",
        paper=True,
        strategy="trend",
        regime="normal",
        decision_price=100.0,
        router_tp=router_tp,
        router_sl=router_sl,
        model_probability=0.7,
        model_threshold=0.6,
        model_applied=False,
        features=features,
        snapshot_file=path,
    )

    row = ml_entry_snapshot.read_ml_entry_snapshots(path)[0]
    assert appended is True
    assert row["planned_tp_pct"] == pytest.approx(0.10)
    assert row["planned_sl_pct"] == pytest.approx(0.05)
    assert row["features"]["direction"] == expected_direction


def test_builder_joins_one_closed_trade_to_one_entry_snapshot_and_keeps_time(tmp_path):
    journal_path = tmp_path / "trade_journal.jsonl"
    snapshot_path = tmp_path / "ml_entry_snapshots.jsonl"
    output_path = tmp_path / "ml_dataset_v2.csv"
    report_path = tmp_path / "ml_dataset_v2_report.json"

    rows = [
        (
            _trade(
                "trade-short",
                direction="short",
                symbol="ETHUSDT",
                entry_ts="2026-07-19T09:00:00+00:00",
                exit_ts="2026-07-19T09:10:00+00:00",
                gross_pnl=-4.0,
            ),
            _snapshot(
                "trade-short",
                direction="short",
                symbol="ETHUSDT",
                captured_at="2026-07-19T09:00:00+00:00",
                seed=2.0,
            ),
        ),
        (_trade("trade-long"), _snapshot("trade-long")),
    ]
    for trade, snapshot in rows:
        trade_journal.append_trade_record(journal_path, trade)
        ml_entry_snapshot.append_ml_entry_snapshot(snapshot_path, snapshot)

    report = dataset_v2.build_from_files(
        journal_path, snapshot_path, output_path, report_path
    )
    frame = pd.read_csv(output_path)

    assert report["rows_output"] == 2
    assert report["join_coverage"] == pytest.approx(1.0)
    assert list(frame["trade_id"]) == ["trade-long", "trade-short"]
    assert list(frame["target"]) == [1, 0]
    assert list(frame["direction"]) == ["long", "short"]
    assert list(frame["feature_direction"]) == [1.0, -1.0]
    assert list(frame["feature_tp_pct_used"]) == pytest.approx([0.04, 0.04])
    assert list(frame["feature_sl_pct_used"]) == pytest.approx([0.02, 0.02])
    assert frame["entry_ts"].str.contains("2026-07-19").all()
    assert not list(tmp_path.glob(".*.tmp"))
    report_text = report_path.read_text(encoding="utf-8").lower()
    assert not any(name in report_text for name in ("api_key", "api_secret", "token"))


def test_builder_reports_missing_orphan_and_conflicting_duplicate_vectors():
    first_trade = _trade("trade-a")
    second_trade = _trade("trade-b", gross_pnl=-2.0)
    missing_trade = _trade("trade-missing", symbol="SOLUSDT")
    first_snapshot = _snapshot("trade-a")
    second_snapshot = _snapshot("trade-b")
    orphan_snapshot = _snapshot("trade-orphan", symbol="XRPUSDT", seed=9.0)

    rows, report = dataset_v2.build_ml_dataset_rows(
        [first_trade, second_trade, missing_trade],
        [first_snapshot, second_snapshot, orphan_snapshot],
    )

    assert len(rows) == 1
    assert report["valid_join_pairs"] == 2
    assert report["join_coverage"] == pytest.approx(2 / 3)
    assert report["retention_after_vector_dedup"] == pytest.approx(0.5)
    assert report["duplicate_feature_vectors"] == 1
    assert report["conflicting_feature_vectors"] == 1
    assert report["missing_snapshot"] == ["trade-missing"]
    assert report["orphan_snapshot"] == ["trade-orphan"]


def _synthetic_dataset(count: int = 150) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        entry = start + timedelta(hours=index)
        exit_time = entry + timedelta(minutes=10)
        direction = "long" if index % 2 == 0 else "short"
        net_pnl = 1.5 if index % 3 else -1.0
        entry_price = 100.0 + index / 100.0
        qty = 1.0
        row = {
            field: None for field in dataset_v2.ML_DATASET_FIELDS
        }
        row.update(
            {
                "dataset_version": dataset_v2.ML_DATASET_VERSION,
                "feature_schema_version": ml_entry_snapshot.ML_ENTRY_SNAPSHOT_VERSION,
                "trade_id": f"synthetic-{index:04d}",
                "session_id": "synthetic-session",
                "git_sha": "c" * 40,
                "symbol": "BTCUSDT" if index % 4 else "ETHUSDT",
                "direction": direction,
                "entry_ts": entry.isoformat(),
                "exit_ts": exit_time.isoformat(),
                "duration_sec": 600.0,
                "entry_price": entry_price,
                "exit_price": entry_price + (1.0 if net_pnl > 0 else -1.0),
                "qty": qty,
                "leverage": 2.0,
                "entry_fee": 0.1,
                "exit_fee": 0.1,
                "funding": 0.0,
                "slippage": 0.0,
                "gross_pnl": net_pnl + 0.2,
                "net_pnl": net_pnl,
                "net_return": net_pnl / (entry_price * qty),
                "target": int(net_pnl > 0.0),
                "exit_reason": "take_profit" if net_pnl > 0 else "stop_loss",
                "strategy": "synthetic",
                "regime": "normal",
                "ml_probability": 0.5,
                "ml_threshold": 0.6,
                "model_applied": False,
                "paper": True,
            }
        )
        for feature_index, name in enumerate(ml_entry_snapshot.ML_ENTRY_FEATURES):
            row[f"feature_{name}"] = (
                (index + 1) * (feature_index + 1) / 100_000.0
            )
        row["feature_last_price"] = entry_price
        row["feature_qty"] = qty
        row["feature_direction"] = 1.0 if direction == "long" else -1.0
        row["feature_hour"] = float(entry.hour)
        row["feature_weekday"] = float(entry.weekday())
        row["feature_tp_pct_used"] = 0.04
        row["feature_sl_pct_used"] = 0.02
        rows.append(row)
    return pd.DataFrame(rows, columns=dataset_v2.ML_DATASET_FIELDS)


def test_temporal_evaluation_deduplicates_purges_and_never_replaces_model():
    frame = _synthetic_dataset()
    duplicate = frame.iloc[20].copy()
    duplicate["trade_id"] = "synthetic-duplicate"
    duplicate_entry = pd.Timestamp(duplicate["entry_ts"]) + pd.Timedelta(days=7)
    duplicate["entry_ts"] = duplicate_entry.isoformat()
    duplicate["exit_ts"] = (duplicate_entry + pd.Timedelta(minutes=10)).isoformat()
    frame = pd.concat([frame, duplicate.to_frame().T], ignore_index=True)

    report = temporal.evaluate_temporal_dataset(
        frame,
        min_rows=100,
        trees=20,
        walk_forward_windows=2,
        min_threshold_trades=3,
    )

    assert report["active_model_replaced"] is False
    assert report["data_quality"]["rows_input"] == 151
    assert report["data_quality"]["rows_valid"] == 150
    assert report["data_quality"]["duplicate_feature_vectors_removed"] == 1
    ranges = report["holdout"]["ranges"]
    assert ranges["train"]["end"] < ranges["probability_calibration"]["start"]
    assert ranges["probability_calibration"]["end"] < ranges["threshold_calibration"]["start"]
    assert ranges["threshold_calibration"]["end"] < ranges["validation"]["start"]
    assert len(report["walk_forward"]) == 2
    assert "strategy_without_ml" in report["holdout"]
    assert "strategy_with_ml_filter" in report["holdout"]


def test_validation_results_cannot_change_selected_threshold():
    original = _synthetic_dataset()
    changed_validation = original.copy()
    validation_start = int(len(changed_validation) * 0.8)
    changed_validation.loc[validation_start:, "net_pnl"] *= -1.0
    changed_validation.loc[validation_start:, "net_return"] *= -1.0
    changed_validation.loc[validation_start:, "target"] = (
        changed_validation.loc[validation_start:, "net_pnl"] > 0.0
    ).astype(int)

    first = temporal.evaluate_temporal_dataset(
        original,
        min_rows=100,
        trees=20,
        walk_forward_windows=0,
        min_threshold_trades=3,
    )
    second = temporal.evaluate_temporal_dataset(
        changed_validation,
        min_rows=100,
        trees=20,
        walk_forward_windows=0,
        min_threshold_trades=3,
    )

    assert first["holdout"]["threshold"] == second["holdout"]["threshold"]


def test_temporal_evaluation_purges_labels_that_were_not_known_at_cutoff():
    frame = _synthetic_dataset()
    frame.loc[10, "exit_ts"] = frame.loc[95, "entry_ts"]
    frame.loc[95, "exit_ts"] = frame.loc[125, "entry_ts"]

    report = temporal.evaluate_temporal_dataset(
        frame,
        min_rows=100,
        trees=10,
        walk_forward_windows=0,
        min_threshold_trades=3,
    )

    purged = report["holdout"]["purged_label_overlap"]
    assert purged["train"] == 1
    assert purged["calibration"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"target": 0},
        {"feature_direction": 1.0},
        {"feature_hour": 99.0},
        {"paper": "maybe"},
    ],
)
def test_temporal_validation_rejects_inconsistent_rows(changes):
    frame = _synthetic_dataset(100)
    for column, value in changes.items():
        if column == "paper":
            frame[column] = frame[column].astype(object)
        frame.loc[1, column] = value
    with pytest.raises(temporal.MLTemporalValidationError):
        temporal.validate_ml_dataset(frame, min_rows=100)


def test_temporal_report_is_written_atomically(tmp_path):
    report_path = tmp_path / "temporal.json"
    report = temporal.evaluate_temporal_dataset(
        _synthetic_dataset(),
        min_rows=100,
        trees=10,
        walk_forward_windows=0,
        min_threshold_trades=3,
    )
    temporal._atomic_json(report_path, report)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["active_model_replaced"] is False
    assert not list(tmp_path.glob(".*.tmp"))
