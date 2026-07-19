"""Честная временная оценка ML-фильтра без замены рабочей модели."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from build_ml_dataset_from_journal import ML_DATASET_FIELDS, ML_DATASET_VERSION
from ml_entry_snapshot import ML_ENTRY_FEATURES, ML_ENTRY_SNAPSHOT_VERSION


class MLTemporalValidationError(ValueError):
    """Выборка не допускает безопасную временную оценку."""


FEATURE_COLUMNS = tuple(f"feature_{name}" for name in ML_ENTRY_FEATURES)
REQUIRED_COLUMNS = ML_DATASET_FIELDS


def validate_ml_dataset(
    frame: pd.DataFrame,
    *,
    min_rows: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Проверить зерно, время, признаки и удалить точные копии векторов."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MLTemporalValidationError(f"отсутствуют обязательные колонки: {missing}")

    unexpected_features = sorted(
        column
        for column in frame.columns
        if column.startswith("feature_") and column not in ML_DATASET_FIELDS
    )
    if unexpected_features:
        raise MLTemporalValidationError(
            f"обнаружены признаки другой схемы: {unexpected_features}"
        )

    clean = frame.copy()
    versions = pd.to_numeric(clean["dataset_version"], errors="coerce")
    feature_versions = pd.to_numeric(clean["feature_schema_version"], errors="coerce")
    if versions.isna().any() or set(versions.astype(int).unique()) != {ML_DATASET_VERSION}:
        raise MLTemporalValidationError(
            f"поддерживается только dataset_version={ML_DATASET_VERSION}"
        )
    if (
        feature_versions.isna().any()
        or set(feature_versions.astype(int).unique()) != {ML_ENTRY_SNAPSHOT_VERSION}
    ):
        raise MLTemporalValidationError(
            f"поддерживается только feature_schema_version={ML_ENTRY_SNAPSHOT_VERSION}"
        )

    clean["trade_id"] = clean["trade_id"].astype(str).str.strip()
    if (clean["trade_id"] == "").any():
        raise MLTemporalValidationError("обнаружен пустой trade_id")
    duplicate_trade_ids = clean["trade_id"][clean["trade_id"].duplicated()].unique().tolist()
    if duplicate_trade_ids:
        raise MLTemporalValidationError(
            f"trade_id повторяется: {sorted(duplicate_trade_ids)[:10]}"
        )

    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    if (clean["symbol"] == "").any():
        raise MLTemporalValidationError("обнаружен пустой symbol")
    clean["direction"] = clean["direction"].astype(str).str.strip().str.lower()
    if not set(clean["direction"].unique()).issubset({"long", "short"}):
        raise MLTemporalValidationError("direction должен быть long или short")

    paper_values = clean["paper"].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes"}
        if str(value).strip().lower() in {"0", "1", "true", "false", "yes", "no"}
        else None
    )
    if paper_values.isna().any():
        raise MLTemporalValidationError("paper должен быть логическим значением")
    clean["paper"] = paper_values.astype(bool)
    if clean["paper"].nunique() != 1:
        raise MLTemporalValidationError(
            "бумажные и реальные сделки нельзя смешивать в одной оценке"
        )

    clean["entry_ts"] = pd.to_datetime(clean["entry_ts"], errors="coerce", utc=True)
    clean["exit_ts"] = pd.to_datetime(clean["exit_ts"], errors="coerce", utc=True)
    if clean[["entry_ts", "exit_ts"]].isna().any().any():
        raise MLTemporalValidationError("временные метки отсутствуют или повреждены")
    if (clean["exit_ts"] < clean["entry_ts"]).any():
        raise MLTemporalValidationError("обнаружена сделка с выходом раньше входа")

    clean["target"] = pd.to_numeric(clean["target"], errors="coerce")
    if clean["target"].isna().any() or not set(clean["target"].astype(int).unique()).issubset({0, 1}):
        raise MLTemporalValidationError("target должен состоять только из 0 и 1")
    clean["target"] = clean["target"].astype(int)

    numeric_columns = list(FEATURE_COLUMNS) + [
        "entry_price",
        "exit_price",
        "qty",
        "gross_pnl",
        "net_pnl",
        "net_return",
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    numeric_values = clean[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise MLTemporalValidationError("в признаках или результатах есть NaN/Inf")
    if (clean[["entry_price", "exit_price", "qty"]] <= 0.0).any().any():
        raise MLTemporalValidationError("цены и количество должны быть больше нуля")
    expected_target = (clean["net_pnl"] > 0.0).astype(int)
    if not clean["target"].equals(expected_target):
        raise MLTemporalValidationError("target не совпадает со знаком net_pnl")
    expected_return = clean["net_pnl"] / (clean["entry_price"] * clean["qty"])
    if not np.allclose(
        clean["net_return"].to_numpy(dtype=float),
        expected_return.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
    ):
        raise MLTemporalValidationError("net_return не совпадает с результатом сделки")

    expected_direction = clean["direction"].map({"long": 1.0, "short": -1.0})
    if not np.allclose(clean["feature_direction"], expected_direction, atol=1e-12):
        raise MLTemporalValidationError("feature_direction не совпадает с direction")
    if (clean[["feature_last_price", "feature_qty"]] <= 0.0).any().any():
        raise MLTemporalValidationError("цена и количество в признаках должны быть положительными")
    if (clean[["feature_tp_pct_used", "feature_sl_pct_used"]] <= 0.0).any().any():
        raise MLTemporalValidationError("плановые TP/SL в признаках должны быть положительными")
    expected_hour = clean["entry_ts"].dt.hour.to_numpy(dtype=float)
    expected_weekday = clean["entry_ts"].dt.weekday.to_numpy(dtype=float)
    if not np.allclose(clean["feature_hour"], expected_hour, atol=1e-12):
        raise MLTemporalValidationError("feature_hour не совпадает со временем входа UTC")
    if not np.allclose(clean["feature_weekday"], expected_weekday, atol=1e-12):
        raise MLTemporalValidationError("feature_weekday не совпадает со временем входа UTC")

    clean = clean.sort_values(["entry_ts", "trade_id"], kind="stable").reset_index(drop=True)
    duplicate_mask = clean.duplicated(subset=list(FEATURE_COLUMNS), keep="first")
    duplicate_count = int(duplicate_mask.sum())
    conflicting_count = 0
    if duplicate_count:
        first_target_by_vector: Dict[Tuple[float, ...], int] = {}
        for row in clean.itertuples(index=False):
            vector = tuple(float(getattr(row, column)) for column in FEATURE_COLUMNS)
            target = int(row.target)
            if vector in first_target_by_vector and first_target_by_vector[vector] != target:
                conflicting_count += 1
            else:
                first_target_by_vector.setdefault(vector, target)
        clean = clean.loc[~duplicate_mask].reset_index(drop=True)

    if len(clean) < max(3, int(min_rows)):
        raise MLTemporalValidationError(
            f"после очистки осталось {len(clean)} строк; требуется минимум {max(3, int(min_rows))}"
        )
    if clean["entry_ts"].nunique() < 3:
        raise MLTemporalValidationError("нужно минимум три разных момента входа")

    report = {
        "rows_input": int(len(frame)),
        "rows_valid": int(len(clean)),
        "duplicate_feature_vectors_removed": duplicate_count,
        "conflicting_duplicate_vectors": conflicting_count,
        "trade_ids_unique": True,
        "feature_count": len(FEATURE_COLUMNS),
        "entry_start": clean["entry_ts"].iloc[0].isoformat(),
        "entry_end": clean["entry_ts"].iloc[-1].isoformat(),
        "positive_rate": float(clean["target"].mean()),
        "paper": bool(clean["paper"].iloc[0]),
    }
    return clean, report


def _split_by_unique_time(
    frame: pd.DataFrame,
    first_fraction: float,
    second_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_times = np.array(sorted(frame["entry_ts"].unique()))
    if len(unique_times) < 3:
        raise MLTemporalValidationError("недостаточно разных временных точек для разделения")
    first_cut = min(len(unique_times) - 2, max(1, int(len(unique_times) * first_fraction)))
    second_cut = min(
        len(unique_times) - 1,
        max(first_cut + 1, int(len(unique_times) * second_fraction)),
    )
    first_end = unique_times[first_cut - 1]
    second_end = unique_times[second_cut - 1]
    first = frame[frame["entry_ts"] <= first_end].copy()
    second = frame[(frame["entry_ts"] > first_end) & (frame["entry_ts"] <= second_end)].copy()
    third = frame[frame["entry_ts"] > second_end].copy()
    if first.empty or second.empty or third.empty:
        raise MLTemporalValidationError("одно из временных окон оказалось пустым")
    return first, second, third


def temporal_train_calibration_validation_split(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить 60/20/20 без пересечения одинаковых временных меток."""
    return _split_by_unique_time(frame, 0.60, 0.80)


def _probabilities_for_positive(model: RandomForestClassifier, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    class_positions = {int(label): index for index, label in enumerate(model.classes_)}
    if 1 not in class_positions:
        return np.zeros(len(values), dtype=float)
    return np.asarray(probabilities[:, class_positions[1]], dtype=float)


def _fit_probability_calibrator(
    raw_probability: np.ndarray,
    target: np.ndarray,
) -> Optional[LogisticRegression]:
    if len(raw_probability) < 10 or len(np.unique(target)) < 2:
        return None
    calibrator = LogisticRegression(random_state=42, solver="lbfgs")
    calibrator.fit(np.asarray(raw_probability, dtype=float).reshape(-1, 1), target)
    return calibrator


def _apply_calibrator(
    calibrator: Optional[LogisticRegression],
    raw_probability: np.ndarray,
) -> np.ndarray:
    raw = np.clip(np.asarray(raw_probability, dtype=float), 0.0, 1.0)
    if calibrator is None:
        return raw
    return np.asarray(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)


def _select_threshold(
    probability: np.ndarray,
    net_return: np.ndarray,
    *,
    min_trades: int,
) -> Dict[str, Any]:
    probability = np.asarray(probability, dtype=float)
    net_return = np.asarray(net_return, dtype=float)
    if len(probability) == 0:
        raise MLTemporalValidationError("пустое окно выбора порога")
    quantiles = np.linspace(0.0, 0.95, 20)
    candidates = np.unique(
        np.concatenate(([0.0, 0.5], np.quantile(probability, quantiles)))
    )
    best: Optional[Dict[str, Any]] = None
    for threshold in sorted((float(value) for value in candidates), reverse=True):
        selected = probability >= threshold
        trades = int(selected.sum())
        if trades < max(1, int(min_trades)):
            continue
        expectancy = float(net_return[selected].mean())
        candidate = {
            "threshold": threshold,
            "trades": trades,
            "expectancy_return": expectancy,
        }
        if best is None:
            best = candidate
        elif expectancy > float(best["expectancy_return"]) + 1e-12:
            best = candidate
        elif math.isclose(expectancy, float(best["expectancy_return"]), abs_tol=1e-12):
            if trades > int(best["trades"]):
                best = candidate
    if best is None:
        return {"threshold": 0.0, "trades": len(probability), "expectancy_return": float(net_return.mean())}
    return best


def strategy_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    pnl = frame["net_pnl"].to_numpy(dtype=float)
    returns = frame["net_return"].to_numpy(dtype=float)
    trades = len(frame)
    if trades == 0:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "expectancy_pnl": None,
            "expectancy_return": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "win_rate": None,
        }
    gains = float(pnl[pnl > 0.0].sum())
    losses = float(-pnl[pnl < 0.0].sum())
    cumulative = np.concatenate(([0.0], np.cumsum(pnl)))
    peaks = np.maximum.accumulate(cumulative)
    drawdown = peaks - cumulative
    return {
        "trades": int(trades),
        "net_pnl": float(pnl.sum()),
        "expectancy_pnl": float(pnl.mean()),
        "expectancy_return": float(returns.mean()),
        "profit_factor": (float(gains / losses) if losses > 0.0 else None),
        "max_drawdown": float(drawdown.max()),
        "win_rate": float((pnl > 0.0).mean()),
    }


def _classification_metrics(target: np.ndarray, probability: np.ndarray) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "roc_auc": None,
        "average_precision": None,
        "brier": None,
    }
    if len(target) == 0:
        return result
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    try:
        result["brier"] = float(brier_score_loss(target, probability))
    except ValueError:
        pass
    if len(np.unique(target)) < 2:
        return result
    result["roc_auc"] = float(roc_auc_score(target, probability))
    result["average_precision"] = float(average_precision_score(target, probability))
    return result


def _time_range(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "start": frame["entry_ts"].iloc[0].isoformat(),
        "end": frame["entry_ts"].iloc[-1].isoformat(),
    }


def _fit_evaluate_window(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    trees: int,
    min_threshold_trades: int,
) -> Dict[str, Any]:
    if train.empty or calibration.empty or validation.empty:
        raise MLTemporalValidationError("одно из временных окон оказалось пустым")

    train_before_purge = len(train)
    calibration_before_purge = len(calibration)
    calibration_start = calibration["entry_ts"].min()
    validation_start = validation["entry_ts"].min()
    train = train[train["exit_ts"] <= calibration_start].copy()
    calibration = calibration[calibration["exit_ts"] <= validation_start].copy()
    if train.empty or calibration.empty:
        raise MLTemporalValidationError(
            "после удаления сделок с результатом из будущего окно стало пустым"
        )
    if train["target"].nunique() < 2:
        raise MLTemporalValidationError("в обучающем окне присутствует только один класс")

    calibration_for_probability, calibration_for_threshold, _unused = _split_by_unique_time(
        calibration,
        0.50,
        0.99,
    )
    # При очень редких одинаковых временных метках третья часть может содержать
    # несколько строк. Присоединяем её к окну выбора порога: в validation она не
    # попадает и утечки будущего не создаёт.
    if not _unused.empty:
        calibration_for_threshold = pd.concat(
            [calibration_for_threshold, _unused], ignore_index=True
        ).sort_values("entry_ts", kind="stable")

    probability_before_purge = len(calibration_for_probability)
    threshold_start = calibration_for_threshold["entry_ts"].min()
    calibration_for_probability = calibration_for_probability[
        calibration_for_probability["exit_ts"] <= threshold_start
    ].copy()
    if calibration_for_probability.empty:
        raise MLTemporalValidationError(
            "после временной очистки нет данных для калибровки вероятности"
        )

    X_train = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    model = RandomForestClassifier(
        n_estimators=max(10, int(trees)),
        random_state=42,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    raw_probability_calibration = _probabilities_for_positive(
        model,
        calibration_for_probability.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float),
    )
    calibrator = _fit_probability_calibrator(
        raw_probability_calibration,
        calibration_for_probability["target"].to_numpy(dtype=int),
    )
    threshold_probability = _apply_calibrator(
        calibrator,
        _probabilities_for_positive(
            model,
            calibration_for_threshold.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float),
        ),
    )
    threshold_stats = _select_threshold(
        threshold_probability,
        calibration_for_threshold["net_return"].to_numpy(dtype=float),
        min_trades=min(min_threshold_trades, max(1, len(calibration_for_threshold))),
    )

    validation_probability = _apply_calibrator(
        calibrator,
        _probabilities_for_positive(
            model,
            validation.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float),
        ),
    )
    threshold = float(threshold_stats["threshold"])
    selected = validation_probability >= threshold
    selected_frame = validation.loc[selected].copy()
    baseline = strategy_metrics(validation)
    filtered = strategy_metrics(selected_frame)
    comparison = {
        "net_pnl_delta": float(filtered["net_pnl"] - baseline["net_pnl"]),
        "max_drawdown_delta": float(filtered["max_drawdown"] - baseline["max_drawdown"]),
        "trades_removed": int(baseline["trades"] - filtered["trades"]),
    }
    return {
        "ranges": {
            "train": _time_range(train),
            "probability_calibration": _time_range(calibration_for_probability),
            "threshold_calibration": _time_range(calibration_for_threshold),
            "validation": _time_range(validation),
        },
        "threshold": threshold_stats,
        "probability_calibration": "logistic" if calibrator is not None else "raw",
        "purged_label_overlap": {
            "train": int(train_before_purge - len(train)),
            "calibration": int(calibration_before_purge - len(calibration)),
            "probability_calibration": int(
                probability_before_purge - len(calibration_for_probability)
            ),
        },
        "classification": _classification_metrics(
            validation["target"].to_numpy(dtype=int), validation_probability
        ),
        "strategy_without_ml": baseline,
        "strategy_with_ml_filter": filtered,
        "comparison": comparison,
    }


def _walk_forward(
    frame: pd.DataFrame,
    *,
    windows: int,
    trees: int,
    min_threshold_trades: int,
) -> List[Dict[str, Any]]:
    windows = max(0, int(windows))
    if windows <= 0:
        return []
    unique_times = np.array(sorted(frame["entry_ts"].unique()))
    chunks = [chunk for chunk in np.array_split(unique_times, windows + 2) if len(chunk)]
    results: List[Dict[str, Any]] = []
    for index in range(min(windows, max(0, len(chunks) - 2))):
        past_times = np.concatenate(chunks[: index + 2])
        validation_times = chunks[index + 2]
        past = frame[frame["entry_ts"].isin(past_times)].copy()
        validation = frame[frame["entry_ts"].isin(validation_times)].copy()
        try:
            train, calibration, _tail = _split_by_unique_time(past, 0.70, 0.99)
            if not _tail.empty:
                calibration = pd.concat([calibration, _tail], ignore_index=True).sort_values(
                    "entry_ts", kind="stable"
                )
            window_report = _fit_evaluate_window(
                train,
                calibration,
                validation,
                trees=trees,
                min_threshold_trades=min_threshold_trades,
            )
            window_report["window"] = index + 1
            results.append(window_report)
        except MLTemporalValidationError as exc:
            results.append({"window": index + 1, "error": str(exc)})
    return results


def evaluate_temporal_dataset(
    frame: pd.DataFrame,
    *,
    min_rows: int = 100,
    trees: int = 200,
    walk_forward_windows: int = 3,
    min_threshold_trades: int = 10,
) -> Dict[str, Any]:
    clean, quality = validate_ml_dataset(frame, min_rows=min_rows)
    train, calibration, validation = temporal_train_calibration_validation_split(clean)
    evaluation = _fit_evaluate_window(
        train,
        calibration,
        validation,
        trees=trees,
        min_threshold_trades=min_threshold_trades,
    )
    walk_forward = _walk_forward(
        clean,
        windows=walk_forward_windows,
        trees=trees,
        min_threshold_trades=min_threshold_trades,
    )
    return {
        "dataset_version": ML_DATASET_VERSION,
        "method": "temporal_train_calibration_validation",
        "active_model_replaced": False,
        "data_quality": quality,
        "holdout": evaluation,
        "walk_forward": walk_forward,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def main() -> int:
    data_root = Path(os.getenv("DATA_ROOT", "./data")).expanduser()
    parser = argparse.ArgumentParser(
        description="Оценить ML-фильтр на последовательных временных окнах"
    )
    parser.add_argument("--dataset", default=str(data_root / "ml_dataset_v2.csv"))
    parser.add_argument("--report", default=str(data_root / "ml_temporal_report.json"))
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--walk-forward", type=int, default=3)
    parser.add_argument("--min-threshold-trades", type=int, default=10)
    args = parser.parse_args()

    try:
        frame = pd.read_csv(args.dataset)
        report = evaluate_temporal_dataset(
            frame,
            min_rows=args.min_rows,
            trees=args.trees,
            walk_forward_windows=args.walk_forward,
            min_threshold_trades=args.min_threshold_trades,
        )
        _atomic_json(Path(args.report).expanduser(), report)
    except Exception as exc:
        print(f"[ML-EVAL] ошибка: {exc}")
        return 2

    baseline = report["holdout"]["strategy_without_ml"]
    filtered = report["holdout"]["strategy_with_ml_filter"]
    print(
        "[ML-EVAL] "
        f"validation={baseline['trades']} "
        f"selected={filtered['trades']} "
        f"net_without_ml={baseline['net_pnl']:.6f} "
        f"net_with_ml={filtered['net_pnl']:.6f} "
        f"report={args.report}"
    )
    print("[ML-EVAL] рабочая модель не изменялась")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
