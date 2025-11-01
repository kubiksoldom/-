# -*- coding: utf-8 -*-
"""Train RandomForest-based ML model with calibrated probabilities and EV-aware threshold."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise RuntimeError("Dataset must contain 'target' column")
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    return df


def _sort_for_time_split(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_entry" in df.columns:
        try:
            order = pd.to_datetime(df["ts_entry"], utc=True, errors="coerce")
            return df.assign(_ts_sort=order).sort_values("_ts_sort", na_position="first").drop(columns="_ts_sort")
        except Exception:
            pass
    if "ts" in df.columns:
        return df.sort_values("ts")
    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float,
    val_size: float,
    seed: int,
    time_split: bool,
) -> SplitData:
    if test_size < 0 or val_size < 0 or (test_size + val_size) >= 1:
        raise ValueError("test_size and val_size must be >=0 and sum to <1")

    features = _ensure_numeric(df[feature_cols])
    targets = df["target"].astype(int)

    if time_split:
        df_sorted = _sort_for_time_split(pd.concat([features, targets], axis=1))
        features_sorted = df_sorted[feature_cols]
        targets_sorted = df_sorted["target"]
        n = len(df_sorted)
        test_n = int(round(n * test_size))
        val_n = int(round(n * val_size))
        test_n = min(max(test_n, 1 if test_size > 0 else 0), n)
        val_n = min(max(val_n, 1 if val_size > 0 else 0), n - test_n)
        train_n = n - test_n - val_n
        if train_n <= 0:
            raise ValueError("Not enough data for requested splits")
        X_train = features_sorted.iloc[:train_n]
        y_train = targets_sorted.iloc[:train_n]
        X_val = features_sorted.iloc[train_n: train_n + val_n]
        y_val = targets_sorted.iloc[train_n: train_n + val_n]
        X_test = features_sorted.iloc[train_n + val_n:]
        y_test = targets_sorted.iloc[train_n + val_n:]
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            targets,
            test_size=test_size,
            random_state=seed,
            stratify=targets if targets.nunique() > 1 else None,
        )
        if val_size > 0:
            remaining = 1.0 - test_size
            val_fraction = val_size / remaining
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_fraction,
                random_state=seed,
                stratify=y_temp if y_temp.nunique() > 1 else None,
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val = X_train.iloc[0:0]
            y_val = y_train.iloc[0:0]
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def _safe_metric(fn, y_true, y_pred, **kwargs):
    default = kwargs.pop("default", float("nan"))
    try:
        val = fn(y_true, y_pred, **kwargs)
        try:
            return float(val)
        except Exception:
            return float(np.asarray(val).item())
    except Exception:
        try:
            return float(default)
        except Exception:
            return float(np.asarray(default).item())


def _unwrap_pipeline(model):
    if isinstance(model, CalibratedClassifierCV):
        return getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
    return model


def _evaluate_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (proba >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "precision": _safe_metric(precision_score, y_true, preds, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, preds, zero_division=0),
        "f1": _safe_metric(f1_score, y_true, preds, zero_division=0),
        "mcc": _safe_metric(matthews_corrcoef, y_true, preds, default=0.0),
        "trades": int(preds.sum()),
    }
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def _threshold_from_youden(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5, {"reason": "constant_proba"}
    pos = y_true.sum()
    neg = len(y_true) - pos
    best_thr = thresholds[0]
    best_score = -1.0
    for thr in thresholds:
        preds = proba >= thr
        tp = float(((preds == 1) & (y_true == 1)).sum())
        fn = float(((preds == 0) & (y_true == 1)).sum())
        fp = float(((preds == 1) & (y_true == 0)).sum())
        tn = float(((preds == 0) & (y_true == 0)).sum())
        tpr = tp / max(tp + fn, 1.0)
        fpr = fp / max(fp + tn, 1.0)
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr), {"youden": float(best_score)}


def _threshold_from_f1(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.unique(proba):
        preds = proba >= thr
        score = _safe_metric(f1_score, y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr
    return float(best_thr), {"f1": float(best_f1)}


def _threshold_from_pat(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(proba)[::-1]
    chosen = thresholds[-1]
    chosen_prec = 0.0
    chosen_rec = 0.0
    chosen_trades = 0
    for thr in thresholds:
        preds = proba >= thr
        trades = int(preds.sum())
        if trades == 0:
            continue
        prec = _safe_metric(precision_score, y_true, preds, zero_division=0)
        if prec >= target_precision:
            chosen = thr
            chosen_prec = prec
            chosen_rec = _safe_metric(recall_score, y_true, preds, zero_division=0)
            chosen_trades = trades
            break
    if chosen_trades == 0:
        preds = proba >= chosen
        chosen_prec = _safe_metric(precision_score, y_true, preds, zero_division=0)
        chosen_rec = _safe_metric(recall_score, y_true, preds, zero_division=0)
        chosen_trades = int(preds.sum())
    return float(chosen), {
        "precision": float(chosen_prec),
        "recall": float(chosen_rec),
        "target_precision": float(target_precision),
        "trades": int(chosen_trades),
    }


def _threshold_from_ev(
    y_true: np.ndarray,
    proba: np.ndarray,
    fee: float,
    r_avg: float,
) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best_ev = -1e9
    best_prec = 0.0
    best_trades = 0
    for thr in np.unique(proba):
        preds = proba >= thr
        trades = int(preds.sum())
        if trades == 0:
            continue
        prec = _safe_metric(precision_score, y_true, preds, zero_division=0)
        ev = prec * r_avg - (1.0 - prec) * 1.0 - fee
        if ev > best_ev:
            best_ev = ev
            best_prec = prec
            best_thr = thr
            best_trades = trades
    if best_trades == 0:
        best_ev = prec = 0.0
    return float(best_thr), {
        "expected_value": float(best_ev),
        "precision": float(best_prec),
        "trades": int(best_trades),
        "fee": float(fee),
        "r_avg": float(r_avg),
    }


def choose_threshold(
    metric: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    pat_target: float,
    ev_fee: float,
    ev_r_avg: float,
) -> Tuple[float, Dict[str, float]]:
    if metric == "youden":
        return _threshold_from_youden(y_true, proba)
    if metric == "f1":
        return _threshold_from_f1(y_true, proba)
    if metric == "pAt":
        return _threshold_from_pat(y_true, proba, pat_target)
    if metric == "ev":
        return _threshold_from_ev(y_true, proba, ev_fee, ev_r_avg)
    raise ValueError(f"Unknown threshold metric: {metric}")


def build_pipeline(seed: int) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=600,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", rf),
    ])
    return pipeline


def get_feature_importances(model, feature_cols: List[str]) -> List[Dict[str, float]]:
    try:
        pipeline = _unwrap_pipeline(model)
        if not isinstance(pipeline, Pipeline):
            return []
        clf = pipeline.named_steps.get("clf")
        if clf is None or not hasattr(clf, "feature_importances_"):
            return []
        importances = clf.feature_importances_
        pairs = list(zip(feature_cols, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:30]
        return [{"feature": name, "importance": float(val)} for name, val in top]
    except Exception:
        return []


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    if X.empty:
        return {
            "rows": 0,
            "auc": 0.0,
            "pr_auc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mcc": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "rows": int(len(X)),
        "auc": _safe_metric(roc_auc_score, y, proba, default=0.0),
        "pr_auc": _safe_metric(average_precision_score, y, proba, default=0.0),
    }
    thr_metrics = _evaluate_threshold(y.to_numpy(), proba, threshold)
    metrics.update({
        "f1": thr_metrics["f1"],
        "precision": thr_metrics["precision"],
        "recall": thr_metrics["recall"],
        "mcc": thr_metrics["mcc"],
        "confusion_matrix": thr_metrics["confusion_matrix"],
        "trades": thr_metrics["trades"],
    })
    return metrics


def train(args: argparse.Namespace) -> None:
    df = load_dataset(args.csv)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise RuntimeError("Dataset must contain feature_* columns")

    splits = split_dataset(
        df,
        feature_cols,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        time_split=bool(args.time_split),
    )

    pipeline = build_pipeline(args.seed)
    pipeline.fit(splits.X_train, splits.y_train)

    calibration_info = {"applied": False, "method": "none"}
    final_model = pipeline
    if args.calibrate and args.calibrate.lower() in {"isotonic", "platt"}:
        if splits.X_val.empty:
            raise RuntimeError("Validation set is empty – cannot calibrate")
        method = "isotonic" if args.calibrate.lower() == "isotonic" else "sigmoid"
        calibrator = CalibratedClassifierCV(final_model, method=method, cv="prefit")
        calibrator.fit(splits.X_val, splits.y_val)
        final_model = calibrator
        calibration_info = {"applied": True, "method": method}

    val_source_X = splits.X_val if not splits.X_val.empty else splits.X_train
    val_source_y = splits.y_val if not splits.y_val.empty else splits.y_train
    val_proba = final_model.predict_proba(val_source_X)[:, 1]
    threshold, threshold_meta = choose_threshold(
        args.threshold_metric,
        val_source_y.to_numpy(),
        val_proba,
        pat_target=args.p_at_target,
        ev_fee=args.ev_fee,
        ev_r_avg=args.ev_r_avg,
    )

    val_metrics = evaluate_model(final_model, val_source_X, val_source_y, threshold)
    test_metrics = evaluate_model(final_model, splits.X_test, splits.y_test, threshold)

    pipe = _unwrap_pipeline(final_model)
    assert pipe is not None, "Pipeline unwrap failed"
    scaler = pipe.named_steps.get("scaler") if isinstance(pipe, Pipeline) else None
    clf = pipe.named_steps.get("clf") if isinstance(pipe, Pipeline) else None
    scaler_params = {
        "mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else [],
        "scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else [],
    }

    feature_importances = get_feature_importances(final_model, feature_cols)

    meta = {
        "version": "2",
        "features": feature_cols,
        "proba_threshold": float(threshold),
        "calibration": calibration_info,
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "train_rows": int(len(splits.X_train)),
        "val_rows": int(len(splits.X_val)),
        "test_rows": int(len(splits.X_test)),
        "train_time_utc": datetime.now(timezone.utc).isoformat(),
        "ev_params": {"fee": float(args.ev_fee), "r_avg": float(args.ev_r_avg)},
        "threshold_metric": args.threshold_metric,
        "threshold_details": threshold_meta,
        "scaler_params": scaler_params,
        "feature_importances": feature_importances,
    }

    joblib.dump(final_model, args.model)
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "model_path": os.path.abspath(args.model),
                "meta_path": os.path.abspath(args.meta),
                "threshold": float(threshold),
                "metrics_test": test_metrics,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train calibrated RandomForest model")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train model from dataset")
    train_p.add_argument("--csv", default="ml_dataset.csv", help="Path to dataset CSV")
    train_p.add_argument("--model", default="rf_model.pkl", help="Where to save trained model")
    train_p.add_argument("--meta", default="model_meta.json", help="Where to save metadata JSON")
    train_p.add_argument("--test_size", type=float, default=0.2, help="Fraction for test split")
    train_p.add_argument("--val_size", type=float, default=0.1, help="Fraction for validation split")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument(
        "--calibrate",
        choices=["isotonic", "platt", "none"],
        default="platt",
        help="Calibration method",
    )
    train_p.add_argument(
        "--threshold_metric",
        choices=["youden", "f1", "pAt", "ev"],
        default="ev",
        help="Metric to choose classification threshold",
    )
    train_p.add_argument("--p_at_target", type=float, default=0.65, help="Target precision for pAt metric")
    train_p.add_argument("--ev_fee", type=float, default=0.0014, help="Fee used for EV calculations")
    train_p.add_argument("--ev_r_avg", type=float, default=1.8, help="Average reward multiple for EV")
    train_p.add_argument("--time_split", type=int, default=1, help="Use chronological split (1) or random (0)")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "main" or __name__ == "__main__":
    main()
