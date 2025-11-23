# -*- coding: utf-8 -*-
"""train_ml_model.py

Базовое обучение RandomForest + CalibratedClassifierCV.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DEFAULT_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [c for c in df.columns if c.startswith("feature_")]
    missing = [c for c in ["target"] + numeric_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Отсутствуют колонки: {missing}")
    return df


def _build_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["target"].astype(int)
    return X, y, feature_cols


def _calc_pr_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    # simple trapezoid
    area = np.trapz(precision, recall)
    return float(area)


def _metrics(y_true, prob) -> Dict:
    roc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.0
    pr = _calc_pr_auc(y_true, prob)
    pred_labels = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred_labels)
    precision_at = {}
    recall_at = {}
    for thr in DEFAULT_THRESHOLDS:
        labels = (prob >= thr).astype(int)
        precision_at[str(thr)] = float(precision_score(y_true, labels, zero_division=0))
        recall_at[str(thr)] = float(recall_score(y_true, labels, zero_division=0))
    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "acc": float(acc),
        "precision_at": precision_at,
        "recall_at": recall_at,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RF model with calibration")
    parser.add_argument("--input", default="ml_dataset.csv")
    parser.add_argument("--model_out", default="rf_model.pkl")
    parser.add_argument("--meta_out", default="model_meta.json")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = _load_dataset(args.input)
    X, y, feature_cols = _build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=args.random_state,
    )
    clf = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)

    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]

    metrics = _metrics(y_test, prob_test)

    joblib.dump(clf, args.model_out)

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features": feature_cols,
        "thresholds": {
            "global": 0.6,
            "soft": 0.5,
            "strict": 0.7,
            "used": 0.6,
        },
        "metrics": metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_split": float(args.test_size),
    }

    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "model_out": os.path.abspath(args.model_out),
                "meta_out": os.path.abspath(args.meta_out),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
