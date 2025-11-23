# -*- coding: utf-8 -*-
"""Offline ML debug report."""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]


def _load(path: str):
    return pd.read_csv(path)


def _bin_key(p: float):
    for lo, hi in BINS:
        if lo <= p < hi or (hi == 1.0 and p <= hi):
            return f"{lo:.1f}-{hi:.1f}"
    return "other"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate offline ML report")
    parser.add_argument("--dataset", default="ml_dataset.csv")
    parser.add_argument("--model", default="rf_model.pkl")
    parser.add_argument("--meta", default="model_meta.json")
    args = parser.parse_args()

    df = _load(args.dataset)
    features = [c for c in df.columns if c.startswith("feature_")]
    model = joblib.load(args.model)
    meta = {}
    try:
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["target"].astype(int)

    prob = model.predict_proba(X)[:, 1]

    bins = defaultdict(list)
    for yt, p in zip(y, prob):
        bins[_bin_key(float(p))].append((yt, p))

    lines = ["# ML offline report", ""]
    lines.append(f"Samples: {len(df)}")
    lines.append(f"Model: {args.model}")
    lines.append(f"Meta thresholds: {(meta.get('thresholds') if isinstance(meta, dict) else {})}")
    lines.append("")
    lines.append("## Probability buckets")
    for label in [f"{lo:.1f}-{hi:.1f}" for lo, hi in BINS]:
        bucket = bins.get(label, [])
        if not bucket:
            lines.append(f"- {label}: empty")
            continue
        wins = sum(1 for yt, _ in bucket if yt == 1)
        wl = wins / max(1, len(bucket))
        avg_prob = sum(p for _, p in bucket) / max(1, len(bucket))
        lines.append(
            f"- {label}: n={len(bucket)} | winrate={wl:.3f} | avg_proba={avg_prob:.3f}"
        )

    report_path = "ML_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
