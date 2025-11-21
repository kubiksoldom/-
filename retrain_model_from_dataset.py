# -*- coding: utf-8 -*-
# retrain_model_from_dataset.py
"""
Переобучение ML-модели с прогрессом и ETA (ASCII-only вывод).

Что делает:
- Загружает датасет (CSV/Parquet).
- Фильтрует выходные (по умолчанию исключает weekday in {5,6}; можно отключить через EXCLUDE_WEEKENDS=0).
- Берёт ТОЛЬКО числовые фичи (строковые, например 'symbol', игнорируются).
- Убирает NaN/Inf в выбранных признаках и таргете (надёжность обучения).
- Делит по времени (60/20/20).
- Обучает RandomForest (warm_start) с прогресс-баром.
- Калибрует вероятности (CalibratedClassifierCV, sigmoid по умолчанию).
- Печатает метрики и сохраняет модель + meta (с описанием применённых фильтров).

ENV:
  DATASET_PATH=ml_dataset.csv
  TARGET_COL=target
  MODEL_FILE=rf_model.pkl
  MODEL_META=model_meta.json
  SEED=42

  RF_TREES=300
  RF_MAX_DEPTH=12
  RF_MIN_SAMPLES=3
  RF_STEP=15
  CALIB_METHOD=sigmoid     # sigmoid | isotonic
  CALIB_CV=3
  ML_THRESHOLD=0.58

  # Новое:
  EXCLUDE_WEEKENDS=1       # 1 = исключать выходные, 0 = не трогать
  WEEKEND_VALUES=5,6       # какие weekday (0=Mon..6=Sun) считать выходными
"""

import os
import sys
import json
import time
import argparse
import pathlib
import datetime as dt
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, brier_score_loss

# ============== Конфиг ==============
DATA_PATH    = os.getenv("DATASET_PATH", "ml_dataset.csv")   # авто по расширению
TARGET_COL   = os.getenv("TARGET_COL", "target")
MODEL_FILE   = os.getenv("MODEL_FILE", "rf_model.pkl")
MODEL_META   = os.getenv("MODEL_META", "model_meta.json")
RANDOM_SEED  = int(os.getenv("SEED", "42"))

RF_TREES       = int(os.getenv("RF_TREES", "300"))
RF_MAX_DEPTH   = int(os.getenv("RF_MAX_DEPTH", "12"))
RF_MIN_SAMPLES = int(os.getenv("RF_MIN_SAMPLES", "3"))
RF_STEP        = int(os.getenv("RF_STEP", "15"))

CALIB_METHOD   = os.getenv("CALIB_METHOD", "sigmoid")  # 'sigmoid' | 'isotonic'
CALIB_CV       = int(os.getenv("CALIB_CV", "3"))

DEFAULT_THR    = float(os.getenv("ML_THRESHOLD", "0.58"))

TP_PCT_DEFAULT       = float(os.getenv("TP_PCT_DEFAULT", "0.006"))
SL_PCT_DEFAULT       = float(os.getenv("SL_PCT_DEFAULT", "0.004"))
ML_COST_PCT          = float(os.getenv("ML_COST_PCT", "0.0010"))
MIN_TRADES_GLOBAL    = int(os.getenv("ML_MIN_TRADES_GLOBAL", "20"))
MIN_TRADES_REGIME    = int(os.getenv("ML_MIN_TRADES_REGIME", "12"))

# Новые флаги фильтра выходных
EXCLUDE_WEEKENDS = int(os.getenv("EXCLUDE_WEEKENDS", "1"))  # 1 = исключать
WEEKEND_VALUES   = os.getenv("WEEKEND_VALUES", "5,6")

# ============== Безопасная печать (ASCII) ==============
def safe_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    end = kwargs.get("end", "\n")
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
    try:
        sys.stdout.write(s + end)
        sys.stdout.flush()
    except Exception:
        sys.stdout.buffer.write((s + end).encode(enc, errors="replace"))
        sys.stdout.flush()

# ============== Утилиты ==============
def _fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

class ProgressBar:
    # ASCII bar: ####----- style
    def __init__(self, total: int, desc: str = "", width: int = 80):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = time.time()
        self.n = 0
        self.width = max(40, width)

    def _line(self) -> str:
        frac = self.n / self.total
        pct = int(frac * 100)
        elapsed = time.time() - self.start
        speed = self.n / elapsed if elapsed > 0 else 0.0
        remain = (self.total - self.n) / speed if speed > 0 else 0.0

        bar_w = max(10, self.width - len(self.desc) - 28)
        filled = int(bar_w * frac)
        bar = "#" * filled + "-" * (bar_w - filled)
        return f"{self.desc} [{bar}] {pct:3d}% | {self.n}/{self.total} | ETA {_fmt_time(remain)}"

    def set(self, n: int):
        self.n = min(self.total, max(0, int(n)))
        safe_print("\r" + self._line(), end="")

    def close(self):
        self.set(self.total)
        safe_print("")

class Spinner:
    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.last = 0
        self.interval = 0.08

    def spin(self):
        now = time.time()
        if now - self.last < self.interval:
            return
        self.last = now
        ch = Spinner.FRAMES[self.i % len(Spinner.FRAMES)]
        self.i += 1
        safe_print("\r" + f"{ch} {self.text}", end="")

    def done(self, ok: bool = True):
        mark = "[OK]" if ok else "[FAIL]"
        safe_print("\r" + f"{mark} {self.text}")

# ============== IO ==============
def load_dataset(path: str) -> pd.DataFrame:
    ext = pathlib.Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _save_pickle(obj: Any, path: str):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _maybe_load_meta(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def pick_threshold_from_meta_or_default(meta: Optional[Dict[str, Any]]) -> float:
    if meta and isinstance(meta, dict):
        thr_block = meta.get("thresholds", {}) or {}
        used = thr_block.get("used") or thr_block.get("global")
        if used is not None:
            try:
                return float(used)
            except Exception:
                pass
    return DEFAULT_THR


def _realized_ev_array(y_true: np.ndarray, tp_arr: np.ndarray, sl_arr: np.ndarray, cost: float) -> np.ndarray:
    """Возвращает массив ожидаемого EV на сделку по фактическому исходу."""
    y_true = np.asarray(y_true).astype(int)
    tp_arr = np.asarray(tp_arr).astype(float)
    sl_arr = np.asarray(sl_arr).astype(float)
    wins = y_true == 1
    return np.where(wins, tp_arr - cost, -sl_arr - cost)


def _best_threshold_stats(
    proba: np.ndarray,
    y_true: np.ndarray,
    realized_ev: np.ndarray,
    min_trades: int,
    default_thr: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Подбор лучшего порога по среднему EV, возвращает метрики или None."""

    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    realized_ev = np.asarray(realized_ev, dtype=float)

    if proba.size == 0 or y_true.size == 0:
        return None

    pos_total = int((y_true == 1).sum())
    if pos_total == 0:
        return None

    thr_candidates = np.unique(proba)
    if default_thr is not None:
        thr_candidates = np.unique(np.append(thr_candidates, float(default_thr)))
    thr_candidates = np.sort(thr_candidates)[::-1]

    best: Optional[Dict[str, Any]] = None
    for thr in thr_candidates:
        mask = proba >= thr
        trades = int(mask.sum())
        if trades < max(1, int(min_trades)):
            continue
        wins = int(((y_true == 1) & mask).sum())
        if trades == 0:
            continue
        precision = float(wins / trades) if trades > 0 else 0.0
        recall = float(wins / max(1, pos_total))
        ev_mean = float(realized_ev[mask].mean()) if trades > 0 else 0.0
        candidate = {
            "threshold": float(thr),
            "ev": ev_mean,
            "precision": precision,
            "recall": recall,
            "trades": trades,
        }
        if best is None:
            best = candidate
            continue
        if ev_mean > best["ev"] + 1e-12:
            best = candidate
        elif abs(ev_mean - best["ev"]) < 1e-12 and trades > best["trades"]:
            best = candidate

    return best

# ============== Основной пайплайн ==============
def main():
    parser = argparse.ArgumentParser(description="Retrain ML model from dataset")
    parser.add_argument("--walk-forward", dest="walk_forward", type=int, default=0,
                        help="number of walk-forward windows (0 = disable)")
    args = parser.parse_args()
    walk_forward_steps = max(0, int(args.walk_forward))

    np.random.seed(RANDOM_SEED)

    # 1) Загрузка
    safe_print("| [LOAD] dataset from %s ..." % DATA_PATH)
    spinner = Spinner("[LOAD] reading dataset ...")
    spinner.spin()
    try:
        df = load_dataset(DATA_PATH)
        spinner.done(True)
    except Exception as e:
        spinner.done(False)
        safe_print("[ERROR] load dataset: %s" % e)
        sys.exit(1)

    safe_print("[LOAD] shape=%s" % (df.shape,))

    if TARGET_COL not in df.columns:
        safe_print("[ERROR] dataset must contain target column '%s'." % TARGET_COL)
        sys.exit(1)

    # 1.1) ФИЛЬТР ВЫХОДНЫХ (если включён и есть колонка 'weekday')
    weekend_vals: List[int] = []
    try:
        weekend_vals = [int(x.strip()) for x in WEEKEND_VALUES.split(",") if x.strip() != ""]
    except Exception:
        weekend_vals = [5, 6]

    filt_info = {
        "exclude_weekends": bool(EXCLUDE_WEEKENDS),
        "weekend_values": weekend_vals,
        "dropped_rows_due_to_nan": 0,
        "source_path": DATA_PATH,
    }

    if EXCLUDE_WEEKENDS:
        if "weekday" in df.columns:
            before = len(df)
            df = df[~df["weekday"].astype(int).isin(weekend_vals)].copy()
            after = len(df)
            safe_print("[FILTER] exclude weekends %s -> kept %d/%d rows" % (weekend_vals, after, before))
        else:
            safe_print("[FILTER] 'weekday' column not found; cannot exclude weekends")

    # 2) Признаки/таргет (только числовые фичи)
    numeric_cols = [c for c, dt in df.dtypes.items() if np.issubdtype(dt, np.number)]
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    if not numeric_cols:
        safe_print("[ERROR] no numeric feature columns found.")
        sys.exit(1)

    # 2.1) Надёжная очистка NaN/Inf — СНАЧАЛА на уровне df, чтобы не ломать индексы
    cols_for_clean = numeric_cols + [TARGET_COL]
    before = len(df)
    df_clean = df[cols_for_clean].replace([np.inf, -np.inf], np.nan).dropna()
    dropped = before - len(df_clean)
    filt_info["dropped_rows_due_to_nan"] = int(max(0, dropped))
    if dropped > 0:
        safe_print("[CLEAN] dropped rows with NaN/Inf in numeric features/target: %d (kept %d/%d)" %
                   (dropped, len(df_clean), before))

    # 2.2) Обновляем X/y по очищенному df
    y = df_clean[TARGET_COL].astype(int).values
    X = df_clean[numeric_cols].values

    tp_all = np.full(len(df_clean), TP_PCT_DEFAULT, dtype=float)
    if "tp_pct_used" in df_clean.columns:
        tp_all = pd.to_numeric(df_clean["tp_pct_used"], errors="coerce").fillna(TP_PCT_DEFAULT).to_numpy(dtype=float)
    sl_all = np.full(len(df_clean), SL_PCT_DEFAULT, dtype=float)
    if "sl_pct_used" in df_clean.columns:
        sl_all = pd.to_numeric(df_clean["sl_pct_used"], errors="coerce").fillna(SL_PCT_DEFAULT).to_numpy(dtype=float)

    # 3) Временной сплит 60/20/20 (порядок уже временной из билдера)
    N = len(df_clean)
    if N < 100:
        safe_print("[WARN] dataset is very small after filtering/cleaning: N=%d" % N)

    i1 = int(N * 0.60)
    i2 = int(N * 0.80)
    X_tr, y_tr = X[:i1], y[:i1]
    X_ca, y_ca = X[i1:i2], y[i1:i2]
    X_va, y_va = X[i2:],   y[i2:]

    def _stats(y_arr: np.ndarray) -> Tuple[int, int, int, float]:
        n = len(y_arr)
        pos = int((y_arr == 1).sum())
        neg = n - pos
        rate = float(pos / max(1, n))
        return n, pos, neg, rate

    tr_n, tr_pos, tr_neg, tr_rate = _stats(y_tr)
    ca_n, ca_pos, ca_neg, ca_rate = _stats(y_ca)
    va_n, va_pos, va_neg, va_rate = _stats(y_va)

    safe_print("[SPLIT] train: rows=%d pos=%d neg=%d rate=%.3f | calib: rows=%d pos=%d neg=%d rate=%.3f | val: rows=%d pos=%d neg=%d rate=%.3f"
               % (tr_n, tr_pos, tr_neg, tr_rate, ca_n, ca_pos, ca_neg, ca_rate, va_n, va_pos, va_neg, va_rate))
    safe_print("[FEATS] using %d numeric features: %s" % (len(numeric_cols), numeric_cols))

    # 4) RF с прогрессом
    safe_print("=== RandomForest (warm_start + progress) ===")
    rf = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        random_state=RANDOM_SEED,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES,
        n_jobs=-1,
        class_weight=None,
    )
    bar = ProgressBar(total=RF_TREES, desc="Train RF", width=80)
    built = 0
    try:
        while built < RF_TREES:
            step = min(RF_STEP, RF_TREES - built)
            rf.n_estimators = built + step
            t0 = time.time()
            rf.fit(X_tr, y_tr)
            built += step
            bar.set(built)
            if time.time() - t0 < 0.02:
                time.sleep(0.02)
        bar.close()
    except KeyboardInterrupt:
        bar.close()
        safe_print("[WARN] training interrupted by user; continue with current forest")

    # 5) Калибровка
    calibration_method = (CALIB_METHOD or "sigmoid").strip().lower()
    if calibration_method not in ("sigmoid", "isotonic"):
        safe_print("[WARN] unsupported CALIB_METHOD=%s -> fallback to sigmoid" % CALIB_METHOD)
        calibration_method = "sigmoid"

    calibration_method_used = "raw"
    if len(X_ca) > 0 and len(np.unique(y_ca)) > 1:
        spinner = Spinner(f"Calibrating probabilities ({calibration_method}) ...")
        spinner.spin()
        try:
            clf = CalibratedClassifierCV(rf, method=calibration_method, cv=CALIB_CV)
            clf.fit(X_ca, y_ca)
            spinner.done(True)
            calibration_method_used = calibration_method
        except Exception as e:
            spinner.done(False)
            safe_print("[WARN] calibration failed: %s; using raw RF probabilities" % e)
            clf = rf
    else:
        safe_print("[WARN] calibration skipped (insufficient calibration data)")
        clf = rf

    # 6) Оценка
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_va)[:, 1]
    else:
        pred = clf.predict(X_va)
        y_proba = pred.astype(float) if isinstance(pred, np.ndarray) else np.zeros(len(X_va), dtype=float)

    calibration_valid_score: Optional[float] = None
    try:
        if len(y_va) > 0:
            calibration_valid_score = float(brier_score_loss(y_va, y_proba))
    except Exception as e:
        safe_print("[WARN] brier_score_loss failed: %s" % e)
        calibration_valid_score = None

    if calibration_valid_score is not None:
        safe_print("[CALIB] method=%s | val_brier=%.6f" % (calibration_method_used, calibration_valid_score))
    else:
        safe_print("[CALIB] method=%s | val_brier=n/a" % calibration_method_used)

    prev_meta = _maybe_load_meta(MODEL_META)
    thr_prev = pick_threshold_from_meta_or_default(prev_meta)

    tp_va = tp_all[i2:]
    sl_va = sl_all[i2:]

    realized_va = _realized_ev_array(y_va, tp_va, sl_va, ML_COST_PCT)

    p50 = p90 = 0.0
    if "atr_norm" in df_clean.columns:
        atr_all = pd.to_numeric(df_clean["atr_norm"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(atr_all) > 0:
            p50 = float(np.percentile(atr_all.values, 50))
            p90 = float(np.percentile(atr_all.values, 90))
    atr_percentiles = {
        "p50": float(p50) if np.isfinite(p50) else None,
        "p90": float(p90) if np.isfinite(p90) else None,
    }

    global_stats = _best_threshold_stats(y_proba, y_va, realized_va, MIN_TRADES_GLOBAL, default_thr=thr_prev)
    global_thr = float(global_stats["threshold"]) if global_stats else float(thr_prev)
    if not np.isfinite(global_thr):
        global_thr = float(DEFAULT_THR)

    fallback_thr = float(global_thr if np.isfinite(global_thr) else DEFAULT_THR)

    regime_thresholds: Dict[str, float] = {
        "regime_low": fallback_thr,
        "regime_high": fallback_thr,
        "regime_ultra": fallback_thr,
    }
    regime_metrics: Dict[str, Optional[Dict[str, Any]]] = {
        "regime_low": None,
        "regime_high": None,
        "regime_ultra": None,
    }
    regime_available: Dict[str, int] = {
        "regime_low": 0,
        "regime_high": 0,
        "regime_ultra": 0,
    }

    if len(y_va) > 0 and "atr_norm" in df_clean.columns:
        atr_va_series = pd.to_numeric(df_clean.iloc[i2:]["atr_norm"], errors="coerce")
        atr_va = np.nan_to_num(atr_va_series.to_numpy(dtype=float), nan=p50, posinf=p90, neginf=0.0)

        masks = {
            "regime_low": atr_va < p50,
            "regime_high": (atr_va >= p50) & (atr_va < p90),
            "regime_ultra": atr_va >= p90,
        }

        for regime_name, mask in masks.items():
            mask = np.asarray(mask, dtype=bool)
            available = int(mask.sum())
            regime_available[regime_name] = available
            if available == 0:
                continue
            stats = _best_threshold_stats(
                y_proba[mask],
                y_va[mask],
                realized_va[mask],
                MIN_TRADES_REGIME,
                default_thr=thr_prev,
            )
            if stats:
                regime_metrics[regime_name] = stats
                regime_thresholds[regime_name] = float(stats["threshold"])

    best_regime = None
    best_ev = -1e9
    for regime_name, stats in regime_metrics.items():
        if not stats:
            continue
        ev_val = float(stats.get("ev", 0.0))
        if (best_regime is None) or (ev_val > best_ev + 1e-12):
            best_regime = regime_name
            best_ev = ev_val

    used_mode = "global"
    used_thr = fallback_thr
    if best_regime:
        used_mode = best_regime
        used_thr = float(regime_thresholds.get(best_regime, fallback_thr))
    elif global_stats:
        used_thr = float(global_thr)

    y_pred = (y_proba >= used_thr).astype(int)

    try:
        rep = classification_report(y_va, y_pred, digits=4, zero_division=0)
        safe_print("\n=== Validation report ===")
        safe_print(rep)
    except Exception as e:
        safe_print("[WARN] classification_report failed: %s" % e)

    try:
        thr_median = float(np.median(y_proba)) if len(y_proba) > 0 else 0.0
        thr_q55 = float(np.quantile(y_proba, 0.55)) if len(y_proba) > 0 else 0.0
        safe_print("[THR] median=%.4f | q55=%.4f | used=%.4f" % (thr_median, thr_q55, used_thr))
    except Exception:
        pass

    safe_print("[ATR] percentiles: p50=%.6g | p90=%.6g" % (p50, p90))
    if global_stats:
        safe_print(
            "[THR:global] thr=%.4f | EV≈%.3f%% | precision=%.4f | recall=%.4f | trades=%d" % (
                global_stats["threshold"],
                global_stats["ev"] * 100.0,
                global_stats["precision"],
                global_stats["recall"],
                global_stats["trades"],
            )
        )
    else:
        safe_print("[THR:global] fallback thr=%.4f" % fallback_thr)

    for regime_name in ("regime_low", "regime_high", "regime_ultra"):
        stats = regime_metrics.get(regime_name)
        available = regime_available.get(regime_name, 0)
        thr_val = float(regime_thresholds.get(regime_name, fallback_thr))
        if stats:
            safe_print(
                "[THR:%s] thr=%.4f | EV≈%.3f%% | precision=%.4f | recall=%.4f | trades=%d (avail=%d)"
                % (
                    regime_name,
                    stats["threshold"],
                    stats["ev"] * 100.0,
                    stats["precision"],
                    stats["recall"],
                    stats["trades"],
                    available,
                )
            )
        else:
            safe_print(
                "[THR:%s] fallback thr=%.4f (avail=%d < %d)"
                % (regime_name, thr_val, available, MIN_TRADES_REGIME)
            )

    safe_print("[THR] selected mode=%s | used=%.4f" % (used_mode, used_thr))

    trades_total = int((y_pred == 1).sum())
    precision_total = None
    if trades_total > 0:
        precision_total = float((y_va[y_pred == 1] == 1).mean())

    weekly_precision = None
    weekly_support = 0
    week_start_iso = None
    week_end_iso = None
    if len(y_va) == len(y_pred):
        weekly_mask = None
        if "ts_entry" in df_clean.columns:
            ts_series = pd.to_numeric(df_clean.iloc[i2:]["ts_entry"], errors="coerce")
            dt_series = pd.to_datetime(ts_series, unit="ms", errors="coerce", utc=True)
            if dt_series.notna().any():
                week_end_dt = dt_series.max()
                if pd.notna(week_end_dt):
                    horizon_start = week_end_dt - pd.Timedelta(days=7)
                    weekly_mask = dt_series >= horizon_start
                    week_start_iso = horizon_start.to_pydatetime().isoformat()
                    week_end_iso = week_end_dt.to_pydatetime().isoformat()

        if weekly_mask is None:
            weekly_mask = np.zeros(len(y_pred), dtype=bool)
            if len(y_pred) > 0:
                start_idx = max(0, int(len(y_pred) * 0.8))
                weekly_mask[start_idx:] = True

        weekly_mask = np.asarray(weekly_mask, dtype=bool)
        weekly_support = int(((y_pred == 1) & weekly_mask).sum())
        if weekly_support > 0:
            weekly_precision = float((y_va[weekly_mask][y_pred[weekly_mask] == 1] == 1).mean())

    if precision_total is not None:
        safe_print(f"[ML] validation precision={precision_total:.4f} (trades={trades_total})")
    else:
        safe_print(f"[ML] validation precision: нет сигналов (trades={trades_total})")

    if weekly_precision is not None:
        safe_print(f"[ML] weekly precision = {weekly_precision:.4f} (support={weekly_support})")

    walk_forward_results: List[Dict[str, Any]] = []
    if walk_forward_steps > 0:
        safe_print("[WF] running walk-forward evaluation: N=%d" % walk_forward_steps)
        indices = np.arange(len(df_clean))
        chunks = np.array_split(indices, walk_forward_steps + 1)
        for wf_idx in range(walk_forward_steps):
            val_idx = chunks[wf_idx + 1]
            train_chunks = chunks[:wf_idx + 1]
            if not len(train_chunks):
                safe_print("[WF] window #%d skipped: no training chunks" % (wf_idx + 1))
                continue
            train_idx = np.concatenate(train_chunks)
            if train_idx.size < 50 or val_idx.size == 0:
                safe_print(
                    "[WF] window #%d skipped: train=%d val=%d" % (wf_idx + 1, train_idx.size, val_idx.size)
                )
                continue

            cut = max(1, int(train_idx.size * 0.8))
            if cut >= train_idx.size:
                cut = train_idx.size - 1
            base_idx = train_idx[:cut]
            calib_idx = train_idx[cut:]
            if base_idx.size == 0 or calib_idx.size == 0:
                safe_print(
                    "[WF] window #%d skipped: insufficient split (train=%d, calib=%d)"
                    % (wf_idx + 1, base_idx.size, calib_idx.size)
                )
                continue

            try:
                wf_rf = RandomForestClassifier(
                    n_estimators=RF_TREES,
                    random_state=RANDOM_SEED,
                    max_depth=RF_MAX_DEPTH,
                    min_samples_leaf=RF_MIN_SAMPLES,
                    n_jobs=-1,
                )
                wf_rf.fit(X[base_idx], y[base_idx])
            except Exception as e:
                safe_print("[WF] window #%d training failed: %s" % (wf_idx + 1, e))
                continue

            wf_model = wf_rf
            wf_method_used = "raw"
            if calib_idx.size > 1 and len(np.unique(y[calib_idx])) > 1:
                try:
                    wf_model = CalibratedClassifierCV(wf_rf, method=calibration_method, cv=CALIB_CV)
                    wf_model.fit(X[calib_idx], y[calib_idx])
                    wf_method_used = calibration_method
                except Exception as e:
                    safe_print("[WF] window #%d calibration failed: %s" % (wf_idx + 1, e))
                    wf_model = wf_rf
                    wf_method_used = "raw"

            wf_thr = used_thr
            wf_thr_source = "fallback"
            if calib_idx.size > 0 and hasattr(wf_model, "predict_proba"):
                try:
                    cal_proba = wf_model.predict_proba(X[calib_idx])[:, 1]
                    cal_realized = _realized_ev_array(y[calib_idx], tp_all[calib_idx], sl_all[calib_idx], ML_COST_PCT)
                    thr_stats = _best_threshold_stats(
                        cal_proba,
                        y[calib_idx],
                        cal_realized,
                        MIN_TRADES_GLOBAL,
                        default_thr=used_thr,
                    )
                    if thr_stats and thr_stats.get("threshold") is not None:
                        wf_thr = float(thr_stats["threshold"])
                        wf_thr_source = "calibration"
                except Exception as e:
                    safe_print("[WF] window #%d threshold search failed: %s" % (wf_idx + 1, e))

            if not hasattr(wf_model, "predict_proba"):
                preds_val = wf_model.predict(X[val_idx])
                proba_val = preds_val.astype(float) if isinstance(preds_val, np.ndarray) else np.zeros(len(val_idx))
            else:
                proba_val = wf_model.predict_proba(X[val_idx])[:, 1]

            decisions = proba_val >= wf_thr
            trades = int(decisions.sum())
            positives_val = int((y[val_idx] == 1).sum())
            hits = int(((y[val_idx] == 1) & decisions).sum())
            precision = float(hits / trades) if trades > 0 else 0.0
            recall = float(hits / max(1, positives_val))
            realized_val = _realized_ev_array(y[val_idx], tp_all[val_idx], sl_all[val_idx], ML_COST_PCT)
            ev_mean = float(realized_val[decisions].mean()) if trades > 0 else 0.0

            wf_entry = {
                "window": int(wf_idx + 1),
                "train_rows": int(train_idx.size),
                "calib_rows": int(calib_idx.size),
                "val_rows": int(val_idx.size),
                "threshold": float(wf_thr),
                "threshold_source": wf_thr_source,
                "ev_mean": ev_mean,
                "precision": precision,
                "recall": recall,
                "trades": int(trades),
                "positives": int(positives_val),
                "calibration": wf_method_used,
                "val_start": int(val_idx[0]),
                "val_end": int(val_idx[-1]),
            }
            walk_forward_results.append(wf_entry)
            safe_print(
                "[WF] window #%d | thr=%.4f | EV≈%.3f%% | precision=%.4f | recall=%.4f | trades=%d/%d"
                % (
                    wf_entry["window"],
                    wf_thr,
                    ev_mean * 100.0,
                    precision,
                    recall,
                    trades,
                    positives_val,
                )
            )

        if walk_forward_results:
            wf_df = pd.DataFrame(walk_forward_results)
            report_dir = pathlib.Path(MODEL_META).resolve().parent
            csv_path = report_dir / "walk_forward_report.csv"
            json_path = report_dir / "walk_forward_report.json"
            try:
                wf_df.to_csv(csv_path, index=False)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(walk_forward_results, f, ensure_ascii=False, indent=2)
                avg_ev = float(wf_df["ev_mean"].mean()) if not wf_df.empty else 0.0
                avg_prec = float(wf_df["precision"].mean()) if not wf_df.empty else 0.0
                safe_print(
                    "[WF] summary: mean_EV≈%.3f%% | mean_precision=%.4f | windows=%d"
                    % (avg_ev * 100.0, avg_prec, len(walk_forward_results))
                )
                safe_print("[WF] reports saved -> %s, %s" % (csv_path, json_path))
            except Exception as e:
                safe_print("[WF] failed to save reports: %s" % e)
        else:
            safe_print("[WF] no walk-forward windows evaluated")

    # 7) Сохранение
    _save_pickle(clf, MODEL_FILE)
    metrics_block = {
        "precision_total": (float(precision_total) if precision_total is not None else None),
        "precision_week": (float(weekly_precision) if weekly_precision is not None else None),
        "trades_total": int(trades_total),
        "trades_week": int(weekly_support),
        "week_window_start": week_start_iso,
        "week_window_end": week_end_iso,
        "weekly_precision": {
            "precision": (float(weekly_precision) if weekly_precision is not None else None),
            "support": int(weekly_support),
        },
    }

    meta = {
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "Calibrated(RandomForest)" if clf is not rf else "RandomForest",
        "features": numeric_cols,
        "thresholds": {
            "used_mode": used_mode,
            "used": float(used_thr),
            "global": float(global_thr),
            "ev_only": float(global_thr),
            "regime_low": float(regime_thresholds.get("regime_low", fallback_thr)),
            "regime_high": float(regime_thresholds.get("regime_high", fallback_thr)),
            "regime_ultra": float(regime_thresholds.get("regime_ultra", fallback_thr)),
        },
        "train_rows": int(tr_n),
        "calib_rows": int(ca_n),
        "val_rows": int(va_n),
        "params": {
            "trees": int(RF_TREES),
            "max_depth": int(RF_MAX_DEPTH),
            "min_samples_leaf": int(RF_MIN_SAMPLES),
            "calibration": str(calibration_method_used),
            "calib_cv": int(CALIB_CV),
            "seed": int(RANDOM_SEED),
        },
        "filters": filt_info,
        "calibration": {
            "method": str(calibration_method_used),
            "valid_score": (float(calibration_valid_score) if calibration_valid_score is not None else None),
        },
        "atr_percentiles": atr_percentiles,
        "metrics": metrics_block,
    }
    _save_json(meta, MODEL_META)

    safe_print("[OK] Model saved -> %s" % MODEL_FILE)
    safe_print("[OK] Meta  saved -> %s (features=%d)" % (MODEL_META, len(numeric_cols)))
    safe_print("[OK] Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\n[INT] stopped by user")
