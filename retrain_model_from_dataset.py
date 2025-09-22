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
import pathlib
import datetime as dt
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

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

# ============== Основной пайплайн ==============
def main():
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
    spinner = Spinner("Calibrating probabilities ...")
    spinner.spin()
    try:
        clf = CalibratedClassifierCV(rf, method=CALIB_METHOD, cv=CALIB_CV)
        clf.fit(X_ca, y_ca)
        spinner.done(True)
    except Exception as e:
        spinner.done(False)
        safe_print("[WARN] calibration failed: %s; using raw RF probabilities" % e)
        clf = rf

    # 6) Оценка
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_va)[:, 1]
    else:
        pred = clf.predict(X_va)
        y_proba = pred.astype(float) if isinstance(pred, np.ndarray) else np.zeros(len(X_va), dtype=float)

    thr_used = pick_threshold_from_meta_or_default(_maybe_load_meta(MODEL_META))
    y_pred = (y_proba >= thr_used).astype(int)

    try:
        rep = classification_report(y_va, y_pred, digits=4, zero_division=0)
        safe_print("\n=== Validation report ===")
        safe_print(rep)
    except Exception as e:
        safe_print("[WARN] classification_report failed: %s" % e)

    try:
        thr_median = float(np.median(y_proba))
        thr_q55 = float(np.quantile(y_proba, 0.55))
        safe_print("[THR] median=%.4f | q55=%.4f | used=%.4f" % (thr_median, thr_q55, thr_used))
    except Exception:
        pass

    # 7) Сохранение
    _save_pickle(clf, MODEL_FILE)
    meta = {
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "Calibrated(RandomForest)" if clf is not rf else "RandomForest",
        "features": numeric_cols,
        "thresholds": {
            "used": float(thr_used),
            "global": float(DEFAULT_THR),
        },
        "train_rows": int(tr_n),
        "calib_rows": int(ca_n),
        "val_rows": int(va_n),
        "params": {
            "trees": int(RF_TREES),
            "max_depth": int(RF_MAX_DEPTH),
            "min_samples_leaf": int(RF_MIN_SAMPLES),
            "calibration": str(CALIB_METHOD),
            "calib_cv": int(CALIB_CV),
            "seed": int(RANDOM_SEED),
        },
        "filters": filt_info,
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
