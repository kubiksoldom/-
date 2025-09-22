# -*- coding: utf-8 -*-
"""
nn_model.py — RF + sigmoid calibration + per-trade EV + regimes + rescue
• Временной сплит устойчивый: валидное окно с обоими классами.
• Порог максимизирует средний EV по выбранным сделкам (учёт комиссий/спрэда),
  при наличии столбцов tp_pct_used/sl_pct_used — используется per-trade EV.
• Кандидаты порогов: global (EV+freq), ev_only, regime_low/high(atr_norm),
  при необходимости — rescue (ослабление порога до min_trades при EV>=min_ev).
• Полные метаданные сохраняются в model_meta.json.
"""

import json, pickle, hashlib
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, accuracy_score

# ==== файлы ====
DATASET    = "ml_dataset.csv"
MODEL_FILE = "rf_model.pkl"
MODEL_META = "model_meta.json"

# ==== временной сплит ====
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20
VAL_FRAC   = 0.20

# ==== торговые параметры (на случай, если в датасете нет tp/sl колонок) ====
TP_PCT              = 0.0040   # +0.40%
SL_PCT              = 0.0025   # -0.25%
COST_PCT_ROUND_TRIP = 0.0010   # 0.10% комиссии+спрэд за круг

# ==== контроль частоты (мягкий) ====
TARGET_TRADES_PER_1000 = 50     # целевая частота входов на 1000 наблюдений
FREQ_PENALTY           = 0.0002 # штраф за отклонение частоты
MIN_TRADES_FOR_THR     = 20     # минимум сделок для кандидата

# ==== финальные требования к выбранному порогу ====
MIN_USED_TRADES        = 20
MIN_USED_EV            = 0.00005     # ≥ +0.005% EV/сделку
MIN_USED_TRADES_REGIM  = 12

# ==== режимы по волатильности ====
USE_REGIMES = True
REGIME_PCTL = 70
ATR_FEATURE = "atr_norm"

# ==== требования к валидации ====
VAL_MIN_POS        = 10
VAL_MIN_NEG        = 10
VAL_BACK_STEP_ROWS = 100

# -------------------- утилиты --------------------

def schema_hash(columns) -> str:
    s = json.dumps(list(columns), ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()

def load_data(path: str = DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "target" in df.columns:
        df["target"] = df["target"].astype(int)
    elif "pnl" in df.columns:
        df["target"] = (df["pnl"] > 0).astype(int)
    elif "direction" in df.columns:
        df["target"] = (df["direction"] > 0).astype(int)
    else:
        raise RuntimeError("Нет target/pnl/direction — не из чего сформировать целевую переменную.")

    # привести всё к числам
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

def _choose_features(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {target_col, "ts", "pnl", "direction", "target"}
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

# -------------------- устойчивый time-split --------------------

def _find_validation_window(df_sorted: pd.DataFrame, target_col: str,
                            desired_len: int, step: int,
                            min_pos: int, min_neg: int) -> Tuple[int,int]:
    n = len(df_sorted)
    start = max(0, n - desired_len)
    best = None
    s = start
    while s >= 0:
        y = df_sorted.iloc[s:][target_col]
        pos = int(y.sum()); neg = int(len(y) - pos)
        score = min(pos, neg)
        if best is None or score > best[0]:
            best = (score, s)
        if pos >= min_pos and neg >= min_neg:
            return s, n
        s -= step
    return (best[1] if best else start), n

def _time_split_3(df: pd.DataFrame, target_col: str):
    if "ts" not in df.columns:
        df = df.copy(); df["ts"] = range(len(df))
    df = df.sort_values("ts").reset_index(drop=True)

    n = len(df)
    desired_val_len = max(200, int(n * VAL_FRAC))
    step = max(50, desired_val_len // 10)

    val_start, val_end = _find_validation_window(df, target_col, desired_val_len, step, VAL_MIN_POS, VAL_MIN_NEG)
    val_df = df.iloc[val_start:val_end].copy()
    prefix = df.iloc[:val_start].copy()

    if len(prefix) < 100:
        train_df = prefix.copy()
        calib_df = prefix.iloc[0:0].copy()
    else:
        frac = TRAIN_FRAC / (TRAIN_FRAC + CALIB_FRAC)
        t_end = max(1, int(len(prefix) * frac))
        train_df = prefix.iloc[:t_end].copy()
        calib_df = prefix.iloc[t_end:].copy()

    feats = _choose_features(df, target_col)
    X_tr, y_tr = train_df[feats].fillna(0.0), train_df[target_col].astype(int)
    X_ca, y_ca = calib_df[feats].fillna(0.0), calib_df[target_col].astype(int)
    X_va, y_va = val_df[feats].fillna(0.0),  val_df[target_col].astype(int)

    def stats(name, y):
        p = int(y.sum()); n0 = int(len(y) - p)
        return f"{name}: rows={len(y)} pos={p} neg={n0} pos_rate={p/max(1,len(y)):.3f}"
    print("[SPLIT]", stats("train", y_tr), "|", stats("calib", y_ca), "|", stats("val", y_va))

    return feats, (X_tr,y_tr), (X_ca,y_ca), (X_va,y_va), train_df, calib_df, val_df

# -------------------- EV helpers & threshold search --------------------

def _ev_per_trade_array(win_prob: np.ndarray, tp_arr: np.ndarray, sl_arr: np.ndarray, cost: float) -> np.ndarray:
    """EV_i = p_i*(tp_i - cost) - (1-p_i)*(sl_i + cost)"""
    return win_prob * (tp_arr - cost) - (1.0 - win_prob) * (sl_arr + cost)

def pick_threshold_by_objective(
    y_true: np.ndarray,
    proba: np.ndarray,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    cost: float,
    min_trades: int,
    target_trades: Optional[int],
    freq_penalty: float,
):
    """
    Максимизируем: mean(EV_selected) - freq_penalty * |trades - target| / N
    Возвращает (thr, ev, precision, recall, trades, score)
    """
    y_true = np.asarray(y_true).astype(int)
    proba  = np.asarray(proba).astype(float)
    tp_arr = np.asarray(tp_arr).astype(float)
    sl_arr = np.asarray(sl_arr).astype(float)

    pos = int(y_true.sum()); neg = int(len(y_true) - pos)
    if pos == 0:
        return (1.0, 0.0, 0.0, 0.0, 0, 0.0)
    if neg == 0:
        ev_all = float(np.mean(_ev_per_trade_array(np.ones_like(proba), tp_arr, sl_arr, cost)))
        return (0.0, ev_all, 1.0, 1.0, int(len(y_true)), ev_all)

    thr_grid = np.unique(proba)  # быстрее, чем проход по PR-кривой
    best = (None, -1e9, 0.0, 0.0, 0, -1e9)
    N = len(y_true)

    for thr in thr_grid:
        preds = proba >= thr
        trades = int(preds.sum())
        if trades < min_trades:
            continue
        precision = float((y_true[preds] == 1).mean()) if trades > 0 else 0.0
        recall    = float(((y_true == 1) & preds).sum() / max(1, pos))
        ev_mean   = float(np.mean(_ev_per_trade_array(proba[preds], tp_arr[preds], sl_arr[preds], cost)))
        score = ev_mean
        if target_trades is not None and N > 0 and freq_penalty > 0.0:
            deviation = abs(trades - target_trades) / float(N)
            score = ev_mean - freq_penalty * deviation
        if (score > best[5]) or (abs(score - best[5]) < 1e-12 and trades > best[4]):
            best = (float(thr), ev_mean, precision, recall, trades, float(score))

    if best[0] is None:
        return (1.0, 0.0, 0.0, 0.0, 0, 0.0)
    return best

def pick_threshold_by_ev_only(
    y_true: np.ndarray,
    proba: np.ndarray,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    cost: float,
):
    """Максимизируем mean(EV_selected) без штрафа."""
    y_true = np.asarray(y_true).astype(int)
    proba  = np.asarray(proba).astype(float)
    tp_arr = np.asarray(tp_arr).astype(float)
    sl_arr = np.asarray(sl_arr).astype(float)

    pos = int(y_true.sum()); neg = int(len(y_true) - pos)
    if pos == 0:
        return (1.0, 0.0, 0.0, 0.0, 0)
    if neg == 0:
        ev_all = float(np.mean(_ev_per_trade_array(np.ones_like(proba), tp_arr, sl_arr, cost)))
        return (0.0, ev_all, 1.0, 1.0, int(len(y_true)))

    thr_grid = np.unique(proba)
    best = (1.0, -1e9, 0.0, 0.0, 0)
    for thr in thr_grid:
        preds = proba >= thr
        trades = int(preds.sum())
        if trades == 0:
            continue
        precision = float((y_true[preds] == 1).mean())
        recall    = float(((y_true == 1) & preds).sum() / max(1, int(y_true.sum())))
        ev_mean   = float(np.mean(_ev_per_trade_array(proba[preds], tp_arr[preds], sl_arr[preds], cost)))
        if (ev_mean > best[1]) or (abs(ev_mean - best[1]) < 1e-12 and trades > best[4]):
            best = (float(thr), ev_mean, precision, recall, trades)
    return best

def rescue_threshold_from_ev_only(
    y_true: np.ndarray,
    proba: np.ndarray,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    cost: float,
    start_thr: float,
    min_trades: int,
    min_ev: float = 0.0,
    max_relax: float = 0.25,
    steps: int = 100,
):
    """Плавно ослабляем порог, пока не получим >= min_trades и EV>=min_ev."""
    if len(y_true) == 0:
        return None
    lo = max(0.0, start_thr * (1.0 - max_relax))
    hi = float(start_thr)
    grid = np.linspace(hi, lo, steps)
    pos = int(y_true.sum())

    for thr in grid:
        preds = proba >= thr
        trades = int(preds.sum())
        if trades < min_trades or trades == 0:
            continue
        precision = float((y_true[preds] == 1).mean())
        recall    = float(((y_true == 1) & preds).sum() / max(1, pos))
        ev_mean   = float(np.mean(_ev_per_trade_array(proba[preds], tp_arr[preds], sl_arr[preds], cost)))
        if ev_mean >= min_ev:
            return (float(thr), ev_mean, precision, recall, trades)
    return None

def _eval_threshold_subset(y_true: np.ndarray, proba: np.ndarray, thr: float,
                           tp_arr: np.ndarray, sl_arr: np.ndarray, cost: float):
    if thr is None or len(y_true) == 0:
        return None
    preds = proba >= thr
    trades = int(preds.sum())
    if trades == 0:
        return (float(thr), 0.0, 0.0, 0.0, 0)
    precision = float((y_true[preds] == 1).mean())
    recall    = float(((y_true == 1) & preds).sum() / max(1, int(y_true.sum())))
    ev_mean   = float(np.mean(_ev_per_trade_array(proba[preds], tp_arr[preds], sl_arr[preds], cost)))
    return (float(thr), ev_mean, precision, recall, trades)

# -------------------- обучение/экспорт --------------------

def train_and_export(df: pd.DataFrame):
    target_col = "target"
    feats, (X_tr,y_tr), (X_ca,y_ca), (X_va,y_va), tr_df, ca_df, va_df = _time_split_3(df, target_col)

    # базовая модель
    base = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    # калибровка на train+calib
    X_tc = pd.concat([X_tr, X_ca], axis=0)
    y_tc = pd.concat([y_tr, y_ca], axis=0)
    tscv = TimeSeriesSplit(n_splits=3)
    calibrator = CalibratedClassifierCV(estimator=base, cv=tscv, method="sigmoid")
    calibrator.fit(X_tc, y_tc)

    # вероятности на валидации
    val_proba = calibrator.predict_proba(X_va)[:, 1]

    # массивы TP/SL на валидации
    if {"tp_pct_used","sl_pct_used"}.issubset(va_df.columns):
        tp_va = va_df["tp_pct_used"].to_numpy(dtype=float)
        sl_va = va_df["sl_pct_used"].to_numpy(dtype=float)
        using_per_trade = True
    else:
        tp_va = np.full(len(y_va), TP_PCT, dtype=float)
        sl_va = np.full(len(y_va), SL_PCT, dtype=float)
        using_per_trade = False

    target_trades_global = int(len(y_va) * (TARGET_TRADES_PER_1000 / 1000.0))

    # глобальный кандидат
    thr_global, ev_glob, p_glob, r_glob, n_glob, score_glob = pick_threshold_by_objective(
        y_va.to_numpy(), val_proba,
        tp_arr=tp_va, sl_arr=sl_va, cost=COST_PCT_ROUND_TRIP,
        min_trades=MIN_TRADES_FOR_THR,
        target_trades=target_trades_global,
        freq_penalty=FREQ_PENALTY,
    )

    # ev_only
    thr_evonly, ev_only, p_only, r_only, n_only = pick_threshold_by_ev_only(
        y_va.to_numpy(), val_proba,
        tp_arr=tp_va, sl_arr=sl_va, cost=COST_PCT_ROUND_TRIP,
    )

    # режимы по atr_norm
    thr_low = thr_high = None
    atr_cut = None
    low_stats = high_stats = None
    if USE_REGIMES and (ATR_FEATURE in va_df.columns):
        atr_base = pd.concat([tr_df, ca_df], axis=0)
        if atr_base[ATR_FEATURE].notna().any():
            atr_cut = float(np.percentile(atr_base[ATR_FEATURE].dropna(), REGIME_PCTL))
            mask_high = va_df[ATR_FEATURE] >= atr_cut
            mask_low  = ~mask_high

            if mask_low.sum() >= MIN_TRADES_FOR_THR:
                thr_low, *_ = pick_threshold_by_objective(
                    y_va[mask_low].to_numpy(), val_proba[mask_low],
                    tp_arr=tp_va[mask_low], sl_arr=sl_va[mask_low],
                    cost=COST_PCT_ROUND_TRIP,
                    min_trades=MIN_TRADES_FOR_THR,
                    target_trades=int(mask_low.sum() * (TARGET_TRADES_PER_1000 / 1000.0)),
                    freq_penalty=FREQ_PENALTY,
                )
                low_stats = _eval_threshold_subset(
                    y_true=y_va[mask_low].to_numpy(), proba=val_proba[mask_low], thr=thr_low,
                    tp_arr=tp_va[mask_low], sl_arr=sl_va[mask_low], cost=COST_PCT_ROUND_TRIP
                )

            if mask_high.sum() >= MIN_TRADES_FOR_THR:
                thr_high, *_ = pick_threshold_by_objective(
                    y_va[mask_high].to_numpy(), val_proba[mask_high],
                    tp_arr=tp_va[mask_high], sl_arr=sl_va[mask_high],
                    cost=COST_PCT_ROUND_TRIP,
                    min_trades=MIN_TRADES_FOR_THR,
                    target_trades=int(mask_high.sum() * (TARGET_TRADES_PER_1000 / 1000.0)),
                    freq_penalty=FREQ_PENALTY,
                )
                high_stats = _eval_threshold_subset(
                    y_true=y_va[mask_high].to_numpy(), proba=val_proba[mask_high], thr=thr_high,
                    tp_arr=tp_va[mask_high], sl_arr=sl_va[mask_high], cost=COST_PCT_ROUND_TRIP
                )

    # финальный выбор
    candidates = [
        ("global",  thr_global, ev_glob, p_glob, r_glob, n_glob),
        ("ev_only", thr_evonly, ev_only, p_only, r_only, n_only),
    ]
    if low_stats is not None:
        t,e,p,r,n = low_stats; candidates.append(("regime_low", t,e,p,r,n))
    if high_stats is not None:
        t,e,p,r,n = high_stats; candidates.append(("regime_high", t,e,p,r,n))

    filt = []
    for name, thr, ev, p, r, n in candidates:
        if name.startswith("regime"):
            if n >= MIN_USED_TRADES_REGIM and ev >= MIN_USED_EV:
                filt.append((name, thr, ev, p, r, n))
        else:
            if n >= MIN_USED_TRADES and ev >= MIN_USED_EV:
                filt.append((name, thr, ev, p, r, n))

    if filt:
        filt.sort(key=lambda x: (x[2], x[5]))    # max EV, при равенстве — больше сделок
        used_mode, used_thr, used_ev, used_p, used_r, used_n = filt[-1]
    else:
        rescue = None
        if ev_only >= MIN_USED_EV and n_only < MIN_USED_TRADES:
            rescue = rescue_threshold_from_ev_only(
                y_true=y_va.to_numpy(), proba=val_proba,
                tp_arr=tp_va, sl_arr=sl_va, cost=COST_PCT_ROUND_TRIP,
                start_thr=thr_evonly,
                min_trades=MIN_USED_TRADES, min_ev=MIN_USED_EV,
                max_relax=0.25, steps=120
            )
        if rescue is not None:
            r_thr, r_ev, r_p, r_r, r_n = rescue
            used_mode, used_thr, used_ev, used_p, used_r, used_n = "ev_rescue", r_thr, r_ev, r_p, r_r, r_n
        else:
            used_mode, used_thr, used_ev, used_p, used_r, used_n = "global", 1.0, 0.0, 0.0, 0.0, 0

    # отчёт
    val_pred_used = (val_proba >= used_thr).astype(int)
    print("=== RandomForest (time-split, sigmoid-calibrated, EV thresholds) ===")
    print(classification_report(y_va, val_pred_used, digits=4, zero_division=0))
    acc = accuracy_score(y_va, val_pred_used)
    f1  = f1_score(y_va, val_pred_used, zero_division=0)

    print(f"[THR:{used_mode}] thr={used_thr:.4f} | precision={used_p:.4f} recall={used_r:.4f} "
          f"trades={used_n} | EV≈{used_ev*100:.3f}%")
    print(f"[THR:global_raw] thr={thr_global:.4f} | precision={p_glob:.4f} recall={r_glob:.4f} "
          f"trades={n_glob} | EV≈{ev_glob*100:.3f}%")
    print(f"[THR:ev_only   ] thr={thr_evonly:.4f} | precision={p_only:.4f} recall={r_only:.4f} "
          f"trades={n_only} | EV≈{ev_only*100:.3f}%")
    if low_stats is not None:
        t,e,p,r,n = low_stats
        print(f"[THR:reg_low  ] thr={t:.4f} | precision={p:.4f} recall={r:.4f} trades={n} | EV≈{e*100:.3f}%")
    if high_stats is not None:
        t,e,p,r,n = high_stats
        print(f"[THR:reg_high ] thr={t:.4f} | precision={p:.4f} recall={r:.4f} trades={n} | EV≈{e*100:.3f}%")
    if {"tp_pct_used","sl_pct_used"}.issubset(va_df.columns):
        print(f"[INFO] val medians: tp={np.median(tp_va):.4f}, sl={np.median(sl_va):.4f}")

    # сохранить
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(calibrator, f)

    meta = {
        "features": feats,
        "hash": schema_hash(feats),
        "thresholds": {
            "used_mode": used_mode,
            "used": float(used_thr),
            "global": float(thr_global),
            "ev_only": float(thr_evonly),
            "low_vol": (None if thr_low is None else float(thr_low)),
            "high_vol": (None if thr_high is None else float(thr_high)),
            "regime_feature": ATR_FEATURE,
            "regime_percentile": REGIME_PCTL,
            "atr_norm_cut": (None if atr_cut is None else float(atr_cut)),
        },
        "calibrated": True,
        "n_rows": int(len(df)),
        "split": {
            "type": "time",
            "fractions": {"train": TRAIN_FRAC, "calib": CALIB_FRAC, "val": VAL_FRAC},
            "train_rows": int(len(tr_df)),
            "calib_rows": int(len(ca_df)),
            "val_rows": int(len(va_df)),
        },
        "metrics": {
            "val_accuracy_at_used": float(acc),
            "val_f1_at_used": float(f1),
        },
        "ev_params": {
            "using_per_trade_ev": using_per_trade,
            "tp_pct_default": float(TP_PCT),
            "sl_pct_default": float(SL_PCT),
            "cost_pct_round_trip": float(COST_PCT_ROUND_TRIP),
            "target_trades_per_1000": int(TARGET_TRADES_PER_1000),
            "freq_penalty": float(FREQ_PENALTY),
            "min_trades_for_thr": int(MIN_TRADES_FOR_THR),
            "min_used_trades": int(MIN_USED_TRADES),
            "min_used_ev": float(MIN_USED_EV),
            "min_used_trades_regime": int(MIN_USED_TRADES_REGIM),
        },
    }
    with open(MODEL_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Model → {MODEL_FILE}")
    print(f"[OK] Meta  → {MODEL_META} (features={len(feats)})")
    return float(used_thr)

# -------------------- online API --------------------

def predict_proba_online(feature_dict: dict) -> float:
    with open(MODEL_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta["features"]
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    row = [[float(feature_dict.get(k, 0.0)) for k in feats]]
    return float(model.predict_proba(row)[0][1])

def predict_signal_online(feature_dict: dict) -> int:
    with open(MODEL_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    thr_cfg = meta.get("thresholds", {})
    feats = meta["features"]
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    p = float(model.predict_proba([[float(feature_dict.get(k, 0.0)) for k in feats]])[0][1])

    used_mode = thr_cfg.get("used_mode", "global")
    thr = float(thr_cfg.get("used", thr_cfg.get("global", 0.5)))

    if used_mode.startswith("regime"):
        atr_cut = thr_cfg.get("atr_norm_cut", None)
        reg_feat = thr_cfg.get("regime_feature", ATR_FEATURE)
        if atr_cut is not None and reg_feat in feature_dict:
            val = float(feature_dict.get(reg_feat, 0.0))
            low_thr  = thr_cfg.get("low_vol", None)
            high_thr = thr_cfg.get("high_vol", None)
            if val >= float(atr_cut) and high_thr is not None:
                thr = float(high_thr)
            elif val < float(atr_cut) and low_thr is not None:
                thr = float(low_thr)
    return int(p >= thr)

# -------------------- cli --------------------
if __name__ == "__main__":
    df = load_data()
    print(f"Форма датасета: {df.shape}")
    train_and_export(df)
