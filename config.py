# -*- coding: utf-8 -*-
# config.py
"""
Все значения подтягиваются из .env (load_dotenv).
Надёжные парсеры для bool/int/float/list, безопасные дефолты.
Сохраняем прежние имена переменных, чтобы не ломать импорт.
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()  # подхватываем .env рядом с проектом

# ----------------------
# helpers
# ----------------------
_TRUE = {"1", "true", "yes", "y", "on", "t"}
_FALSE = {"0", "false", "no", "n", "off", "f"}

def env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def env_bool(key: str, default: bool = False) -> int:
    """
    Возвращаем 0/1 (как раньше), но понимаем 1/0/true/false/yes/no…
    """
    raw = os.getenv(key)
    if raw is None:
        return 1 if default else 0
    s = raw.strip().lower()
    if s in _TRUE:
        return 1
    if s in _FALSE:
        return 0
    # fallback: непустая строка -> True
    return 1 if bool(s) else 0

def env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(float(raw))  # чтобы "10.0" не падало
    except Exception:
        return int(default)

def env_float(key: str, default: float = 0.0) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)

def env_list(key: str, default_csv: str = "", sep: str = ",") -> List[str]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        raw = default_csv
    return [x.strip() for x in raw.split(sep) if x.strip()]

# ================== БАЗОВЫЕ ПАРЫ/ОБЪЁМ РЫНКА ==================
TOP_LIQUID_PAIRS = env_list(
    "TOP_LIQUID_PAIRS",
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,ADAUSDT,OPUSDT,AVAXUSDT,LINKUSDT,DOTUSDT,MATICUSDT,LTCUSDT"
)
PAIRS_COUNT       = env_int("PAIRS_COUNT", 5)
AUTO_SELECT_PAIRS = env_bool("AUTO_SELECT_PAIRS", 1)

# (2) авто-правило «баланс → кол-во пар» (использует main.py)
AUTO_PAIRS_RULE = env_str("AUTO_PAIRS_RULE", "0:2,25:3,60:4,120:5")

# (3) фильтр пар по окну нотационала [мин ; баланс*доля] (использует main.py)
PAIR_FILTER_MIN_NOTIONAL = env_float("PAIR_FILTER_MIN_NOTIONAL", 0.50)
PAIR_FILTER_HCAP_FRAC    = env_float("PAIR_FILTER_HCAP_FRAC", 1.0)
EXPLORATION_FRAC         = env_float("EXPLORATION_FRAC", 0.35)
EXPLORATION_ATR_BONUS    = env_float("EXPLORATION_ATR_BONUS", 0.15)

# ================== РЕЖИМ ЗАПУСКА ==================
# 0 — реальная торговля, 1 — бумажный режим (виртуальные сделки по реальному рынку)
PAPER_MODE = env_bool("PAPER_MODE", False)
SAFE_MODE  = env_bool("SAFE_MODE",  False)

# Принудительная кодировка консоли/сабпроцессов (полезно на Windows)
PYTHONIOENCODING = env_str("PYTHONIOENCODING", "UTF-8")
SUBPROC_ENCODING = env_str("SUBPROC_ENCODING", "UTF-8")

# PAPER-кошелёк
PAPER_SYNC_BALANCE    = env_bool("PAPER_SYNC_BALANCE", True)
PAPER_BALANCE_MULT    = env_float("PAPER_BALANCE_MULT", 1.0)
VIRTUAL_START_BALANCE = env_float("VIRTUAL_START_BALANCE", 100.0)

# ================== СЕССИИ РАБОТА/ПЕРЕРЫВ ==================
WORK_DURATION_SEC   = env_int("WORK_DURATION_SEC",  3600)
BREAK_DURATION_SEC  = env_int("BREAK_DURATION_SEC", 600)
ENTRY_COOLDOWN_SEC  = env_int("ENTRY_COOLDOWN_SEC", 12)

# Расписание торговли (локальное время)
EXCLUDE_WEEKENDS   = env_int("EXCLUDE_WEEKENDS", 1)
TRADE_HOURS_LOCAL  = env_str("TRADE_HOURS_LOCAL", "22:00-08:00")
FORCE_SCHEDULE_OFF = env_int("FORCE_SCHEDULE_OFF", 0)

# ================== ДАННЫЕ/ЛОГИ ==================
DATA_ROOT          = env_str("DATA_ROOT", "").strip() or "./data"
RECORD_MARKET_DATA = env_bool("RECORD_MARKET_DATA", True)
LOG_JSONL          = env_str("LOG_JSONL", "bot_cycle_log.jsonl")
LOG_ENABLED        = env_bool("LOG_ENABLED", True)
LOG_RU             = env_bool("LOG_RU", True)
KLINE_HISTORY_LIMIT = env_int("KLINE_HISTORY_LIMIT", 300)
ROUTER_HEARTBEAT_SEC = env_int("ROUTER_HEARTBEAT_SEC", 60)
MARGIN_POLL_SEC      = env_int("MARGIN_POLL_SEC", 15)
MIN_BARS             = env_int("MIN_BARS", 210)
TG_DAILY_REPORT    = env_bool("TG_DAILY_REPORT", False)

# ================== APK MANAGER / SECURITY ==================
ENABLE_APK_MANAGER        = bool(env_bool("ENABLE_APK_MANAGER", 1))
APK_DEFAULT_PORT          = env_int("APK_DEFAULT_PORT", 8787)
APK_DEFAULT_AUTOSTOP_MIN  = env_int("APK_DEFAULT_AUTOSTOP_MIN", 15)
APK_BIND_ALL              = bool(env_bool("APK_BIND_ALL", 0))
APK_ENABLE_BASIC_AUTH     = bool(env_bool("APK_ENABLE_BASIC_AUTH", 0))
APK_BASIC_AUTH_USER       = env_str("APK_BASIC_AUTH_USER", "tradeapp")
APK_BASIC_AUTH_PASS       = env_str("APK_BASIC_AUTH_PASS", "changeme")
APK_MANAGER_LOG           = env_str("APK_MANAGER_LOG", "logs/apk_manager.jsonl")
TRUSTED_IPS               = env_list("TRUSTED_IPS", "127.0.0.1,192.168.1.0/24")
ENABLE_PIN_FOR_REAL       = bool(env_bool("ENABLE_PIN_FOR_REAL", 1))
PIN_HASH                  = env_str("PIN_HASH", "")
ENABLE_TG_2FA             = bool(env_bool("ENABLE_TG_2FA", 0))
TG_2FA_TTL                = env_int("TG_2FA_TTL", 300)
ONE_TIME_APK_LINK         = bool(env_bool("ONE_TIME_APK_LINK", 1))

# ================== TELEGRAM ==================
TELEGRAM_TOKEN   = env_str("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = env_str("TELEGRAM_CHAT_ID", "")

# ================== BYBIT КРЕДЫ / ЗАЩИТА REAL ==================
BYBIT_API_KEY    = env_str("BYBIT_API_KEY", "")
BYBIT_API_SECRET = env_str("BYBIT_API_SECRET", "")
CONFIRM_REAL     = env_int("CONFIRM_REAL", 0)

# ================== СТРАТЕГИЯ ==================
STRATEGY_MODE    = env_str("STRATEGY_MODE", "prod")
TEST_SIGNAL_MODE = env_str("TEST_SIGNAL_MODE", "alt")

EMA_FAST          = env_int("EMA_FAST", 9)
EMA_SLOW          = env_int("EMA_SLOW", 21)
MOM_LOOKBACK      = env_int("MOM_LOOKBACK", 5)
MOM_MIN_PCT       = env_float("MOM_MIN_PCT", 0.0003)
BREAKOUT_LOOKBACK = env_int("BREAKOUT_LOOKBACK", 20)
BREAKOUT_PAD_PCT  = env_float("BREAKOUT_PAD_PCT", 0.0002)
DEBUG_TRADING     = env_bool("DEBUG_TRADING", 0)

STRAT_VOL_LOOKBACK = env_int("STRAT_VOL_LOOKBACK", 30)
STRAT_VOL_BOOST_MIN = env_float("STRAT_VOL_BOOST_MIN", 1.02)
STRAT_CONFIRM_BARS = env_int("STRAT_CONFIRM_BARS", 1)

# ================== РИСК-МЕНЕДЖМЕНТ / ИСПОЛНЕНИЕ ==================
DEFAULT_LEVERAGE    = env_int("DEFAULT_LEVERAGE", 10)
MAX_BALANCE_SHARE   = env_float("MAX_BALANCE_SHARE", 0.10)   # потолок по нотационалу сделки
COMMISSION_PER_SIDE = env_float("COMMISSION_PER_SIDE", 0.0006)
ADAPTIVE_LEV_ENABLED        = env_bool("ADAPTIVE_LEV_ENABLED", 1)
ADAPTIVE_LEV_MIN            = env_int("ADAPTIVE_LEV_MIN", 5)
ADAPTIVE_LEV_MAX            = env_int("ADAPTIVE_LEV_MAX", 100)
ADAPTIVE_LEV_TIER_RULE      = env_str("ADAPTIVE_LEV_TIER_RULE", "0:85,25:75,50:60,100:45,250:30,500:20")
ADAPTIVE_LEV_ATR_REF_PCT    = env_float("ADAPTIVE_LEV_ATR_REF_PCT", 0.001)
ADAPTIVE_LEV_SPREAD_PENALTY = env_float("ADAPTIVE_LEV_SPREAD_PENALTY", 1.5)
ADAPTIVE_LEV_REEVAL_SEC     = env_int("ADAPTIVE_LEV_REEVAL_SEC", 300)
ADAPTIVE_LEV_REQUIRE_AFFORDABLE = env_bool("ADAPTIVE_LEV_REQUIRE_AFFORDABLE", 1)
LEV_STEP_MAX                = env_float("LEV_STEP_MAX", 2.0)
BANDIT_AGING                = env_float("BANDIT_AGING", 0.995)
MIN_OBS_BEFORE_EXPLOIT      = env_int("MIN_OBS_BEFORE_EXPLOIT", 20)
FORCED_EXPLORATION_RATE     = env_float("FORCED_EXPLORATION_RATE", 0.10)

# Порог достаточной волатильности: ATR >= MIN_ATR_PCT * price
MIN_ATR_PCT         = env_float("MIN_ATR_PCT", 0.00022)

# Наш пользовательский минимум по нотационалу (дополнительно к биржевому minOrderAmt)
MIN_NOTIONAL_USDT   = env_float("MIN_NOTIONAL_USDT", 5.0)

# Анти-тильт
MAX_DRAWDOWN_PCT        = env_float("MAX_DRAWDOWN_PCT", 8.0)
MAX_CONSECUTIVE_LOSSES  = env_int("MAX_CONSECUTIVE_LOSSES", 3)
LOSS_COOLDOWN_SEC       = env_int("LOSS_COOLDOWN_SEC", 900)

# Доли доступного/хард-кап (для попыток «поднять до min_qty»)
USEABLE_BAL_SHARE = env_float("USEABLE_BAL_SHARE", 0.95)
HARD_CAP_SHARE    = env_float("HARD_CAP_SHARE", 0.60)

# Динамический выход (трейл от пика UPNL)
TRAIL_DROP_PCT = env_float("TRAIL_DROP_PCT", 0.004)

# Фолбэки для пары, если API не вернул фильтры
DEFAULT_MIN_QTY_FALLBACK      = env_float("DEFAULT_MIN_QTY_FALLBACK", 0.001)
DEFAULT_QTY_STEP_FALLBACK     = env_float("DEFAULT_QTY_STEP_FALLBACK", 0.001)
DEFAULT_MIN_NOTIONAL_FALLBACK = env_float("DEFAULT_MIN_NOTIONAL_FALLBACK", 5.0)

# ---- Новые параметры под main.py ----
# Гейт по спрэду: если относительный спрэд > SPREAD_MAX_PCT — пропускаем вход
SPREAD_MAX_PCT = env_float("SPREAD_MAX_PCT", 0.0012)  # 0.12%
SPREAD_DEPTH   = env_int("SPREAD_DEPTH", 1)

# Позиционирование от риска и ATR:
# размер ≈ (баланс * RISK_PER_TRADE_FRAC) / (ATR_STOP_K * ATR)
RISK_PER_TRADE_FRAC = env_float("RISK_PER_TRADE_FRAC", 0.0065)  # 0.65% от баланса
ATR_STOP_K          = env_float("ATR_STOP_K", 1.2)
MIN_SHARE           = env_float("MIN_SHARE", 0.001)
MAX_SHARE           = env_float("MAX_SHARE", MAX_BALANCE_SHARE)

# Маржинальные граничные уровни
MAX_IM_PERCENT  = env_float("MAX_IM_PERCENT", 30.0)
CRIT_IM_PERCENT = env_float("CRIT_IM_PERCENT", 60.0)

# ================== ПАРАМЕТРЫ PAPER-ДВИЖКА ==================
SLIPPAGE_BPS = env_float("SLIPPAGE_BPS", 2.0)
TAKER_FEE    = env_float("TAKER_FEE", 0.0006)

# ================== МАШИННОЕ ОБУЧЕНИЕ ==================
ML_THRESHOLD         = env_float("ML_THRESHOLD", 0.58)
AUTO_RETRAIN_ON_EXIT = env_bool("AUTO_RETRAIN_ON_EXIT", False)

# --- ML "VETO" ---
ML_VETO_ENABLED = env_bool("ML_VETO_ENABLED", True)
ML_VETO_THR     = env_float("ML_VETO_THR", 0.30)
ML_VETO_LOG     = env_bool("ML_VETO_LOG", True)
ML_CONF_HIGH    = env_float("ML_CONF_HIGH", 0.80)
ML_CONF_MID     = env_float("ML_CONF_MID", 0.65)
ML_MIN_WEEKLY_PREC = env_float("ML_MIN_WEEKLY_PREC", 0.52)

# --- Candle patterns ---
ENABLE_CANDLE_PATTERNS = env_bool("ENABLE_CANDLE_PATTERNS", 1)
CANDLE_MIN_CONF = env_float("CANDLE_MIN_CONF", 0.55)
CANDLE_LOOKBACK_BARS = env_int("CANDLE_LOOKBACK_BARS", 60)
PATTERN_VOL_FILTER = env_bool("PATTERN_VOL_FILTER", 1)

# Пути к артефактам модели
MODEL_FILE = env_str("MODEL_FILE", "rf_model.pkl")
MODEL_META = env_str("MODEL_META", "model_meta.json")

# ================== API GUARD (rate-limit для dataset/сетевых утилит) ==================
API_GUARD_RATE  = env_float("API_GUARD_RATE", 4.8)  # токенов в секунду
API_GUARD_BURST = env_int("API_GUARD_BURST", 10)    # ёмкость «взрыва»
API_GUARD_METRICS_SEC = env_float("API_GUARD_METRICS_SEC", 30.0)

# ===========================================================
# ============ manage_ml / dataset builder ==================
# ===========================================================
BYBIT_CSV_PATH    = env_str("BYBIT_CSV_PATH", "fills_all.csv")
DATASET_SINCE     = env_str("DATASET_SINCE", "2025-01-01")

HISTORY_MINUTES   = env_int("HISTORY_MINUTES", 60)
LABEL_HORIZON_MIN = env_int("LABEL_HORIZON_MIN", 60)

PARALLELISM           = env_int("PARALLELISM", 10)
MIN_DELAY_BETWEEN_REQ = env_float("MIN_DELAY_BETWEEN_REQ", 0.08)
HTTP_RETRIES          = env_int("HTTP_RETRIES", 6)
MAX_BACKOFF           = env_float("MAX_BACKOFF", 8.0)

ENABLE_LRU_CACHE   = env_bool("ENABLE_LRU_CACHE", 1)
KLINE_CACHE_MAX    = env_int("KLINE_CACHE_MAX", 16384)
SNAPSHOT_CACHE_TTL = env_float("SNAPSHOT_CACHE_TTL", 15.0)

LIMIT_ROWS = env_int("LIMIT_ROWS", 0)

TP_MODE       = env_str("TP_MODE", "adaptive")  # adaptive | fixed
TP_PCT_FIXED  = env_float("TP_PCT_FIXED", 0.0030)
SL_PCT_FIXED  = env_float("SL_PCT_FIXED", 0.0025)
TP_ATR_K      = env_float("TP_ATR_K", 5.0)
SL_ATR_K      = env_float("SL_ATR_K", 4.0)
TP_CLAMP_LO   = env_float("TP_CLAMP_LO", 0.0020)
TP_CLAMP_HI   = env_float("TP_CLAMP_HI", 0.0060)
SL_CLAMP_LO   = env_float("SL_CLAMP_LO", 0.0015)
SL_CLAMP_HI   = env_float("SL_CLAMP_HI", 0.0040)

MICRO_SLEEP = env_float("MICRO_SLEEP", 0.002)

