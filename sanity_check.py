# sanity_check.py
# Ничего не торгует. Проверяет конфиг, модель, наличие ключевых функций.
# Онлайн-запросы к бирже отключены по умолчанию. Включить: RUN_ONLINE_CHECKS=1

import os, json, pickle, importlib, sys, traceback, platform

RUN_ONLINE = str(os.getenv("RUN_ONLINE_CHECKS", "0")).strip().lower() in ("1","true","yes")

def safe_import(name):
    try:
        m = importlib.import_module(name)
        print(f"[OK] import {name}")
        return m
    except Exception as e:
        print(f"[FAIL] import {name}: {e.__class__.__name__}: {e}")
        traceback.print_exc(limit=1)
        return None

def check_file(path):
    if os.path.exists(path):
        print(f"[OK] file: {path} ({os.path.getsize(path)} bytes)")
        return True
    else:
        print(f"[MISS] file: {path}")
        return False

def check_attrs(module, expected):
    if not module:
        return
    missing = [a for a in expected if not hasattr(module, a)]
    if missing:
        print(f"[WARN] {module.__name__}: нет атрибутов: {missing}")
    else:
        print(f"[OK] {module.__name__}: все ключевые атрибуты присутствуют")

def main():
    print("=== 0) Env ===")
    print(f"Python: {platform.python_version()}  |  Executable: {sys.executable}")
    print(f"RUN_ONLINE_CHECKS={int(RUN_ONLINE)}")

    print("\n=== 1) Конфиг ===")
    cfg = safe_import("config")
    if cfg:
        keys_core = [
            "PAPER_MODE","PAIRS_COUNT","TOP_LIQUID_PAIRS","MAX_BALANCE_SHARE",
            "MIN_NOTIONAL_USDT","DEFAULT_LEVERAGE","MIN_ATR_PCT","TRAIL_DROP_PCT",
            "ML_THRESHOLD","ML_VETO_ENABLED","ML_VETO_THR","DEBUG_TRADING",
        ]
        keys_pairs = [
            "AUTO_SELECT_PAIRS","AUTO_PAIRS_RULE",
            "PAIR_FILTER_MIN_NOTIONAL","PAIR_FILTER_HCAP_FRAC",
        ]
        keys_risk_spread = [
            "USEABLE_BAL_SHARE","HARD_CAP_SHARE",
            "SPREAD_MAX_PCT","SPREAD_DEPTH",
            "RISK_PER_TRADE_FRAC","ATR_STOP_K",
        ]
        all_keys = keys_core + keys_pairs + keys_risk_spread
        for k in all_keys:
            print(f"  {k} = {getattr(cfg, k, '<нет>')}")
        if not isinstance(getattr(cfg, "TOP_LIQUID_PAIRS", []), list):
            print("[WARN] TOP_LIQUID_PAIRS не list")
        # мягкая подсказка по удалению тестнета из окружения
        if os.getenv("BYBIT_TESTNET"):
            print("[INFO] BYBIT_TESTNET найден в окружении, но код больше его не использует.")

    print("\n=== 2) Модель ===")
    model_ok = check_file(getattr(cfg, "MODEL_FILE", "rf_model.pkl") if cfg else "rf_model.pkl")
    meta_ok  = check_file(getattr(cfg, "MODEL_META", "model_meta.json") if cfg else "model_meta.json")
    model = None; meta = None
    if model_ok:
        try:
            with open(getattr(cfg, "MODEL_FILE", "rf_model.pkl"),"rb") as f:
                model = pickle.load(f)
            cls = getattr(model, "__class__", type("X", (object,), {})).__name__
            print(f"[OK] model loaded: {cls}")
            has_proba = hasattr(model, "predict_proba")
            nfeat = getattr(model, "n_features_in_", None)
            print(f"  predict_proba={has_proba}, n_features_in_={nfeat}")
        except Exception as e:
            print(f"[FAIL] load model: {e}")
    if meta_ok:
        try:
            with open(getattr(cfg, "MODEL_META", "model_meta.json"),"r",encoding="utf-8") as f:
                meta = json.load(f)
            feats = (meta or {}).get("features")
            thr   = ((meta or {}).get("thresholds") or {}).get("used") or ((meta or {}).get("thresholds") or {}).get("global")
            print(f"[OK] meta loaded: features={len(feats) if feats else 'None'}, thr={thr}")
            if model is not None and hasattr(model, "n_features_in_") and feats:
                if model.n_features_in_ != len(feats):
                    print(f"[WARN] n_features_in_ ({model.n_features_in_}) != len(meta.features) ({len(feats)})")
        except Exception as e:
            print(f"[FAIL] load meta: {e}")

    print("\n=== 3) Стратегия/утилиты ===")
    strat = safe_import("strategy")
    if strat:
        if hasattr(strat, "detect_impulse"):
            try:
                # 10 свечей-заглушек OHLCV: open, high, low, close, volume
                dummy = [
                    [100,101,99,100.5,10],[100.5,101.2,100,101,12],
                    [101,102,100.5,101.5,9],[101.5,103,101,102.7,15],
                    [102.7,103.5,102,103.2,11],[103.2,104,102.9,103.6,13],
                    [103.6,104.3,102.8,103.1,9],[103.1,103.7,102.4,103.4,8],
                    [103.4,104.1,102.7,103.9,10],[103.9,105,103.1,104.8,16],
                ]
                sig = strat.detect_impulse(dummy)
                print(f"[OK] detect_impulse() отработал: {sig}")
            except Exception as e:
                print(f"[FAIL] detect_impulse(): {e}")
        else:
            print("[MISS] strategy.detect_impulse")

    utils = safe_import("utils")
    if utils:
        for name in ["log","tg_send","write_cycle_log","adjust_qty","SAFE_MODE"]:
            print(f"  utils.{name}: {'OK' if hasattr(utils,name) else 'MISS'}")

    print("\n=== 4) Брокерные интерфейсы ===")
    paper = safe_import("paper_engine")
    if paper:
        check_attrs(paper, [
            "get_balance","get_kline_any","get_ticker_snapshot","get_min_order_filters",
            "set_leverage","place_market_order","has_open_position","force_close_all_positions_absolute"
        ])
        try:
            bal = paper.get_balance()
            print(f"[OK] paper_engine.get_balance() -> {bal}")
        except Exception as e:
            print(f"[FAIL] paper_engine.get_balance(): {e}")

    bybit = safe_import("bybit_api")
    if bybit:
        check_attrs(bybit, [
            "get_kline_any","get_ticker_snapshot","get_orderbook_spread",
            "get_min_order_filters","get_current_price","get_tickers_linear",
            "place_market_order","has_open_position","force_close_all_positions_absolute",
            "fetch_price_history","get_server_time"
        ])
        if RUN_ONLINE:
            try:
                ts = bybit.get_server_time()
                print(f"[OK] bybit.get_server_time() -> {ts}")
            except Exception as e:
                print(f"[WARN] bybit.get_server_time(): {e}")

    print("\n=== 5) Финал ===")
    print("Если есть [FAIL]/[WARN] — пришли вывод, дам фикс-патчи.")
    print("Если всё [OK] — запускай бота в PAPER:  python main.py paper")

if __name__ == "__main__":
    main()
