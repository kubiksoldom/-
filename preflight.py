# -*- coding: utf-8 -*-
"""
preflight.py
------------
Одноразовая «техготовность» перед REAL:
- проверяет доступ к Bybit (баланс, тикеры)
- пытается определить режим позиций (One-Way/Hedge) по positionIdx
- показывает выбранные к торговле пары и их фильтры (min_qty/step)
- пингует Telegram (опционально)
- проверяет DATA_ROOT (создает при необходимости)

НЕ создаёт ордера. Безопасно.
Запуск:
    python preflight.py              # со всеми проверками, включая TG (если указан)
    python preflight.py --skip-tg    # пропустить TG-пинг
"""
import os
import sys
import argparse
from dotenv import load_dotenv

# загружаем .env заранее
load_dotenv()

# локальные модули бота
import config
from utils import tg_send, log
import bybit_api as bb

OK = "✅"
WARN = "⚠️"
ERR = "❌"

def check_api():
    print("\n[1/6] Проверка API-доступа к Bybit…")
    ok = True
    # баланс
    try:
        bal = bb.get_balance()
        print(f"   {OK} Баланс (equity): {bal:.2f} USDT")
    except Exception as e:
        print(f"   {ERR} Баланс: {e}")
        ok = False
    # тикеры
    try:
        t = bb.get_tickers_linear()
        n = len((t.get("result", {}) or {}).get("list", []) or [])
        print(f"   {OK} Тикеры linear: получено {n}")
    except Exception as e:
        print(f"   {ERR} Тикеры: {e}")
        ok = False
    return ok

def infer_position_mode():
    print("\n[2/6] Попытка определить режим позиций (One-Way/Hedge)…")
    try:
        data = bb.get_positions()
        lst = (data.get("result", {}) or {}).get("list", []) or []
        idx_vals = set()
        for p in lst:
            try:
                idx_vals.add(int(p.get("positionIdx", 0)))
            except Exception:
                pass
        mode = "unknown"
        if 1 in idx_vals or 2 in idx_vals:
            mode = "hedge"
        elif idx_vals == {0} or idx_vals == set() or 0 in idx_vals:
            mode = "one-way"  # по ответу API так бывает и без открытых позиций
        icon = OK if mode == "one-way" else (WARN if mode == "unknown" else WARN)
        print(f"   {icon} Режим позиций (по positionIdx): {mode} (idx={sorted(list(idx_vals)) or '∅'})")
        if mode != "one-way":
            print("     Рекомендация: в интерфейсе Bybit включить One-Way (не Hedge).")
        return True
    except Exception as e:
        print(f"   {WARN} Не удалось определить: {e}")
        print("     Проверь вручную в интерфейсе Bybit → Derivatives → Position mode: One-Way.")
        return False

def pick_pairs():
    print("\n[3/6] Подбор торговых пар и проверка фильтров…")
    try:
        base = getattr(config, "TOP_LIQUID_PAIRS", ["BTCUSDT","ETHUSDT","SOLUSDT"])
        want = int(getattr(config, "PAIRS_COUNT", 2))
        top = bb.fast_pick_top_pairs(count=want)
        if not top:
            print(f"   {WARN} fast_pick_top_pairs вернул пусто. Возьму первые {want} из конфигурации.")
            top = base[:want]
        print(f"   {OK} Будем работать с: {top}")
        # таблица фильтров
        for s in top:
            try:
                min_qty, step, min_notional = bb.get_min_order_filters(s)
                price = bb.get_current_price(s)
                print(f"      • {s:8s} | price≈{price:.6f} | min_qty={min_qty} | step={step} | min_notional={min_notional or '—'}")
            except Exception as e:
                print(f"      {WARN} {s}: {e}")
        return True
    except Exception as e:
        print(f"   {ERR} Подбор пар/фильтров: {e}")
        return False

def check_data_root():
    print("\n[4/6] Проверка DATA_ROOT…")
    root = (getattr(config, "DATA_ROOT", None) or os.getenv("DATA_ROOT","")).strip()
    if not root:
        print(f"   {WARN} DATA_ROOT не задан. Запись поминутных свечей отключена.")
        return True
    try:
        os.makedirs(root, exist_ok=True)
        test_path = os.path.join(root, "preflight_ok.txt")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok\n")
        print(f"   {OK} Папка доступна: {os.path.abspath(root)}")
        try:
            os.remove(test_path)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"   {ERR} Не удалось создать/писать в DATA_ROOT: {e}")
        return False

def check_telegram(run_ping=True):
    print("\n[5/6] Проверка Telegram…")
    if not run_ping:
        print("   (пропущено по флагу --skip-tg)")
        return True
    token = os.getenv("TELEGRAM_TOKEN","").strip()
    chat  = os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat:
        print(f"   {WARN} TELEGRAM_TOKEN/CHAT_ID не заданы — алерты не будут приходить.")
        return False
    try:
        tg_send("✅ [preflight] Тестовое сообщение: связь есть!")
        print(f"   {OK} Сообщение отправлено (проверь чат).")
        return True
    except Exception as e:
        print(f"   {ERR} Не удалось отправить в Telegram: {e}")
        return False

def print_risk_summary():
    print("\n[6/6] Контрольный обзор риска…")
    max_share = float(getattr(config, "MAX_BALANCE_SHARE", 0.02))
    lev = int(getattr(config, "DEFAULT_LEVERAGE", 2))
    safe = int(os.getenv("SAFE_MODE", str(getattr(config, "SAFE_MODE", 1))))
    print(f"   • SAFE_MODE={safe}  (1=теневой режим, ордеров нет)")
    print(f"   • DEFAULT_LEVERAGE={lev}x")
    print(f"   • MAX_BALANCE_SHARE={max_share*100:.1f}% на вход")
    print(f"   • MIN_ATR_PCT={float(getattr(config,'MIN_ATR_PCT',0.001))*100:.3f}% (фильтр волатильности)")
    print(f"   • ML threshold (fallback)={float(os.getenv('ML_THRESHOLD', getattr(config,'ML_THRESHOLD',0.58))):.3f}")
    print("   Рекомендация на первый час REAL: SAFE_MODE=1, затем переключить в 0.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tg", action="store_true", help="Пропустить Telegram-пинг")
    args = ap.parse_args()

    print("=== PRE-FLIGHT CHECK ===")
    ok_all = True
    ok_all &= check_api()
    infer_position_mode()  # не критично
    ok_all &= pick_pairs()
    ok_all &= check_data_root()
    ok_all &= check_telegram(run_ping=not args.skip_tg)
    print_risk_summary()

    print("\nИТОГО:", OK if ok_all else WARN, "(см. предупреждения выше)")
    if not ok_all:
        sys.exit(1)

if __name__ == "__main__":
    main()
