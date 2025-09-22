from bybit_api import client, close_position_by_market
from utils import log, tg_send
import time

def get_position_size(symbol):
    """Возвращает текущий размер позиции по символу (0 если закрыта)."""
    try:
        data = client.get_positions(category="linear", settleCoin="USDT")
        if data["retCode"] == 0:
            for pos in data["result"]["list"]:
                if pos["symbol"] == symbol:
                    return float(pos["size"])
        return 0.0
    except Exception as e:
        log(f"[❌] Ошибка get_position_size: {e}")
        return 0.0

def force_close_all_positions():
    """
    Принудительно закрывает все открытые позиции по всем linear USDT контрактам, пока size не станет 0.
    """
    try:
        data = client.get_positions(category="linear", settleCoin="USDT")
        if data["retCode"] == 0:
            count = 0
            for pos in data["result"]["list"]:
                symbol = pos["symbol"]
                size = float(pos["size"])
                if size == 0:
                    continue
                log(f"[FORCE CLOSE] {symbol}: Начинаю поэтапное закрытие по рынку (size={size})!")
                attempt = 1
                max_attempts = 5
                while attempt <= max_attempts:
                    close_position_by_market(symbol)
                    time.sleep(2)
                    size_after = get_position_size(symbol)
                    if size_after == 0:
                        log(f"[✅] {symbol}: Позиция полностью закрыта!")
                        tg_send(f"[FORCE CLOSE] {symbol}: Позиция полностью закрыта!")
                        break
                    else:
                        log(f"[❗] {symbol}: Осталось size={size_after} (попытка {attempt}/{max_attempts})")
                        attempt += 1
                if size_after != 0:
                    log(f"[ALERT] {symbol}: Не удалось полностью закрыть позицию! Осталось size={size_after}")
                    tg_send(f"[ALERT] {symbol}: Не удалось полностью закрыть позицию! Осталось size={size_after}")
                count += 1
            if count == 0:
                log("[INFO] Нет открытых позиций для закрытия!")
            else:
                log("[🏁] Все возможные позиции обработаны! UPL должен быть равен нулю (если нигде не осталось size>0).")
        else:
            log(f"[❌] Ошибка ответа API: {data}")
    except Exception as e:
        log(f"[❌] Ошибка force close: {e}")

if __name__ == "__main__":
    log("🚨 Ручной сброс рисков: начинаю поэтапно закрывать ВСЕ позиции!")
    force_close_all_positions()
    log("✅ Ручной сброс рисков завершён.")
