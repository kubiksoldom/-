import json
import os
from datetime import datetime
from bybit_api import client, log

BOT_LOG = "bot_cycle_log.jsonl"
MERGED_LOG = "combined_log.jsonl"

def fetch_closed_trades_from_bybit(limit=200):
    try:
        data = client.get_closed_pnl(category="linear", limit=limit)
        deals = []
        for d in data.get("result", {}).get("list", []):
            try:
                pnl = float(d.get("closedPnl", 0))
                side = d.get("side", "Buy")
                qty = float(d.get("qty", 0))
                symbol = d.get("symbol", "")
                entry_price = float(d.get("entryPrice", 0))
                exit_price = float(d.get("exitPrice", 0))
                ts = int(d.get("updatedTime", 0)) // 1000
                direction = "long" if side == "Buy" else "short"

                deal_obj = {
                    "symbol": symbol,
                    "direction": direction,
                    "buy_price": entry_price if direction == "long" else None,
                    "sell_price": exit_price if direction == "short" else None,
                    "qty": qty,
                    "pnl": pnl,
                    "event": "bybit_history",
                    "ts": datetime.utcfromtimestamp(ts).isoformat()
                }
                deals.append(deal_obj)
            except Exception as e:
                log(f"[!] Ошибка разбора сделки из истории: {e}")
        return deals
    except Exception as e:
        log(f"[❌] Ошибка получения истории сделок Bybit: {e}")
        return []

def merge_logs():
    existing = []
    if os.path.exists(BOT_LOG):
        with open(BOT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.append(json.loads(line))
                except:
                    continue

    history = fetch_closed_trades_from_bybit()
    combined = existing + history

    with open(MERGED_LOG, "w", encoding="utf-8") as f:
        for d in combined:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[✔] Объединено {len(existing)} логов и {len(history)} историй → {MERGED_LOG}")

if __name__ == "__main__":
    merge_logs()