import time, json
from bybit_api import fetch_executions_range, normalize_fill

# последние 14 дней по деривативам (linear). Поставь symbol="BTCUSDT" чтобы сузить.
end_ms = int(time.time() * 1000)
start_ms = end_ms - 14 * 24 * 60 * 60 * 1000

fills = fetch_executions_range(category="linear", start_ms=start_ms, end_ms=end_ms, symbol=None)
print("rows:", len(fills))

# нормализуем и сохраняем в JSONL
with open("bybit_import_fills.jsonl", "w", encoding="utf-8") as f:
    for r in fills:
        f.write(json.dumps(normalize_fill(r), ensure_ascii=False) + "\n")
print("saved: bybit_import_fills.jsonl")
