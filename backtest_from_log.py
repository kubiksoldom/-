import json
import pandas as pd

LOG_FILE = "bot_cycle_log.jsonl"

CLOSE_EVENTS = {"take_profit", "stop_loss", "manual_close", "dynamic_tp_exit", "no_profit_exit", "paper_close"}

def analyze_trades(log_file=LOG_FILE):
    deals = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("event") in CLOSE_EVENTS:
                    deals.append(obj)
            except:
                continue
    df = pd.DataFrame(deals)
    if df.empty:
        print("Нет сделок для анализа.")
        return
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    df['timestamp'] = pd.to_datetime(df.get('closed_at') or df.get('timestamp'))
    df = df.sort_values('timestamp')
    df['equity'] = df['pnl'].cumsum()
    total_profit = df['pnl'].sum()
    winrate = (df['pnl'] > 0).mean() * 100
    avg_pnl = df['pnl'].mean()
    print(f"📊 Сделок: {len(df)}")
    print(f"💰 Общий PnL: {total_profit:.2f} USDT")
    print(f"🏆 Winrate: {winrate:.2f}%")
    print(f"📈 Средний PnL: {avg_pnl:.2f}")
    print("✅ Equity по времени (последние 10):")
    print(df[['timestamp', 'pnl', 'equity']].tail(10))

if __name__ == "__main__":
    analyze_trades()
