# gen_synthetic_fills.py
# Usage:
#   python gen_synthetic_fills.py --out fills_all.csv --n 20000 --make_model_friendly
# or append to existing:
#   python gen_synthetic_fills.py --out fills_all.csv --n 20000 --append

import argparse, csv, os, random, time
from datetime import datetime, timedelta
import math

DEFAULT_COLS = [
    "ts",              # unix seconds
    "ts_ms",           # unix ms
    "symbol",
    "side",            # buy/sell
    "price",
    "qty",
    "notional_usd",
    "leverage",
    "order_id",
    "exec_id",
    "exec_type",       # Trade / Fill / Cancel
    "fill_type",       # market/limit
    "is_taker",        # 0/1
    "fee",
    "fee_asset",
    "fee_usd",
    "liquidation",     # 0/1
    "reason",          # tp/sl/market/close
    "tp_hit",          # 0/1 whether tp was hit for this trade lifecycle
    "sl_hit",          # 0/1
    "pnl_usd",         # realized pnl on this fill (can be cumulative)
    "src"              # source (bybit/binance/synthetic)
]

SYMBOL_BASES = [
    ("BTCUSDT", 45000.0, 200.0),
    ("ETHUSDT", 3200.0, 20.0),
    ("SOLUSDT", 120.0, 3.0),
    ("ADAUSDT", 0.35, 0.01),
    ("DOGEUSDT", 0.18, 0.01),
    ("DOTUSDT", 6.0, 0.2),
    ("LINKUSDT", 7.5, 0.3),
    ("OPUSDT", 2.8, 0.12),
    ("MATICUSDT", 0.85, 0.03),
    ("LTCUSDT", 85.0, 1.5),
]

def gen_one(ts_base, symbol, bias_win=0.5):
    base, vol = symbol[1], symbol[2]
    # small random walk for price
    delta = random.gauss(0, vol)
    price = max(0.0001, round(base + delta + random.uniform(-vol, vol), 8))
    side = random.choice(["buy", "sell"])
    # qty chosen so that notional is reasonable for symbol
    target_notional = random.uniform(5, 200) if base > 100 else random.uniform(5, 1000)
    qty = max(0.000001, round(target_notional / price, 6))
    notional = round(price * qty, 6)
    leverage = random.choice([1, 2, 5, 10, 20])
    order_id = f"ord_{int(ts_base)}_{random.randint(1000,999999)}"
    exec_id = f"ex_{random.randint(10**6,10**9-1)}"
    exec_type = "Trade"
    fill_type = random.choices(["market", "limit"], weights=[0.65, 0.35])[0]
    is_taker = 1 if fill_type=="market" else 0
    fee_pct = 0.0006 if is_taker else 0.0002
    fee = round(notional * fee_pct, 6)
    fee_asset = "USDT"
    fee_usd = fee
    liquid = 0
    # determine trade outcome (tp/sl/market)
    # bias_win increases probability that TP is hit rather than SL
    win = random.random() < bias_win
    if win:
        reason = random.choices(["tp","market"], weights=[0.85,0.15])[0]
        tp_hit = 1
        sl_hit = 0
        pnl_usd = round(notional * (random.uniform(0.0008, 0.01)) * leverage, 6)  # positive
    else:
        reason = "sl"
        tp_hit = 0
        sl_hit = 1
        pnl_usd = round(-abs(notional * (random.uniform(0.0008, 0.01)) * leverage), 6)
    row = {
        "ts": int(ts_base),
        "ts_ms": int(ts_base*1000),
        "symbol": symbol[0],
        "side": side,
        "price": price,
        "qty": qty,
        "notional_usd": round(notional,6),
        "leverage": leverage,
        "order_id": order_id,
        "exec_id": exec_id,
        "exec_type": exec_type,
        "fill_type": fill_type,
        "is_taker": is_taker,
        "fee": fee,
        "fee_asset": fee_asset,
        "fee_usd": fee_usd,
        "liquidation": liquid,
        "reason": reason,
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "pnl_usd": pnl_usd,
        "src": "synthetic"
    }
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fills_all.synthetic.csv", help="output file (will append if --append)")
    p.add_argument("--n", type=int, default=20000, help="number of records to generate")
    p.add_argument("--start", default=None, help="start timestamp (YYYY-mm-dd) or leave for now-ndays")
    p.add_argument("--days", type=int, default=90, help="spread records across last N days if start not provided")
    p.add_argument("--append", action="store_true")
    p.add_argument("--make_model_friendly", action="store_true", help="bias toward more winning trades (for testing only)")
    args = p.parse_args()

    out_exists = os.path.exists(args.out) and args.append
    mode = "a" if out_exists else "w"
    header = DEFAULT_COLS

    # time spread
    now = datetime.utcnow()
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_dt = now - timedelta(days=args.days)

    total = args.n
    bias = 0.65 if args.make_model_friendly else 0.48

    # if appending and file exists, try to detect header
    if out_exists:
        # we will append with the same header
        pass

    with open(args.out, mode, newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not out_exists:
            writer.writeheader()
        for i in range(total):
            # spread timestamps uniformly across window + small clustering
            frac = i / max(1, total-1)
            ts = start_dt + (now - start_dt) * frac
            # jitter seconds
            ts = ts + timedelta(seconds=random.randint(0, 3600))
            ts_base = ts.timestamp()
            symbol = random.choice(SYMBOL_BASES)
            row = gen_one(ts_base, symbol, bias_win=bias)
            writer.writerow({k: row.get(k, "") for k in header})
            if i % 2000 == 0 and i>0:
                print(f"[gen] {i} rows generated...")

    print(f"[done] wrote {total} rows to {args.out}")
    print("NOTE: this file is synthetic — mark it as such if used for training/evaluation.")

if __name__ == "__main__":
    main()
