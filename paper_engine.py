# -*- coding: utf-8 -*-
"""
paper_engine.py — бумажный брокер с тем же интерфейсом, что bybit_api.py.

Идея:
- Рыночные данные и биржевые фильтры берём из реального bybit_api (read-only).
- Сделки, баланс и позиции моделируем локально (комиссии, проскальзывание).
- Если real.get_current_price(...) вернёт 0.0, подстраховываемся снапшотом.
- Фолбек LINEAR→SPOT для снапшотов/свечей реализован внутри bybit_api.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, List

import config
from utils import log, write_cycle_log, adjust_qty
import bybit_api as real  # рынок/фильтры/снапшоты из реала


@dataclass
class Position:
    symbol: str
    side: str            # "Buy" / "Sell"
    qty: float
    entry_price: float
    opened_at: float
    max_upnl: float = 0.0


class PaperBroker:
    def __init__(self):
        # Инициализация виртуального баланса
        sync = int(getattr(config, "PAPER_SYNC_BALANCE", 1)) == 1
        if sync:
            try:
                real_bal = float(real.get_balance())
                mult = float(getattr(config, "PAPER_BALANCE_MULT", 1.0))
                self.equity = real_bal * max(0.0, mult)
                log(f"[PAPER] sync balance: real {real_bal:.2f} → paper {self.equity:.2f}")
            except Exception as e:
                self.equity = float(getattr(config, "VIRTUAL_START_BALANCE", 100.0))
                log(f"[PAPER] real balance unavailable → using fixed {self.equity:.2f}. reason: {e}")
        else:
            self.equity = float(getattr(config, "VIRTUAL_START_BALANCE", 100.0))
            log(f"[PAPER] using fixed start balance: {self.equity:.2f}")

        self._positions: Dict[str, Position] = {}
        self._leverage: Dict[str, int] = defaultdict(lambda: int(getattr(config, "DEFAULT_LEVERAGE", 10)))

    # ========= Счёт / позиции =========
    def get_balance(self) -> float:
        return float(self.equity)

    def _safe_now_price(self, symbol: str) -> float:
        """Цена с подстраховкой: сначала current_price, если 0 — берём из снапшота last_price."""
        p = float(real.get_current_price(symbol) or 0.0)
        if p > 0:
            return p
        snap = real.get_ticker_snapshot(symbol) or {}
        lp = float(snap.get("last_price", 0.0) or 0.0)
        return lp

    def get_positions(self, symbol: Optional[str] = None) -> Dict:
        """
        Возвращаем структуру максимально похожую на bybit v5:
        {"result":{"list":[{"symbol":...,"side":...,"size":"...","avgPrice":"...","unrealisedPnl":"..."}]}}
        """
        lst: List[Dict] = []
        now_price_cache: Dict[str, float] = {}
        syms = [symbol] if symbol else list(self._positions.keys())

        for s in syms:
            pos = self._positions.get(s)
            if not pos:
                continue
            if s not in now_price_cache:
                now_price_cache[s] = self._safe_now_price(s)
            mark = float(now_price_cache[s] or 0.0)

            upl = (mark - pos.entry_price) * pos.qty if pos.side == "Buy" else (pos.entry_price - mark) * pos.qty
            pos.max_upnl = max(pos.max_upnl, upl)

            lst.append({
                "symbol": s,
                "side": pos.side,
                "size": str(pos.qty),
                "avgPrice": str(pos.entry_price),
                "unrealisedPnl": str(upl),
            })
        return {"result": {"list": lst}}

    def has_open_position(self, symbol: str) -> bool:
        p = self._positions.get(symbol)
        return bool(p and p.qty > 0.0)

    # ========= Торговля =========
    def _fill_price(self, symbol: str, side: str) -> float:
        """
        Простая модель исполнения: last ± slippage_bps.
        """
        last = float(self._safe_now_price(symbol) or 0.0)
        bps = float(getattr(config, "SLIPPAGE_BPS", 2.0)) / 10_000.0  # 2 bps = 0.02%
        if side == "Buy":
            return last * (1 + bps)
        else:
            return last * (1 - bps)

    def place_market_order(self, symbol: str, side: str, qty: float):
        """
        Открываем позицию, если по символу ещё нет (не допускаем flip).
        """
        if self.has_open_position(symbol):
            log(f"[PAPER] position already open on {symbol}, skip new entry — order not sent")
            return False

        qty = float(qty)
        if qty <= 0:
            log(f"[PAPER] qty<=0, skip")
            return False

        # Биржевые фильтры (минималки/шаг) — как в реале
        min_qty, step, min_notional = real.get_min_order_filters(symbol)
        price_now = float(self._safe_now_price(symbol) or 0.0)
        if price_now <= 0:
            log(f"[PAPER] no price for {symbol}")
            return False

        # Аккуратное приведение количества под шаг/минималки
        adj_qty = adjust_qty(price_now, qty, min_qty=min_qty, qty_step=step, min_notional=(min_notional or 0.0))
        if adj_qty <= 0:
            log(f"[PAPER] qty {qty} adjusted→{adj_qty} not tradable for {symbol}")
            return False
        qty = adj_qty

        # Комиссия на вход
        fill = self._fill_price(symbol, side)
        fee_rate = float(getattr(config, "TAKER_FEE", 0.0006))
        fee = fill * qty * fee_rate
        self.equity -= fee

        self._positions[symbol] = Position(
            symbol=symbol, side=side, qty=float(qty),
            entry_price=float(fill), opened_at=time.time(), max_upnl=0.0
        )
        log(f"[PAPER-OPEN] {side} {qty} {symbol} @ {fill:.6f} (fee {fee:.6f})")
        write_cycle_log({
            "symbol": symbol,
            "direction": "long" if side == "Buy" else "short",
            "buy_price": fill if side == "Buy" else None,
            "sell_price": fill if side == "Sell" else None,
            "qty": qty,
            "event": "open",
            "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "paper": True
        })
        return {"result": {"orderId": f"paper-{int(time.time()*1000)}"}}

    def close_position_by_market(self, symbol: str, qty: Optional[float] = None):
        pos = self._positions.get(symbol)
        if not pos or pos.qty <= 0:
            log(f"[PAPER] no position on {symbol} to close")
            return

        close_qty = pos.qty if qty is None else min(float(qty), pos.qty)
        exit_side = "Sell" if pos.side == "Buy" else "Buy"
        fill = self._fill_price(symbol, exit_side)
        fee_rate = float(getattr(config, "TAKER_FEE", 0.0006))
        fee = fill * close_qty * fee_rate

        # PnL по закрываемой части
        pnl = (fill - pos.entry_price) * close_qty if pos.side == "Buy" else (pos.entry_price - fill) * close_qty

        self.equity += pnl
        self.equity -= fee

        pos.qty -= close_qty
        if pos.qty <= 0:
            log_payload = {
                "symbol": symbol,
                "direction": "long" if pos.side == "Buy" else "short",
                "qty": close_qty,
                "pnl": pnl - fee,  # net
                "event": "paper_close",
                "closed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reason": "manual/paper",
                "paper": True,
            }
            if pos.side == "Buy":
                log_payload["buy_price"] = pos.entry_price
                log_payload["sell_price"] = fill
            else:
                log_payload["sell_price"] = pos.entry_price
                log_payload["buy_price"] = fill

            write_cycle_log(log_payload)
            log(f"[PAPER-CLOSE] {symbol} pnl={pnl - fee:.6f} equity={self.equity:.2f}")
            self._positions.pop(symbol, None)
        else:
            log(f"[PAPER-PARTIAL CLOSE] {symbol} closed {close_qty}, remain {pos.qty}")

    def force_close_all_positions_absolute(self):
        for s in list(self._positions.keys()):
            self.close_position_by_market(s)

    def set_leverage(self, symbol: str, leverage: int = 10):
        try:
            lev_to_set = int(float(leverage or 0))
        except Exception as e:
            log(f"[PAPER-LEV] {symbol}: invalid leverage {leverage!r} ({e})")
            return False

        sym = str(symbol or "").strip().upper()
        if not sym:
            log("[PAPER-LEV] пустой символ при установке плеча")
            return False

        if lev_to_set <= 0:
            log(f"[PAPER-LEV] {sym}: leverage must be >0, got {lev_to_set}")
            return False

        self._leverage[sym] = lev_to_set
        log(f"[PAPER-LEV] {sym}: {lev_to_set}x (virtual)")
        return True

    # ========= Прокси к рыночным данным/утилитам =========
    def get_min_order_filters(self, symbol: str) -> Tuple[float, float, Optional[float]]:
        return real.get_min_order_filters(symbol)

    def get_current_price(self, symbol: str) -> float:
        # используем ту же подстраховку, что и выше
        return float(self._safe_now_price(symbol) or 0.0)

    def get_symbol_leverage_limits(self, symbol: str) -> Tuple[int, int]:
        return real.get_symbol_leverage_limits(symbol)

    def get_max_leverage(self, symbol: str) -> int:
        return real.get_max_leverage(symbol)

    def get_kline_any(self, symbol: str, interval: str = "1", limit: int = 60,
                      end_ms: Optional[int] = None) -> Tuple[List[list], str]:
        return real.get_kline_any(symbol, interval=interval, limit=limit, end_ms=end_ms)

    def get_ticker_snapshot(self, symbol: str) -> Dict[str, float]:
        return real.get_ticker_snapshot(symbol)

    def get_atr(self, symbol: str, interval: str = "15", period: int = 14) -> float:
        return real.get_atr(symbol, interval=interval, period=period)

    def get_tickers_linear(self):
        return real.get_tickers_linear()

    def get_orderbook_spread(self, symbol: str, depth: int = 1) -> float:
        try:
            return float(real.get_orderbook_spread(symbol, depth=depth) or 0.0)
        except Exception:
            return 0.0

    # ——— Сервисно: ручной ресинк equity c реала ———
    def resync_from_real(self, mult: Optional[float] = None):
        try:
            real_bal = float(real.get_balance())
            if mult is None:
                mult = float(getattr(config, "PAPER_BALANCE_MULT", 1.0))
            self.equity = real_bal * max(0.0, float(mult))
            log(f"[PAPER] re-sync balance: real {real_bal:.2f} × {mult} → paper {self.equity:.2f}")
            return self.equity
        except Exception as e:
            log(f"[PAPER] resync failed: {e}")
            return self.equity


# ===== Экспорт совместимого интерфейса на уровне модуля (как в bybit_api) =====
_broker = PaperBroker()

def get_balance() -> float: return _broker.get_balance()
def get_positions(symbol: Optional[str] = None): return _broker.get_positions(symbol)
def has_open_position(symbol: str) -> bool: return _broker.has_open_position(symbol)
def place_market_order(symbol: str, side: str, qty: float): return _broker.place_market_order(symbol, side, qty)
def close_position_by_market(symbol: str, qty: Optional[float] = None): return _broker.close_position_by_market(symbol, qty)
def force_close_all_positions_absolute(): return _broker.force_close_all_positions_absolute()
def set_leverage(symbol: str, leverage: int = 10): return _broker.set_leverage(symbol, leverage)

# Прокси-рыночные
def get_min_order_filters(symbol: str): return _broker.get_min_order_filters(symbol)
def get_current_price(symbol: str) -> float: return _broker.get_current_price(symbol)
def get_kline_any(symbol: str, interval: str = "1", limit: int = 60, end_ms: Optional[int] = None): return _broker.get_kline_any(symbol, interval, limit, end_ms)
def get_ticker_snapshot(symbol: str): return _broker.get_ticker_snapshot(symbol)
def get_atr(symbol: str, interval: str = "15", period: int = 14) -> float: return _broker.get_atr(symbol, interval, period)
def get_tickers_linear(): return _broker.get_tickers_linear()
def get_orderbook_spread(symbol: str, depth: int = 1) -> float: return _broker.get_orderbook_spread(symbol, depth)

# Опционально: ручной ресинк с реала
def resync_from_real(mult: Optional[float] = None):
    return _broker.resync_from_real(mult)
