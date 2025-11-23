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
from typing import Dict, Optional, List, Tuple

import config
from utils import log, adjust_qty, append_trade_event
import bybit_api as real  # рынок/фильтры/снапшоты из реала


@dataclass
class Position:
    symbol: str
    side: str            # "Buy" / "Sell"
    qty: float
    entry_price: float
    max_upnl: float = 0.0
    position_id: str = ""
    client_id: str = ""
    source: str = "MANUAL"
    realized_pnl: float = 0.0


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
        self._last_entry_ts: Dict[str, float] = {}

    # ========= Счёт / позиции =========
    def get_balance(self) -> float:
        return float(self.equity)

    def get_equity(self) -> float:
        return float(self.equity)

    def get_margin_info(self) -> Dict[str, float]:
        equity = max(float(self.equity), 0.0)
        im_used = 0.0
        for sym, pos in list(self._positions.items()):
            try:
                price_now = float(self._safe_now_price(sym) or 0.0)
            except Exception:
                price_now = 0.0
            if price_now <= 0:
                continue
            lev = max(1, int(self._leverage.get(sym, int(getattr(config, "DEFAULT_LEVERAGE", 10)))))
            notional = price_now * float(pos.qty)
            im_used += notional / max(lev, 1)
        im_pct = (im_used / equity * 100.0) if equity > 0 else 0.0
        mm_pct = im_pct * 0.5
        return {"IM": round(im_pct, 2), "MM": round(mm_pct, 2), "equity": round(equity, 2)}

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

    def place_market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False):
        """
        Открываем позицию, если по символу ещё нет (не допускаем flip).
        """
        qty = float(qty)
        if qty <= 0:
            log(f"[PAPER] qty<=0, skip")
            return False

        existing = self._positions.get(symbol)

        cooldown = max(0.0, float(getattr(config, "STRATEGY_COOLDOWN", 0)))
        now_ts = time.time()
        if cooldown > 0 and (now_ts - float(self._last_entry_ts.get(symbol, 0.0))) < cooldown:
            log(f"[PAPER] cooldown {symbol}: {now_ts - float(self._last_entry_ts.get(symbol, 0.0)):.1f}/{cooldown}s")
            return False

        if reduce_only:
            if not existing or existing.qty <= 0:
                log(f"[PAPER] reduce-only order for {symbol} ignored — no open position")
                return False
            expected_side = "Sell" if existing.side == "Buy" else "Buy"
            if side and side != expected_side:
                log(
                    f"[PAPER] reduce-only {side} does not match open position side {existing.side} on {symbol}")
                return False
            if not self.close_position_by_market(symbol, qty):
                return False
            return {"result": {"orderId": f"paper-reduce-{int(time.time()*1000)}"}}

        if existing and existing.qty > 0:
            log(f"[PAPER] position already open on {symbol}, skip new entry — order not sent")
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

        hard_cap_abs = float(getattr(config, "HARD_NOTIONAL_CAP", 0.0))
        notional = price_now * qty
        if hard_cap_abs > 0 and notional > hard_cap_abs:
            capped_qty = adjust_qty(price_now, hard_cap_abs / max(price_now, 1e-9), min_qty=min_qty, qty_step=step, min_notional=(min_notional or 0.0))
            if capped_qty <= 0:
                log(f"[PAPER] notional cap blocks trade for {symbol}: cap={hard_cap_abs:.4f}")
                return False
            qty = capped_qty

        # Комиссия на вход
        fill = self._fill_price(symbol, side)
        fee_rate = float(getattr(config, "TAKER_FEE", 0.0006))
        fee = fill * qty * fee_rate
        self.equity -= fee

        millis = int(time.time() * 1000)
        order_id = f"paper-{millis}"
        client_id = f"paper-{symbol}-{millis}"
        position_id = f"paper-{symbol}-{int(time.time())}"

        self._positions[symbol] = Position(
            symbol=symbol,
            side=side,
            qty=float(qty),
            entry_price=float(fill),
            max_upnl=0.0,
            position_id=position_id,
            client_id=client_id,
            source="MANUAL",
        )
        direction = "LONG" if str(side).upper() in ("BUY", "LONG") or float(qty) > 0 else "SHORT"
        log(f"[PAPER-OPEN] {side} {qty} {symbol} @ {fill:.6f} (fee {fee:.6f}) [{direction}]")
        self._last_entry_ts[symbol] = now_ts
        append_trade_event({
            "ts": None,
            "symbol": symbol,
            "side": "long" if side == "Buy" else "short",
            "status": "open",
            "qty": qty,
            "price": fill,
            "fee": fee,
            "realized_pnl": 0.0,
            "order_id": order_id,
            "client_id": client_id,
            "source": "MANUAL",
            "note": "PAPER_OPEN",
            "position_id": position_id,
        })
        return {"result": {"orderId": order_id}}

    def close_position_by_market(self, symbol: str, qty: Optional[float] = None, max_attempts: int = 1):
        pos = self._positions.get(symbol)
        if not pos or pos.qty <= 0:
            log(f"[PAPER] no position on {symbol} to close")
            return False

        close_qty = pos.qty if qty is None else min(float(qty), pos.qty)
        exit_side = "Sell" if pos.side == "Buy" else "Buy"
        fill = self._fill_price(symbol, exit_side)
        fee_rate = float(getattr(config, "TAKER_FEE", 0.0006))
        fee = fill * close_qty * fee_rate

        # PnL по закрываемой части
        pnl = (fill - pos.entry_price) * close_qty if pos.side == "Buy" else (pos.entry_price - fill) * close_qty

        self.equity += pnl
        self.equity -= fee

        remaining_qty = max(0.0, pos.qty - close_qty)
        direction = "LONG" if pos.side == "Buy" else "SHORT"
        net_pnl = pnl - fee
        pos.realized_pnl += net_pnl

        order_id = f"paper-close-{int(time.time()*1000)}"
        status = "closed" if remaining_qty <= 1e-12 else "partial"

        append_trade_event({
            "symbol": symbol,
            "side": "long" if pos.side == "Buy" else "short",
            "status": status,
            "qty": close_qty,
            "price": fill,
            "fee": fee,
            "realized_pnl": net_pnl,
            "order_id": order_id,
            "client_id": pos.client_id,
            "source": pos.source,
            "note": "PAPER_CLOSE" if status == "closed" else "PAPER_PARTIAL",
            "position_id": pos.position_id,
        })

        if remaining_qty <= 1e-12:
            log(f"[PAPER-CLOSE] {symbol} pnl={net_pnl:.6f} equity={self.equity:.2f} [{direction}]")
            self._positions.pop(symbol, None)
        else:
            pos.qty = remaining_qty
            log(f"[PAPER-PARTIAL CLOSE] {symbol} closed {close_qty}, remain {pos.qty} [{direction}]")
        return True

    def force_close_all_positions_absolute(self):
        for s in list(self._positions.keys()):
            self.close_position_by_market(s)

    def close_all_positions(self):
        self.force_close_all_positions_absolute()

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
def get_equity() -> float: return _broker.get_equity()
def get_margin_info() -> Dict[str, float]: return _broker.get_margin_info()
def get_positions(symbol: Optional[str] = None): return _broker.get_positions(symbol)
def has_open_position(symbol: str) -> bool: return _broker.has_open_position(symbol)
def place_market_order(symbol: str, side: str, qty: float, reduce_only: bool = False):
    return _broker.place_market_order(symbol, side, qty, reduce_only)


def close_position_by_market(symbol: str, qty: Optional[float] = None, max_attempts: int = 5):
    return _broker.close_position_by_market(symbol, qty, max_attempts)
def force_close_all_positions_absolute(): return _broker.force_close_all_positions_absolute()
def close_all_positions(): return _broker.close_all_positions()
def set_leverage(symbol: str, leverage: int = 10): return _broker.set_leverage(symbol, leverage)

# Прокси-рыночные
def get_min_order_filters(symbol: str): return _broker.get_min_order_filters(symbol)
def filters_reliable(symbol: str) -> bool: return real.filters_reliable(symbol)
def get_current_price(symbol: str) -> float: return _broker.get_current_price(symbol)
def get_kline_any(symbol: str, interval: str = "1", limit: int = 60, end_ms: Optional[int] = None): return _broker.get_kline_any(symbol, interval, limit, end_ms)
def get_ticker_snapshot(symbol: str): return _broker.get_ticker_snapshot(symbol)
def get_atr(symbol: str, interval: str = "15", period: int = 14) -> float: return _broker.get_atr(symbol, interval, period)
def get_tickers_linear(): return _broker.get_tickers_linear()
def get_orderbook_spread(symbol: str, depth: int = 1) -> float: return _broker.get_orderbook_spread(symbol, depth)

# Опционально: ручной ресинк с реала
def resync_from_real(mult: Optional[float] = None):
    return _broker.resync_from_real(mult)
