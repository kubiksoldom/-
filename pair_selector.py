# -*- coding: utf-8 -*-
"""Shared helpers for choosing trading pairs based on config/env."""
from typing import Iterable, List, Sequence, Set

import bybit_api as bb
import config
from utils import log


def _normalize_base_pairs(raw_pairs: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in raw_pairs:
        sym = str(raw).strip().upper()
        if sym:
            normalized.append(sym)
    return normalized


def _exchange_universe() -> Set[str]:
    try:
        symbols = bb.get_symbols()
        return {str(s.get("symbol") or "").upper() for s in symbols or [] if s.get("symbol")}
    except Exception as exc:
        log(f"[PAIR SELECT] Не удалось получить список символов: {exc}")
        return set()


def _rank_by_liquidity(candidates: Iterable[str]) -> List[str]:
    try:
        tickers = ((bb.get_tickers_linear() or {}).get("result", {}) or {}).get("list", []) or []
        volumes = {}
        for t in tickers:
            sym = str(t.get("symbol") or "").upper()
            if sym not in candidates:
                continue
            try:
                volumes[sym] = float(t.get("turnover24h", 0) or 0.0)
            except Exception:
                volumes[sym] = 0.0
        if volumes:
            return sorted(candidates, key=lambda s: volumes.get(s, 0.0), reverse=True)
    except Exception as exc:
        log(f"[PAIR SELECT] Рейтинг ликвидности недоступен: {exc}")
    return list(candidates)


def _fill_to_cap(selected: List[str], ordered_pool: List[str], available: Set[str], cap: int) -> List[str]:
    filtered = [s for s in selected if s in available]
    for sym in ordered_pool:
        if len(filtered) >= cap:
            break
        if sym in filtered:
            continue
        if available and sym in available:
            filtered.append(sym)
    return filtered


def select_pairs_from_config() -> List[str]:
    base_pairs = _normalize_base_pairs(getattr(config, "TOP_LIQUID_PAIRS", []))
    if not base_pairs:
        base_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

    cap = int(getattr(config, "PAIRS_COUNT", len(base_pairs)) or len(base_pairs))
    cap = max(1, cap)
    auto_select = bool(int(getattr(config, "AUTO_SELECT_PAIRS", 1)))

    available_on_exchange = _exchange_universe()
    candidates = [p for p in base_pairs if not available_on_exchange or p in available_on_exchange]
    if not candidates:
        candidates = list(base_pairs)

    ordered = _rank_by_liquidity(candidates) if auto_select else list(candidates)
    selected = ordered[:cap]

    if available_on_exchange:
        selected = _fill_to_cap(selected, ordered or candidates, available_on_exchange, cap)
        max_possible = len([s for s in ordered if s in available_on_exchange])
        if len(selected) < cap and max_possible >= cap:
            selected = selected + [s for s in ordered if s in available_on_exchange and s not in selected][: cap - len(selected)]
    else:
        filler = [p for p in candidates if p not in selected]
        selected = (selected + filler)[:cap]

    selected = selected[: min(cap, len(selected))] if selected else []
    log(f"Работаем с: {selected} (cap={cap})")
    return selected
