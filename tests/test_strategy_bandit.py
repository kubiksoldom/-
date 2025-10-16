import random

import numpy as np

from strategy import StrategyRouter, Candidate


def _make_router(tmp_path):
    router = StrategyRouter(store_path=str(tmp_path / "stats.json"))
    router.stats.clear()
    return router


def test_bandit_forced_exploration_prefers_cold_start(monkeypatch, tmp_path):
    router = _make_router(tmp_path)
    router.stats["__global__"] = {"plays": 100}

    cold = Candidate("buy", 1.0, 0.9, 1.1, "cold", 0.55, "cold_strat")
    hot = Candidate("buy", 1.0, 0.9, 1.1, "hot", 0.90, "hot_strat")

    key_cold = router._key("BTCUSDT", "1", "trend", cold.strategy)
    router.stats[key_cold] = {
        "alpha": 2.0,
        "beta": 3.0,
        "bandit_count": 1.0,
        "count": 1.0,
        "history": [],
        "ewma_wr": 0.5,
        "ewma_r": 0.0,
        "mean_r": 0.0,
        "m2": 1e-4,
    }

    key_hot = router._key("BTCUSDT", "1", "trend", hot.strategy)
    router.stats[key_hot] = {
        "alpha": 25.0,
        "beta": 5.0,
        "bandit_count": 80.0,
        "count": 80.0,
        "history": [0.2] * 10,
        "ewma_wr": 0.65,
        "ewma_r": 0.02,
        "mean_r": 0.015,
        "m2": 0.01,
    }

    monkeypatch.setattr(random, "random", lambda: 0.5)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(np.random, "beta", lambda a, b: a / (a + b))

    chosen = router._pick_by_bandit("BTCUSDT", "1", "trend", [cold, hot])

    assert chosen is cold


def test_bandit_prefers_high_score_when_warm(monkeypatch, tmp_path):
    router = _make_router(tmp_path)
    router.stats["__global__"] = {"plays": 500}

    underperformer = Candidate("buy", 1.0, 0.9, 1.1, "weak", 0.30, "weak_strat")
    performer = Candidate("buy", 1.0, 0.9, 1.1, "strong", 0.85, "strong_strat")

    key_weak = router._key("BTCUSDT", "1", "trend", underperformer.strategy)
    router.stats[key_weak] = {
        "alpha": 5.0,
        "beta": 15.0,
        "bandit_count": 60.0,
        "count": 60.0,
        "history": [0.0] * 10,
        "ewma_wr": 0.45,
        "ewma_r": -0.01,
        "mean_r": -0.005,
        "m2": 0.02,
    }

    key_strong = router._key("BTCUSDT", "1", "trend", performer.strategy)
    router.stats[key_strong] = {
        "alpha": 30.0,
        "beta": 5.0,
        "bandit_count": 80.0,
        "count": 80.0,
        "history": [0.1] * 10,
        "ewma_wr": 0.70,
        "ewma_r": 0.03,
        "mean_r": 0.02,
        "m2": 0.015,
    }

    monkeypatch.setattr(random, "random", lambda: 0.99)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(np.random, "beta", lambda a, b: a / (a + b))

    chosen = router._pick_by_bandit("BTCUSDT", "1", "trend", [underperformer, performer])

    assert chosen is performer
