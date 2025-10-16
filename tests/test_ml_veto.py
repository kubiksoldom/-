import math

import config
from ml_veto import _confidence_to_factor


def test_confidence_to_factor_full_size(monkeypatch):
    monkeypatch.setattr(config, "ML_CONF_HIGH", 0.80, raising=False)
    monkeypatch.setattr(config, "ML_CONF_MID", 0.65, raising=False)

    factor, band = _confidence_to_factor(0.92)

    assert math.isclose(factor, 1.0)
    assert band == "full"


def test_confidence_to_factor_reduced_and_block(monkeypatch):
    monkeypatch.setattr(config, "ML_CONF_HIGH", 0.85, raising=False)
    monkeypatch.setattr(config, "ML_CONF_MID", 0.70, raising=False)

    reduced_factor, reduced_band = _confidence_to_factor(0.75)
    blocked_factor, blocked_band = _confidence_to_factor(0.45)

    assert math.isclose(reduced_factor, 0.5)
    assert reduced_band == "reduced"
    assert math.isclose(blocked_factor, 0.0)
    assert blocked_band == "blocked"


def test_confidence_to_factor_invalid_value():
    factor, band = _confidence_to_factor(float("nan"))

    assert math.isclose(factor, 0.0)
    assert band == "invalid"
