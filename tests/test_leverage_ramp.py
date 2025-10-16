from utils import apply_leverage_ramp, fallback_leverage


def test_apply_leverage_ramp_caps_growth():
    new_lev, reason = apply_leverage_ramp(10, 50, 2.0)
    assert new_lev == 20
    assert "ramp_limit_up" in reason


def test_apply_leverage_ramp_caps_drop():
    new_lev, reason = apply_leverage_ramp(20, 2, 2.0)
    assert new_lev == 10
    assert "ramp_limit_down" in reason


def test_fallback_leverage_prefers_previous():
    assert fallback_leverage(12, 8) == 8
    assert fallback_leverage(5, None) == 5
