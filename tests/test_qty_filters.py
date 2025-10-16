import pytest

from utils import adjust_qty


def test_adjust_qty_respects_step():
    assert adjust_qty(10.0, 1.234, qty_step=0.01) == pytest.approx(1.23)


def test_adjust_qty_fails_below_min_qty():
    assert adjust_qty(100.0, 0.0005, min_qty=0.001, qty_step=0.0001) == 0.0


def test_adjust_qty_enforces_min_notional():
    ok_qty = adjust_qty(10.0, 0.4, min_notional=3.0)
    assert ok_qty == pytest.approx(0.4)
    assert adjust_qty(10.0, 0.1, min_notional=5.0) == 0.0
