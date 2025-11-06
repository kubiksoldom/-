import hashlib
import math
import socket
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (
    adjust_qty,
    affordable_min_order,
    compute_sha256,
    get_local_ips,
    is_port_free,
)  # noqa: E402


def test_compute_sha256(tmp_path):
    path = tmp_path / "sample.bin"
    data = b"trade-app-test"
    path.write_bytes(data)
    assert compute_sha256(str(path)) == hashlib.sha256(data).hexdigest()


def test_is_port_free_roundtrip():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        assert not is_port_free(port, "127.0.0.1")
    assert is_port_free(port, "127.0.0.1")


def test_get_local_ips_contains_loopback():
    ips = get_local_ips()
    assert "127.0.0.1" in ips
    assert isinstance(ips, list)


@pytest.mark.parametrize(
    "price, qty, min_qty, qty_step, min_notional, expected",
    [
        (100.0, 0.1234, 0.01, 0.01, 0.0, 0.12),
        (150.0, 0.005, 0.01, 0.001, 0.0, 0.0),
        (10.0, 0.2, 0.01, 0.01, 5.0, 0.0),
    ],
)
def test_adjust_qty_rounds_and_filters(price, qty, min_qty, qty_step, min_notional, expected):
    result = adjust_qty(price, qty, min_qty=min_qty, qty_step=qty_step, min_notional=min_notional)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), None])
def test_adjust_qty_handles_non_finite(non_finite):
    out = adjust_qty(100.0, non_finite, min_qty=0.01, qty_step=0.01, min_notional=1.0)
    assert out == 0.0


def test_affordable_min_order_ok_case():
    res = affordable_min_order(
        price=100.0,
        min_qty=0.01,
        min_notional_usdt=5.0,
        balance_usdt=100.0,
        max_balance_share=0.5,
        hard_cap_share=0.5,
        leverage=5.0,
        qty_step=0.001,
        taker_fee=0.0006,
    )
    assert res["ok"] is True
    assert math.isclose(res["qty"], 0.01, rel_tol=1e-9)
    assert res["margin_required"] <= res["margin_cap"]


def test_affordable_min_order_insufficient_balance():
    res = affordable_min_order(
        price=100.0,
        min_qty=0.05,
        min_notional_usdt=5.0,
        balance_usdt=2.0,
        max_balance_share=0.25,
        hard_cap_share=0.2,
        leverage=1.0,
        qty_step=0.01,
        taker_fee=0.0006,
    )
    assert res["ok"] is False
    assert res["qty"] == 0.0
    assert res["margin_required"] > res["margin_cap"]
