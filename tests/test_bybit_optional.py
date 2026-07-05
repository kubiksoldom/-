import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bybit_api  # noqa: E402


def test_bybit_market_data_fallback_when_pybit_unavailable(monkeypatch):
    monkeypatch.setattr(bybit_api, "HTTP", None, raising=False)
    monkeypatch.setattr(bybit_api, "_PYBIT_IMPORT_ERROR", ImportError("missing pybit"), raising=False)
    monkeypatch.setattr(bybit_api, "log", lambda *args, **kwargs: None)

    assert bybit_api.get_kline("BTCUSDT") == {"result": {"list": []}}
    assert bybit_api.get_positions("BTCUSDT") == {"result": {"list": []}}
    assert bybit_api.get_wallet_balance() == {"result": {"list": []}}
    assert bybit_api.ping_credentials("key", "secret")["ok"] is False
