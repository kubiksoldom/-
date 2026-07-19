import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PAPER_MODE"] = "1"
os.environ["SAFE_MODE"] = "1"
os.environ["TELEGRAM_ENABLED"] = "0"
os.environ["PAPER_SYNC_BALANCE"] = "0"
os.environ["RUN_ONLINE_CHECKS"] = "0"
os.environ["BYBIT_API_KEY"] = ""
os.environ["BYBIT_API_SECRET"] = ""
os.environ["TELEGRAM_TOKEN"] = ""
os.environ["TELEGRAM_CHAT_ID"] = ""

import config  # noqa: E402
import main  # noqa: E402
import paper_engine  # noqa: E402
import sanity_check  # noqa: E402


def setup_function():
    sanity_check._reset_summary()


def test_position_lifecycle_sanity_contracts_are_offline_and_green():
    assert sanity_check.check_position_lifecycle_contracts(main, paper_engine, config) is True
    assert sanity_check._SUMMARY["error"] == []
    assert len(sanity_check._SUMMARY["ok"]) == 7


def test_missing_lifecycle_interface_makes_sanity_fail():
    incomplete_main = SimpleNamespace()

    assert sanity_check.check_position_lifecycle_contracts(
        incomplete_main,
        paper_engine,
        config,
    ) is False
    assert sanity_check.sanity_exit_code() == 1
    assert "отсутствуют функции" in sanity_check._SUMMARY["error"][0]


def test_sanity_exit_code_is_zero_for_warnings_and_nonzero_for_errors():
    sanity_check.record_warn("предупреждение")
    assert sanity_check.sanity_exit_code() == 0

    sanity_check.record_error("критическая ошибка")
    assert sanity_check.sanity_exit_code() == 1


def test_number_value_preserves_numeric_balance(monkeypatch):
    monkeypatch.setenv("SANITY_BALANCE", "123.45")
    assert sanity_check._number_value("SANITY_BALANCE", config, default=1.0) == 123.45
