import importlib
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

adb_utils = importlib.import_module("trade_app.adb_utils")  # noqa: E402


def _fake_run(stdout: str = "", stderr: str = ""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)


def test_list_devices_parses_output():
    output = """List of devices attached\nABC123\tdevice product:sdk\nXYZ\toffline\n"""
    with patch.object(adb_utils, "find_adb_binary", return_value="/usr/bin/adb"), \
            patch.object(adb_utils, "_run", return_value=_fake_run(stdout=output)):
        devices = adb_utils.list_devices()
    assert len(devices) == 2
    assert devices[0].serial == "ABC123"
    assert devices[0].state == "device"
    assert devices[1].state == "offline"


def test_list_devices_missing_adb():
    with patch.object(adb_utils, "find_adb_binary", return_value=None):
        with pytest.raises(adb_utils.AdbError):
            adb_utils.list_devices()


def test_install_apk_success():
    with patch.object(adb_utils, "find_adb_binary", return_value="/usr/bin/adb"), \
            patch.object(adb_utils, "_run", return_value=_fake_run(stdout="Success\n")):
        ok, output = adb_utils.install_apk("ABC123", "/tmp/app.apk")
    assert ok
    assert "Success" in output


def test_install_apk_failure():
    with patch.object(adb_utils, "find_adb_binary", return_value="/usr/bin/adb"), \
            patch.object(adb_utils, "_run", return_value=_fake_run(stdout="Failure\n", stderr="INSTALL_FAILED")):
        ok, output = adb_utils.install_apk("ABC123", "/tmp/app.apk")
    assert not ok
    assert "INSTALL_FAILED" in output


def test_connect_wifi():
    with patch.object(adb_utils, "find_adb_binary", return_value="/usr/bin/adb"), \
            patch.object(adb_utils, "_run", return_value=_fake_run(stdout="connected to 1.2.3.4:5555\n")):
        ok, output = adb_utils.connect_wifi("1.2.3.4:5555")
    assert ok
    assert "connected" in output
