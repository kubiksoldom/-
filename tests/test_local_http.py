import sys
from pathlib import Path

import requests
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trade_app.local_http import LocalApkHTTPServer  # noqa: E402


@pytest.fixture
def apk_file(tmp_path):
    path = tmp_path / "app.apk"
    path.write_bytes(b"dummy-apk")
    return path


def test_single_use_token(apk_file):
    server = LocalApkHTTPServer(
        str(apk_file),
        host="127.0.0.1",
        port=0,
        trusted_ips=["127.0.0.1"],
    )
    token = server.create_token(single_use=True)
    server.start()
    try:
        url = server.build_url(token, host_override="127.0.0.1")
        first = requests.get(url, timeout=5)
        assert first.status_code == 200
        second = requests.get(url, timeout=5)
        assert second.status_code == 404
        forbidden = requests.get(f"http://127.0.0.1:{server.port}/bad", timeout=5)
        assert forbidden.status_code == 403
    finally:
        server.stop()


def test_basic_auth_required(apk_file):
    server = LocalApkHTTPServer(
        str(apk_file),
        host="127.0.0.1",
        port=0,
        trusted_ips=["127.0.0.1"],
        basic_auth=("user", "pass"),
    )
    token = server.create_token(single_use=False)
    server.start()
    try:
        url = server.build_url(token, host_override="127.0.0.1")
        unauth = requests.get(url, timeout=5)
        assert unauth.status_code == 401
        auth = requests.get(url, timeout=5, auth=("user", "pass"))
        assert auth.status_code == 200
    finally:
        server.stop()
