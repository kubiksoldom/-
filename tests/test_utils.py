import hashlib
import socket
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import compute_sha256, get_local_ips, is_port_free  # noqa: E402


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
