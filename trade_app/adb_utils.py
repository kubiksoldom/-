"""Thin adb wrapper used by the APK manager."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class AdbDevice:
    serial: str
    state: str
    description: str = ""


class AdbError(RuntimeError):
    pass


def find_adb_binary(extra_paths: Optional[Sequence[str]] = None) -> Optional[str]:
    candidates = []
    env_path = os.getenv("ADB_PATH")
    if env_path:
        candidates.append(env_path)
    for item in extra_paths or []:
        candidates.append(item)
    which = shutil.which("adb")
    if which:
        candidates.append(which)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _run(cmd: Sequence[str], timeout: int = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise AdbError("adb binary not found") from exc


def list_devices(timeout: int = 10, extra_paths: Optional[Sequence[str]] = None) -> List[AdbDevice]:
    adb = find_adb_binary(extra_paths)
    if not adb:
        raise AdbError("adb binary not found")
    result = _run([adb, "devices", "-l"], timeout=timeout)
    devices: List[AdbDevice] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        desc = " ".join(parts[2:]) if len(parts) > 2 else ""
        devices.append(AdbDevice(serial=serial, state=state, description=desc))
    return devices


def install_apk(serial: str, apk_path: str, *, timeout: int = 180, reinstall: bool = True, extra_paths: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
    adb = find_adb_binary(extra_paths)
    if not adb:
        raise AdbError("adb binary not found")
    cmd = [adb, "-s", serial, "install"]
    if reinstall:
        cmd.append("-r")
    cmd.append(apk_path)
    result = _run(cmd, timeout=timeout)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    success = "Success" in result.stdout
    return success, output.strip()


def connect_wifi(target: str, *, timeout: int = 15, extra_paths: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
    adb = find_adb_binary(extra_paths)
    if not adb:
        raise AdbError("adb binary not found")
    result = _run([adb, "connect", target], timeout=timeout)
    success = "connected" in result.stdout.lower()
    return success, result.stdout.strip() or result.stderr.strip()


__all__ = [
    "AdbDevice",
    "AdbError",
    "find_adb_binary",
    "list_devices",
    "install_apk",
    "connect_wifi",
]
