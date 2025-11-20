"""Restricted local HTTP server for APK distribution."""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse


@dataclass
class TokenInfo:
    token: str
    single_use: bool
    downloads: int = 0


class LocalApkHTTPServer:
    """Threaded HTTP server that exposes a single APK download endpoint."""

    def __init__(
        self,
        file_path: str,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        trusted_ips: Optional[Iterable[str]] = None,
        basic_auth: Optional[Tuple[str, str]] = None,
        log_path: str = "logs/apk_manager.jsonl",
    ) -> None:
        self.file_path = Path(file_path)
        self.host = host
        self.port = port
        self.basic_auth = basic_auth
        self._log_path = Path(log_path)
        self._tokens: Dict[str, TokenInfo] = {}
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        trusted = list(trusted_ips or [])
        self._trusted_networks = []
        self._trusted_addresses = set()
        for item in trusted:
            item = item.strip()
            if not item:
                continue
            try:
                if "/" in item:
                    self._trusted_networks.append(ip_network(item, strict=False))
                else:
                    self._trusted_addresses.add(ip_address(item))
            except ValueError:
                continue

    # --------------------------- public API ---------------------------
    def start(self) -> None:
        if self._httpd is not None:
            raise RuntimeError("Server already running")
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, int(self.port)), handler)
        self._httpd.daemon_threads = True
        _ = self._httpd.daemon_threads
        # update port (0 -> assigned)
        self.port = self._httpd.server_port
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.log_event("server_start", {"host": self.host, "port": self.port})

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None
        self.log_event("server_stop", {})

    def is_running(self) -> bool:
        return self._httpd is not None

    def create_token(self, single_use: bool = True) -> str:
        token = secrets.token_hex(32)
        with self._lock:
            self._tokens[token] = TokenInfo(token=token, single_use=single_use)
        self.log_event("token_created", {"token": token, "single_use": single_use})
        return token

    def revoke_token(self, token: str) -> None:
        with self._lock:
            if token in self._tokens:
                self._tokens.pop(token)
                self.log_event("token_revoked", {"token": token})

    def build_url(self, token: str, *, host_override: Optional[str] = None) -> str:
        host = host_override or self.host
        return f"http://{host}:{self.port}/dl/{token}/app.apk"

    def log_event(self, event: str, payload: Dict[str, object]) -> None:
        self._log_event(event, payload)

    # --------------------------- internals ---------------------------
    def _make_handler(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # pragma: no cover - disable console spam
                return

            def _require_auth(self) -> bool:
                if not server_ref.basic_auth:
                    return True
                user, password = server_ref.basic_auth
                header = self.headers.get("Authorization")
                if not header or not header.startswith("Basic "):
                    self.send_response(HTTPStatus.UNAUTHORIZED)
                    self.send_header("WWW-Authenticate", 'Basic realm="TradeApp"')
                    self.end_headers()
                    server_ref.log_event(
                        "basic_auth_required",
                        {"path": self.path, "client_ip": self.client_address[0]},
                    )
                    return False
                encoded = header.split(" ", 1)[1]
                try:
                    decoded = base64.b64decode(encoded).decode("utf-8")
                except Exception:
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    server_ref.log_event(
                        "basic_auth_decode_failed",
                        {"path": self.path, "client_ip": self.client_address[0]},
                    )
                    return False
                if decoded != f"{user}:{password}":
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    server_ref.log_event(
                        "basic_auth_invalid",
                        {"path": self.path, "client_ip": self.client_address[0]},
                    )
                    return False
                return True

            def _check_ip(self) -> bool:
                client_ip = self.client_address[0]
                if server_ref._is_ip_allowed(client_ip):
                    return True
                self.send_error(HTTPStatus.FORBIDDEN)
                server_ref.log_event(
                    "ip_blocked",
                    {"path": self.path, "client_ip": client_ip},
                )
                return False

            def do_GET(self):
                if not self._check_ip():
                    return
                if not self._require_auth():
                    return
                parsed = urlparse(self.path)
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) != 3 or parts[0] != "dl" or parts[2] != "app.apk":
                    self.send_error(HTTPStatus.FORBIDDEN)
                    server_ref.log_event(
                        "invalid_path",
                        {"path": parsed.path, "client_ip": self.client_address[0]},
                    )
                    return
                token = parts[1]
                info = server_ref._tokens.get(token)
                if info is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    server_ref.log_event(
                        "token_missing",
                        {"path": parsed.path, "client_ip": self.client_address[0]},
                    )
                    return
                if info.single_use and info.downloads >= 1:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    server_ref.log_event(
                        "token_consumed",
                        {"path": parsed.path, "client_ip": self.client_address[0]},
                    )
                    return
                try:
                    file_size = server_ref.file_path.stat().st_size
                    with server_ref.file_path.open("rb") as fh:
                        data = fh.read()
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    server_ref.log_event(
                        "file_missing",
                        {"token": token, "client_ip": self.client_address[0]},
                    )
                    return
                except Exception as exc:  # pragma: no cover - unexpected filesystem issues
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    server_ref.log_event(
                        "file_error",
                        {
                            "token": token,
                            "client_ip": self.client_address[0],
                            "error": str(exc),
                        },
                    )
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/vnd.android.package-archive")
                self.send_header("Content-Length", str(file_size))
                self.end_headers()
                self.wfile.write(data)

                with server_ref._lock:
                    info.downloads += 1
                    if info.single_use and info.downloads >= 1:
                        server_ref._tokens.pop(token, None)
                server_ref._log_event(
                    "download_ok",
                    {
                        "token": token,
                        "client_ip": self.client_address[0],
                        "single_use": info.single_use,
                        "downloads": info.downloads,
                    },
                )

        _ = Handler.log_message
        _ = Handler.do_GET
        return Handler

    def _is_ip_allowed(self, client_ip: str) -> bool:
        try:
            ip = ip_address(client_ip)
        except ValueError:
            return False
        if not self._trusted_networks and not self._trusted_addresses:
            return ip.is_loopback or ip.is_private
        if ip in self._trusted_addresses:
            return True
        for network in self._trusted_networks:
            if ip in network:
                return True
        return False

    def _log_event(self, event: str, payload: Dict[str, object]) -> None:
        record = {"ts": time.time(), "event": event}
        record.update(payload)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + os.linesep)
        except Exception:
            # logging must never crash server
            pass


__all__ = ["LocalApkHTTPServer", "TokenInfo"]
