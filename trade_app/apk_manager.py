"""APK manager screen for TradeApp."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import requests
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import config
from trade_app.adb_utils import AdbError, AdbDevice, install_apk, list_devices
from trade_app.local_http import LocalApkHTTPServer
from trade_app.qr_utils import generate_qr_pixmap
from utils import compute_sha256, get_local_ips, is_port_free, log


class ApkManagerScreen(QWidget):
    SOURCES = ["Локальный файл", "URL", "GitHub Release"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName("ApkManagerScreen")
        self.server: Optional[LocalApkHTTPServer] = None
        self.server_token: Optional[str] = None
        self.display_host: Optional[str] = None
        self.current_file: Optional[Path] = None
        self.current_sha: Optional[str] = None
        self._temp_files: List[Path] = []
        self.autostop_timer = QTimer(self)
        self.autostop_timer.setSingleShot(True)
        self.autostop_timer.timeout.connect(self._on_autostop)
        self.devices: List[AdbDevice] = []
        self.trusted_ips = list(config.TRUSTED_IPS or [])
        self.log_path = Path(getattr(config, "APK_MANAGER_LOG", "logs/apk_manager.jsonl"))
        self._setup_ui()
        self._refresh_mode_label()
        self._update_controls()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        main = QVBoxLayout(self)
        title = QLabel("<h2>APK Manager</h2>")
        main.addWidget(title)

        self.lbl_mode = QLabel()
        main.addWidget(self.lbl_mode)

        source_box = QGroupBox("Источник APK")
        source_layout = QVBoxLayout(source_box)
        top_row = QHBoxLayout()
        self.cmb_source = QComboBox()
        self.cmb_source.addItems(self.SOURCES)
        top_row.addWidget(self.cmb_source)
        source_layout.addLayout(top_row)

        self.source_stack = QStackedWidget()

        # Local file widget
        local_widget = QWidget()
        lw = QHBoxLayout(local_widget)
        self.le_local_path = QLineEdit()
        self.btn_browse = QPushButton("Выбрать…")
        lw.addWidget(self.le_local_path)
        lw.addWidget(self.btn_browse)
        self.source_stack.addWidget(local_widget)

        # URL widget
        url_widget = QWidget()
        ul = QHBoxLayout(url_widget)
        self.le_url = QLineEdit()
        self.le_url.setPlaceholderText("https://example.com/app.apk")
        ul.addWidget(self.le_url)
        self.source_stack.addWidget(url_widget)

        # GitHub widget
        gh_widget = QWidget()
        gh_form = QFormLayout(gh_widget)
        self.le_repo = QLineEdit()
        self.le_repo.setPlaceholderText("owner/repo")
        self.le_asset = QLineEdit()
        self.le_asset.setPlaceholderText("asset substring e.g. arm64")
        gh_form.addRow("Репозиторий:", self.le_repo)
        gh_form.addRow("Фильтр по имени:", self.le_asset)
        self.source_stack.addWidget(gh_widget)

        source_layout.addWidget(self.source_stack)

        action_row = QHBoxLayout()
        self.btn_fetch = QPushButton("Скачать")
        self.btn_check_sha = QPushButton("Проверить SHA256")
        action_row.addWidget(self.btn_fetch)
        action_row.addWidget(self.btn_check_sha)
        source_layout.addLayout(action_row)

        self.lbl_file_info = QLabel("Файл: —")
        source_layout.addWidget(self.lbl_file_info)

        main.addWidget(source_box)

        # Server controls
        server_box = QGroupBox("Раздача APK")
        server_layout = QFormLayout(server_box)

        self.cmb_interface = QComboBox()
        self._populate_interfaces()
        server_layout.addRow("Интерфейс:", self.cmb_interface)

        self.spn_port = QSpinBox()
        self.spn_port.setRange(0, 65535)
        self.spn_port.setValue(int(getattr(config, "APK_DEFAULT_PORT", 8787)))
        server_layout.addRow("Порт:", self.spn_port)

        self.spn_autostop = QSpinBox()
        self.spn_autostop.setRange(0, 240)
        self.spn_autostop.setValue(int(getattr(config, "APK_DEFAULT_AUTOSTOP_MIN", 15)))
        server_layout.addRow("Авто-стоп (мин):", self.spn_autostop)

        self.chk_single_use = QCheckBox("Одноразовая ссылка")
        self.chk_single_use.setChecked(bool(getattr(config, "ONE_TIME_APK_LINK", True)))
        server_layout.addRow("Безопасность:", self.chk_single_use)

        self.lbl_basic_auth = QLabel(self._basic_auth_text())
        self.lbl_basic_auth.setWordWrap(True)
        server_layout.addRow("Basic Auth:", self.lbl_basic_auth)

        self.lbl_trusted = QLabel(self._trusted_ips_text())
        self.lbl_trusted.setWordWrap(True)
        server_layout.addRow("Доверенные IP:", self.lbl_trusted)

        server_buttons = QHBoxLayout()
        self.btn_start_server = QPushButton("Запустить раздачу")
        self.btn_stop_server = QPushButton("Остановить")
        self.btn_generate_qr = QPushButton("Сгенерировать QR")
        server_buttons.addWidget(self.btn_start_server)
        server_buttons.addWidget(self.btn_stop_server)
        server_buttons.addWidget(self.btn_generate_qr)
        server_layout.addRow(server_buttons)

        self.lbl_server_status = QLabel("Сервер: не запущен")
        self.lbl_server_status.setWordWrap(True)
        server_layout.addRow("Статус:", self.lbl_server_status)

        main.addWidget(server_box)

        # QR display
        qr_box = QGroupBox("QR")
        qr_layout = QVBoxLayout(qr_box)
        self.lbl_qr = QLabel()
        self.lbl_qr.setAlignment(Qt.AlignCenter)
        qr_layout.addWidget(self.lbl_qr)
        main.addWidget(qr_box)

        # ADB controls
        adb_box = QGroupBox("Установка через ADB")
        adb_layout = QHBoxLayout(adb_box)
        self.cmb_devices = QComboBox()
        self.btn_refresh_devices = QPushButton("Обновить список")
        self.btn_install_adb = QPushButton("Установить")
        adb_layout.addWidget(self.cmb_devices, 1)
        adb_layout.addWidget(self.btn_refresh_devices)
        adb_layout.addWidget(self.btn_install_adb)
        main.addWidget(adb_box)

        # Log area
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(160)
        main.addWidget(self.txt_log, 1)

        # wiring
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        self.btn_browse.clicked.connect(self._pick_local_file)
        self.btn_fetch.clicked.connect(self._fetch_apk)
        self.btn_check_sha.clicked.connect(self._check_sha)
        self.btn_start_server.clicked.connect(self._start_server)
        self.btn_stop_server.clicked.connect(self._stop_server)
        self.btn_generate_qr.clicked.connect(self._generate_qr)
        self.btn_refresh_devices.clicked.connect(self._refresh_devices)
        self.btn_install_adb.clicked.connect(self._install_selected_device)

    # ------------------------------------------------------------------ helpers
    def _refresh_mode_label(self) -> None:
        safe = int(os.getenv("SAFE_MODE", "1" if getattr(config, "SAFE_MODE", True) else "0"))
        paper = int(os.getenv("PAPER_MODE", "1" if getattr(config, "PAPER_MODE", True) else "0"))
        self.lbl_mode.setText(
            f"Текущий режим: PAPER_MODE={paper} / SAFE_MODE={safe}. "
            "Для REAL требуется PIN/2FA."
        )

    def _populate_interfaces(self) -> None:
        self.cmb_interface.clear()
        ips = get_local_ips()
        for ip in ips:
            label = ip
            if ip == "127.0.0.1":
                label += " (localhost)"
            self.cmb_interface.addItem(label, ip)
        if getattr(config, "APK_BIND_ALL", False):
            self.cmb_interface.addItem("0.0.0.0 (все интерфейсы)", "0.0.0.0")

    def _basic_auth_text(self) -> str:
        if getattr(config, "APK_ENABLE_BASIC_AUTH", False):
            user = getattr(config, "APK_BASIC_AUTH_USER", "tradeapp")
            return f"Включено (user: {user})"
        return "Выключено"

    def _trusted_ips_text(self) -> str:
        if not self.trusted_ips:
            return "Не задано (разрешены локальные/private IP)"
        return ", ".join(self.trusted_ips)

    def _update_controls(self) -> None:
        has_file = self.current_file is not None and self.current_file.exists()
        running = self.server is not None and self.server.is_running()
        self.btn_fetch.setEnabled(self.cmb_source.currentText() != "Локальный файл")
        self.btn_check_sha.setEnabled(has_file)
        self.btn_start_server.setEnabled(has_file and not running)
        self.btn_stop_server.setEnabled(running)
        self.btn_generate_qr.setEnabled(running)
        if not has_file:
            self.lbl_file_info.setText("Файл: —")
        if not running:
            self.lbl_server_status.setText("Сервер: не запущен")
            self.lbl_qr.clear()

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{timestamp}] {message}")
        log(f"[APK] {message}")
        self._write_log_entry("ui", {"message": message})

    def _write_log_entry(self, event: str, payload: dict) -> None:
        record = {"ts": time.time(), "event": event}
        record.update(payload)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _on_source_changed(self, index: int) -> None:
        self.source_stack.setCurrentIndex(index)
        self._update_controls()

    def _pick_local_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбери APK", "", "APK files (*.apk);;All files (*.*)")
        if not path:
            return
        self.le_local_path.setText(path)
        self._set_current_file(Path(path), is_temp=False)

    def _fetch_apk(self) -> None:
        source = self.cmb_source.currentText()
        if source == "URL":
            self._download_from_url()
        elif source == "GitHub Release":
            self._download_from_github()
        else:
            QMessageBox.information(self, "Локальный файл", "Укажи путь к APK через кнопку 'Выбрать'.")

    def _download_from_url(self) -> None:
        url = self.le_url.text().strip()
        if not url:
            QMessageBox.warning(self, "URL", "Укажи ссылку на APK.")
            return
        self._append_log(f"Скачиваю APK из {url}")
        try:
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            temp = tempfile.NamedTemporaryFile(suffix=".apk", delete=False)
            with temp as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
            path = Path(temp.name)
            self._set_current_file(path, is_temp=True)
            self._append_log("Загрузка завершена.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка скачивания", str(exc))
            self._append_log(f"Ошибка скачивания: {exc}")

    def _download_from_github(self) -> None:
        repo = self.le_repo.text().strip()
        if "/" not in repo:
            QMessageBox.warning(self, "GitHub", "Формат репо: owner/repo")
            return
        asset_filter = self.le_asset.text().strip()
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        headers = {"Accept": "application/vnd.github+json"}
        self._append_log(f"Запрашиваю релиз GitHub: {api_url}")
        try:
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            assets = data.get("assets", [])
            if not assets:
                raise RuntimeError("Нет артефактов в релизе")
            asset = None
            if asset_filter:
                for item in assets:
                    if asset_filter in item.get("name", ""):
                        asset = item
                        break
            if asset is None:
                asset = assets[0]
            download_url = asset.get("browser_download_url")
            if not download_url:
                raise RuntimeError("Не найден download URL")
            self.le_url.setText(download_url)
            self.cmb_source.setCurrentText("URL")
            self._download_from_url()
        except Exception as exc:
            QMessageBox.critical(self, "GitHub", str(exc))
            self._append_log(f"Ошибка GitHub: {exc}")

    def _set_current_file(self, path: Path, *, is_temp: bool) -> None:
        self.current_file = path
        if is_temp:
            self._temp_files.append(path)
        size = path.stat().st_size if path.exists() else 0
        self.current_sha = None
        self.lbl_file_info.setText(f"Файл: {path} ({size / 1024:.1f} КБ)")
        self._append_log(f"Выбран файл {path}")
        self._update_controls()

    def _check_sha(self) -> None:
        if not self.current_file or not self.current_file.exists():
            QMessageBox.warning(self, "SHA256", "Сначала выбери файл.")
            return
        try:
            sha = compute_sha256(str(self.current_file))
            self.current_sha = sha
            self.lbl_file_info.setText(f"Файл: {self.current_file} — SHA256: {sha}")
            self._append_log(f"SHA256: {sha}")
        except Exception as exc:
            QMessageBox.critical(self, "SHA256", str(exc))
            self._append_log(f"Ошибка SHA256: {exc}")

    def _start_server(self) -> None:
        if not self.current_file or not self.current_file.exists():
            QMessageBox.warning(self, "Раздача", "Нет APK файла для раздачи.")
            return
        if self.server and self.server.is_running():
            QMessageBox.information(self, "Раздача", "Сервер уже запущен.")
            return
        bind_host = self.cmb_interface.currentData() or "127.0.0.1"
        display_host = bind_host
        if bind_host == "0.0.0.0":
            display_host = self.cmb_interface.itemText(self.cmb_interface.currentIndex()).split()[0]
        port = int(self.spn_port.value())
        if port and not is_port_free(port, bind_host):
            QMessageBox.critical(self, "Порт", f"Порт {port} занят на {bind_host}.")
            return
        basic_auth = None
        if getattr(config, "APK_ENABLE_BASIC_AUTH", False):
            basic_auth = (
                getattr(config, "APK_BASIC_AUTH_USER", "tradeapp"),
                getattr(config, "APK_BASIC_AUTH_PASS", "changeme"),
            )
        self.server = LocalApkHTTPServer(
            str(self.current_file),
            host=bind_host,
            port=port,
            trusted_ips=self.trusted_ips,
            basic_auth=basic_auth,
            log_path=str(self.log_path),
        )
        single_use = bool(self.chk_single_use.isChecked())
        self.server_token = self.server.create_token(single_use=single_use)
        try:
            self.server.start()
        except Exception as exc:
            QMessageBox.critical(self, "Сервер", str(exc))
            self._append_log(f"Ошибка запуска сервера: {exc}")
            self.server = None
            self.server_token = None
            return
        self.display_host = display_host
        url = self.server.build_url(self.server_token, host_override=display_host)
        self.lbl_server_status.setText(f"Сервер запущен: <a href='{url}'>{url}</a>")
        self.lbl_server_status.setTextFormat(Qt.RichText)
        self.lbl_server_status.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.lbl_server_status.setOpenExternalLinks(True)
        self._append_log(f"Сервер запущен на {bind_host}:{self.server.port}")
        if self.current_sha:
            self._write_log_entry("sha", {"sha256": self.current_sha})
        if self.spn_autostop.value() > 0:
            self.autostop_timer.start(self.spn_autostop.value() * 60 * 1000)
        else:
            self.autostop_timer.stop()
        self._update_controls()

    def _stop_server(self) -> None:
        if not self.server:
            return
        try:
            self.server.stop()
        except Exception as exc:
            self._append_log(f"Ошибка остановки: {exc}")
        self.server = None
        self.server_token = None
        self.display_host = None
        self.autostop_timer.stop()
        self._update_controls()
        self._append_log("Сервер остановлен")

    def _generate_qr(self) -> None:
        if not self.server or not self.server.is_running() or not self.server_token:
            QMessageBox.information(self, "QR", "Сначала запусти раздачу.")
            return
        host = self.display_host or (self.cmb_interface.currentData() or "127.0.0.1")
        url = self.server.build_url(self.server_token, host_override=host)
        pix = generate_qr_pixmap(url)
        scaled = pix.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_qr.setPixmap(scaled)
        self._append_log(f"QR обновлён для {url}")

    def _on_autostop(self) -> None:
        if self.server and self.server.is_running():
            self._append_log("Авто-стоп: таймер истёк")
            self._stop_server()

    def _refresh_devices(self) -> None:
        try:
            self.devices = list_devices()
        except AdbError as exc:
            QMessageBox.warning(self, "ADB", str(exc))
            self._append_log(f"ADB: {exc}")
            return
        self.cmb_devices.clear()
        if not self.devices:
            self.cmb_devices.addItem("Нет подключенных устройств")
            self.cmb_devices.setEnabled(False)
            self.btn_install_adb.setEnabled(False)
            return
        self.cmb_devices.setEnabled(True)
        self.btn_install_adb.setEnabled(True)
        for dev in self.devices:
            label = f"{dev.serial} ({dev.state})"
            if dev.description:
                label += f" — {dev.description}"
            self.cmb_devices.addItem(label, dev.serial)
        self._append_log(f"Найдено устройств: {len(self.devices)}")

    def _install_selected_device(self) -> None:
        if not self.current_file or not self.current_file.exists():
            QMessageBox.warning(self, "ADB", "Нет APK файла.")
            return
        if not self.devices:
            self._refresh_devices()
            if not self.devices:
                return
        idx = self.cmb_devices.currentIndex()
        serial = self.cmb_devices.itemData(idx)
        if not serial:
            QMessageBox.information(self, "ADB", "Выбери устройство.")
            return
        try:
            ok, output = install_apk(serial, str(self.current_file))
        except AdbError as exc:
            QMessageBox.critical(self, "ADB", str(exc))
            self._append_log(f"ADB ошибка: {exc}")
            return
        if ok:
            QMessageBox.information(self, "ADB", "Установка завершена: Success")
            self._append_log(f"ADB install success: {serial}")
        else:
            QMessageBox.warning(self, "ADB", output or "Неизвестная ошибка")
            self._append_log(f"ADB install failed: {output}")

    def cleanup(self) -> None:
        self.autostop_timer.stop()
        if self.server and self.server.is_running():
            try:
                self.server.stop()
            except Exception:
                pass
        for path in self._temp_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        self._temp_files.clear()

    def __del__(self):  # pragma: no cover - cleanup helper
        self.cleanup()


__all__ = ["ApkManagerScreen"]
