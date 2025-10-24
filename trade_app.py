# trade_app.py
# v3.2 — Полный UI апгрейд: Паника/Пауза/Инструменты/Фильтры/Отчёты/Хоткеи/Безопасность/Конфиг/Подсветка
import sys, os, json, re, time, pathlib, csv, math, statistics, shutil, random, uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from PyQt5.QtCore import Qt, QTimer, QProcess, QProcessEnvironment, QUrl, QByteArray
from PyQt5.QtGui import (
    QFont, QTextCursor, QDesktopServices, QKeySequence,
    QSyntaxHighlighter, QTextCharFormat, QColor, QPalette, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QMessageBox, QPlainTextEdit, QComboBox, QCheckBox,
    QFileDialog, QLineEdit, QFormLayout, QSpinBox, QShortcut, QFrame,
    QMainWindow, QAction, QToolBar, QDialog, QDialogButtonBox, QTabWidget,
    QTextBrowser, QStyle, QInputDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter
)

import pyqtgraph as pg
from pyqtgraph import DateAxisItem

import config
from trade_app.apk_manager import ApkManagerScreen
from utils import (
    log,
    ts_to_epoch,
    safe_read_jsonl,
    now_iso,
    get_sessions_root,
    list_session_directories,
    env_bool,
    verify_pin_hash,
    tg_send,
)

import bybit_api

try:  # pragma: no cover - keyring может отсутствовать
    import keyring  # type: ignore
except Exception:  # pragma: no cover
    keyring = None

# ----------------------------- константы -----------------------------
APP_TITLE         = "🔥 КриптоБот v3.2"
LOG_DEFAULT       = "bot_cycle_log.jsonl"
MAIN_PY           = "main.py"
BUILD_DATASET_PY  = "build_ml_dataset_from_fills.py"
TRAIN_MODEL_PY    = "nn_model.py"
CONFIG_FILE       = os.path.join(os.path.expanduser("~"), ".trade_app_config.json")

STOP_TIMEOUT_SEC  = 6
PANIC_WAIT_SEC    = 2
MARGIN_STATUS_FILE = os.path.join("logs", "metrics", "margin_status.json")
ML_STATUS_FILE     = os.path.join("logs", "metrics", "ml_status.json")

# ----------------------------- стили -----------------------------
COMMON_STYLESHEET = """
* {
    font-family: 'Segoe UI', 'Noto Sans', 'Roboto', sans-serif;
}
QWidget {
    font-size: 11pt;
}
QPushButton {
    border: 1px solid rgba(22, 119, 255, 160);
    border-radius: 8px;
    padding: 6px 12px;
    background-color: rgba(22, 119, 255, 0.85);
    color: #ffffff;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(64, 152, 255, 0.95);
}
QPushButton:pressed {
    background-color: rgba(11, 94, 215, 0.95);
}
QPushButton[destructive="true"] {
    background-color: #ff4d4f;
    border-color: #ff7875;
}
QPushButton[destructive="true"]:hover {
    background-color: #ff7875;
}
QFrame#StatusFrame {
    border: 1px solid rgba(22, 119, 255, 120);
    border-radius: 10px;
    padding: 6px;
    margin-bottom: 6px;
}
QPlainTextEdit, QTextEdit {
    border-radius: 8px;
    padding: 8px;
}
QLabel#SecondaryLabel {
    font-size: 10pt;
}
"""

DARK_STYLESHEET = """
QWidget {
    background-color: #181b26;
    color: #f2f5ff;
}
QPlainTextEdit, QTextEdit {
    background-color: #10131b;
    border: 1px solid #2c3142;
    color: #f2f5ff;
}
QComboBox, QLineEdit, QSpinBox {
    background-color: #161a27;
    border: 1px solid #2c3142;
    border-radius: 6px;
    padding: 4px 6px;
    color: #f2f5ff;
    selection-background-color: rgba(22, 119, 255, 0.7);
}
QCheckBox {
    padding: 2px;
}
QFrame#StatusFrame {
    background-color: rgba(22, 119, 255, 0.12);
}
QToolTip {
    color: #f2f5ff;
    background-color: #1f2331;
    border: 1px solid #2c3142;
}
"""

LIGHT_STYLESHEET = """
QWidget {
    background-color: #f7f9fc;
    color: #101320;
}
QPlainTextEdit, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #d6dbe6;
    color: #101320;
}
QComboBox, QLineEdit, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #d6dbe6;
    border-radius: 6px;
    padding: 4px 6px;
}
QFrame#StatusFrame {
    background-color: rgba(22, 119, 255, 0.10);
}
QToolTip {
    color: #101320;
    background-color: #ffffff;
    border: 1px solid #d6dbe6;
}
"""

# ----------------------------- конфиг -----------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "PAPER",
    "unsafe": False,
    "autoscroll": True,
    "ignore_schedule": False,
    "log_path": LOG_DEFAULT,
    "pairs": "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
    "default_lev": 5,
    "win_w": 1000,
    "win_h": 680,
    "pos_x": None,
    "pos_y": None,
    "theme": "Auto",
    "log_lang": "RU",
}

def load_config() -> Dict[str, Any]:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if isinstance(cfg, dict):
                    out = DEFAULT_CONFIG.copy()
                    out.update(cfg)
                    # sanity
                    if not out.get("log_path"):
                        out["log_path"] = LOG_DEFAULT
                    if not out.get("pairs"):
                        out["pairs"] = DEFAULT_CONFIG["pairs"]
                    theme = str(out.get("theme", "Auto") or "Auto").title()
                    if theme not in {"Auto", "Dark", "Light"}:
                        theme = "Auto"
                    out["theme"] = theme
                    return out
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]) -> None:
    try:
        data = DEFAULT_CONFIG.copy()
        data.update(cfg or {})
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[CONFIG] save error:", e, file=sys.stderr)

# ----------------------------- хранилище ключей -----------------------------
class KeysDialog(QDialog):
    SERVICE_NAME = "tradeapp.bybit"
    ENV_FILE = os.path.abspath(os.path.join(os.getcwd(), "dotenv.env"))

    def __init__(self, parent: "TradeApp"):
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("🔑 API-ключи Bybit")
        self.resize(540, 380)

        self._stored_key: Optional[str] = None
        self._stored_secret: Optional[str] = None
        self._pending_key: Optional[str] = None
        self._pending_secret: Optional[str] = None
        self._storage_label = "—"

        layout = QVBoxLayout(self)
        title = QLabel("<h2>Управление API-ключами</h2>")
        layout.addWidget(title)

        self.lbl_hint = QLabel(
            "Ключ должен иметь права <b>Trade</b>. Отзыв секретов приведёт к остановке торговли."
        )
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.lbl_storage = QLabel()
        self.lbl_storage.setObjectName("SecondaryLabel")
        layout.addWidget(self.lbl_storage)

        form = QFormLayout()
        key_row = QHBoxLayout()
        self.lbl_key_value = QLabel("—")
        self.lbl_key_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_edit_key = QPushButton("Изменить ключ…")
        self.btn_edit_key.setProperty("secondary", True)
        key_row.addWidget(self.lbl_key_value, 1)
        key_row.addWidget(self.btn_edit_key)
        key_container = QWidget()
        key_container.setLayout(key_row)
        form.addRow("API Key", key_container)

        secret_row = QHBoxLayout()
        self.lbl_secret_status = QLabel("—")
        self.btn_edit_secret = QPushButton("Обновить секрет…")
        self.btn_edit_secret.setProperty("secondary", True)
        secret_row.addWidget(self.lbl_secret_status, 1)
        secret_row.addWidget(self.btn_edit_secret)
        secret_container = QWidget()
        secret_container.setLayout(secret_row)
        form.addRow("API Secret", secret_container)

        self.lbl_rights = QLabel("Права ключа: Trade ✔  Withdraw ✖")
        self.lbl_rights.setObjectName("SecondaryLabel")
        form.addRow("Подсказка", self.lbl_rights)

        layout.addLayout(form)

        self.lbl_status = QLabel("Статус: —")
        layout.addWidget(self.lbl_status)

        self.lbl_last_check = QLabel("Последняя проверка: —")
        self.lbl_last_check.setObjectName("SecondaryLabel")
        layout.addWidget(self.lbl_last_check)

        btns = QHBoxLayout()
        self.btn_check = QPushButton("🔍 Проверить подключение")
        self.btn_save = QPushButton("💾 Сохранить")
        self.btn_delete = QPushButton("🗑 Удалить")
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.setProperty("secondary", True)
        btns.addWidget(self.btn_check)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        layout.addLayout(btns)

        self.btn_close.clicked.connect(self.reject)
        self.btn_edit_key.clicked.connect(self._edit_key)
        self.btn_edit_secret.clicked.connect(self._edit_secret)
        self.btn_save.clicked.connect(self._save)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_check.clicked.connect(self._check)

        self._load_credentials()
        self._update_labels()

    # --- helpers -----------------------------------------------------
    def _mask_key(self, value: Optional[str]) -> str:
        if not value:
            return "—"
        value = value.strip()
        if len(value) <= 8:
            return value
        return f"{value[:4]}…{value[-4:]}"

    def _load_env_map(self) -> Dict[str, str]:
        data: Dict[str, str] = {}
        path = self.ENV_FILE
        if not os.path.exists(path):
            return data
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip()
        except Exception:
            return {}
        return data

    def _write_env_map(self, data: Dict[str, str]) -> None:
        try:
            with open(self.ENV_FILE, "w", encoding="utf-8") as fh:
                for key in sorted(data.keys()):
                    fh.write(f"{key}={data[key]}\n")
        except Exception as exc:
            QMessageBox.warning(self, "dotenv", f"Не удалось сохранить dotenv.env: {exc}")

    def _load_credentials(self) -> None:
        key = secret = None
        source = "none"
        if keyring is not None:
            try:
                key = keyring.get_password(self.SERVICE_NAME, "api_key")
                secret = keyring.get_password(self.SERVICE_NAME, "api_secret")
                if key or secret:
                    source = "keyring"
            except Exception:
                key = secret = None
        if not key or not secret:
            env_map = self._load_env_map()
            key = env_map.get("BYBIT_API_KEY")
            secret = env_map.get("BYBIT_API_SECRET")
            if key and secret:
                source = "dotenv"
        self._stored_key = key or None
        self._stored_secret = secret or None
        self._storage_label = source

    def _update_labels(self) -> None:
        self.lbl_key_value.setText(self._mask_key(self._pending_key or self._stored_key))
        if self._pending_secret:
            secret_state = "Изменён (не сохранён)"
        elif self._stored_secret:
            secret_state = "Секрет задан"
        else:
            secret_state = "—"
        self.lbl_secret_status.setText(secret_state)
        if self._storage_label == "keyring":
            storage_text = "Хранилище: системный keyring"
            self.lbl_hint.setStyleSheet("")
        elif self._storage_label == "dotenv":
            storage_text = f"Хранилище: dotenv.env ({self.ENV_FILE})"
            self.lbl_hint.setStyleSheet("color: #ff7875;")
        else:
            storage_text = "Хранилище: не найдено"
            self.lbl_hint.setStyleSheet("color: #ff7875;")
        self.lbl_storage.setText(storage_text)

        cfg = dict(self.parent_app._cfg)
        last_ts = cfg.get("keys_last_check")
        status = cfg.get("keys_last_status")
        if last_ts:
            self.lbl_last_check.setText(f"Последняя проверка: {last_ts} ({status or 'нет данных'})")
        else:
            self.lbl_last_check.setText("Последняя проверка: —")

    # --- actions -----------------------------------------------------
    def _edit_key(self) -> None:
        current = self._pending_key or self._stored_key or ""
        text, ok = QInputDialog.getText(
            self,
            "Обновить API Key",
            "Введите новый API Key",
            QLineEdit.Normal,
            current,
        )
        if ok and text:
            self._pending_key = text.strip()
            self.lbl_status.setText("Статус: ключ обновлён (не сохранён)")
            self._update_labels()

    def _edit_secret(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Обновить API Secret",
            "Введите новый API Secret",
            QLineEdit.Password,
        )
        if ok and text:
            self._pending_secret = text.strip()
            self.lbl_status.setText("Статус: секрет обновлён (не сохранён)")
            self._update_labels()

    def _save_to_keyring(self, api_key: str, api_secret: str) -> bool:
        if keyring is None:
            return False
        try:
            keyring.set_password(self.SERVICE_NAME, "api_key", api_key)
            keyring.set_password(self.SERVICE_NAME, "api_secret", api_secret)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Keyring", f"Не удалось сохранить в keyring: {exc}")
            return False

    def _save_to_env(self, api_key: str, api_secret: str) -> None:
        data = self._load_env_map()
        data["BYBIT_API_KEY"] = api_key
        data["BYBIT_API_SECRET"] = api_secret
        self._write_env_map(data)

    def _save(self) -> None:
        api_key = (self._pending_key or self._stored_key or "").strip()
        api_secret = (self._pending_secret or self._stored_secret or "").strip()
        if not api_key or not api_secret:
            QMessageBox.warning(self, "Сохранение", "Укажи и ключ, и секрет для сохранения.")
            return
        stored_in_keyring = self._save_to_keyring(api_key, api_secret)
        if stored_in_keyring:
            data = self._load_env_map()
            if "BYBIT_API_KEY" in data:
                data.pop("BYBIT_API_KEY", None)
                data.pop("BYBIT_API_SECRET", None)
                self._write_env_map(data)
            storage = "keyring"
        else:
            self._save_to_env(api_key, api_secret)
            storage = "dotenv"

        bybit_api.configure_client(api_key, api_secret)
        cfg = dict(self.parent_app._cfg)
        cfg.pop("keys_last_check", None)
        cfg.pop("keys_last_status", None)
        save_config(cfg)
        self.parent_app._cfg = cfg

        self._stored_key = api_key
        self._stored_secret = api_secret
        self._pending_key = None
        self._pending_secret = None
        self._storage_label = storage
        self.lbl_status.setText("Статус: ключи сохранены")
        self._update_labels()

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Удаление ключей",
            "Удалить сохранённые ключи?",
        )
        if confirm != QMessageBox.Yes:
            return
        if keyring is not None:
            try:
                keyring.delete_password(self.SERVICE_NAME, "api_key")
            except Exception:
                pass
            try:
                keyring.delete_password(self.SERVICE_NAME, "api_secret")
            except Exception:
                pass
        data = self._load_env_map()
        changed = False
        for field in ("BYBIT_API_KEY", "BYBIT_API_SECRET"):
            if field in data:
                data.pop(field, None)
                changed = True
        if changed:
            self._write_env_map(data)
        bybit_api.configure_client("", "")
        cfg = dict(self.parent_app._cfg)
        cfg.pop("keys_last_check", None)
        cfg.pop("keys_last_status", None)
        save_config(cfg)
        self.parent_app._cfg = cfg
        self._stored_key = None
        self._stored_secret = None
        self._pending_key = None
        self._pending_secret = None
        self._storage_label = "dotenv"
        self.lbl_status.setText("Статус: ключи удалены")
        self._update_labels()

    def _check(self) -> None:
        api_key = (self._pending_key or self._stored_key or "").strip()
        api_secret = (self._pending_secret or self._stored_secret or "").strip()
        if not api_key or not api_secret:
            QMessageBox.warning(self, "Проверка", "Сначала укажи ключ и секрет.")
            return
        self.btn_check.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = bybit_api.ping_credentials(api_key=api_key, api_secret=api_secret)
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_check.setEnabled(True)

        cfg = dict(self.parent_app._cfg)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if result.get("ok"):
            balance = float(result.get("balance") or 0.0)
            msg = f"Статус: ОК (баланс {balance:.2f} USDT)"
            cfg["keys_last_status"] = f"OK / {balance:.2f} USDT"
        else:
            error = str(result.get("error") or "ошибка")
            msg = f"Статус: Ошибка — {error}"
            cfg["keys_last_status"] = f"Ошибка: {error}"
        cfg["keys_last_check"] = ts
        save_config(cfg)
        self.parent_app._cfg = cfg
        self.lbl_status.setText(msg)
        self._update_labels()

# ----------------------------- утилиты -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def ts_to_epoch(ts: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None

def safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # защищаемся от битых строк
                    continue
    except Exception:
        return []
    return rows

def control_file_for(log_path: str) -> str:
    """файл управления рядом с логом"""
    folder = os.path.abspath(os.path.dirname(log_path) or ".")
    return os.path.join(folder, "control.json")

def write_control(log_path: str, payload: Dict[str, Any]) -> None:
    """дописываем команду в control.json (main.py должен читать его раз в N сек)"""
    try:
        cpath = control_file_for(log_path)
        data = []
        if os.path.exists(cpath):
            try:
                with open(cpath, "r", encoding="utf-8") as f:
                    cur = json.load(f)
                    if isinstance(cur, list):
                        data = cur
            except Exception:
                data = []
        payload = dict(payload)
        payload.setdefault("cmd", str(payload.get("cmd") or "custom"))
        payload.setdefault("ts", now_iso())
        payload.setdefault("cmd_id", uuid.uuid4().hex)
        cmd_ids = {str((row or {}).get("cmd_id")) for row in data if isinstance(row, dict)}
        if payload["cmd_id"] in cmd_ids:
            payload["cmd_id"] = uuid.uuid4().hex
        data.append(payload)
        target = pathlib.Path(cpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target)
    except Exception as e:
        print("[CTRL] write_control error:", e, file=sys.stderr)

def fetch_linear_pairs(timeout=6.0) -> List[str]:
    """Пробуем подтянуть список USDT-перпов с Bybit, мягко деградируем."""
    try:
        import urllib.request, json as _json
        url = "https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
        data = _json.loads(raw)
        lst = (((data or {}).get("result") or {}).get("list") or [])
        syms = []
        for it in lst:
            s = (it or {}).get("symbol") or ""
            if s.endswith("USDT"):
                syms.append(s)
        syms = sorted(list(dict.fromkeys(syms)))
        return syms or ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","TONUSDT","DOGEUSDT","ARBUSDT","ADAUSDT"]
    except Exception:
        # дефолтный набор
        return ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","TONUSDT","DOGEUSDT","ARBUSDT","ADAUSDT"]

# ----------------------------- подсветка лога -----------------------------
class LogHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []

        def mk(fmt_fg: str = None, bold=False):
            f = QTextCharFormat()
            if fmt_fg:
                c = QColor(fmt_fg)
                f.setForeground(c)
            if bold:
                f.setFontWeight(QFont.Bold)
            return f

        # ключевые теги
        self.rules.append((re.compile(r"\[(?:ERR|ERROR)\].*"), mk("#ff4d4f", True)))   # ERROR красный
        self.rules.append((re.compile(r"\bTraceback\b.*"),        mk("#ff4d4f")))       # traceback
        self.rules.append((re.compile(r"\[WARN(?:ING)?\].*"),     mk("#fa8c16", True))) # WARNING оранж
        self.rules.append((re.compile(r"\[TRADE\].*"),            mk("#1677ff", True))) # TRADE синий
        self.rules.append((re.compile(r"\[ML\].*"),               mk("#b026ff", True))) # ML фиолет
        self.rules.append((re.compile(r"\[CTRL\].*"),             mk("#13c2c2")))       # CTRL бирюз
        self.rules.append((re.compile(r"\[APP\].*"),              mk("#8c8c8c")))       # APP серый

        # временные метки YYYY-...
        self.rules.append((re.compile(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b"), mk("#595959")))

    def highlightBlock(self, text: str):
        for rx, fmt in self.rules:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)

# ----------------------------- экраны -----------------------------
class MainMenu(QWidget):
    """Главное меню: режимы, пресеты, пары/плечо, файл лога, переходы"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._cfg = load_config()
        self._ignore_schedule_user_defined = False
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                    raw_cfg = json.load(fh)
                if isinstance(raw_cfg, dict) and "ignore_schedule" in raw_cfg:
                    self._ignore_schedule_user_defined = True
        except Exception:
            self._ignore_schedule_user_defined = False
        self._pin_verified_at: Optional[float] = None

        layout = QVBoxLayout()
        self.label = QLabel("<h2>🔥 КриптоБот</h2><p>Запускай и смотри, как он дышит.</p>")
        layout.addWidget(self.label)

        form = QFormLayout()

        # Режимы и пресеты
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["PAPER", "REAL"])
        self.cmb_mode.setCurrentText(str(self._cfg.get("mode", "PAPER")))
        form.addRow("Режим:", self.cmb_mode)

        self.cmb_theme = QComboBox()
        self.cmb_theme.addItems(["Auto", "Dark", "Light"])
        cur_theme = str(self._cfg.get("theme", "Auto") or "Auto")
        if cur_theme.title() not in {"Auto", "Dark", "Light"}:
            cur_theme = "Auto"
        self.cmb_theme.setCurrentText(cur_theme.title())
        form.addRow("Тема оформления:", self.cmb_theme)

        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(["RU", "EN"])
        lang_cur = str(self._cfg.get("log_lang", "RU") or "RU").upper()
        if lang_cur not in {"RU", "EN"}:
            lang_cur = "RU"
        self.cmb_lang.setCurrentText(lang_cur)
        form.addRow("Язык логов:", self.cmb_lang)

        # Пресеты: быстрые кнопки
        preset_box = QHBoxLayout()
        self.btn_preset_scalp = QPushButton("⚡ Скальп (PAPER)")
        self.btn_preset_real = QPushButton("🛡️ Реал (SAFE)")
        self.btn_preset_scalp.setProperty("secondary", True)
        self.btn_preset_real.setProperty("secondary", True)
        preset_box.addWidget(self.btn_preset_scalp)
        preset_box.addWidget(self.btn_preset_real)
        w_preset = QWidget(); w_preset.setLayout(preset_box)
        form.addRow("Пресеты:", w_preset)

        # Безопасность/лог
        self.chk_unsafe = QCheckBox("Отключить SAFE_MODE (добавить --unsafe)")
        self.chk_unsafe.setChecked(bool(self._cfg.get("unsafe", False)))
        self.chk_autoscroll = QCheckBox("Автоскролл лога")
        self.chk_autoscroll.setChecked(bool(self._cfg.get("autoscroll", True)))

        self.le_log = QLineEdit(self._cfg.get("log_path", LOG_DEFAULT))
        self.btn_pick_log = QPushButton("…")
        self.btn_clear_control = QPushButton("Clear control")
        self.btn_clear_control.setProperty("secondary", True)
        log_row = QHBoxLayout()
        log_row.addWidget(self.le_log, 1)
        log_row.addWidget(self.btn_pick_log)
        log_row.addWidget(self.btn_clear_control)
        wrap = QWidget(); wrap.setLayout(log_row)
        form.addRow("Файл лога:", wrap)

        # Пары и плечо (простая форма)
        self.le_pairs = QLineEdit(self._cfg.get("pairs", DEFAULT_CONFIG["pairs"]))
        self.btn_load_pairs = QPushButton("Загрузить пары (Bybit)")
        pairs_row = QHBoxLayout()
        pairs_row.addWidget(self.le_pairs, 1)
        pairs_row.addWidget(self.btn_load_pairs)
        w_pairs = QWidget(); w_pairs.setLayout(pairs_row)
        form.addRow("Пары:", w_pairs)

        self.sp_lev = QSpinBox(); self.sp_lev.setRange(1, 100)
        self.sp_lev.setValue(int(self._cfg.get("default_lev", 5)))
        form.addRow("Плечо по умолчанию:", self.sp_lev)

        layout.addLayout(form)
        layout.addWidget(self.chk_unsafe)
        layout.addWidget(self.chk_autoscroll)
        mode_now = self.cmb_mode.currentText().upper()
        ignore_default = bool(self._cfg.get("ignore_schedule", False))
        if not self._ignore_schedule_user_defined:
            ignore_default = (mode_now == "PAPER")
        self.chk_ignore_schedule = QCheckBox("Игнорировать расписание (24/7)")
        self.chk_ignore_schedule.setChecked(ignore_default)
        layout.addWidget(self.chk_ignore_schedule)

        # Кнопки перехода
        btns = QHBoxLayout()
        self.btn_start = QPushButton("🛠 За работу")
        self.btn_report = QPushButton("📊 Отчёты")
        self.btn_tools = QPushButton("🧰 Инструменты (ML)")
        self.btn_report.setProperty("secondary", True)
        self.btn_tools.setProperty("secondary", True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_report)
        btns.addWidget(self.btn_tools)
        self.btn_apk = None
        if getattr(self.parent, "enable_apk", False):
            self.btn_apk = QPushButton("📦 APK Manager")
            self.btn_apk.setProperty("secondary", True)
            btns.addWidget(self.btn_apk)
        layout.addLayout(btns)

        self.setLayout(layout)

        # wiring
        self.btn_start.clicked.connect(self.on_start)
        self.btn_report.clicked.connect(lambda: self.parent.goto_screen("report"))
        self.btn_tools.clicked.connect(lambda: self.parent.goto_screen("tools"))
        if self.btn_apk:
            self.btn_apk.clicked.connect(lambda: self.parent.goto_screen("apk"))
        self.btn_pick_log.clicked.connect(self.pick_log)
        self.btn_preset_scalp.clicked.connect(self.apply_preset_scalp)
        self.btn_preset_real.clicked.connect(self.apply_preset_real)
        self.btn_load_pairs.clicked.connect(self.load_pairs_from_exchange)
        self.btn_clear_control.clicked.connect(self.clear_control_file)

        # авто-сейв конфига при изменениях
        self.cmb_mode.currentTextChanged.connect(self.on_mode_change)
        self.chk_unsafe.toggled.connect(self._save_cfg)
        self.chk_autoscroll.toggled.connect(self._save_cfg)
        self.chk_ignore_schedule.toggled.connect(self._save_cfg)
        self.le_log.textChanged.connect(self._save_cfg)
        self.le_pairs.textChanged.connect(self._save_cfg)
        self.sp_lev.valueChanged.connect(self._save_cfg)
        self.cmb_theme.currentTextChanged.connect(self.on_theme_change)
        self.cmb_lang.currentTextChanged.connect(self._save_cfg)

    def _save_cfg(self, *args):
        self._cfg.update({
            "mode": self.cmb_mode.currentText(),
            "unsafe": bool(self.chk_unsafe.isChecked()),
            "autoscroll": bool(self.chk_autoscroll.isChecked()),
            "ignore_schedule": bool(self.chk_ignore_schedule.isChecked()),
            "log_path": self.le_log.text().strip() or LOG_DEFAULT,
            "pairs": self.le_pairs.text().strip(),
            "default_lev": int(self.sp_lev.value()),
            "theme": self.cmb_theme.currentText(),
            "log_lang": self.cmb_lang.currentText().upper(),
        })
        self._ignore_schedule_user_defined = True
        save_config(self._cfg)

    def on_mode_change(self, value: str):
        mode = str(value or "").upper()
        if mode == "PAPER":
            self.chk_ignore_schedule.setChecked(True)
        elif mode == "REAL":
            self.chk_ignore_schedule.setChecked(False)
        self._save_cfg()

    def _show_status_message(self, text: str, timeout: int = 4000):
        parent = getattr(self, "parent", None)
        if parent is not None and hasattr(parent, "statusBar"):
            bar = parent.statusBar()
            if bar is not None:
                bar.showMessage(str(text), timeout)

    def clear_control_file(self):
        log_path = self.le_log.text().strip() or LOG_DEFAULT
        cpath = pathlib.Path(control_file_for(log_path))
        lock_path = cpath.parent / f"{cpath.name}.lock"
        try:
            if cpath.exists():
                cpath.unlink()
            if lock_path.exists():
                lock_path.unlink()
            self._show_status_message("control.json очищен")
        except Exception as e:
            self._show_status_message(f"Не удалось очистить control.json: {e}")

    def gather_launch_options(self) -> Dict[str, Any]:
        mode = self.cmb_mode.currentText()
        unsafe = self.chk_unsafe.isChecked()
        autoscroll = self.chk_autoscroll.isChecked()
        log_path = self.le_log.text().strip() or LOG_DEFAULT
        pairs = [s.strip() for s in self.le_pairs.text().split(",") if s.strip()]
        default_lev = int(self.sp_lev.value())
        return {
            "mode": mode,
            "unsafe": unsafe,
            "autoscroll": autoscroll,
            "ignore_schedule": bool(self.chk_ignore_schedule.isChecked()),
            "log_path": log_path,
            "pairs": pairs,
            "default_lev": default_lev,
            "log_lang": self.cmb_lang.currentText().upper(),
        }

    def _ensure_real_security(self) -> bool:
        require_pin = bool(env_bool("ENABLE_PIN_FOR_REAL", getattr(config, "ENABLE_PIN_FOR_REAL", True)))
        tg_required = bool(env_bool("ENABLE_TG_2FA", getattr(config, "ENABLE_TG_2FA", False)))
        try:
            ttl = int(os.getenv("TG_2FA_TTL", str(getattr(config, "TG_2FA_TTL", 300))) or 300)
        except Exception:
            ttl = 300
        reuse_window = min(300, ttl) if tg_required else 300
        now_ts = time.time()
        if require_pin:
            if self._pin_verified_at and now_ts - self._pin_verified_at < reuse_window:
                return True
            stored_hash = os.getenv("PIN_HASH") or getattr(config, "PIN_HASH", "")
            if not stored_hash:
                QMessageBox.warning(
                    self,
                    "Безопасность",
                    "PIN не задан. Установи переменную PIN_HASH (sha256) перед запуском REAL.",
                )
                log("[SECURITY_EVENT] REAL blocked: PIN_HASH not configured")
                return False
            pin, ok = QInputDialog.getText(
                self,
                "PIN",
                "Введите PIN для включения REAL:",
                QLineEdit.Password,
            )
            if not ok:
                log("[SECURITY_EVENT] REAL cancelled: PIN dialog closed")
                return False
            if not verify_pin_hash(str(pin), stored_hash):
                log("[SECURITY_EVENT] REAL denied: invalid PIN")
                QMessageBox.critical(self, "PIN", "Неверный PIN. Переход в REAL запрещён.")
                return False
            if tg_required:
                code = f"{random.randint(0, 999999):06d}"
                tg_send(
                    "<b>TradeApp 2FA</b>\n"
                    f"Код: <code>{code}</code>\n"
                    f"Действителен {ttl} секунд."
                )
                start_ts = time.time()
                entered, ok = QInputDialog.getText(self, "2FA", "Введите код из Telegram:")
                if not ok:
                    log("[SECURITY_EVENT] REAL cancelled: 2FA dialog closed")
                    return False
                if time.time() - start_ts > ttl or str(entered).strip() != code:
                    log("[SECURITY_EVENT] REAL denied: 2FA failed")
                    QMessageBox.critical(self, "2FA", "Неверный или истёкший код.")
                    return False
        self._pin_verified_at = now_ts
        log("[SECURITY_EVENT] REAL security check passed")
        return True

    def ask_real_launch(self, log_path: str, pairs: List[str], default_lev: int, safe_default: bool = True) -> Optional[bool]:
        dlg = QDialog(self)
        dlg.setWindowTitle("Подтверждение REAL")
        layout = QVBoxLayout(dlg)
        pairs_txt = ", ".join(pairs) if pairs else "—"
        summary = QLabel(
            "<b>Режим REAL</b><br>"
            f"Лог: {log_path}<br>"
            f"Пары: {pairs_txt}<br>"
            f"Плечо: {default_lev}x"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        safe_checkbox = QCheckBox("SAFE_MODE (ордеры не отправляются)")
        safe_checkbox.setChecked(bool(safe_default))
        layout.addWidget(safe_checkbox)

        warning = QLabel("Подтверди запуск. SAFE_MODE=0 отправит реальные ордера.")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Старт")
        buttons.button(QDialogButtonBox.Cancel).setText("Отмена")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return None
        return bool(safe_checkbox.isChecked())

    def on_theme_change(self, value: str):
        value = str(value or "Auto").title()
        prev = self._cfg.get("theme")
        self._save_cfg()
        if value != prev:
            try:
                self.parent.apply_theme(value)
            except AttributeError:
                pass

    def pick_log(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Выбери/задай файл лога", self.le_log.text(),
            "JSON Lines (*.jsonl);;All files (*.*)"
        )
        if fn:
            self.le_log.setText(fn)

    def apply_preset_scalp(self):
        self.cmb_mode.setCurrentText("PAPER")
        self.chk_unsafe.setChecked(False)
        self.sp_lev.setValue(5)
        # типичные ликвидные пары
        self.le_pairs.setText("BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,ARBUSDT")

    def apply_preset_real(self):
        self.cmb_mode.setCurrentText("REAL")
        self.chk_unsafe.setChecked(False)
        self.sp_lev.setValue(3)

    def load_pairs_from_exchange(self):
        syms = fetch_linear_pairs()
        if syms:
            self.le_pairs.setText(",".join(syms[:20]))

    def on_start(self):
        options = self.gather_launch_options()
        mode = options["mode"]
        unsafe = options["unsafe"]
        autoscroll = options["autoscroll"]
        ignore_schedule = options["ignore_schedule"]
        log_path = options["log_path"]
        pairs = options["pairs"]
        default_lev = options["default_lev"]
        log_lang = options["log_lang"]

        if mode.upper() == "REAL":
            if not self._ensure_real_security():
                self.cmb_mode.setCurrentText("PAPER")
                return
            safe_choice = self.ask_real_launch(log_path, pairs, default_lev, safe_default=not unsafe)
            if safe_choice is None:
                log("[SECURITY_EVENT] REAL launch cancelled after dialog")
                return
            unsafe = not safe_choice
            self.chk_unsafe.setChecked(unsafe)

        # перед стартом положим команды управления (пары/плечо)
        try:
            write_control(log_path, {"cmd": "set_pairs", "set_pairs": pairs, "default_lev": default_lev})
        except Exception:
            pass

        self.parent.run_screen.set_options(
            mode=mode, unsafe=unsafe, autoscroll=autoscroll, log_path=log_path,
            preset_pairs=pairs, default_lev=default_lev, ignore_schedule=ignore_schedule,
            log_lang=log_lang
        )
        self.parent.report_screen.set_log_path(log_path)
        self.parent.goto_screen("run")
        self.parent.run_screen.start_bot()

class RunScreen(QWidget):
    """Экран запуска/управления: поток лога, паника/пауза, фильтры"""
    LINE_RX_PAIRS = re.compile(r"\[PAIRS\]\s+Работаем с:\s*(.+)")
    LINE_RX_LEV   = re.compile(r"\[LEV\]\s+([A-Z0-9_]+):\s*([0-9]+)x")
    LINE_RX_MODE  = re.compile(r"Старт\s+\[(PAPER|REAL)\]")

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.proc: Optional[QProcess] = None
        self.autoscroll = True
        self.log_path = LOG_DEFAULT
        self.mode = "PAPER"
        self.unsafe = False
        self.safe_mode = True
        self.ignore_schedule = False
        self.pairs: List[str] = []
        self.default_lev = 5
        self.leverage: Dict[str, int] = {}  # symbol -> int
        self.last_line_ts: Optional[float] = None
        self.session_started_at: Optional[str] = None
        self.session_started_ts: Optional[float] = None
        self.session_lookup_iso: Optional[str] = None
        self.filter_error = False
        self.filter_trade = False
        self.filter_ml = False
        self.entries_paused = False
        self.log_lang = "RU"
        self.user_requested_stop = False
        self.stop_requested_at: Optional[float] = None
        self.forced_kill = False
        self.received_stats_line = False
        self.stats_data: Optional[Dict[str, Any]] = None
        self.stats_session_dir: Optional[str] = None
        self.stats_shown = False
        self.latest_stats_mtime: Optional[float] = None
        self.working_dir = os.path.abspath(os.path.dirname(MAIN_PY) or ".")

        self.stats_watch_timer = QTimer(self)
        self.stats_watch_timer.setInterval(1000)
        self.stats_watch_timer.timeout.connect(self.poll_session_stats)
        self.stop_notice_timer = QTimer(self)
        self.stop_notice_timer.setSingleShot(True)
        self.stop_notice_timer.timeout.connect(self._show_stop_notice)
        self.stop_notice_shown = False
        self.stop_prompt_shown = False
        self.stop_poll_timer = QTimer(self)
        self.stop_poll_timer.setInterval(250)
        self.stop_poll_timer.timeout.connect(self._check_stop_completion)
        self.stop_prompt_timer = QTimer(self)
        self.stop_prompt_timer.setSingleShot(True)
        self.stop_prompt_timer.timeout.connect(self._ask_for_panic)
        self.panic_timer = QTimer(self)
        self.panic_timer.setSingleShot(True)
        self.panic_timer.timeout.connect(self._force_kill_if_alive)

        layout = QVBoxLayout()
        layout.setSpacing(12)

        header = QHBoxLayout()
        lbl_title = QLabel("<b>Командный пункт</b>")
        lbl_hint = QLabel("Настрой запуск на главном экране, затем контролируй сессию здесь.")
        lbl_hint.setObjectName("SecondaryLabel")
        self.lbl_pause_badge = QLabel("ПАУЗА")
        self.lbl_pause_badge.setVisible(False)
        self.lbl_pause_badge.setStyleSheet(
            "QLabel {"
            "background-color: #faad14;"
            "color: #000;"
            "padding: 2px 8px;"
            "border-radius: 8px;"
            "font-weight: 700;"
            "}"
        )
        self.btn_help = QPushButton("?")
        self.btn_help.setFixedWidth(32)
        self.btn_help.setProperty("secondary", True)
        header.addWidget(lbl_title)
        header.addSpacing(8)
        header.addWidget(lbl_hint, 1)
        header.addWidget(self.lbl_pause_badge, 0, Qt.AlignRight)
        header.addWidget(self.btn_help, 0, Qt.AlignRight)
        layout.addLayout(header)

        # Статусная строка
        status_frame = QFrame()
        status_frame.setObjectName("StatusFrame")
        top = QHBoxLayout(status_frame)
        top.setContentsMargins(12, 8, 12, 8)
        top.setSpacing(16)
        self.lbl_status  = QLabel("Статус: <b>остановлен</b>")
        self.lbl_mode    = QLabel("Режим: —")
        self.lbl_pairs   = QLabel("Пары: —")
        self.lbl_lev     = QLabel("Плечо: —")
        self.lbl_margin  = QLabel("Маржа: —")
        self.lbl_margin.setObjectName("SecondaryLabel")
        self.lbl_ml_state = QLabel("ML: —")
        self.lbl_ml_state.setObjectName("SecondaryLabel")
        self.lbl_session = QLabel("Сессия: —")
        self.lbl_session.setObjectName("SecondaryLabel")
        top.addWidget(self.lbl_status, 1)
        top.addWidget(self.lbl_mode, 1)
        top.addWidget(self.lbl_pairs, 2)
        top.addWidget(self.lbl_lev, 1)
        top.addWidget(self.lbl_margin, 1)
        top.addWidget(self.lbl_ml_state, 1)
        top.addWidget(self.lbl_session, 1)
        layout.addWidget(status_frame)

        # Лог
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas", 10))
        self.txt.setObjectName("LogView")
        layout.addWidget(self.txt, 1)
        # Подсветка лога
        self._highlighter = LogHighlighter(self.txt.document())

        # Кнопки управления (верхний ряд)
        btns = QHBoxLayout()
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_panic = QPushButton("🛑 PANIC")
        self.btn_pause = QPushButton("⏸ Пауза сигналов")
        self.btn_update_pairs = QPushButton("🔁 Обновить пары/плечо")
        self.btn_stop.setProperty("destructive", True)
        self.btn_panic.setProperty("destructive", True)
        self.btn_pause.setProperty("secondary", True)
        self.btn_update_pairs.setProperty("secondary", True)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_panic)
        btns.addWidget(self.btn_pause)
        btns.addWidget(self.btn_update_pairs)
        layout.addLayout(btns)

        # Нижняя панель: фильтры/лог/репорты
        btns2 = QHBoxLayout()
        self.btn_clear = QPushButton("🧹 Очистить лог-файл")
        self.btn_open_folder = QPushButton("📂 Папка лога")
        self.btn_open_log_file = QPushButton("📄 Открыть лог")
        self.btn_copy_tail = QPushButton("📋 Копировать 200 строк")
        self.btn_open_control = QPushButton("📝 control.json")
        self.btn_report = QPushButton("📊 Отчёты")
        for btn in (self.btn_clear, self.btn_open_folder, self.btn_open_log_file,
                    self.btn_copy_tail, self.btn_open_control, self.btn_report):
            btn.setProperty("secondary", True)
        btns2.addWidget(self.btn_clear)
        btns2.addWidget(self.btn_open_folder)
        btns2.addWidget(self.btn_open_log_file)
        btns2.addWidget(self.btn_copy_tail)
        btns2.addWidget(self.btn_open_control)
        btns2.addStretch(1)
        btns2.addWidget(self.btn_report)
        layout.addLayout(btns2)

        # Переключатели/heartbeat
        bot = QHBoxLayout()
        self.chk_autoscroll = QCheckBox("Автоскролл"); self.chk_autoscroll.setChecked(True)
        self.chk_only_error = QCheckBox("только [ERROR]")
        self.chk_only_trade = QCheckBox("только [TRADE]")
        self.chk_only_ml = QCheckBox("только [ML]")
        self.lbl_heartbeat = QLabel("Линии: 0")
        self.lbl_heartbeat.setObjectName("SecondaryLabel")
        bot.addWidget(self.chk_autoscroll)
        bot.addWidget(self.chk_only_error)
        bot.addWidget(self.chk_only_trade)
        bot.addWidget(self.chk_only_ml)
        bot.addStretch(1)
        bot.addWidget(self.lbl_heartbeat)
        layout.addLayout(bot)

        self.setLayout(layout)

        # Триггеры
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_panic.clicked.connect(self.on_panic)
        self.btn_pause.clicked.connect(self.toggle_pause_resume)
        self.btn_update_pairs.clicked.connect(self.update_pairs_control)
        self.btn_clear.clicked.connect(self.clear_log_file)
        self.btn_open_folder.clicked.connect(self.open_log_folder)
        self.btn_open_log_file.clicked.connect(self.open_log_file)
        self.btn_copy_tail.clicked.connect(self.copy_tail)
        self.btn_open_control.clicked.connect(self.open_control_file)
        self.btn_report.clicked.connect(lambda: self.parent.goto_screen("report"))
        self.chk_autoscroll.toggled.connect(lambda v: setattr(self, "autoscroll", v))
        self.chk_only_error.toggled.connect(lambda v: setattr(self, "filter_error", v))
        self.chk_only_trade.toggled.connect(lambda v: setattr(self, "filter_trade", v))
        self.chk_only_ml.toggled.connect(lambda v: setattr(self, "filter_ml", v))
        self.btn_help.clicked.connect(self.show_command_help)

        # heartbeat
        self.lines_total = 0
        self.t_heartbeat = QTimer(self); self.t_heartbeat.setInterval(1000)
        self.t_heartbeat.timeout.connect(self.tick_heartbeat)
        self.t_heartbeat.start()

        self.margin_frozen = False
        self.margin_timer = QTimer(self)
        margin_interval = max(3, int(getattr(config, "MARGIN_POLL_SEC", 15))) * 1000
        self.margin_timer.setInterval(margin_interval)
        self.margin_timer.timeout.connect(self.refresh_margin_status)
        self.margin_timer.timeout.connect(self.refresh_ml_status)
        self.margin_timer.start()
        self.refresh_margin_status()
        self.refresh_ml_status()

        # хоткеи экрана Run
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_pause_resume)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self.on_panic)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.on_stop)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.copy_tail)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=lambda: self.parent.goto_screen("report"))

    # --- внешние установки из MainMenu ---
    def set_options(self, mode: str, unsafe: bool, autoscroll: bool, log_path: str,
                    preset_pairs: List[str], default_lev: int, ignore_schedule: bool,
                    log_lang: str):
        self.mode = mode
        self.unsafe = unsafe
        if self.mode.upper() == "PAPER":
            self.safe_mode = True
        else:
            self.safe_mode = not bool(self.unsafe)
        self.autoscroll = autoscroll
        self.chk_autoscroll.setChecked(autoscroll)
        self.log_path = log_path
        self.pairs = list(preset_pairs or [])
        self.default_lev = int(default_lev)
        self.ignore_schedule = bool(ignore_schedule)
        self.log_lang = str(log_lang or "RU").upper()
        safe_label = f"SAFE_MODE={1 if self.safe_mode else 0}"
        color = "#52c41a" if self.safe_mode else "#ff4d4f"
        self.lbl_mode.setText(f'Режим: <b>{self.mode}</b> / <span style="color:{color}">{safe_label}</span>')

    # --- запуск подпроцесса ---
    def start_bot(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Уже запущен", "Бот уже бежит.")
            return

        # каталог для лога
        try:
            lp = pathlib.Path(self.log_path)
            if lp.parent and not lp.parent.exists():
                lp.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # проверим main.py
        main_path = os.path.abspath(MAIN_PY)
        if not os.path.exists(main_path):
            QMessageBox.critical(self, "Файл не найден",
                                 f"Не нашёл MAIN_PY: {main_path}\nПоправь константу MAIN_PY.")
            return
        workdir = os.path.dirname(main_path) or os.getcwd()

        self.txt.clear()
        self.leverage.clear()
        self.lines_total = 0
        self.last_line_ts = None
        self.session_started_at = now_iso()
        self.session_lookup_iso = self.session_started_at
        self.session_started_ts = time.time()
        self.lbl_session.setText(f"Сессия: {self.session_started_at}")
        self.lbl_status.setText("Статус: <b>запуск…</b>")
        self.lbl_heartbeat.setText("Линии: 0  |  последний лог: —")
        self.lbl_mode.setText("Режим: —")
        self.lbl_pairs.setText("Пары: —")
        self.lbl_lev.setText("Плечо: —")
        self.user_requested_stop = False
        self.stop_requested_at = None
        self.forced_kill = False
        self.received_stats_line = False
        self.stats_data = None
        self.stats_session_dir = None
        self.stats_shown = False
        self.latest_stats_mtime = None
        self.stop_notice_shown = False
        self.stop_prompt_shown = False
        self.entries_paused = False
        self.set_pause_state(False, update_ctrl=False)
        if self.stop_notice_timer.isActive():
            self.stop_notice_timer.stop()
        if self.stop_prompt_timer.isActive():
            self.stop_prompt_timer.stop()
        if self.stop_poll_timer.isActive():
            self.stop_poll_timer.stop()
        if self.panic_timer.isActive():
            self.panic_timer.stop()
        if not self.stats_watch_timer.isActive():
            self.stats_watch_timer.start()

        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(workdir)
        self.proc.setProgram(sys.executable)

        # Включаем UTF-8 и пробрасываем LOG_JSONL
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("LANG", "C.UTF-8")
        env.insert("LC_ALL", "C.UTF-8")
        env.insert("LOG_JSONL", os.path.abspath(self.log_path))
        env.insert("LOG_RU", "1" if self.log_lang.upper() == "RU" else "0")
        env.insert("PAPER_MODE", "1" if self.mode.upper() == "PAPER" else "0")
        env.insert("SAFE_MODE", "1" if self.safe_mode else "0")
        env.insert("MAX_ACTIVE_PAIRS", "2")
        env.insert("ALLOWED_PAIRS", "ETHUSDT,SOLUSDT")
        if self.ignore_schedule:
            env.insert("EXCLUDE_WEEKENDS", "0")
            env.insert("TRADE_HOURS_LOCAL", "00:00-24:00")
            env.insert("FORCE_SCHEDULE_OFF", "1")
        else:
            env.insert("FORCE_SCHEDULE_OFF", "0")
            if env.contains("EXCLUDE_WEEKENDS"):
                env.remove("EXCLUDE_WEEKENDS")
            if env.contains("TRADE_HOURS_LOCAL"):
                env.remove("TRADE_HOURS_LOCAL")
        self.proc.setProcessEnvironment(env)

        # python -X utf8 main.py [paper|real ...]
        args = ["-X", "utf8", main_path]
        if self.mode.upper() == "PAPER":
            args.append("paper")
        else:
            args += ["real", "--yes"]
            if not self.safe_mode:
                args.append("--unsafe")
        self.proc.setArguments(args)

        # stderr → stdout
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_ready)
        self.proc.started.connect(lambda: self.lbl_status.setText("Статус: <b>запущен</b>"))
        self.proc.errorOccurred.connect(self.on_process_error)
        self.proc.finished.connect(self.on_finished)

        self.append_line(f"[APP] ▶ Стартую: {sys.executable} {' '.join(args)}")
        self.proc.start()

        # Перед стартом обновим статусные лейблы по пресетам
        self.lbl_pairs.setText(f"Пары: {', '.join(self.pairs) if self.pairs else '—'}")
        self.lbl_lev.setText(f"Плечо: {self.default_lev}x (дефолт)")

    def on_ready(self):
        try:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            data = ""
        if not data:
            return
        for raw in data.splitlines():
            line = raw.rstrip()
            self.parse_status(line)
            if self.passes_filter(line):
                self.append_line(line)

    def passes_filter(self, line: str) -> bool:
        if not (self.filter_error or self.filter_trade or self.filter_ml):
            return True
        ok = True
        if self.filter_error:
            ok = ok and ("[ERR" in line or "ERROR" in line or "Traceback" in line)
        if self.filter_trade:
            ok = ok and ("[TRADE]" in line or "close" in line or "TP" in line or "SL" in line)
        if self.filter_ml:
            ok = ok and ("[ML]" in line or "ml_veto" in line or "prob=" in line)
        return ok

    def on_process_error(self, error: QProcess.ProcessError):
        if self.user_requested_stop:
            return
        hints = {
            QProcess.FailedToStart: "не удалось запустить Python или main.py",
            QProcess.Crashed: "процесс упал сразу после старта",
            QProcess.Timedout: "таймаут при запуске процесса",
            QProcess.WriteError: "ошибка записи в stdin процесса",
            QProcess.ReadError: "ошибка чтения вывода процесса",
        }
        hint = hints.get(error, "неизвестная ошибка запуска")
        self.append_line(f"[APP] ❌ Ошибка запуска: {hint}")
        QMessageBox.critical(
            self,
            "Ошибка запуска",
            "Не получилось стартовать main.py (режим {0}).\n{1}.".format(self.mode, hint),
        )

    def on_finished(self, code, status):
        self.append_line(f"[APP] ⏹ Завершено. code={code}, status={status}")
        self.lbl_status.setText("Статус: <b>остановлен</b>")

        if self.stop_notice_timer.isActive():
            self.stop_notice_timer.stop()
        if self.stop_prompt_timer.isActive():
            self.stop_prompt_timer.stop()
        if self.stop_poll_timer.isActive():
            self.stop_poll_timer.stop()
        if self.panic_timer.isActive():
            self.panic_timer.stop()

        elapsed = None
        if self.stop_requested_at is not None:
            elapsed = time.time() - self.stop_requested_at
        elif self.session_started_ts is not None:
            elapsed = time.time() - self.session_started_ts

        if not self.session_lookup_iso:
            self.session_lookup_iso = self.session_started_at
        self.session_started_ts = None
        self.session_started_at = None
        self.lbl_session.setText("Сессия: —")
        self.stop_prompt_shown = False

        user_stop = self.user_requested_stop or (self.stop_requested_at is not None)
        if status != QProcess.NormalExit and code == 0:
            user_stop = True

        if user_stop:
            if code != 0 and status != QProcess.NormalExit and not self.forced_kill:
                QMessageBox.information(
                    self,
                    "Остановлено",
                    "Процесс был завершён пользователем (принудительно).",
                )
            if not self.stop_notice_shown:
                self.stop_notice_timer.start(1500)
        else:
            if code != 0:
                if elapsed is not None and elapsed < 3:
                    QMessageBox.critical(
                        self,
                        "Ошибка запуска",
                        "Процесс завершился слишком быстро. Проверь конфигурацию и логи.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Процесс завершился с ошибкой",
                        f"Код возврата: {code}\nПроверь последние строки лога.",
                    )

        self.poll_session_stats()

        if self.proc:
            self.proc.deleteLater()
            self.proc = None

    def on_stop(self):
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            self.parent.goto_screen("main")
            return
        self.append_line("[CTRL] stop=True → control.json")
        self.send_ctrl({"cmd": "stop", "stop": True})
        self.user_requested_stop = True
        self.stop_requested_at = time.time()
        self.stop_prompt_shown = False
        self.lbl_status.setText("Статус: <b>останавливаю…</b>")
        if self.stop_poll_timer.isActive():
            self.stop_poll_timer.stop()
        self.stop_poll_timer.start()
        if self.stop_prompt_timer.isActive():
            self.stop_prompt_timer.stop()
        self.stop_prompt_timer.start(int(STOP_TIMEOUT_SEC * 1000))

    def toggle_pause_resume(self):
        self.set_pause_state(not self.entries_paused)

    def set_pause_state(self, paused: bool, update_ctrl: bool = True):
        paused = bool(paused)
        if paused == self.entries_paused and update_ctrl:
            # ничего не меняем
            return
        self.entries_paused = paused
        if paused:
            self.btn_pause.setText("▶ Возобновить")
            self.lbl_pause_badge.setVisible(True)
            if update_ctrl:
                self.send_ctrl({"cmd": "pause_on", "pause_entries": True})
                self.append_line("[CTRL] pause_entries=True → control.json")
        else:
            self.btn_pause.setText("⏸ Пауза сигналов")
            self.lbl_pause_badge.setVisible(False)
            if update_ctrl:
                self.send_ctrl({"cmd": "pause_off", "pause_entries": False})
                self.append_line("[CTRL] pause_entries=False → control.json")

    def _check_stop_completion(self):
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            if self.stop_poll_timer.isActive():
                self.stop_poll_timer.stop()
            if self.stop_prompt_timer.isActive():
                self.stop_prompt_timer.stop()
            return
        if not self.stop_requested_at:
            if self.stop_poll_timer.isActive():
                self.stop_poll_timer.stop()
            return
        if time.time() - self.stop_requested_at >= STOP_TIMEOUT_SEC and not self.stop_prompt_shown:
            self.stop_poll_timer.stop()
            self.stop_prompt_shown = True
            if self.stop_prompt_timer.isActive():
                self.stop_prompt_timer.stop()
            self._ask_for_panic()

    def _ask_for_panic(self):
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("main.py всё ещё работает")
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            "Подпроцесс не завершился в течение {0} секунд.".format(int(STOP_TIMEOUT_SEC))
        )
        msg.setInformativeText("Перейти к аварийному завершению?")
        panic_btn = msg.addButton("Авария", QMessageBox.DestructiveRole)
        wait_btn = msg.addButton("Подождать", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == panic_btn:
            self.on_panic()
        else:
            self.stop_prompt_shown = False
            self.stop_requested_at = time.time()
            self.stop_poll_timer.start()
            self.stop_prompt_timer.start(int(STOP_TIMEOUT_SEC * 1000))

    def _force_kill_if_alive(self):
        if not self.proc:
            return
        if self.proc.state() != QProcess.NotRunning:
            self.append_line("[APP] ⚠️ Жёстко завершаю подпроцесс после аварии…")
            self.forced_kill = True
            self.proc.kill()

    def on_panic(self):
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            self.append_line("[APP] Процесс уже остановлен — авария не требуется.")
            return
        self.append_line("[CTRL] cmd=panic_stop → control.json")
        self.send_ctrl({"cmd": "panic_stop", "panic": True, "close_all": True})
        self.user_requested_stop = True
        self.stop_requested_at = time.time()
        self.stop_prompt_shown = True
        self.lbl_status.setText("Статус: <b>аварийное завершение…</b>")
        if self.stop_prompt_timer.isActive():
            self.stop_prompt_timer.stop()
        if self.stop_poll_timer.isActive():
            self.stop_poll_timer.stop()
        if self.panic_timer.isActive():
            self.panic_timer.stop()
        self.panic_timer.start(int(PANIC_WAIT_SEC * 1000))

    def update_pairs_control(self):
        # перечитываем пары/плечо из MainMenu через parent
        pairs = self.parent.screens["main"].le_pairs.text()
        pairs = [s.strip() for s in pairs.split(",") if s.strip()]
        lev = int(self.parent.screens["main"].sp_lev.value())
        self.pairs = pairs
        self.default_lev = lev
        self.lbl_pairs.setText(f"Пары: {', '.join(self.pairs) if self.pairs else '—'}")
        self.lbl_lev.setText(f"Плечо: {self.default_lev}x (дефолт)")
        self.send_ctrl({"cmd": "set_pairs", "set_pairs": pairs, "default_lev": lev})
        self.append_line("[CTRL] обновил пары/плечо через control.json")

    def send_ctrl(self, payload: Dict[str, Any]):
        try:
            write_control(self.log_path, payload)
        except Exception as e:
            self.append_line(f"[CTRL] Ошибка записи control.json: {e}")

    def show_command_help(self):
        cpath = control_file_for(self.log_path)
        html = """
        <h3>Командный пункт</h3>
        <p>Этот экран помогает мягко управлять текущей сессией main.py.</p>
        <h4>Кнопки</h4>
        <ul>
            <li><b>⏹ Остановить</b> — записывает <code>{{"stop": true}}</code> в control.json и ждёт завершения процесса.</li>
            <li><b>🛑 Авария</b> — мгновенно отправляет <code>{{"panic": true, "close_all": true}}</code> и принудительно гасит подпроцесс.</li>
            <li><b>⏸ / ▶</b> — переключает паузу входов (флаг <code>pause_entries</code>).</li>
            <li><b>🔁 Обновить пары/плечо</b> — записывает выбранные пары и плечо в control.json.</li>
        </ul>
        <h4>Горячие клавиши</h4>
        <ul>
            <li><b>Space</b> — пауза/возобновление сигналов.</li>
            <li><b>Ctrl+S</b> — мягкая остановка.</li>
            <li><b>Ctrl+P</b> — аварийное завершение.</li>
            <li><b>Ctrl+C</b> — копия последних строк лога.</li>
            <li><b>Ctrl+E</b> — перейти во вкладку «Отчёты».</li>
        </ul>
        <h4>Файлы</h4>
        <ul>
            <li>Лог текущей сессии: <code>{log}</code></li>
            <li>Файл управления: <code>{ctrl}</code></li>
            <li>Папка отчётов: <code>{sessions}</code></li>
        </ul>
        <h4>FAQ</h4>
        <ul>
            <li>Если процесс не завершается, приложение предложит аварийное завершение автоматически.</li>
            <li>Историю сессий можно открыть через кнопку «📊 Отчёты» → вкладка «История сессий».</li>
            <li>Логи и control.json сохраняются в UTF-8, удобно открывать в любом редакторе.</li>
        </ul>
        """.format(log=os.path.abspath(self.log_path or LOG_DEFAULT), ctrl=os.path.abspath(cpath),
                   sessions=os.path.abspath(get_sessions_root()))

        box = QMessageBox(self)
        box.setWindowTitle("Справка: Командный пункт")
        box.setIcon(QMessageBox.Information)
        box.setTextFormat(Qt.RichText)
        box.setStandardButtons(QMessageBox.Ok)
        box.setText(html)
        box.exec_()

    def clear_log_file(self):
        if not self.log_path:
            return
        if os.path.exists(self.log_path):
            ask = QMessageBox.question(self, "Подтвердить очистку",
                                       f"Удалить лог-файл?\n{self.log_path}",
                                       QMessageBox.Yes | QMessageBox.No)
            if ask != QMessageBox.Yes:
                return
            try:
                os.remove(self.log_path)
                self.append_line(f"[APP] Лог очищен: {self.log_path}")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось удалить лог: {e}")
        else:
            self.append_line("[APP] Лога ещё нет — нечего чистить.")

    def open_log_folder(self):
        if not self.log_path:
            return
        folder = os.path.abspath(os.path.dirname(self.log_path) or ".")
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def open_log_file(self):
        if not self.log_path:
            return
        path = os.path.abspath(self.log_path)
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(self, "Лог", f"Файл ещё не создан:\n{path}")

    def open_control_file(self):
        cpath = control_file_for(self.log_path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(cpath)))

    def copy_tail(self):
        text = self.txt.toPlainText().splitlines()[-200:]
        QApplication.clipboard().setText("\n".join(text))
        self.append_line("[APP] Скопировал последние 200 строк в буфер.")

    # --- парсер статуса из строк бота ---
    def parse_status(self, line: str):
        try:
            if "[STATS]" in line:
                self._handle_stats_line(line)
            # пары
            m = self.LINE_RX_PAIRS.search(line)
            if m:
                txt = m.group(1)
                # безопасный разбор: без eval
                txt2 = txt.strip().strip("[]")
                arr = [x.strip().strip("'\"") for x in txt2.split(",") if x.strip()]
                self.pairs = list(map(str, arr))
                self.lbl_pairs.setText(f"Пары: {', '.join(self.pairs) if self.pairs else '—'}")
                return

            # плечо
            m = self.LINE_RX_LEV.search(line)
            if m:
                sym, lev = m.group(1), m.group(2)
                self.leverage[sym] = int(lev)
                pretty = ", ".join(f"{k}:{v}x" for k, v in self.leverage.items()) if self.leverage else "—"
                self.lbl_lev.setText(f"Плечо: {pretty}")
                return

            # режим
            m = self.LINE_RX_MODE.search(line)
            if m:
                self.mode = m.group(1)
                unsafe_txt = "UNSAFE" if self.unsafe else "SAFE"
                color = "#ff4d4f" if self.unsafe else "#52c41a"
                self.lbl_mode.setText(f'Режим: <b>{self.mode}</b> / <span style="color:{color}">{unsafe_txt}</span>')
                return
        except Exception:
            pass  # не ломаем UI при парсинге

    def append_line(self, line: str):
        self.lines_total += 1
        self.txt.appendPlainText(line)
        self.last_line_ts = time.time()
        if self.autoscroll:
            self.txt.moveCursor(QTextCursor.End)

    def tick_heartbeat(self):
        age = 0 if self.last_line_ts is None else int(time.time() - self.last_line_ts)
        uptime_txt = ""
        if self.session_started_ts:
            seconds = max(0, int(time.time() - self.session_started_ts))
            hh = seconds // 3600
            mm = (seconds % 3600) // 60
            ss = seconds % 60
            uptime_txt = f"  |  аптайм: {hh:02d}:{mm:02d}:{ss:02d}"
            if self.session_started_at:
                self.lbl_session.setText(
                    f"Сессия: {self.session_started_at} (аптайм {hh:02d}:{mm:02d}:{ss:02d})"
                )
        self.lbl_heartbeat.setText(
            f"Линии: {self.lines_total}  |  последний лог: {age}s назад{uptime_txt}"
        )

    def refresh_margin_status(self):
        try:
            with open(MARGIN_STATUS_FILE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except FileNotFoundError:
            self.lbl_margin.setText("Маржа: —")
            self.lbl_margin.setStyleSheet("")
            self.margin_frozen = False
            return
        except Exception as exc:
            self.lbl_margin.setText(f"Маржа: ошибка ({exc})")
            self.lbl_margin.setStyleSheet("color:#faad14;")
            return

        try:
            im_pct = float(state.get("IM") or state.get("im_pct") or 0.0)
        except Exception:
            im_pct = 0.0
        try:
            mm_pct = float(state.get("MM") or state.get("mm_pct") or 0.0)
        except Exception:
            mm_pct = 0.0
        frozen = bool(state.get("frozen"))
        text = f"Маржа: IM {im_pct:.1f}% / MM {mm_pct:.1f}%"
        if frozen:
            text += " — FROZEN"
        self.lbl_margin.setText(text)

        if frozen:
            self.lbl_margin.setStyleSheet("color:#ff4d4f; font-weight:600;")
        else:
            self.lbl_margin.setStyleSheet("")

        if frozen and not self.margin_frozen:
            self.append_line("[MARGIN] FROZEN: высокий расход IM — новые входы остановлены")
        if self.margin_frozen and not frozen:
            self.append_line("[MARGIN] нормализация маржи, входы разрешены")
        self.margin_frozen = frozen

    def refresh_ml_status(self):
        try:
            with open(ML_STATUS_FILE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except FileNotFoundError:
            self.lbl_ml_state.setText("ML: —")
            self.lbl_ml_state.setStyleSheet("")
            return
        except Exception as exc:
            self.lbl_ml_state.setText(f"ML: ошибка ({exc})")
            self.lbl_ml_state.setStyleSheet("color:#faad14;")
            return

        paused = bool(state.get("paused", True))
        precision = state.get("precision_week")
        threshold = state.get("threshold")
        reason = str(state.get("reason") or "")
        reason_map = {
            "artifacts_missing": "нет модели",
            "weekly_precision_missing": "нет weekly precision",
            "precision_drop": "precision < порога",
        }
        human_reason = reason_map.get(reason, reason)

        parts = ["ML:"]
        parts.append("PAUSED" if paused else "OK")

        if isinstance(precision, (int, float)) and math.isfinite(precision):
            if isinstance(threshold, (int, float)) and math.isfinite(threshold):
                parts.append(f"({precision:.3f}/{float(threshold):.3f})")
            else:
                parts.append(f"({precision:.3f})")
        if human_reason:
            parts.append(f"— {human_reason}")

        self.lbl_ml_state.setText(" ".join(parts))
        if paused:
            self.lbl_ml_state.setStyleSheet("color:#ff4d4f; font-weight:600;")
        else:
            self.lbl_ml_state.setStyleSheet("color:#52c41a; font-weight:600;")

    # --- работа со статистикой сессии ---
    def poll_session_stats(self):
        if self.stats_shown:
            if self.stats_watch_timer.isActive():
                self.stats_watch_timer.stop()
            return
        if not (self.session_lookup_iso or self.stats_session_dir):
            return
        session_dir = self.stats_session_dir or self._guess_session_dir()
        if not session_dir:
            return
        self.stats_session_dir = session_dir
        stats = self._load_stats_from_disk(session_dir)
        if stats:
            self.stats_data = stats
            self._show_stats_modal()

    def _guess_session_dir(self) -> Optional[str]:
        try:
            root = get_sessions_root(create=False)
        except Exception:
            return None
        if not root.exists():
            return None
        ref_iso = self.session_lookup_iso or self.session_started_at
        target_ts = ts_to_epoch(ref_iso) if ref_iso else None
        best_path = None
        best_diff = None
        best_start = None
        try:
            entries = sorted(root.iterdir())
        except Exception:
            return None
        for entry in entries:
            if not entry.is_dir():
                continue
            meta_ts = None
            meta_path = entry / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    if isinstance(meta, dict):
                        meta_ts = ts_to_epoch(meta.get("start_ts"))
                except Exception:
                    meta_ts = None
            if meta_ts is None:
                try:
                    meta_ts = entry.stat().st_mtime
                except Exception:
                    continue
            if target_ts is not None:
                diff = abs(meta_ts - target_ts)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_path = entry
            else:
                if best_start is None or meta_ts > best_start:
                    best_start = meta_ts
                    best_path = entry
        return str(best_path) if best_path else None

    def _load_stats_from_disk(self, session_dir: str) -> Optional[Dict[str, Any]]:
        stats_path = os.path.join(session_dir, "stats.json")
        if not os.path.exists(stats_path):
            return None
        try:
            mtime = os.path.getmtime(stats_path)
            if self.latest_stats_mtime and mtime <= self.latest_stats_mtime:
                return None
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.latest_stats_mtime = mtime
                data.setdefault("session_dir", session_dir)
                data.setdefault("log_path", os.path.join(session_dir, "log.txt"))
                return data
        except Exception:
            return None
        return None

    def _handle_stats_line(self, line: str):
        self.received_stats_line = True
        stats = self.stats_data.copy() if isinstance(self.stats_data, dict) else {}
        try:
            pnl = re.search(r"Итог:\s*([+\-]?\d+(?:\.\d+)?)\s*USDT", line)
            trades = re.search(r"trades=([0-9]+)", line)
            wins = re.search(r"\(\+([0-9]+)\/-([0-9]+)\)", line)
            duration = re.search(r"duration=([0-9:]+)", line)
            maxdd = re.search(r"maxDD=([0-9\.]+)", line)
            winrate = re.search(r"winrate=([0-9\.]+)%", line)
            sharpe = re.search(r"sharpe=([0-9\.\-]+)", line)
            if pnl:
                stats["pnl_total"] = float(pnl.group(1))
            if trades:
                stats["trades"] = int(trades.group(1))
            if wins:
                stats["wins"] = int(wins.group(1))
                stats["losses"] = int(wins.group(2))
            if duration:
                stats["duration_human"] = duration.group(1)
            if maxdd:
                stats["max_drawdown"] = float(maxdd.group(1))
            if winrate:
                stats["winrate"] = float(winrate.group(1))
            if sharpe:
                stats["sharpe"] = float(sharpe.group(1))
        except Exception:
            pass
        session_dir = self._guess_session_dir()
        if session_dir:
            disk_stats = self._load_stats_from_disk(session_dir)
            if disk_stats:
                stats.update({k: v for k, v in disk_stats.items() if v is not None})
            else:
                stats.setdefault("session_dir", session_dir)
        self.stats_data = stats
        self._show_stats_modal()

    def _show_stats_modal(self):
        if not isinstance(self.stats_data, dict) or self.stats_shown:
            return
        if self.stop_notice_timer.isActive():
            self.stop_notice_timer.stop()
        self.stop_notice_shown = True
        stats = self.stats_data
        pnl = stats.get("pnl_total")
        trades = stats.get("trades")
        wins = stats.get("wins")
        losses = stats.get("losses")
        winrate = stats.get("winrate")
        duration = stats.get("duration_human") or self._format_duration(stats.get("duration_sec"))
        maxdd = stats.get("max_drawdown")
        sharpe = stats.get("sharpe")
        mode = stats.get("mode")
        text_lines = []
        if mode:
            text_lines.append(f"Режим: {mode}")
        if pnl is not None:
            text_lines.append(f"PnL: {pnl:+.2f} USDT")
        if trades is not None:
            text_lines.append(f"Сделки: {int(trades)} (побед {wins or 0}, поражений {losses or 0})")
        if winrate is not None:
            text_lines.append(f"Winrate: {float(winrate):.1f}%")
        if maxdd is not None:
            text_lines.append(f"Max DD: {float(maxdd):.2f} USDT")
        if sharpe is not None:
            text_lines.append(f"Sharpe: {float(sharpe):.2f}")
        if duration:
            text_lines.append(f"Длительность: {duration}")
        info = "\n".join(text_lines) if text_lines else "Итоговая статистика доступна."

        msg = QMessageBox(self)
        msg.setWindowTitle("Итог сессии")
        msg.setIcon(QMessageBox.Information)
        msg.setText("Итоги завершившейся сессии")
        msg.setInformativeText(info)
        btn_log = msg.addButton("Открыть лог", QMessageBox.ActionRole)
        btn_folder = msg.addButton("Открыть папку", QMessageBox.ActionRole)
        btn_stats = msg.addButton("Открыть отчёт", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        msg.exec_()
        clicked = msg.clickedButton()
        session_dir = stats.get("session_dir")
        log_path = stats.get("log_path") or (os.path.join(session_dir, "log.txt") if session_dir else None)
        if clicked == btn_log and log_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(log_path)))
        elif clicked == btn_folder and session_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(session_dir)))
        elif clicked == btn_stats:
            try:
                report = self.parent.report_screen
                self.parent.goto_screen("report")
                report.sessions_tab.refresh()
                report.tabs.setCurrentWidget(report.sessions_tab)
            except Exception:
                pass
        self.stats_shown = True
        self.session_lookup_iso = None
        if self.stats_watch_timer.isActive():
            self.stats_watch_timer.stop()

    def _format_duration(self, seconds: Optional[float]) -> Optional[str]:
        if seconds is None:
            return None
        try:
            seconds = int(seconds)
            hh = seconds // 3600
            mm = (seconds % 3600) // 60
            ss = seconds % 60
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        except Exception:
            return None

    def _show_stop_notice(self) -> None:
        if self.stats_shown or self.stats_data or self.stop_notice_shown:
            return
        self.stop_notice_shown = True
        msg = QMessageBox(self)
        msg.setWindowTitle("Остановлено")
        msg.setIcon(QMessageBox.Information)
        msg.setText("Остановлено пользователем")
        msg.setInformativeText("Итоги сессии не получены (stats.json отсутствует).")
        btn_tail = msg.addButton("Показать хвост лога", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        msg.exec_()
        if msg.clickedButton() == btn_tail:
            self.copy_tail()

class SessionsTab(QWidget):
    """Вкладка истории сессий и логов."""

    TAIL_LINES = 200

    def __init__(self, parent: "ReportScreen"):
        super().__init__(parent)
        self.parent_screen = parent
        self.records: List[Dict[str, Any]] = []
        self.filtered_records: List[Dict[str, Any]] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        filters = QHBoxLayout()
        self.cmb_period = QComboBox()
        self.cmb_period.addItem("Все", None)
        self.cmb_period.addItem("24 часа", 1)
        self.cmb_period.addItem("7 дней", 7)
        self.cmb_period.addItem("30 дней", 30)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Все режимы", None)
        self.cmb_mode.addItem("PAPER", "PAPER")
        self.cmb_mode.addItem("REAL", "REAL")
        self.chk_fav = QCheckBox("Только ⭐ избранные")
        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.setProperty("secondary", True)

        filters.addWidget(QLabel("Период:"))
        filters.addWidget(self.cmb_period)
        filters.addWidget(QLabel("Режим:"))
        filters.addWidget(self.cmb_mode)
        filters.addWidget(self.chk_fav)
        filters.addStretch(1)
        filters.addWidget(self.btn_refresh)
        layout.addLayout(filters)

        self.table = QTableWidget(0, 8)
        headers = ["Старт", "Финиш", "Длительность", "Режим", "PnL", "Сделки", "Winrate", "⭐"]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.lbl_preview = QLabel("Предпросмотр log.txt (последние 200 строк)")
        self.lbl_preview.setObjectName("SecondaryLabel")
        preview_widget = QWidget()
        pv_layout = QVBoxLayout(preview_widget)
        pv_layout.addWidget(self.lbl_preview)
        pv_layout.addWidget(self.preview)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        self.lbl_summary = QLabel("Сессий: 0")
        self.lbl_summary.setObjectName("SecondaryLabel")
        layout.addWidget(self.lbl_summary)

        buttons = QHBoxLayout()
        self.btn_open_log = QPushButton("📄 Открыть лог")
        self.btn_open_folder = QPushButton("📂 Папка")
        self.btn_copy_tail = QPushButton("📋 Копировать 200 строк")
        self.btn_clear_log = QPushButton("🧹 Очистить лог")
        self.btn_toggle_fav = QPushButton("⭐ В избранное")
        self.btn_delete = QPushButton("🗑 Удалить")
        self.btn_export_stats = QPushButton("💾 Экспорт stats.json")
        for btn in (
            self.btn_open_log,
            self.btn_open_folder,
            self.btn_copy_tail,
            self.btn_clear_log,
            self.btn_toggle_fav,
            self.btn_export_stats,
        ):
            btn.setProperty("secondary", True)
        self.btn_delete.setProperty("destructive", True)

        buttons.addWidget(self.btn_open_log)
        buttons.addWidget(self.btn_open_folder)
        buttons.addWidget(self.btn_copy_tail)
        buttons.addWidget(self.btn_clear_log)
        buttons.addWidget(self.btn_toggle_fav)
        buttons.addWidget(self.btn_export_stats)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_delete)
        layout.addLayout(buttons)

        self.cmb_period.currentIndexChanged.connect(self.refresh)
        self.cmb_mode.currentIndexChanged.connect(self.refresh)
        self.chk_fav.toggled.connect(self.refresh)
        self.btn_refresh.clicked.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self._update_preview)
        self.btn_open_log.clicked.connect(self._open_log)
        self.btn_open_folder.clicked.connect(self._open_folder)
        self.btn_copy_tail.clicked.connect(self._copy_tail)
        self.btn_clear_log.clicked.connect(self._clear_log)
        self.btn_delete.clicked.connect(self._delete_sessions)
        self.btn_toggle_fav.clicked.connect(self._toggle_favorite)
        self.btn_export_stats.clicked.connect(self._export_stats)

    def refresh(self) -> None:
        dirs = list_session_directories()
        records: List[Dict[str, Any]] = []
        for path in dirs:
            rec = self._build_record(path)
            if rec:
                records.append(rec)
        self.records = records
        self._apply_filters()
        self._update_preview()

    def _build_record(self, path: pathlib.Path) -> Optional[Dict[str, Any]]:
        meta = self._read_json(path / "meta.json")
        stats = self._read_json(path / "stats.json")
        start_iso = None
        if isinstance(meta, dict):
            start_iso = meta.get("start_ts") or meta.get("start_time")
        if not start_iso and isinstance(stats, dict):
            start_iso = stats.get("start_ts")
        start_epoch = ts_to_epoch(start_iso) if start_iso else None
        if start_epoch is None:
            try:
                start_epoch = path.stat().st_mtime
            except Exception:
                start_epoch = time.time()
        end_iso = stats.get("end_ts") if isinstance(stats, dict) else None
        duration_sec = stats.get("duration_sec") if isinstance(stats, dict) else None
        pnl = stats.get("pnl_total") if isinstance(stats, dict) else None
        trades = stats.get("trades") if isinstance(stats, dict) else None
        winrate = stats.get("winrate") if isinstance(stats, dict) else None
        wins = stats.get("wins") if isinstance(stats, dict) else None
        losses = stats.get("losses") if isinstance(stats, dict) else None
        favorite = (path / "favorites.flag").exists()
        mode = None
        if isinstance(stats, dict):
            mode = stats.get("mode")
        if not mode and isinstance(meta, dict):
            mode = meta.get("mode")
        log_path = path / "log.txt"
        stats_path = path / "stats.json"
        return {
            "path": path,
            "start_iso": start_iso,
            "start_epoch": start_epoch,
            "end_iso": end_iso,
            "duration_sec": duration_sec,
            "mode": (str(mode).upper() if mode else None),
            "pnl": pnl,
            "trades": trades,
            "winrate": winrate,
            "wins": wins,
            "losses": losses,
            "favorite": favorite,
            "log_path": log_path if log_path.exists() else None,
            "stats_path": stats_path if stats_path.exists() else None,
        }

    def _apply_filters(self) -> None:
        days = self.cmb_period.currentData()
        cutoff_ts = None
        if isinstance(days, int) and days:
            cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        mode = self.cmb_mode.currentData()
        only_fav = self.chk_fav.isChecked()
        filtered: List[Dict[str, Any]] = []
        for rec in self.records:
            if cutoff_ts is not None and rec.get("start_epoch") and rec["start_epoch"] < cutoff_ts:
                continue
            if mode and rec.get("mode") and rec.get("mode") != mode:
                continue
            if mode and rec.get("mode") is None:
                continue
            if only_fav and not rec.get("favorite"):
                continue
            filtered.append(rec)
        self.filtered_records = filtered
        self._populate_table(filtered)

    def _populate_table(self, rows: List[Dict[str, Any]]) -> None:
        self.table.setRowCount(len(rows))
        for r_idx, rec in enumerate(rows):
            start_txt = self._fmt_iso(rec.get("start_iso"))
            end_txt = self._fmt_iso(rec.get("end_iso"))
            dur_txt = self._fmt_duration(rec.get("duration_sec"))
            pnl = rec.get("pnl")
            winrate = rec.get("winrate")
            trades = rec.get("trades")
            values = [
                start_txt,
                end_txt or "—",
                dur_txt or "—",
                rec.get("mode") or "—",
                f"{pnl:+.2f}" if isinstance(pnl, (int, float)) else "—",
                str(int(trades)) if isinstance(trades, (int, float)) else "—",
                f"{float(winrate):.1f}%" if isinstance(winrate, (int, float)) else "—",
                "⭐" if rec.get("favorite") else "",
            ]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, rec)
                if c == 4 and isinstance(pnl, (int, float)):
                    color = QColor("#52c41a") if pnl >= 0 else QColor("#ff7875")
                    item.setForeground(color)
                if c == 7:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r_idx, c, item)
        self.table.resizeRowsToContents()
        summary = f"Сессий: {len(rows)}"
        if rows:
            pnl_sum = sum(float(r.get("pnl") or 0.0) for r in rows if isinstance(r.get("pnl"), (int, float)))
            summary += f"  |  ΣPnL: {pnl_sum:+.2f}"
        self.lbl_summary.setText(summary)

    def _fmt_iso(self, iso: Optional[str]) -> str:
        if not iso:
            return "—"
        return iso.replace("T", " ").replace("Z", "")

    def _fmt_duration(self, seconds: Optional[float]) -> Optional[str]:
        if seconds is None:
            return None
        try:
            seconds = int(float(seconds))
        except Exception:
            return None
        return time.strftime("%H:%M:%S", time.gmtime(max(0, seconds)))

    def _read_json(self, path: pathlib.Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _update_preview(self) -> None:
        records = self._selected_records()
        if not records and self.filtered_records:
            records = [self.filtered_records[0]]
            self.table.selectRow(0)
        if not records:
            self.preview.clear()
            self.btn_toggle_fav.setText("⭐ В избранное")
            return
        rec = records[0]
        log_path = rec.get("log_path")
        if log_path and os.path.exists(str(log_path)):
            self.preview.setPlainText(self._read_tail(str(log_path)))
        else:
            self.preview.setPlainText("Лог-файл не найден.")
        if records and all(r.get("favorite") for r in records):
            self.btn_toggle_fav.setText("⭐ Убрать из избранного")
        else:
            self.btn_toggle_fav.setText("⭐ В избранное")

    def _read_tail(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            return "".join(lines[-self.TAIL_LINES:])
        except Exception as exc:
            return f"[APP] Не удалось прочитать лог: {exc}"

    def _selected_records(self) -> List[Dict[str, Any]]:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        out: List[Dict[str, Any]] = []
        for row in rows:
            if 0 <= row < len(self.filtered_records):
                out.append(self.filtered_records[row])
        return out

    def _open_log(self) -> None:
        recs = self._selected_records()
        if not recs:
            return
        log_path = recs[0].get("log_path")
        if log_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(str(log_path))))

    def _open_folder(self) -> None:
        recs = self._selected_records()
        if not recs:
            return
        folder = recs[0].get("path")
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(str(folder))))

    def _copy_tail(self) -> None:
        text = self.preview.toPlainText().strip()
        if not text:
            recs = self._selected_records()
            if recs and recs[0].get("log_path"):
                text = self._read_tail(str(recs[0]["log_path"]))
        if text:
            QApplication.clipboard().setText(text)

    def _clear_log(self) -> None:
        recs = self._selected_records()
        if not recs:
            return
        log_path = recs[0].get("log_path")
        if not log_path:
            QMessageBox.information(self, "Очистка", "Лог отсутствует.")
            return
        if QMessageBox.question(self, "Очистить лог", "Обнулить log.txt?") != QMessageBox.Yes:
            return
        try:
            open(str(log_path), "w", encoding="utf-8").close()
            self.preview.clear()
        except Exception as exc:
            QMessageBox.warning(self, "Очистка", f"Не удалось очистить лог: {exc}")

    def _delete_sessions(self) -> None:
        recs = self._selected_records()
        if not recs:
            return
        if QMessageBox.question(
            self,
            "Удалить сессии",
            f"Удалить выбранные сессии ({len(recs)})?",
        ) != QMessageBox.Yes:
            return
        errors = []
        for rec in recs:
            folder = rec.get("path")
            if not folder:
                continue
            try:
                shutil.rmtree(folder)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            QMessageBox.warning(self, "Удаление", "\n".join(errors[:3]))
        self.refresh()

    def _toggle_favorite(self) -> None:
        recs = self._selected_records()
        if not recs:
            return
        make_fav = any(not r.get("favorite") for r in recs)
        for rec in recs:
            folder = rec.get("path")
            if not folder:
                continue
            flag = pathlib.Path(folder) / "favorites.flag"
            try:
                if make_fav:
                    flag.touch(exist_ok=True)
                else:
                    if flag.exists():
                        flag.unlink()
            except Exception:
                continue
        self.refresh()

    def _export_stats(self) -> None:
        recs = self._selected_records()
        if not recs:
            return
        stats_path = recs[0].get("stats_path")
        if not stats_path:
            QMessageBox.information(self, "Экспорт", "Файл stats.json не найден.")
            return
        default_name = os.path.join(os.getcwd(), f"stats_{os.path.basename(str(recs[0]['path']))}.json")
        target, _ = QFileDialog.getSaveFileName(self, "Сохранить stats.json", default_name, "JSON (*.json)")
        if not target:
            return
        try:
            shutil.copyfile(str(stats_path), target)
            QMessageBox.information(self, "Экспорт", f"Сохранено: {target}")
        except Exception as exc:
            QMessageBox.warning(self, "Экспорт", f"Не удалось сохранить файл: {exc}")


class ReportScreen(QWidget):
    """Экран отчётов: equity, фильтры, экспорт, PNG"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.log_path = LOG_DEFAULT
        self.session_only = False
        self.session_start_iso = None
        self.symbol_filter = None

        outer = QVBoxLayout()
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        equity_page = QWidget()
        equity_layout = QVBoxLayout(equity_page)

        header = QFrame()
        header.setObjectName("StatusFrame")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(4)
        self.label = QLabel("<h3>📈 Equity</h3>")
        self.lbl_updated = QLabel("Обновлено: —")
        self.lbl_updated.setObjectName("SecondaryLabel")
        header_layout.addWidget(self.label)
        header_layout.addWidget(self.lbl_updated)
        equity_layout.addWidget(header)

        filters = QHBoxLayout()
        self.chk_session = QCheckBox("Только текущая сессия")
        self.cmb_symbol = QComboBox(); self.cmb_symbol.addItem("Все символы")
        self.btn_export = QPushButton("⬇ Экспорт отчёта (CSV)")
        self.chk_autorefresh = QCheckBox("Автообновление"); self.chk_autorefresh.setChecked(True)
        self.btn_reset_zoom = QPushButton("🔎 Сброс зума")
        self.btn_save_png = QPushButton("📷 Сохранить PNG")
        for btn in (self.btn_export, self.btn_reset_zoom, self.btn_save_png):
            btn.setProperty("secondary", True)
        filters.addWidget(self.chk_session)
        filters.addWidget(QLabel("Символ:"))
        filters.addWidget(self.cmb_symbol, 1)
        filters.addWidget(self.chk_autorefresh)
        filters.addWidget(self.btn_reset_zoom)
        filters.addWidget(self.btn_save_png)
        filters.addStretch(1)
        filters.addWidget(self.btn_export)
        equity_layout.addLayout(filters)

        axis = DateAxisItem(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': axis})
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        equity_layout.addWidget(self.plot, 1)

        self.lbl_stats = QLabel("Trades: 0 | P+: 0.00 | P-: 0.00 | Win%: 0.0 | MaxDD: 0.00 | Median dur: 0m")
        self.lbl_stats.setObjectName("SecondaryLabel")
        equity_layout.addWidget(self.lbl_stats)

        refresh_row = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Обновить")
        refresh_row.addWidget(self.btn_refresh)
        refresh_row.addStretch(1)
        equity_layout.addLayout(refresh_row)

        self.tabs.addTab(equity_page, "Equity")
        self.sessions_tab = SessionsTab(self)
        self.tabs.addTab(self.sessions_tab, "Сессии")

        bottom = QHBoxLayout()
        self.btn_back = QPushButton("⬅ Назад")
        self.btn_back.setProperty("secondary", True)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_back)
        outer.addLayout(bottom)

        self.setLayout(outer)

        self.btn_back.clicked.connect(lambda: self.parent.goto_screen("main"))
        self.btn_refresh.clicked.connect(self.plot_balance)
        self.chk_session.toggled.connect(self.on_session_toggle)
        self.cmb_symbol.currentIndexChanged.connect(self.plot_balance)
        self.btn_export.clicked.connect(self.export_report)
        self.chk_autorefresh.toggled.connect(self._toggle_autorefresh)
        self.btn_reset_zoom.clicked.connect(lambda: self.plot.enableAutoRange('xy', True))
        self.btn_save_png.clicked.connect(self.save_png)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # автообновление
        self.t = QTimer(self); self.t.setInterval(3000)
        self.t.timeout.connect(self.plot_balance); self.t.start()

        self.plot_balance()

    def set_log_path(self, path: str):
        self.log_path = path

    def _toggle_autorefresh(self, v: bool):
        self.t.setInterval(3000)
        self.t.setSingleShot(False)
        self.t.stop()
        if v:
            self.t.start()

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget is self.sessions_tab:
            self.sessions_tab.refresh()

    def on_session_toggle(self, v: bool):
        self.session_only = v
        # привязываем старт сессии к RunScreen
        rs = self.parent.run_screen
        self.session_start_iso = rs.session_started_at
        self.plot_balance()

    def _load_rows_filtered(self) -> List[Dict[str, Any]]:
        rows = safe_read_jsonl(self.log_path)
        if not rows:
            return []
        # символы
        syms = sorted({r.get("symbol") for r in rows if isinstance(r, dict) and r.get("symbol")})
        # наполняем комбобокс один раз/динамически
        if self.cmb_symbol.count() <= 1 and syms:
            self.cmb_symbol.addItems(syms)
        chosen = self.cmb_symbol.currentText()
        self.symbol_filter = None if chosen in ("", "Все символы") else chosen

        # фильтры событий
        EVENTS_CLOSE = {"paper_close", "close", "dynamic_tp_exit", "no_profit_exit"}
        out = []
        start_ts_epoch = ts_to_epoch(self.session_start_iso) if (self.session_only and self.session_start_iso) else None
        for r in rows:
            if not isinstance(r, dict):
                continue
            ev = r.get("event")
            if ev not in EVENTS_CLOSE:
                continue
            if self.symbol_filter and r.get("symbol") != self.symbol_filter:
                continue
            if start_ts_epoch:
                ts0 = r.get("closed_at") or r.get("timestamp") or r.get("ts_utc")
                t = ts_to_epoch(ts0) if ts0 else None
                if not t or t < start_ts_epoch:
                    continue
            out.append(r)
        return out

    def plot_balance(self):
        if not (self.isVisible()):
            return
        self.plot.clear()
        rows = self._load_rows_filtered()
        if not rows:
            self.label.setText("Данных пока нет (ожидаю сделки)…")
            self.lbl_stats.setText("Trades: 0 | P+: 0.00 | P-: 0.00 | Win%: 0.0 | MaxDD: 0.00 | Median dur: 0m")
            return

        ts_list, equity, pnl_list, durations = [], [], [], []
        s = 0.0
        pos_cnt_p = pos_cnt_n = 0
        sum_p = sum_n = 0.0
        for r in rows:
            pnl = float(r.get("pnl", 0.0))
            ts = r.get("closed_at") or r.get("timestamp") or r.get("ts_utc")
            t = ts_to_epoch(ts) if ts else None
            if t is None:  # пропускаем битые
                continue
            ts_list.append(t)
            s += pnl
            equity.append(s)
            pnl_list.append(pnl)
            if pnl >= 0:
                pos_cnt_p += 1; sum_p += pnl
            else:
                pos_cnt_n += 1; sum_n += pnl
            # duration, если есть entry_ts
            try:
                ent = r.get("opened_at") or r.get("entry_ts") or r.get("signal_ts")
                te = ts_to_epoch(ent) if ent else None
                if te is not None:
                    durations.append(max(0, t - te))
            except Exception:
                pass

        # Max Drawdown и Ulcer Index по equity
        max_dd = 0.0
        peak = -1e18
        ulcer_sum = 0.0
        ulcer_count = 0
        for v in equity:
            peak = max(peak, v)
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
            if peak > 0:
                ulcer_sum += (dd / peak) ** 2
                ulcer_count += 1
        ulcer_index = math.sqrt(ulcer_sum / ulcer_count) if ulcer_count else 0.0

        # median duration
        if durations:
            durations.sort()
            med = durations[len(durations)//2]
        else:
            med = 0
        med_min = int(med // 60)

        winrate = (pos_cnt_p / max(1, pos_cnt_p + pos_cnt_n)) * 100.0
        sharpe = 0.0
        if len(pnl_list) > 1:
            mean_p = statistics.mean(pnl_list)
            std_p = statistics.stdev(pnl_list)
            if std_p > 0:
                sharpe = (mean_p / std_p) * math.sqrt(len(pnl_list))

        self.plot.plot(ts_list, equity, pen=pg.mkPen(width=2), symbol='o', symbolSize=5)
        self.lbl_stats.setText(
            f"Trades: {pos_cnt_p+pos_cnt_n}  |  P+: {sum_p:+.2f}  |  P-: {sum_n:+.2f}  |  "
            f"Win%: {winrate:.1f}  |  MaxDD: {max_dd:.2f}  |  Ulcer: {ulcer_index:.3f}  |  "
            f"Sharpe: {sharpe:.2f}  |  Median dur: {med_min}m"
        )
        self.label.setText("📈 Equity (кумулятивный PnL)")
        self.lbl_updated.setText(f"Обновлено: {now_iso()}")

    def export_report(self):
        rows = self._load_rows_filtered()
        if not rows:
            QMessageBox.information(self, "Экспорт", "Нет данных для экспорта.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт CSV", "trade_report.csv",
                                            "CSV (*.csv);;All files (*.*)")
        if not fn:
            return
        cols = ["symbol","event","pnl","closed_at","exit_reason","opened_at","entry_ts","signal_ts"]
        try:
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in rows:
                    w.writerow([r.get(c,"") for c in cols])
            QMessageBox.information(self, "Экспорт", f"Сохранено: {fn}")
        except Exception as e:
            QMessageBox.warning(self, "Экспорт", f"Ошибка сохранения: {e}")

    def save_png(self):
        # простой скриншот виджета графика
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить график как PNG", "equity.png",
                                            "PNG (*.png);;All files (*.*)")
        if not fn:
            return
        try:
            pix = self.plot.grab()  # QPixmap
            pix.save(fn)
            QMessageBox.information(self, "PNG", f"Сохранено: {fn}")
        except Exception as e:
            QMessageBox.warning(self, "PNG", f"Ошибка сохранения: {e}")

class ToolsScreen(QWidget):
    """Инструменты ML: сборка датасета, обучение модели, просмотр файлов"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.proc: Optional[QProcess] = None
        self.working_dir = os.path.abspath(os.path.dirname(MAIN_PY) or ".")

        layout = QVBoxLayout()
        self.label = QLabel("<h3>🧰 Инструменты (ML)</h3>")
        layout.addWidget(self.label)

        # Кнопки задач
        btns = QHBoxLayout()
        self.btn_build = QPushButton("📦 Собрать датасет")
        self.btn_train = QPushButton("🤖 Переобучить модель")
        self.btn_stop_task = QPushButton("🛑 Прервать задачу")
        self.btn_open_meta = QPushButton("📄 Открыть model_meta.json")
        self.btn_open_fills = QPushButton("📄 Открыть fills_all.csv")
        self.btn_stop_task.setProperty("destructive", True)
        for btn in (self.btn_open_meta, self.btn_open_fills):
            btn.setProperty("secondary", True)
        btns.addWidget(self.btn_build)
        btns.addWidget(self.btn_train)
        btns.addWidget(self.btn_stop_task)
        btns.addStretch(1)
        btns.addWidget(self.btn_open_meta)
        btns.addWidget(self.btn_open_fills)
        layout.addLayout(btns)

        info_frame = QFrame()
        info_frame.setObjectName("StatusFrame")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(6)
        self.lbl_ml_info = QLabel("Загружаю model_meta.json…")
        self.lbl_ml_info.setObjectName("SecondaryLabel")
        self.lbl_ml_info.setWordWrap(True)
        self.btn_cycle_threshold = QPushButton("🔁 Переключить режим порога")
        self.btn_cycle_threshold.setProperty("secondary", True)
        info_layout.addWidget(self.lbl_ml_info)
        info_layout.addWidget(self.btn_cycle_threshold, alignment=Qt.AlignLeft)
        layout.addWidget(info_frame)

        # Лог задач
        self.txt = QPlainTextEdit(); self.txt.setReadOnly(True); self.txt.setFont(QFont("Consolas", 10))
        self.txt.setObjectName("LogView")
        layout.addWidget(self.txt, 1)

        # Панель
        btns2 = QHBoxLayout()
        self.btn_back = QPushButton("⬅ Назад")
        self.btn_back.setProperty("secondary", True)
        btns2.addStretch(1)
        btns2.addWidget(self.btn_back)
        layout.addLayout(btns2)

        self.setLayout(layout)

        # wiring
        self.btn_back.clicked.connect(lambda: self.parent.goto_screen("main"))
        self.btn_build.clicked.connect(lambda: self.run_task(BUILD_DATASET_PY))
        self.btn_train.clicked.connect(lambda: self.run_task(TRAIN_MODEL_PY))
        self.btn_stop_task.clicked.connect(self.stop_task)
        self.btn_open_meta.clicked.connect(lambda: self.open_file("model_meta.json"))
        self.btn_open_fills.clicked.connect(lambda: self.open_file("fills_all.csv"))
        self.btn_cycle_threshold.clicked.connect(self.switch_threshold_mode)

        self.refresh_meta_info()

    def run_task(self, script_name: str):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Задача уже идёт", "Дождись завершения текущей задачи или прерви её.")
            return
        script_path = os.path.join(self.working_dir, script_name)
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "Файл не найден", f"Не нашёл {script_path}")
            return
        self.txt.clear()
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(self.working_dir)
        self.proc.setProgram(sys.executable)
        args = ["-X","utf8", script_path]
        self.proc.setArguments(args)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8","1"); env.insert("PYTHONIOENCODING","utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_ready)
        self.proc.finished.connect(self.on_finished)
        self.txt.appendPlainText(f"[APP] ▶ {sys.executable} {' '.join(args)}")
        self.proc.start()

    def stop_task(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.txt.appendPlainText("[APP] Запрашиваю остановку фоновой задачи…")
            self.proc.terminate()
            if not self.proc.waitForFinished(2000):
                self.txt.appendPlainText("[APP] Жёсткая остановка (kill).")
                self.proc.kill()

    def on_ready(self):
        try:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            data = ""
        if data:
            self.txt.appendPlainText(data.rstrip())

    def on_finished(self, code, status):
        self.txt.appendPlainText(f"[APP] ⏹ Завершено. code={code}, status={status}")
        if code != 0:
            QMessageBox.warning(self, "Задача завершилась с ошибкой",
                                f"Код возврата: {code}\nПроверь лог.")

    def open_file(self, fname: str):
        path = os.path.join(self.working_dir, fname)
        if not os.path.exists(path):
            QMessageBox.information(self, "Открыть", f"Файл не найден: {path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(path)))

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_meta_info()

    def _meta_path(self) -> str:
        return os.path.join(self.working_dir, "model_meta.json")

    def _load_meta(self) -> Optional[Dict[str, Any]]:
        path = self._meta_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.txt.appendPlainText(f"[APP] Ошибка чтения model_meta.json: {e}")
            return None

    def refresh_meta_info(self):
        meta = self._load_meta()
        if not meta:
            self.lbl_ml_info.setText("model_meta.json не найден.")
            return

        thresholds = meta.get("thresholds", {}) or {}
        calibration = meta.get("calibration", {}) or {}
        atr_pct = meta.get("atr_percentiles", {}) or {}

        def _fmt(value: Any, digits: int = 4) -> str:
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "—"

        method = str(calibration.get("method", "—"))
        score = calibration.get("valid_score")
        if isinstance(score, (int, float)) and math.isfinite(score):
            calib_line = f"<b>Калибровка:</b> {method} (Brier={score:.6f})"
        else:
            calib_line = f"<b>Калибровка:</b> {method}"

        used_mode = str(thresholds.get("used_mode", "global"))
        used_thr = thresholds.get("used", thresholds.get("global"))
        thr_line = f"<b>Активный порог:</b> {used_mode} → {_fmt(used_thr)}"

        regime_line = (
            f"<b>Режимы:</b> low={_fmt(thresholds.get('regime_low'))} | "
            f"high={_fmt(thresholds.get('regime_high'))} | "
            f"ultra={_fmt(thresholds.get('regime_ultra'))}"
        )

        p50 = atr_pct.get("p50")
        p90 = atr_pct.get("p90")
        perc_line = f"<b>ATR p50/p90:</b> {_fmt(p50, digits=6)} / {_fmt(p90, digits=6)}"

        self.lbl_ml_info.setText("<br>".join([calib_line, thr_line, regime_line, perc_line]))

    def switch_threshold_mode(self):
        meta = self._load_meta()
        if not meta:
            QMessageBox.warning(self, "Переключение порога", "model_meta.json не найден.")
            return

        thresholds = meta.get("thresholds")
        if not isinstance(thresholds, dict):
            QMessageBox.warning(self, "Переключение порога", "В model_meta.json отсутствует блок thresholds.")
            return

        modes_cycle = ["global", "regime_low", "regime_high", "regime_ultra", "ev_only"]

        def _as_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
            try:
                if value is None:
                    raise ValueError
                v = float(value)
                if not math.isfinite(v):
                    raise ValueError
                return v
            except (TypeError, ValueError):
                return fallback

        current_mode = str(thresholds.get("used_mode", "global"))
        try:
            idx = modes_cycle.index(current_mode)
        except ValueError:
            idx = -1
        next_mode = modes_cycle[(idx + 1) % len(modes_cycle)]

        fallback_thr = _as_float(thresholds.get("used"), _as_float(thresholds.get("global"), 0.5))

        def _resolve(mode: str) -> Optional[float]:
            if mode == "global":
                return _as_float(thresholds.get("global"), fallback_thr)
            if mode == "ev_only":
                return _as_float(thresholds.get("ev_only"), fallback_thr)
            return _as_float(thresholds.get(mode), fallback_thr)

        new_thr = _resolve(next_mode)
        if new_thr is None:
            QMessageBox.warning(self, "Переключение порога", f"Не удалось получить порог для режима {next_mode}.")
            return

        thresholds["used_mode"] = next_mode
        thresholds["used"] = float(new_thr)

        meta_path = self._meta_path()
        backup_path = meta_path + ".bak"
        try:
            shutil.copyfile(meta_path, backup_path)
        except Exception as e:
            QMessageBox.warning(self, "Переключение порога", f"Не удалось создать бэкап: {e}")
            return

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.txt.appendPlainText(
                f"[APP] Порог обновлён → {next_mode} ({new_thr:.4f}); backup: {backup_path}"
            )
            QMessageBox.information(
                self,
                "Порог обновлён",
                f"Режим: {next_mode}\nПорог: {new_thr:.4f}\nБэкап: {backup_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Переключение порога", f"Ошибка записи model_meta.json: {e}")
            return

        self.refresh_meta_info()

# ----------------------------- контейнер приложения -----------------------------
class TradeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.screens: Dict[str, QWidget] = {}
        self.enable_apk = self._is_apk_enabled()
        self._cfg = load_config()
        self.current_theme = str(self._cfg.get("theme", "Auto") or "Auto")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.statusBar().showMessage("Готово", 1500)
        self.init_ui()
        self.install_hotkeys()
        self.setup_menus()

    def apply_theme(self, theme: str):
        app = QApplication.instance()
        if app is None:
            return
        selection = str(theme or "Auto").title()
        if selection not in {"Auto", "Dark", "Light"}:
            selection = "Auto"
        self.current_theme = selection

        applied = selection
        if selection == "Auto":
            system_palette = app.style().standardPalette()
            applied = "Dark" if system_palette.color(QPalette.Window).lightness() < 128 else "Light"

        app.setStyle("Fusion")
        extra_styles = ""

        if applied == "Dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(28, 31, 44))
            palette.setColor(QPalette.WindowText, QColor(242, 245, 255))
            palette.setColor(QPalette.Base, QColor(16, 19, 27))
            palette.setColor(QPalette.AlternateBase, QColor(28, 31, 45))
            palette.setColor(QPalette.ToolTipBase, QColor(31, 35, 49))
            palette.setColor(QPalette.ToolTipText, QColor(242, 245, 255))
            palette.setColor(QPalette.Text, QColor(242, 245, 255))
            palette.setColor(QPalette.Button, QColor(36, 41, 58))
            palette.setColor(QPalette.ButtonText, QColor(242, 245, 255))
            palette.setColor(QPalette.Link, QColor(126, 180, 255))
            palette.setColor(QPalette.Highlight, QColor(22, 119, 255))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(110, 116, 138))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(110, 116, 138))
            extra_styles = DARK_STYLESHEET
        else:
            palette = app.style().standardPalette()
            palette.setColor(QPalette.Window, QColor(247, 249, 252))
            palette.setColor(QPalette.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.AlternateBase, QColor(236, 241, 250))
            palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            palette.setColor(QPalette.ToolTipText, QColor(16, 19, 32))
            palette.setColor(QPalette.Text, QColor(16, 19, 32))
            palette.setColor(QPalette.Button, QColor(240, 244, 252))
            palette.setColor(QPalette.ButtonText, QColor(16, 19, 32))
            palette.setColor(QPalette.Link, QColor(22, 119, 255))
            palette.setColor(QPalette.Highlight, QColor(22, 119, 255))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(168, 174, 188))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(168, 174, 188))
            extra_styles = LIGHT_STYLESHEET

        app.setPalette(palette)
        app.setStyleSheet(COMMON_STYLESHEET + extra_styles)

    def init_ui(self):
        self.screens["main"] = MainMenu(self)
        self.run_screen = RunScreen(self)
        self.report_screen = ReportScreen(self)
        self.tools_screen = ToolsScreen(self)
        self.screens["run"] = self.run_screen
        self.screens["report"] = self.report_screen
        self.screens["tools"] = self.tools_screen
        if self.enable_apk:
            self.apk_screen = ApkManagerScreen(self)
            self.screens["apk"] = self.apk_screen

        for s in self.screens.values():
            self.stack.addWidget(s)

        self.apply_theme(self._cfg.get("theme", "Auto"))

        # размеры/позиция из конфига
        w = int(self._cfg.get("win_w", 1000))
        h = int(self._cfg.get("win_h", 680))
        self.resize(w, h)
        if self._cfg.get("pos_x") is not None and self._cfg.get("pos_y") is not None:
            try:
                self.move(int(self._cfg["pos_x"]), int(self._cfg["pos_y"]))
            except Exception:
                pass

        self.goto_screen("main")
        self.setWindowTitle(APP_TITLE)

    def goto_screen(self, name):
        w = self.screens.get(name)
        if w:
            self.stack.setCurrentWidget(w)

    def install_hotkeys(self):
        # F5 — перезапуск бота (если мы на Run экране)
        QShortcut(QKeySequence("F5"), self, activated=self.hotkey_restart)
        # Ctrl+L — очистка окна лога
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.hotkey_clear_log_view)
        # Ctrl+O — открыть папку лога (если задан)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.hotkey_open_log_folder)
        # Esc — назад в главное меню
        QShortcut(QKeySequence("Esc"), self, activated=lambda: self.goto_screen("main"))

    def setup_menus(self):
        bar = self.menuBar()
        if bar is not None:
            try:
                bar.setNativeMenuBar(False)
            except Exception:
                pass
            launch_menu = bar.addMenu("Запуск")
            self.action_start_paper = QAction("Старт PAPER", self)
            self.action_start_real = QAction("Старт REAL", self)
            launch_menu.addAction(self.action_start_paper)
            launch_menu.addAction(self.action_start_real)
            self.action_start_paper.triggered.connect(self.start_paper_from_menu)
            self.action_start_real.triggered.connect(self.start_real_from_menu)

        toolbar = QToolBar("Навигация", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        key_icon = QIcon.fromTheme("dialog-password")
        if key_icon.isNull():
            key_icon = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        self.action_keys = QAction(key_icon, "🔑 Ключи", self)
        self.action_keys.triggered.connect(self.show_keys_dialog)
        toolbar.addAction(self.action_keys)
        toolbar.addSeparator()
        if self.enable_apk:
            apk_icon = QIcon.fromTheme("document-share")
            if apk_icon.isNull():
                apk_icon = self.style().standardIcon(QStyle.SP_DirIcon)
            self.action_apk = QAction(apk_icon, "📦 APK", self)
            self.action_apk.triggered.connect(lambda: self.goto_screen("apk"))
            toolbar.addAction(self.action_apk)
            toolbar.addSeparator()
        help_icon = QIcon.fromTheme("help-browser")
        if help_icon.isNull():
            help_icon = self.style().standardIcon(QStyle.SP_DialogHelpButton)
        self.action_help = QAction(help_icon, "? Помощь", self)
        self.action_help.setShortcut(QKeySequence("F1"))
        self.action_help.triggered.connect(self.show_help_dialog)
        toolbar.addAction(self.action_help)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def _is_apk_enabled(self) -> bool:
        default = bool(getattr(config, "ENABLE_APK_MANAGER", True))
        return bool(env_bool("ENABLE_APK_MANAGER", default))

    def start_paper_from_menu(self):
        main = self.screens.get("main")
        if not isinstance(main, MainMenu):
            return
        self.goto_screen("main")
        main.cmb_mode.setCurrentText("PAPER")
        main.on_start()

    def start_real_from_menu(self):
        main = self.screens.get("main")
        if not isinstance(main, MainMenu):
            return
        self.goto_screen("main")
        main.cmb_mode.setCurrentText("REAL")
        main.on_start()

    def show_keys_dialog(self):
        dlg = KeysDialog(self)
        dlg.exec_()
        # перезагрузим конфиг, если диалог его обновил
        self._cfg = load_config()

    def show_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Помощь")
        dlg.resize(560, 520)
        layout = QVBoxLayout(dlg)
        tabs = QTabWidget(dlg)

        sections = [
            ("Быстрый старт",
             "<h3>Быстрый старт</h3>"
             "<ul>"
             "<li>Выбери режим (PAPER или REAL) и задай пары/плечо на главном экране.</li>"
             "<li>Меню <b>Запуск → Старт PAPER</b> мгновенно поднимет бота в бумажном режиме."
             " Или нажми кнопку \"🛠 За работу\".</li>"
             "<li>Для REAL используй меню <b>Старт REAL</b> или кнопку \"Запуск\" — появится подтверждение и переключатель SAFE_MODE.</li>"
             "<li>В левой панели Run видно состояние, активные пары и текущую сессию.</li>"
             "<li>Логи выводятся в окне снизу; файл задаётся в поле \"Файл лога\".</li>"
             "</ul>"),
            ("Горячие клавиши",
             "<h3>Горячие клавиши</h3>"
             "<ul>"
             "<li><b>Space</b> — пауза/возобновление входов.</li>"
             "<li><b>Esc</b> — вернуться в главное меню.</li>"
             "<li><b>Ctrl+P</b> — паника (panic_close + остановка процесса).</li>"
             "<li><b>Ctrl+S</b> — остановить процесс.</li>"
             "<li><b>Ctrl+C</b> — скопировать последние 200 строк лога.</li>"
             "<li><b>Ctrl+E</b> — перейти к отчётам.</li>"
             "<li><b>F5</b> — перезапустить бота (остановка + старт).</li>"
             "<li><b>F1</b> — открыть это окно помощи.</li>"
             "</ul>"),
            ("Панель управления",
             "<h3>Панель управления</h3>"
             "<ul>"
             "<li><b>⏹ Остановить</b> — мягко завершить подпроцесс main.py.</li>"
             "<li><b>🛑 Закрыть всё</b> — panic_close + остановка процесса.</li>"
             "<li><b>⏸ / ▶</b> — пауза и возобновление сигналов через control.json.</li>"
             "<li><b>🔁 Обновить пары/плечо</b> — записывает текущие настройки в control.json.</li>"
             "<li>Нижние кнопки открывают/очищают лог, копируют хвост и ведут к отчётам.</li>"
             "</ul>"),
            ("Переменные окружения",
             "<h3>Ключевые переменные окружения</h3>"
             "<ul>"
             "<li><b>MIN_ATR_PCT</b> — минимальная волатильность для входа.</li>"
             "<li><b>SPREAD_MAX_PCT</b> — допустимый спрэд для сделок.</li>"
             "<li><b>ENTRY_COOLDOWN_SEC</b> — пауза между входами по инструменту.</li>"
             "<li><b>KLINE_HISTORY_LIMIT</b> — длина истории свечей для индикаторов (по умолчанию 300).</li>"
             "<li><b>LOG_RU</b> — 1 включает русские сообщения в логах.</li>"
             "<li><b>ROUTER_HEARTBEAT_SEC</b> — троттлинг повторяющихся сообщений роутера.</li>"
             "<li><b>SAFE_MODE</b> и <b>PAPER_MODE</b> — режимы безопасности и моделирования.</li>"
             "</ul>"),
            ("FAQ / Ошибки",
             "<h3>FAQ и сообщения</h3>"
             "<ul>"
             "<li><b>удержание — волатильность вне диапазона</b> — ATR% ниже/выше допустимого, вход отложен.</li>"
             "<li><b>удержание — нет условий для входа</b> — роутер не нашёл подходящую стратегию.</li>"
             "<li><b>скорректировал размер до min_qty</b> — объём поднят до минимально допустимого.</li>"
             "<li><b>позиция уже открыта</b> — повторный вход не отправляется.</li>"
             "<li><b>place_market_order False</b> — биржа отклонила ордер; проверь лимиты и баланс.</li>"
             "</ul>"),
        ]

        for title, html in sections:
            page = QWidget()
            vbox = QVBoxLayout(page)
            view = QTextBrowser()
            view.setOpenExternalLinks(True)
            view.setHtml(html)
            vbox.addWidget(view)
            tabs.addTab(page, title)

        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.setText("Закрыть")
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        dlg.exec_()

    def hotkey_restart(self):
        if self.stack.currentWidget() is self.run_screen:
            self.run_screen.on_stop()
            # маленькая пауза чтобы порт/файлы освободились
            QTimer.singleShot(400, self.run_screen.start_bot)

    def hotkey_clear_log_view(self):
        if self.stack.currentWidget() is self.run_screen:
            self.run_screen.txt.clear()

    def hotkey_open_log_folder(self):
        if self.stack.currentWidget() is self.run_screen:
            self.run_screen.open_log_folder()

    # корректно гасим подпроцесс при выходе и сохраняем конфиг
    def closeEvent(self, e):
        try:
            if self.run_screen.proc and self.run_screen.proc.state() != QProcess.NotRunning:
                self.run_screen.on_stop()
        except Exception:
            pass
        # снимем геометрию и часть состояния из MainMenu
        try:
            cfg = load_config()
            cfg.update({
                "win_w": int(self.width()),
                "win_h": int(self.height()),
                "pos_x": int(self.x()),
                "pos_y": int(self.y()),
                # дубль важных полей с текущего экрана Main
                "mode": self.screens["main"].cmb_mode.currentText(),
                "unsafe": bool(self.screens["main"].chk_unsafe.isChecked()),
                "autoscroll": bool(self.screens["main"].chk_autoscroll.isChecked()),
                "log_path": self.screens["main"].le_log.text().strip() or LOG_DEFAULT,
                "pairs": self.screens["main"].le_pairs.text().strip(),
                "default_lev": int(self.screens["main"].sp_lev.value()),
                "theme": self.current_theme,
            })
            save_config(cfg)
        except Exception:
            pass
        super().closeEvent(e)

# ----------------------------- entrypoint -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TradeApp()
    window.show()
    sys.exit(app.exec_())
