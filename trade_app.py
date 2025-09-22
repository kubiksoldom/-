# trade_app.py
# v3.2 — Полный UI апгрейд: Паника/Пауза/Инструменты/Фильтры/Отчёты/Хоткеи/Безопасность/Конфиг/Подсветка
import sys, os, json, re, time, pathlib, csv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from PyQt5.QtCore import Qt, QTimer, QProcess, QProcessEnvironment, QUrl, QByteArray
from PyQt5.QtGui import (
    QFont, QTextCursor, QDesktopServices, QKeySequence,
    QSyntaxHighlighter, QTextCharFormat, QColor, QPalette
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QMessageBox, QPlainTextEdit, QComboBox, QCheckBox,
    QFileDialog, QLineEdit, QFormLayout, QSpinBox, QShortcut, QFrame
)

import pyqtgraph as pg
from pyqtgraph import DateAxisItem

# ----------------------------- константы -----------------------------
APP_TITLE         = "🔥 КриптоБот v3.2"
LOG_DEFAULT       = "bot_cycle_log.jsonl"
MAIN_PY           = "main.py"
BUILD_DATASET_PY  = "build_ml_dataset_from_fills.py"
TRAIN_MODEL_PY    = "nn_model.py"
CONFIG_FILE       = os.path.join(os.path.expanduser("~"), ".trade_app_config.json")

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
    "log_path": LOG_DEFAULT,
    "pairs": "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT",
    "default_lev": 5,
    "win_w": 1000,
    "win_h": 680,
    "pos_x": None,
    "pos_y": None,
    "theme": "Auto",
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
        payload["ts"] = now_iso()
        data.append(payload)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
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
        log_row = QHBoxLayout()
        log_row.addWidget(self.le_log, 1)
        log_row.addWidget(self.btn_pick_log)
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

        # Кнопки перехода
        btns = QHBoxLayout()
        self.btn_start = QPushButton("🚀 Запуск")
        self.btn_report = QPushButton("📊 Отчёты")
        self.btn_tools = QPushButton("🧰 Инструменты (ML)")
        self.btn_report.setProperty("secondary", True)
        self.btn_tools.setProperty("secondary", True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_report)
        btns.addWidget(self.btn_tools)
        layout.addLayout(btns)

        self.setLayout(layout)

        # wiring
        self.btn_start.clicked.connect(self.on_start)
        self.btn_report.clicked.connect(lambda: self.parent.goto_screen("report"))
        self.btn_tools.clicked.connect(lambda: self.parent.goto_screen("tools"))
        self.btn_pick_log.clicked.connect(self.pick_log)
        self.btn_preset_scalp.clicked.connect(self.apply_preset_scalp)
        self.btn_preset_real.clicked.connect(self.apply_preset_real)
        self.btn_load_pairs.clicked.connect(self.load_pairs_from_exchange)

        # авто-сейв конфига при изменениях
        self.cmb_mode.currentTextChanged.connect(self._save_cfg)
        self.chk_unsafe.toggled.connect(self._save_cfg)
        self.chk_autoscroll.toggled.connect(self._save_cfg)
        self.le_log.textChanged.connect(self._save_cfg)
        self.le_pairs.textChanged.connect(self._save_cfg)
        self.sp_lev.valueChanged.connect(self._save_cfg)
        self.cmb_theme.currentTextChanged.connect(self.on_theme_change)

    def _save_cfg(self, *args):
        self._cfg.update({
            "mode": self.cmb_mode.currentText(),
            "unsafe": bool(self.chk_unsafe.isChecked()),
            "autoscroll": bool(self.chk_autoscroll.isChecked()),
            "log_path": self.le_log.text().strip() or LOG_DEFAULT,
            "pairs": self.le_pairs.text().strip(),
            "default_lev": int(self.sp_lev.value()),
            "theme": self.cmb_theme.currentText(),
        })
        save_config(self._cfg)

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
        mode = self.cmb_mode.currentText()
        unsafe = self.chk_unsafe.isChecked()
        autoscroll = self.chk_autoscroll.isChecked()
        log_path = self.le_log.text().strip() or LOG_DEFAULT
        pairs = [s.strip() for s in self.le_pairs.text().split(",") if s.strip()]
        default_lev = int(self.sp_lev.value())

        # REAL подтверждение
        if mode.upper() == "REAL":
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Подтверждение REAL")
            pairs_txt = ", ".join(pairs) if pairs else "—"
            msg.setText(
                f"<b>Режим REAL</b><br>"
                f"Лог: {log_path}<br>"
                f"Пары: {pairs_txt}<br>"
                f"Плечо: {default_lev}x<br><br>"
                f"Ты подтверждаешь запуск на реале?"
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if msg.exec_() != QMessageBox.Yes:
                return

        # перед стартом положим команды управления (пары/плечо)
        try:
            write_control(log_path, {"set_pairs": pairs, "default_lev": default_lev})
        except Exception:
            pass

        self.parent.run_screen.set_options(
            mode=mode, unsafe=unsafe, autoscroll=autoscroll, log_path=log_path,
            preset_pairs=pairs, default_lev=default_lev
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
        self.pairs: List[str] = []
        self.default_lev = 5
        self.leverage: Dict[str, int] = {}  # symbol -> int
        self.last_line_ts: Optional[float] = None
        self.session_started_at: Optional[str] = None
        self.session_started_ts: Optional[float] = None
        self.filter_error = False
        self.filter_trade = False
        self.filter_ml = False

        layout = QVBoxLayout()
        layout.setSpacing(12)

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
        self.lbl_session = QLabel("Сессия: —")
        self.lbl_session.setObjectName("SecondaryLabel")
        top.addWidget(self.lbl_status, 1)
        top.addWidget(self.lbl_mode, 1)
        top.addWidget(self.lbl_pairs, 2)
        top.addWidget(self.lbl_lev, 1)
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
        self.btn_panic = QPushButton("🛑 Закрыть всё и остановить")
        self.btn_pause = QPushButton("⏸ Пауза сигналов")
        self.btn_resume = QPushButton("▶ Возобновить сигналы")
        self.btn_update_pairs = QPushButton("🔁 Обновить пары/плечо")
        self.btn_stop.setProperty("destructive", True)
        self.btn_panic.setProperty("destructive", True)
        self.btn_pause.setProperty("secondary", True)
        self.btn_resume.setProperty("secondary", True)
        self.btn_update_pairs.setProperty("secondary", True)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_panic)
        btns.addWidget(self.btn_pause)
        btns.addWidget(self.btn_resume)
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
        self.btn_pause.clicked.connect(lambda: self.send_ctrl({"pause_entries": True}))
        self.btn_resume.clicked.connect(lambda: self.send_ctrl({"pause_entries": False}))
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

        # heartbeat
        self.lines_total = 0
        self.t_heartbeat = QTimer(self); self.t_heartbeat.setInterval(1000)
        self.t_heartbeat.timeout.connect(self.tick_heartbeat)
        self.t_heartbeat.start()

        # хоткеи экрана Run
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_pause_resume)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self.on_panic)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.on_stop)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.copy_tail)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=lambda: self.parent.goto_screen("report"))

    # --- внешние установки из MainMenu ---
    def set_options(self, mode: str, unsafe: bool, autoscroll: bool, log_path: str,
                    preset_pairs: List[str], default_lev: int):
        self.mode = mode
        self.unsafe = unsafe
        self.autoscroll = autoscroll
        self.chk_autoscroll.setChecked(autoscroll)
        self.log_path = log_path
        self.pairs = list(preset_pairs or [])
        self.default_lev = int(default_lev)
        unsafe_txt = "UNSAFE" if self.unsafe else "SAFE"
        color = "#ff4d4f" if self.unsafe else "#52c41a"
        self.lbl_mode.setText(f'Режим: <b>{self.mode}</b> / <span style="color:{color}">{unsafe_txt}</span>')

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
        self.session_started_at = now_iso()
        self.session_started_ts = time.time()
        self.lbl_session.setText(f"Сессия: {self.session_started_at}")

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
        self.proc.setProcessEnvironment(env)

        # python -X utf8 main.py [paper|real ...]
        args = ["-X", "utf8", main_path]
        if self.mode.upper() == "PAPER":
            args.append("paper")
        else:
            args += ["real", "--yes"]
            if self.unsafe:
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
        self.session_started_ts = None
        self.session_started_at = None
        self.lbl_session.setText("Сессия: —")
        if code != 0:
            QMessageBox.warning(self, "Процесс завершился с ошибкой",
                                f"Код возврата: {code}\nПроверь последние строки лога.")

    def on_stop(self):
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            self.parent.goto_screen("main")
            return
        self.append_line("[APP] Запрашиваю остановку…")
        self.proc.terminate()
        if not self.proc.waitForFinished(3000):
            self.append_line("[APP] Жёсткая остановка (kill).")
            self.proc.kill()

    def toggle_pause_resume(self):
        # простое переключение по Space
        if self.filter_error or self.filter_trade or self.filter_ml:
            # Space не меняет фильтры; управляем сигналами
            pass
        # Спросим состояние у пользователя? Тут просто шлём "pause_entries: True" / False по очереди не храним
        self.send_ctrl({"toggle_pause": True})
        self.append_line("[CTRL] toggle_pause → control.json")

    def on_panic(self):
        # команда в control.json + остановка процесса
        self.send_ctrl({"panic_close": True})
        self.append_line("[CTRL] panic_close → control.json")
        self.on_stop()

    def update_pairs_control(self):
        # перечитываем пары/плечо из MainMenu через parent
        pairs = self.parent.screens["main"].le_pairs.text()
        pairs = [s.strip() for s in pairs.split(",") if s.strip()]
        lev = int(self.parent.screens["main"].sp_lev.value())
        self.pairs = pairs
        self.default_lev = lev
        self.lbl_pairs.setText(f"Пары: {', '.join(self.pairs) if self.pairs else '—'}")
        self.lbl_lev.setText(f"Плечо: {self.default_lev}x (дефолт)")
        self.send_ctrl({"set_pairs": pairs, "default_lev": lev})
        self.append_line("[CTRL] обновил пары/плечо через control.json")

    def send_ctrl(self, payload: Dict[str, Any]):
        try:
            write_control(self.log_path, payload)
        except Exception as e:
            self.append_line(f"[CTRL] Ошибка записи control.json: {e}")

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

class ReportScreen(QWidget):
    """Экран отчётов: equity, фильтры, экспорт, PNG"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.log_path = LOG_DEFAULT
        self.session_only = False
        self.session_start_iso = None
        self.symbol_filter = None

        layout = QVBoxLayout()
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
        layout.addWidget(header)

        # фильтры
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
        layout.addLayout(filters)

        axis = DateAxisItem(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': axis})
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot, 1)

        # Статистика
        self.lbl_stats = QLabel("Trades: 0 | P+: 0.00 | P-: 0.00 | Win%: 0.0 | MaxDD: 0.00 | Median dur: 0m")
        self.lbl_stats.setObjectName("SecondaryLabel")
        layout.addWidget(self.lbl_stats)

        btns = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_back = QPushButton("⬅ Назад")
        self.btn_back.setProperty("secondary", True)
        btns.addWidget(self.btn_refresh)
        btns.addStretch(1)
        btns.addWidget(self.btn_back)
        layout.addLayout(btns)

        self.setLayout(layout)

        self.btn_back.clicked.connect(lambda: self.parent.goto_screen("main"))
        self.btn_refresh.clicked.connect(self.plot_balance)
        self.chk_session.toggled.connect(self.on_session_toggle)
        self.cmb_symbol.currentIndexChanged.connect(self.plot_balance)
        self.btn_export.clicked.connect(self.export_report)
        self.chk_autorefresh.toggled.connect(self._toggle_autorefresh)
        self.btn_reset_zoom.clicked.connect(lambda: self.plot.enableAutoRange('xy', True))
        self.btn_save_png.clicked.connect(self.save_png)

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

        # Max Drawdown по equity
        max_dd = 0.0
        peak = -1e18
        for v in equity:
            peak = max(peak, v)
            dd = peak - v
            if dd > max_dd:
                max_dd = dd

        # median duration
        if durations:
            durations.sort()
            med = durations[len(durations)//2]
        else:
            med = 0
        med_min = int(med // 60)

        winrate = (pos_cnt_p / max(1, pos_cnt_p + pos_cnt_n)) * 100.0

        self.plot.plot(ts_list, equity, pen=pg.mkPen(width=2), symbol='o', symbolSize=5)
        self.lbl_stats.setText(
            f"Trades: {pos_cnt_p+pos_cnt_n}  |  P+: {sum_p:+.2f}  |  P-: {sum_n:+.2f}  |  "
            f"Win%: {winrate:.1f}  |  MaxDD: {max_dd:.2f}  |  Median dur: {med_min}m"
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

# ----------------------------- контейнер приложения -----------------------------
class TradeApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.screens: Dict[str, QWidget] = {}
        self._cfg = load_config()
        self.current_theme = str(self._cfg.get("theme", "Auto") or "Auto")
        self.init_ui()
        self.install_hotkeys()

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

        for s in self.screens.values():
            self.addWidget(s)

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
            self.setCurrentWidget(w)

    def install_hotkeys(self):
        # F5 — перезапуск бота (если мы на Run экране)
        QShortcut(QKeySequence("F5"), self, activated=self.hotkey_restart)
        # Ctrl+L — очистка окна лога
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.hotkey_clear_log_view)
        # Ctrl+O — открыть папку лога (если задан)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.hotkey_open_log_folder)
        # Esc — назад в главное меню
        QShortcut(QKeySequence("Esc"), self, activated=lambda: self.goto_screen("main"))

    def hotkey_restart(self):
        if self.currentWidget() is self.run_screen:
            self.run_screen.on_stop()
            # маленькая пауза чтобы порт/файлы освободились
            QTimer.singleShot(400, self.run_screen.start_bot)

    def hotkey_clear_log_view(self):
        if self.currentWidget() is self.run_screen:
            self.run_screen.txt.clear()

    def hotkey_open_log_folder(self):
        if self.currentWidget() is self.run_screen:
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
