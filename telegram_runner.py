import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from telegram import KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import bybit_api
import config
import paper_engine
from pair_selector import select_pairs_from_config

TELEGRAM_TOKEN = str(getattr(config, "TELEGRAM_TOKEN", "") or "").strip()
TELEGRAM_CHAT_ID = str(getattr(config, "TELEGRAM_CHAT_ID", "") or "").strip()
TG_REPLY_KEYBOARD = bool(int(getattr(config, "TG_REPLY_KEYBOARD", 1)))

BTN_START = "▶️ Запустить"
BTN_STOP = "⛔ Стоп входов"
BTN_STATUS = "📊 Статус"
BTN_ML = "🧠 ML"
BTN_PAPER = "🧪 Paper"
BTN_HELP = "❓ Помощь"
BTN_ML_ON = "🟢 ML ON"
BTN_ML_BYPASS = "🔴 ML BYPASS"
BTN_BACK = "↩️ Назад"
BTN_PAPER_REFRESH = "🔄 Обновить paper статус"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _control_path() -> str:
    log_path = os.getenv("LOG_JSONL", getattr(config, "LOG_JSONL", "bot_cycle_log.jsonl"))
    folder = os.path.abspath(os.path.dirname(log_path) or ".")
    return os.path.join(folder, "control.json")


def _control_state_path() -> str:
    folder = os.path.abspath(os.path.dirname(_control_path()) or ".")
    return os.path.join(folder, "control_state.json")


def _load_control_data() -> List[Dict[str, Any]]:
    path = _control_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    except Exception:
        pass
    return []


def _parse_ts(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _write_control(payload: Dict[str, Any]) -> None:
    cpath = _control_path()
    data = _load_control_data()
    cmd = dict(payload)
    cmd.setdefault("cmd", "telegram_control")
    cmd.setdefault("ts", _utc_now_iso())
    cmd.setdefault("cmd_id", uuid.uuid4().hex)
    data.append(cmd)

    target = Path(cpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def _control_state_data() -> Dict[str, Any]:
    state_path = _control_state_path()
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return dict(data)
    except Exception:
        pass
    return {}


def _read_ml_block_disabled() -> bool:
    state_data = _control_state_data()
    if "ml_block_disabled" in state_data:
        return bool(state_data.get("ml_block_disabled"))
    return bool(int(getattr(config, "DISABLE_ML_BLOCK", 1)))


def _read_pause_entries() -> bool:
    state_path = _control_state_path()
    try:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict) and "pause_entries" in state:
                return bool(state.get("pause_entries"))
    except Exception:
        pass

    latest_pause: Optional[bool] = None
    latest_ts = 0.0
    for row in _load_control_data():
        if "pause_entries" not in row:
            continue
        ts_val = _parse_ts(row.get("ts"))
        if ts_val >= latest_ts:
            latest_ts = ts_val
            latest_pause = bool(row.get("pause_entries"))
    return bool(latest_pause) if latest_pause is not None else False


def _set_pause_entries(paused: bool, cmd_name: str) -> None:
    _write_control(
        {
            "cmd": cmd_name,
            "source": "telegram_runner",
            "pause_entries": bool(paused),
        }
    )


def _set_ml_block_disabled(block_disabled: bool, cmd_name: str) -> None:
    _write_control(
        {
            "cmd": cmd_name,
            "source": "telegram_runner",
            "set_ml_block_disabled": bool(block_disabled),
        }
    )


def _active_pairs_preview() -> str:
    try:
        pairs = list(select_pairs_from_config())
    except Exception:
        pairs = []
    if not pairs:
        return "-"
    shown = pairs[:6]
    tail = "…" if len(pairs) > 6 else ""
    return ",".join(shown) + tail


def _render_status() -> str:
    paper_mode = bool(int(getattr(config, "PAPER_MODE", 1)))
    pause_entries = _read_pause_entries()
    mode = "PAPER" if paper_mode else "REAL"
    trade_status = "PAUSED" if pause_entries else "RUNNING"
    ml_enabled = bool(int(getattr(config, "ML_VETO_ENABLED", 1)))
    ml_block_disabled = _read_ml_block_disabled()
    ml_mode = "BYPASSED" if ml_block_disabled else "ACTIVE"

    current_balance = 0.0
    balance_source = "n/a"
    try:
        if paper_mode:
            current_balance = float(paper_engine.get_balance())
            balance_source = str(getattr(paper_engine, "get_balance_source", lambda: "virtual")() or "virtual")
        else:
            current_balance = float(bybit_api.get_balance())
            balance_source = "exchange"
    except Exception:
        balance_source = "fallback"

    pairs = _active_pairs_preview()
    return (
        f"MODE: {mode}\n"
        f"TRADE_STATUS: {trade_status}\n"
        f"PAUSE_ENTRIES: {pause_entries}\n"
        f"PAPER_BALANCE_SOURCE: {balance_source}\n"
        f"CURRENT_BALANCE: {current_balance:.2f}\n"
        f"ML_ENABLED: {ml_enabled}\n"
        f"ML_BLOCK_DISABLED: {ml_block_disabled}\n"
        f"ML_MODE: {ml_mode}\n"
        f"ACTIVE_PAIRS: {pairs}"
    )


def _render_paper_status() -> str:
    paper_mode = bool(int(getattr(config, "PAPER_MODE", 1)))
    paper_sync = bool(int(getattr(config, "PAPER_SYNC_BALANCE", 1)))
    source = "fallback"
    balance = 0.0
    try:
        source = str(getattr(paper_engine, "get_balance_source", lambda: "virtual")() or "virtual")
    except Exception:
        source = "fallback"
    try:
        balance = float(paper_engine.get_balance())
    except Exception:
        source = "fallback"
    return (
        f"PAPER_MODE: {paper_mode}\n"
        f"PAPER_SYNC_BALANCE: {paper_sync}\n"
        f"PAPER_BALANCE_SOURCE: {source}\n"
        f"CURRENT_PAPER_BALANCE: {balance:.2f}"
    )


def _main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(BTN_START), KeyboardButton(BTN_STOP)],
        [KeyboardButton(BTN_STATUS), KeyboardButton(BTN_ML)],
        [KeyboardButton(BTN_PAPER), KeyboardButton(BTN_HELP)],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, is_persistent=True)


def _ml_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(BTN_ML_ON), KeyboardButton(BTN_ML_BYPASS)],
        [KeyboardButton(BTN_BACK)],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, is_persistent=True)


def _paper_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(BTN_PAPER_REFRESH)],
        [KeyboardButton(BTN_BACK)],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, is_persistent=True)


async def _show_main_keyboard(update: Update, text: str) -> None:
    message = update.effective_message
    if not message:
        return
    if TG_REPLY_KEYBOARD:
        print("[TG] Reply keyboard shown")
        await message.reply_text(text, reply_markup=_main_keyboard())
        return
    await message.reply_text(text)


async def _handle_start_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("[TG] Button start pressed")
    _set_pause_entries(False, "start_bot")
    await _show_main_keyboard(update, "Новые входы разрешены")


async def _handle_stop_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("[TG] Button stop pressed")
    _set_pause_entries(True, "stop_bot")
    await _show_main_keyboard(update, "Новые входы остановлены, открытые позиции сопровождаются")


async def _handle_status_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("[TG] Button status pressed")
    await _show_main_keyboard(update, _render_status())


def _is_authorized_chat(update: Update) -> bool:
    chat_id = str((update.effective_chat.id if update.effective_chat else "") or "").strip()
    if TELEGRAM_CHAT_ID and chat_id == TELEGRAM_CHAT_ID:
        return True
    print(f"[TG] Unauthorized chat ignored: {chat_id}")
    return False


async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    print("[TG] /start_bot received")
    await _handle_start_action(update, context)


async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    print("[TG] /stop_bot received")
    await _handle_stop_action(update, context)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    print("[TG] /status received")
    await _handle_status_action(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    await _show_main_keyboard(
        update,
        "Доступные команды:\n"
        f"{BTN_START} — разрешить новые входы\n"
        f"{BTN_STOP} — запретить новые входы\n"
        f"{BTN_STATUS} — краткий статус бота\n"
        f"{BTN_ML} — ML active/bypass\n"
        f"{BTN_PAPER} — paper статус\n"
        "/start_bot — разрешить новые входы\n"
        "/stop_bot — запретить новые входы\n"
        "/status — краткий статус бота\n"
        "/help — список команд"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    await _show_main_keyboard(update, "Пульт управления активирован. Используйте кнопки ниже.")


def _normalize_text(update: Update) -> str:
    msg = update.effective_message
    if not msg or msg.text is None:
        return ""
    return str(msg.text).strip()


def _ml_mode_line() -> str:
    return "ML_MODE: BYPASSED" if _read_ml_block_disabled() else "ML_MODE: ACTIVE"


async def _show_ml_menu(update: Update) -> None:
    message = update.effective_message
    if not message:
        return
    if TG_REPLY_KEYBOARD:
        await message.reply_text(
            "ML управление:\n"
            "🟢 ML ON — ML блокирует сделки\n"
            "🔴 ML BYPASS — ML не блокирует\n"
            f"{_ml_mode_line()}",
            reply_markup=_ml_keyboard(),
        )
        return
    await message.reply_text("ML управление недоступно без TG_REPLY_KEYBOARD=1")


async def _show_paper_menu(update: Update) -> None:
    message = update.effective_message
    if not message:
        return
    if TG_REPLY_KEYBOARD:
        await message.reply_text(_render_paper_status(), reply_markup=_paper_keyboard())
        return
    await message.reply_text(_render_paper_status())


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    txt = _normalize_text(update)
    if not txt:
        return

    if txt == BTN_START:
        await _handle_start_action(update, context)
        return
    if txt == BTN_STOP:
        await _handle_stop_action(update, context)
        return
    if txt == BTN_STATUS:
        await _handle_status_action(update, context)
        return
    if txt == BTN_ML:
        await _show_ml_menu(update)
        return
    if txt == BTN_PAPER:
        print("[TG] Button paper status pressed")
        await _show_paper_menu(update)
        return
    if txt == BTN_HELP:
        await help_command(update, context)
        return
    if txt == BTN_ML_BYPASS:
        print("[TG] Button ML bypass pressed")
        _set_ml_block_disabled(True, "set_ml_bypass")
        print("[TG] ML mode set to BYPASSED")
        await _show_ml_menu(update)
        return
    if txt == BTN_ML_ON:
        print("[TG] Button ML active pressed")
        _set_ml_block_disabled(False, "set_ml_active")
        print("[TG] ML mode set to ACTIVE")
        await _show_ml_menu(update)
        return
    if txt == BTN_PAPER_REFRESH:
        print("[TG] Button paper status pressed")
        await _show_paper_menu(update)
        return
    if txt == BTN_BACK:
        await _show_main_keyboard(update, "Главное меню")
        return


def main() -> None:
    if not TELEGRAM_TOKEN:
        print("[TG] TELEGRAM_TOKEN missing — telegram_runner disabled")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("start_bot", start_bot))
    app.add_handler(CommandHandler("stop_bot", stop_bot))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, button_handler))
    print("Запускаю Telegram-бота…")
    app.run_polling()


if __name__ == "__main__":
    main()
