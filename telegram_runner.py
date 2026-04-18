import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import bybit_api
import config
import paper_engine
from pair_selector import select_pairs_from_config

TELEGRAM_TOKEN = str(getattr(config, "TELEGRAM_TOKEN", "") or "").strip()
TELEGRAM_CHAT_ID = str(getattr(config, "TELEGRAM_CHAT_ID", "") or "").strip()


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
    ml_block_disabled = bool(int(getattr(config, "DISABLE_ML_BLOCK", 1)))
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
    _set_pause_entries(False, "start_bot")
    print("[TG] Bot started via Telegram")
    await update.message.reply_text("Бот запущен: новые входы разрешены")


async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    print("[TG] /stop_bot received")
    _set_pause_entries(True, "stop_bot")
    print("[TG] Bot stopped via Telegram (new entries disabled)")
    await update.message.reply_text("Бот остановлен: новые входы запрещены, открытые позиции сопровождаются")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    print("[TG] /status received")
    await update.message.reply_text(_render_status())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized_chat(update):
        return
    await update.message.reply_text(
        "Доступные команды:\n"
        "/start_bot — разрешить новые входы\n"
        "/stop_bot — запретить новые входы\n"
        "/status — краткий статус бота\n"
        "/help — список команд"
    )


def main() -> None:
    if not TELEGRAM_TOKEN:
        print("[TG] TELEGRAM_TOKEN missing — telegram_runner disabled")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start_bot", start_bot))
    app.add_handler(CommandHandler("stop_bot", stop_bot))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("help", help_command))
    print("Запускаю Telegram-бота…")
    app.run_polling()


if __name__ == "__main__":
    main()
