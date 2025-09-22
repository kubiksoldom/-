import os
import asyncio
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from dotenv import load_dotenv
from bybit_api import get_balance, has_open_position, get_margin_info
import importlib

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
bot_is_running = False

async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_is_running
    bot_is_running = False
    await update.message.reply_text("⏸ Бот поставлен на паузу.")

async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_is_running
    bot_is_running = True
    await update.message.reply_text("▶️ Бот возобновил работу.")

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bal = get_balance()
    await update.message.reply_text(f"Баланс: {bal:.2f} USDT")

async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Открытые позиции: (реализуй позже или добавим вывод из bybit_api)")

async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = get_margin_info()
    await update.message.reply_text(f"IM: {m['IM']}%, MM: {m['MM']}%\nEquity: {m['equity']} USDT")

async def reload_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    importlib.reload(__import__("config"))
    await update.message.reply_text("✅ Конфиг обновлён (hot-reload).")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("pause", pause))
    app.add_handler(CommandHandler("resume", resume))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("positions", positions))
    app.add_handler(CommandHandler("risk", risk))
    app.add_handler(CommandHandler("reload", reload_config))
    print("Запускаю Telegram-бота…")
    app.run_polling()

if __name__ == "__main__":
    main()
