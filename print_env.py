import os

print("=== ENV (os.environ) ===")
print("API_KEY:", os.getenv("BYBIT_API_KEY"))
print("SECRET:", os.getenv("BYBIT_API_SECRET"))
print("TG:", os.getenv("TELEGRAM_TOKEN"))
print("CHAT:", os.getenv("TELEGRAM_CHAT_ID"))
