import os
from dotenv import load_dotenv

print("Path .env:", os.path.abspath(".env"))
load_dotenv(".env")

print("AFTER LOAD:")
for k in ["BYBIT_API_KEY", "BYBIT_API_SECRET", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]:
    print(k, "=", os.getenv(k))
