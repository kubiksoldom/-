# Trading Bot

Автоматизированный трейдинг-бот для линейных USDT-инструментов Bybit. Код ориентирован на бумажный режим и безопасный запуск по умолчанию.

## Быстрый старт

1. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Скопируйте пример `.env`:
   ```bash
   cp .env.example .env
   ```
3. Заполните `BYBIT_API_KEY` и `BYBIT_API_SECRET`. Для уведомлений укажите `TELEGRAM_TOKEN` и `TELEGRAM_CHAT_ID` (необязательно).
4. Проверьте окружение:
   ```bash
   python sanity_check.py
   ```

## Безопасный запуск

- Бумажный режим (по умолчанию безопасный):
  ```bash
  python main.py paper
  ```
- Реальный режим (требует явного подтверждения):
  ```bash
  python main.py real --yes
  ```
- Быстрый разовый прогон (одна итерация цикла):
  ```bash
  python main.py paper --once
  ```
- CI/автопроверка (форсирует SAFE_MODE и PAPER_MODE, отключает Telegram):
  ```bash
  python main.py paper --ci --once
  ```

В режиме CI логирование ускорено, SAFE_MODE и PAPER_MODE всегда включены. Флаг `--unsafe` доступен только в реальном режиме без `--ci`.

## Управление во время работы

- `Space` в UI (`trade_app.py`) отправляет `{"toggle_pause": true}` в `control.json` — bot переключает паузу входов и сохраняет состояние в `control_state.json`.
- Кнопки «Пауза» и «Возобновить» отправляют `{"pause_entries": true|false}`.
- Кнопка «Паника» посылает `{"panic_close": true}` и инициирует закрытие позиций.

## Проверки и CI

- `python sanity_check.py` — статическая проверка окружения, моделей и директорий.
- GitHub Actions workflow `.github/workflows/bot-ci.yml` запускает sanity check и бумажный smoke-тест (`python main.py paper --ci --once`).

## Журналы и данные

Логи и данные сохраняются в директории, заданной переменной `DATA_ROOT` (по умолчанию `./data`). JSONL-лог цикла определяется переменной `LOG_JSONL`. Скрипт `sanity_check.py` создаёт недостающие директории автоматически.

