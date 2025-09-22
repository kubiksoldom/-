# Changelog Bybit Grid Bot


## 2025-09-22
- [UPD] Обновлён requirements.txt: удалены фиктивные зависимости и добавлены реальные модули UI/ML.
- [DOC] Перед запуском trade_app установите зависимости: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## 2025-06-17
- [NEW] Добавлен план: управление через Telegram (/start, /stop, /status)
- [NEW] Будет таймер в одном сообщении, поддержка testnet/mainnet статуса
- [FIX] Сделана финальная автоочистка через force_close_all_positions

## 2025-06-16
- [NEW] Добавлен Telegram-отчёт о старте/финише
- [NEW] Авто-выбор лучших пар для торговли
- [UPD] Переход к работе без ручного подтверждения очистки

---

## TODO:
- [ ] Полная интеграция Telegram-команд
- [ ] Ведение changelog автоматически
- [ ] Больше аналитики после работы
