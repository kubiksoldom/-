# Итоги аудита

## Критичные / блокирующие
- **Неверная разметка таргета в `build_ml_dataset_from_fills.py`** — функция `label_trade` читала high/low из индексов `[2]` и `[3]`, хотя свечи нормализуются как `[open, high, low, close, volume]`. Из-за этого high бралось из low, а low — из close, что обнуляло достижение TP и могло делать весь датасет отрицательным (`rows_ok=0`). Исправлено на корректные индексы с безопасным фолбэком для нестандартных строк. 【F:build_ml_dataset_from_fills.py†L533-L550】

## Существенные
- **ML-метаданные не полны** — `sanity_check.py` предупреждает об отсутствующих `thresholds.regime_ultra` и `atr_percentiles.p90`, поэтому при загрузке артефактов статус модели остаётся «degraded»/«unsafe» и трейдинг блокируется. Рекомендуется переобучить модель и обновить `model_meta.json`, чтобы закрыть предупреждения. 【F:sanity_check.py†L285-L290】【F:ml_veto.py†L325-L363】
- **Почему `rows_ok=0` при сборе датасета** — каждая причина фильтра записывается в `DROP_COUNTERS` и логируется: типичные блоки — пустая история (`no_history`), нулевой объём (`thin_volume`) и пустые снапшоты (`empty_snapshot`). Для диагностики стоит: (1) проверить, что `fills_all.csv` содержит положительные `qty`; (2) запустить скрипт с `ENABLE_DISK_CACHE=0` и временно печатать `DROP_COUNTERS` чаще; (3) убедиться, что API-доступ не режется прокси/ключами, иначе Bybit вернёт пустые свечи/снапшоты. 【F:build_ml_dataset_from_fills.py†L553-L592】【F:build_ml_dataset_from_fills.py†L836-L849】

## Минорные / рефакторинг
- **Импортные циклы `utils ↔ bybit_api ↔ api_guard`** — цикл не падает, потому что импорт Bybit внутри `utils.adjust_qty` ленивый, но для читаемости можно вынести фильтры в отдельный модуль или инжектить зависимости. 【F:utils.py†L811-L826】【F:bybit_api.py†L14-L61】【F:api_guard.py†L1-L207】
- **`sanity_check` не проверяет `trade_app` финальные резюме** — скрипт покрывает ключевые файлы, но можно расширить его: например, дернуть `trade_app.write_control`/`RunScreen._calc_session_stats` с синтетическим логом, чтобы гарантировать корректность UI-резюме. 【F:sanity_check.py†L294-L308】【F:trade_app.py†L1360-L1420】

## Таблица контрактов
| Caller | Callee | Контракт | Статус |
| --- | --- | --- | --- |
| `main.decide_cycle` (`main.py`) | `strategy.decide_with_router` | `(symbol: str, timeframe: str, candles: List[List[float]], ctx: Dict) -> {action, reason, sl, tp, meta}` | OK — сигнатура и структура результата совпадают. 【F:main.py†L2268-L2320】【F:strategy.py†L1059-L1099】 |
| `main` | `utils.write_cycle_log` | `write_cycle_log(data: Dict[str, Any])` пишет JSONL с `ts_utc`. | OK — функция потокобезопасна и добавляет таймштамп. 【F:main.py†L1928-L1954】【F:utils.py†L145-L164】 |
| `main` | `ml_veto.load_model_and_meta` / `predict_ok` | Возвращает `(model, meta)` с валидацией weekly precision; `predict_ok` → `(ok, proba, thr, factor, band)` и подгружает свечи/снапшоты. | OK, но при деградации модель возвращается `None`, поэтому требуется корректная мета. 【F:main.py†L2268-L2331】【F:ml_veto.py†L325-L458】 |
| `main` | `broker` (`paper_engine`/`bybit_api`) | Используются `get_balance`, `get_tickers_linear`, `place_market_order`, `filters_reliable`. Интерфейс синхронизирован. | OK — `paper_engine` проксирует функции реального брокера. 【F:main.py†L1169-L1236】【F:paper_engine.py†L40-L210】【F:bybit_api.py†L258-L422】 |
| `bybit_api` | `api_guard.safe_request` | Все REST-вызовы идут через ретраи/бекофф, retCode 10001 пробрасывается наверх. | OK. 【F:bybit_api.py†L184-L233】【F:api_guard.py†L220-L342】 |
| `bybit_api` | `utils.log/tg_send/adjust_qty` | Утилиты используются при логировании и нормализации объёма. | OK. 【F:bybit_api.py†L16-L61】【F:utils.py†L131-L205】 |
| `trade_app.RunScreen` | `main.py` (через `control.json`, QProcess) | Ожидает поддержку команд `stop`, `panic`, `set_pairs`, `pause_entries`. | OK — `main` обрабатывает эти команды в `_process_control`. 【F:trade_app.py†L1285-L1700】【F:main.py†L1479-L1899】 |
| `config.py` | Все модули | Константы/ENV читаются через `env_*` и остаются совместимыми с прежними именами. | OK. 【F:config.py†L1-L233】 |

## Таблица проблем
| Файл:строка | Severity | Описание | Решение |
| --- | --- | --- | --- |
| `build_ml_dataset_from_fills.py:533-550` | Critical | Неверные индексы high/low в `label_trade` искажали таргет (TP/SL никогда не срабатывал) → датасет пустой/смещённый. | Переключить индексы на `[1]`/`[2]`, добавить комментарий и более безопасный фолбэк. 【F:build_ml_dataset_from_fills.py†L533-L550】 |
| `model_meta.json` (проверка `sanity_check.py`) | Major | Нет `thresholds.regime_ultra` и `atr_percentiles.p90`, `ml_veto.load_model_and_meta` помечает артефакты как небезопасные. | Переобучить модель (`manage_ml.py retrain`) и обновить `model_meta.json`, либо вручную добавить пороги согласно актуальной статистике. 【F:sanity_check.py†L285-L290】【F:ml_veto.py†L325-L363】 |
| `build_ml_dataset_from_fills.py:553-592` | Major | При `rows_ok=0` фильтры `no_history/thin_volume/empty_snapshot` выбрасывают строки; без диагностики сложно найти первопричину. | Использовать `DROP_COUNTERS` из лога, проверить целостность `fills_all.csv`, отключить кеш для проверки, валидировать сетевые ответы Bybit. 【F:build_ml_dataset_from_fills.py†L553-L592】【F:build_ml_dataset_from_fills.py†L836-L849】 |
| `utils.py` ↔ `bybit_api.py` | Minor | Импортный цикл затрудняет сопровождение, хоть и безопасен из-за ленивого импорта. | Вынести фильтры/логику в отдельный модуль или передавать зависимости функциями. 【F:utils.py†L811-L826】【F:bybit_api.py†L14-L61】 |
| `sanity_check.py` | Minor | Не покрывает UI-потоки `trade_app`, поэтому регрессии в финальном резюме не ловятся автоматикой. | Добавить лёгкие вызовы `trade_app` утилит (например, `write_control` и `_calc_session_stats`) с тестовыми данными. 【F:sanity_check.py†L294-L308】【F:trade_app.py†L1360-L1420】 |

## Граф зависимостей (основные модули)
- `main.py` → `config`, `utils`, `strategy`, `ml_veto`, `bybit_api`/`paper_engine`, `api_guard` (через брокера)
- `strategy.py` → `utils`, `config`
- `ml_veto.py` → `config`, `utils`, `bybit_api`
- `bybit_api.py` → `api_guard`, `utils`, `config`
- `paper_engine.py` → `bybit_api`, `config`, `utils`
- `trade_app.py` → `config`, `utils`, `bybit_api`, `main.py` (через управление)
- `api_guard.py` → `utils`

Циклы: `utils` ⇄ `bybit_api` (ленивый импорт внутри утилит), `utils` ⇄ `bybit_api` ⇄ `api_guard`. 【8825fc†L1-L3】
