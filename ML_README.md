# ML pipeline overview

## Данные
- Источник сделок: `fills_all.csv` (или путь из `FILLS_PATH`).
- Конвертация в датасет: `python build_ml_dataset_from_fills.py --input=fills_all.csv --out=ml_dataset.csv --timeframe=1m --force`.
- Итоги сборки фиксируются в `ml_dataset_summary.json` и `ml_dataset_summary.txt`.

## Обучение
```
python train_ml_model.py --input=ml_dataset.csv --model_out=rf_model.pkl --meta_out=model_meta.json --test_size=0.2 --random_state=42
```
- Модель: `RandomForestClassifier` + `CalibratedClassifierCV(sigmoid)`.
- Метрики (ROC/PR/accuracy/precision/recall на порогах) и список фичей сохраняются в `model_meta.json`.

## Отладка офлайн
```
python ml_debug_report.py --dataset=ml_dataset.csv --model=rf_model.pkl --meta=model_meta.json
```
- Генерирует `ML_REPORT.md` с разбиением по интервалам вероятностей и фактической винрейтом.

## Включение/выключение в боте
- Флаги в `config.py`: `ML_USE_NEW_ON`, `ML_THRESHOLD`, `ML_IGNORE_WEEKLY`, `ML_SHADOW_MODE`.
- Порог можно задать явно через `ML_THRESHOLD`, иначе берётся рекомендация из `model_meta.json`.
- Shadow-режим (`ML_USE_NEW_ON=0` или `ML_SHADOW_MODE=1`) пишет решения в `ml_shadow.log`, не влияя на маршрутизацию сделок.
