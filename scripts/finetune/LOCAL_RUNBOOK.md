# Local fine-tune runbook (RTX 2060 S / 3060 / 4060, 8 GB VRAM)

Сценарий: один прогон на домашней машине с NVIDIA GPU,
результат на первом же запуске, объективные метрики до и после.

## Требования

- Windows 10/11 или Linux с актуальным NVIDIA-драйвером
- GPU ≥ 8 GB VRAM (RTX 2060 S / 3060 / 4060 и выше)
- 2 GB свободного диска (`.finetune_trainer/` + чекпоинты)
- ≈ 45 минут времени не трогать машину

## Установка зависимостей (один раз)

Открыть PowerShell в папке `OCR-tool` (Windows) или bash (Linux):

```powershell
# 1. Проверить, что уже установлено
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Ожидаемый вывод: `CUDA: True NVIDIA GeForce RTX 2060 SUPER`.

Если `False` (PyTorch без CUDA):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Trainer-зависимости (не входят в основной `requirements.txt`):

```powershell
pip install pandas natsort lmdb nltk fire jiwer
```

## Подготовка датасета

Распаковать готовый ZIP из `colab_release/` (быстрее, чем bootstrap по новой):

```powershell
python -c "import zipfile; zipfile.ZipFile('colab_release/finetune_ru_colab.zip').extractall('datasets')"
```

Опционально — увеличить вес real-crops (против synth-доминирования):

```powershell
# Real крепы начинают весить 3x → доля real в тренировочных update'ах растёт с 11% до 27%
python scripts/finetune/oversample_real.py --multiplier 3
```

## Prefetch stock-весов (если ещё не делали)

```powershell
python scripts/prefetch_easyocr_models.py
```

Загружает `cyrillic_g2.pth` + `craft_mlt_25k.pth` в `resources/easyocr_models/`.

## Полный прогон одной командой

```powershell
python scripts/finetune/run_local.py
```

Что произойдёт:

1. **Preflight-checks** (несколько секунд):
   - Проверяет CUDA, VRAM, версию PyTorch
   - Проверяет наличие всех Python-пакетов trainer'а
   - Проверяет `datasets/finetune_ru/{training,validation}/labels.csv`
   - Проверяет базовые веса в `resources/easyocr_models/`
   - Проверяет свободное место на диске

2. **Baseline metrics** (2-5 мин):
   - Копирует stock-веса в EasyOCR-кеш
   - Прогоняет `real_ocr_metrics.py` на `inputs/*.pdf`
   - Сохраняет `reports/finetune_run/metrics_baseline.json`
   - Печатает stock CER/WER/conf

3. **Fine-tune** (30-45 мин на RTX 2060 S):
   - Автоматически клонит EasyOCR trainer, патчит его под PyTorch 2.x
   - batch_size=16 (safe для 8 GB), valInterval=100, num_iter=30000
   - Сохраняет лучшие веса в `resources/easyocr_models/cyrillic_g2.pth`
   - Stock бэкапится в `.pth.stock`

4. **Post-train metrics** (2-5 мин):
   - Копирует новые веса в EasyOCR-кеш
   - Прогоняет метрики ещё раз
   - Сохраняет `reports/finetune_run/metrics_trained.json`

5. **Comparison report**:

```
═══════════════════════════════════════════════════════════════════
  Baseline (stock) vs. Fine-tuned — average over golden corpus
═══════════════════════════════════════════════════════════════════
  metric         before      after          Δ
  ────────────────────────────────────────────
  CER            XXX.X%      XX.X%     -XX.Xpp
  WER            XXX.X%      XX.X%     -XX.Xpp
  mean_conf       XX.X%      XX.X%     +XX.Xpp
═══════════════════════════════════════════════════════════════════
  ✓ Fine-tune IMPROVED accuracy — keep the new weights.
```

## Auto-apply вариант

Не хотите руками решать — передайте порог улучшения:

```powershell
python scripts/finetune/run_local.py --auto-apply-if-better 3.0
```

Если CER упал хотя бы на 3 процентных пункта — новые веса остаются.
Иначе — автоматический rollback на stock.

## Что делать с результатом

**Если accuracy улучшилась** (Δ CER отрицательный):

```powershell
python build.py
```

Пересобирает инсталлятор с дообученными весами.

**Если accuracy не изменилась** (Δ ≈ 0, ±1 pp):

- Попробовать `--epochs 60` (больше итераций)
- Или запустить `python scripts/finetune/active_learning.py --top 200`
  и вручную разметить самые информативные ошибки, затем второй заход

**Если accuracy ухудшилась** (Δ CER положительный):

```powershell
# Rollback на stock-веса
copy resources\easyocr_models\cyrillic_g2.pth.stock resources\easyocr_models\cyrillic_g2.pth
```

Это гарантированно не хуже, чем было.

## Флаги run_local.py

| Флаг | Для чего |
|------|----------|
| `--preflight-only` | Только проверка окружения, без тренировки |
| `--skip-baseline` | Не считать baseline (если уже делали) |
| `--skip-metrics` | Только train, без до/после метрик (быстрее на 5-10 мин) |
| `--epochs N` | Число 1000-итерационных «эпох» (default 30) |
| `--auto-apply-if-better PP` | Rollback если Δ CER улучшение меньше порога в процентных пунктах |

## Если поймали OOM

Признак: `RuntimeError: CUDA out of memory` во время training.

Причина: другие процессы захватили VRAM (игры, браузер с GPU-рендерингом, Discord).

Фикс:
1. Закрыть всё что использует GPU (игры, браузеры)
2. Перезапустить: `python scripts/finetune/run_local.py`

Если всё равно OOM — в `scripts/finetune/finetune_recognizer.py` в YAML-шаблоне изменить `batch_size: 16` на `batch_size: 8` и перезапустить.

## Мониторинг во время training

Training пишет в stdout каждые 100 итераций. Полезные сигналы:

- **Train loss ↓, Valid loss ↓, Accuracy ↑** — всё хорошо, идёт обучение
- **Train loss ↓, Valid loss ↑ (после 2000-5000 iter)** — overfitting на тренировочный set,
  надо остановить и уменьшить `--epochs`
- **Loss NaN** — взорвалась gradient-norm, редко на FT, если увидели — уменьшить `lr` до 0.0001 в cfg

## Контрольная точка

После `[100/30000] Train loss: ...` должен быть **первый Current_accuracy** в районе **40-70%** (сравнимо с stock). Если Current_accuracy < 10% и loss застрял — что-то не так с дата-set'ом, прервать и проверить `datasets/finetune_ru/training/labels.csv` вручную.
