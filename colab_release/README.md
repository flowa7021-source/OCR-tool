# Colab release: готовые файлы для fine-tune EasyOCR

Содержимое этой папки — снепшот на момент коммита, чтобы запустить
fine-tune `cyrillic_g2.pth` в Google Colab без локальной подготовки.

## Файлы

- **`finetune_ru_colab.zip`** (~57 МБ) — датасет `finetune_ru/` с
  **22 541 word-crop'ами** = 2539 real (GT-verified через Левенштейн-2
  для ТЕКСТА, exact-only для ЧИСЕЛ) + **20 000 synth** (`synth_data.py`
  в 9 шрифтах DejaVu/Liberation/FreeFont, размеры 18-36 px, degradation
  intensity до 0.6). Split 80/20: 17969 train + 4571 val.
  Real stats: seen=8242, dropped_lowconf=2492, dropped_nogt=3205,
  exact=1223, fuzzy=1316. Numeric-strict фильтр (commit ca923f6)
  выкинул ~390 false-positive подстановок типа `22,391 → 2239`.
  Synth weighting: ``freq^0.6`` — редкие слова получают 5-15 примеров,
  hot words (цифры, ``ооо``, ``упд``) не перекрывают хвост.
- **`colab_cell.py`** (~5.6 КБ) — единая ячейка для Colab notebook:
  mount Drive → pip install deps → clone EasyOCR trainer (sparse)
  → patch `dataset.py` для PyTorch 2.x / Python 3 → fetch stock
  weights → write YAML config → train → скопировать
  `best_accuracy.pth` обратно в Drive.

- **`kaggle_cell.py`** (~8 КБ) — вариант для Kaggle Notebook.
  Отличия от Colab: (1) нет Drive mount — всё в `/kaggle/working/`
  персистентно на сессию, (2) **background-поток синхронит
  `best_accuracy.pth` каждые 60 с в output-папку** — даже если
  session помрёт, последний чекпоинт скачиваемый, (3) **12-часовые
  сессии без idle-отключения** → можно закрыть ноутбук и идти
  спать, (4) P100 / T4 × 2 доступны, 30 ч/нед free tier.

## Быстрый запуск (Colab)

1. `colab.research.google.com` → New notebook → Runtime → Change
   runtime type → T4 GPU.
2. Скопировать содержимое `colab_cell.py` в одну Colab-ячейку →
   Shift+Enter. Ячейка сама скачает ZIP + веса с GitHub.
3. ~25-30 мин → `best_accuracy.pth` в `My Drive/finetune_ru/out/`.
4. Скачать `best_accuracy.pth` → положить в
   `resources/easyocr_models/cyrillic_g2.pth` → `python build.py`.

**Важно про Colab**: тариф free отключает runtime через 90 мин
idle. Не сворачивайте вкладку надолго — чекпоинт копируется на
Drive только в конце. Если нужна устойчивость к закрытию
браузера, берите Kaggle.

## Быстрый запуск (Kaggle — рекомендуется если хотите уйти спать)

1. `kaggle.com` → Create → New Notebook.
2. Правая панель → Accelerator → **GPU P100** (быстрее T4) или T4 x 2.
   Internet → **ON**.
3. Вставить содержимое `kaggle_cell.py` в одну ячейку → **Run All**
   (Ctrl+F9). Можно закрывать крышку ноутбука — Kaggle держит
   сессию до 12 ч.
4. Во время обучения `best_accuracy.pth` синхронится в
   `/kaggle/working/out/` каждые 60 с. Если сессия умрёт — скачаете
   последний чекпоинт из правой панели → Output.
5. После `end the training` → скачать `best_accuracy.pth` из
   Output-панели ноутбука → положить в
   `resources/easyocr_models/cyrillic_g2.pth` → `python build.py`.

Подробности, известные грабли и rollback — в
`scripts/finetune/README.md`.

## Пересборка файлов

Оба файла можно пересоздать из исходников репо:

```bash
# ZIP с датасетом (если inputs/*.pdf + *.txt на месте)
python scripts/finetune/bootstrap_dataset.py
cd datasets && zip -r ../colab_release/finetune_ru_colab.zip \
  finetune_ru/training/labels.csv \
  finetune_ru/training/*.jpg \
  finetune_ru/validation/labels.csv \
  finetune_ru/validation/*.jpg

# Colab cell
python scripts/finetune/colab_finetune.py > colab_release/colab_cell.py
```
