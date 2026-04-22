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
  weights → write YAML config → train 100 эпох → скопировать
  `best_accuracy.pth` обратно в Drive.

## Быстрый запуск

1. Распаковать `finetune_ru_colab.zip` у себя локально →
   получится папка `finetune_ru/`.
2. Залить `finetune_ru/` в Google Drive как `My Drive/finetune_ru/`.
3. `colab.research.google.com` → New notebook → Runtime → Change
   runtime type → T4 GPU.
4. Скопировать содержимое `colab_cell.py` в одну Colab-ячейку →
   Shift+Enter → ~1.5 ч → `best_accuracy.pth` в
   `My Drive/finetune_ru/out/`.
5. Скачать `best_accuracy.pth` → положить в
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
