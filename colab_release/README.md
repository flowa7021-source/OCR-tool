# Colab release: готовые файлы для fine-tune EasyOCR

Содержимое этой папки — снепшот на момент коммита, чтобы запустить
fine-tune `cyrillic_g2.pth` в Google Colab без локальной подготовки.

## Файлы

- **`finetune_ru_colab.zip`** (~27 МБ) — датасет `finetune_ru/` с
  6630 word-crop'ами (4662 train + 1139 val, по `labels.csv`) плюс
  лишние jpg-файлы предыдущего прогона — trainer читает только то,
  что перечислено в `labels.csv`, остальное игнорируется.
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
