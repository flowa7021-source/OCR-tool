# Colab release: готовые файлы для fine-tune EasyOCR

Содержимое этой папки — снепшот на момент коммита, чтобы запустить
fine-tune `cyrillic_g2.pth` в Google Colab без локальной подготовки.

## Файлы

- **`finetune_ru_colab.zip`** (~56 МБ) — датасет `finetune_ru/` с
  **22 525 уникальных word-crop'ов** + **3× oversample** на real
  (чтобы real-crops не подавлялись synth'ом) = **27 575 rows** в
  `labels.csv` (22 014 training + 5 563 validation). Real: 2525
  (1223 exact + 1302 fuzzy text-only). Synth: 20 000 (`synth_data.py`,
  9 шрифтов, degradation до 0.6).
  Фильтры: numeric-exact + **digit-ratio ≥ 60%** (блокирует
  `(109.2022 → 01.09.2022`, `21},52 → 20,52` и подобные мусорные
  матчи через embedded-скобки). False-positive rate на Lev-2
  аудите ≈ 10% (было 15% после только numeric-exact, было 20%
  без фильтров).
  Real vs synth share в training: 27.5% (было 11% без oversample —
  модель могла overfittить на synth-пиксели, не обобщая на реальные
  сканы).
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
