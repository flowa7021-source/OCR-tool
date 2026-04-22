# Fine-tune EasyOCR on Russian business documents

Инструментарий для дообучения recognizer'а EasyOCR (`cyrillic_g2.pth`)
на корпусе `inputs/*.pdf` + `inputs/*.txt`.

**Быстрый старт (бесплатный T4 GPU в Colab — рекомендуется):**

```bash
# 1. Подготовить dataset локально (CPU, ~10-15 мин на 5 PDF):
python scripts/finetune/bootstrap_dataset.py

# 2. Залить datasets/finetune_ru/ в Google Drive
#    (прямо архивом из UI — ~100 МБ)

# 3. Открыть новый Colab notebook, выбрать T4 GPU
#    (Runtime → Change runtime type → GPU → T4), вставить
#    содержимое scripts/finetune/colab_finetune.py единой ячейкой и Run.
#    ~1.5h на 100 эпох, 4h на 300 (рекомендуется для 2k crop'ов).

# 4. Скачать best_accuracy.pth с Google Drive
#    ({DRIVE_DATASET_DIR}/out/) и положить в
#    resources/easyocr_models/cyrillic_g2.pth → python build.py
```

Альтернатива Colab — Kaggle Notebook (тоже T4/P100, 30ч/нед бесплатно).
Workflow идентичен, только Drive заменяется на `/kaggle/input/*.zip`.

На production-CPU-машине без GPU fine-tune занимает 50-100× дольше —
не рекомендуется даже для smoke-run; используйте Colab/Kaggle.

## Зачем

Stock-модель EasyOCR `cyrillic_g2.pth` обучена на смешанном
Cyrillic-корпусе (новости, Wikipedia). На наших документах —
транспортных накладных и УПД — она показывает CER 174 % (Этап 0),
т.к. не видела:

- шаблонные шрифты налоговых форм
- штампы, подписи и рукописные пометки
- ИНН/КПП/ОГРН в типовом расположении

Fine-tune вытягивает точность на наш domain без переобучения с нуля.

## Workflow

```bash
# 1. Pre-download stock weights (если ещё не сделано)
python scripts/prefetch_easyocr_models.py

# 2. Bootstrap training dataset из inputs/*.pdf + inputs/*.txt
python scripts/finetune/bootstrap_dataset.py
# → datasets/finetune_ru/{training,validation}/{*.jpg,gt.txt}

# 3. (Опционально) вручную проверить / поправить лейблы в gt.txt
#    Особенно строки где предсказание сильно отличалось от GT.

# 4. Fine-tune recognizer (GPU крайне желателен)
python scripts/finetune/finetune_recognizer.py --epochs 100
# → клонирует trainer, обучает, копирует best_accuracy.pth
# → resources/easyocr_models/cyrillic_g2.pth

# 5. Проверить регрессию / улучшение на golden-корпусе
python scripts/real_ocr_metrics.py
```

После шага 4 следующий `python build.py` автоматически забандлит
дообученные веса в инсталлятор — никаких дополнительных действий.

Stock-веса сохраняются в `cyrillic_g2.pth.stock` (на случай rollback).

## Подготовка данных: что делает bootstrap_dataset.py

- Растеризует каждую страницу каждого PDF в `inputs/` при 300 DPI.
- Запускает текущий EasyOCR, получает word-level bounding boxes.
- Для каждого box'а:
  - вырезает crop, сохраняет как `img_NNNNNNNN.jpg`
  - сверяет предсказание с tokens из `inputs/<pdf>.txt` через
    Левенштейна (budget 1 для коротких, 2 для длинных слов)
  - если есть GT-совпадение — подставляет GT как label
- Детерминированный 80 / 20 split по SHA1 ключа — одна и та же
  пара (PDF, page, idx) всегда попадает в один bucket между
  прогонами.

Формат `gt.txt`: `<filename>\t<label>`, по одному crop'у на строку
(как у `deep-text-recognition-benchmark`).

## Требования

- GPU с CUDA для разумной скорости training'а (на CPU — часы / крохи
  epoch'ов).
- Диск: ~1–2 GB под crops, ~500 MB под EasyOCR trainer repo clone.
- RAM: 4+ GB.
- Python-пакеты для запуска trainer'а (не входят в основной
  `requirements.txt` — нужны только для fine-tune):

  ```bash
  pip install pandas natsort lmdb nltk fire
  ```

## Ограничения

- `inputs/*.txt` — page-level транскрипция, не per-word bbox-annotations,
  поэтому auto-correction покрывает только тот word, что EasyOCR уже
  извлёк «близко». Сильно искажённые слова остаются с shaky labels —
  поэтому ручная коррекция `gt.txt` перед training'ом окупается.
- Рукописный русский требует отдельного корпуса; эта пайплайна
  натренирует recognizer только на том, что EasyOCR смог детектировать
  как текст — рукопись часто теряется ещё на этапе CRAFT-детектора.
  Для handwriting нужен fine-tune и detector'а (`craft_mlt_25k.pth`) —
  сделано отдельной итерацией.

## Откат на stock-веса

```bash
cp resources/easyocr_models/cyrillic_g2.pth.stock \
   resources/easyocr_models/cyrillic_g2.pth
```
