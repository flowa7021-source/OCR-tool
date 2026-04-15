# Системные требования

OCR Studio — десктопное OCR-приложение. Ниже — минимальные и
рекомендуемые требования для разных сценариев использования.

## Операционная система

**Поддерживаются:**
- **Windows 10** версии 2004+ (build 19041) — 64-битная
- **Windows 11** — все версии, 64-битные
- **x64 (AMD64)** архитектура

Инсталлятор (`OCR-Studio-Setup-*.exe`) официально тестируется только
на Windows 10 и 11. `MinVersion=10.0.19041` закреплён в `setup.iss`;
на более старых сборках Windows 10 установщик откажется ставиться.

**Разработка** возможна также на Linux (Ubuntu 22.04+) и macOS
(12+) — там приложение запускается через `python -m src.main`.
Официальных бинарных релизов под Linux/macOS пока нет.

**Не поддерживаются:**
- Windows 7, 8, 8.1, Server 2012/2016
- 32-битные системы (x86, ARM32)
- Windows 10 LTSC 1809 (build < 19041)

## Аппаратные требования

### Минимальные

| Компонент | Значение |
| --- | --- |
| CPU | x64, 2 ядра / 2 потока, ≥ 1.5 GHz |
| RAM | **3 GB** (6 GB при использовании HTR/GOT-OCR 2.0) |
| Диск | 1 GB свободно (без HTR), 2.5 GB (с HTR-моделью); HDD поддерживается |
| Экран | **1024×768** |
| GPU | не требуется — CPU-only OCR полностью работает |

На машинах с RAM < 6 ГБ приложение автоматически переключается в
**режим экономии памяти**: число параллельных воркеров = 1, кэш OCR
снижен до 512 МБ. Это видно в `Справка → Системная информация`.

### Рекомендуемые

| Компонент | Значение |
| --- | --- |
| CPU | 4+ ядер / 8 потоков, AVX2 |
| RAM | 16 GB |
| Диск | **SSD**, 10 GB свободно (temp-файлы + cache + recovery) |
| Экран | 1920×1080 |
| GPU | NVIDIA CUDA 11.8+ (5+ GB VRAM) — только для ускорения HTR |

### Большие задания (книги 500+ страниц, 600 DPI)

| Компонент | Значение |
| --- | --- |
| CPU | 6+ ядер |
| RAM | 16–32 GB |
| Диск | 30+ GB свободно — один 600 DPI A4 scan ≈ 8–12 MB PNG; временные файлы для 500 страниц × 4 воркера могут занять 20+ GB |

## Программные зависимости

### Включены в инсталлятор

Ничего дополнительно ставить не требуется — всё в `OCR-Studio-Setup-*.exe`:

- **Python 3.11** — embedded runtime в PyInstaller-бандле
- **PySide6 6.11** (Qt 6) — GUI
- **Tesseract 5.5.0** (UB Mannheim build) + `tessdata_best` для русского и английского + `osd`
- **OCRmyPDF 16.x** — встраивание searchable-слоя в PDF
- **OpenCV 4.9+** (headless) — препроцессинг изображений
- **scikit-image 0.22+** — Sauvola-бинаризация
- **PyMuPDF (fitz) 1.24+** — rasterize + overlay
- **deskew**, **page-dewarp** — коррекция геометрии
- **python-docx 1.1+** — DOCX-экспорт
- **numpy 1.26+**, **Pillow 10+**, **psutil 5.9+**
- **pytesseract 0.3.10+** — Python-обёртка вокруг CLI

### HTR-вариант инсталлятора дополнительно включает

- **torch 2.1+** (CPU-only wheel от PyTorch)
- **transformers 4.40+**, **tokenizers**, **tiktoken 0.5+**, **safetensors 0.4+**
- **Веса GOT-OCR 2.0** (~580 MB, `model.safetensors`) — уже в инсталляторе, докачивать ничего не нужно

Размер HTR-инсталлятора: ~700–900 MB (compressed lzma2/ultra64).

### Для запуска из исходников

Если не используете готовый инсталлятор:

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
# опционально:
pip install -e ".[htr]"
```

Дополнительно нужен системный Tesseract 5.x:

```bash
# Ubuntu / Debian:
sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# macOS:
brew install tesseract tesseract-lang

# Windows: поместить tesseract.exe + DLL в resources/tesseract/,
# *.traineddata в resources/tessdata/
```

## Права доступа

- **Установка** — стандартный Inno Setup: либо Admin-права (в
  `Program Files`), либо per-user (в `%LOCALAPPDATA%\Programs`).
- **Запуск** — обычный пользователь, без UAC.
- **Сеть** — **не требуется** в штатном режиме. Опциональные
  обращения к сети только по явной команде пользователя:
  1. `Справка → Проверить обновления` — GitHub `/releases/latest`.
  2. `Движок OCR → Скачать GOT-OCR 2.0` — HuggingFace (только если
     не использован HTR-инсталлятор).

## Дисковые пути

| Назначение | Путь (Windows) |
| --- | --- |
| Программа | `C:\Program Files\OCR Studio\` (или `%LOCALAPPDATA%\Programs\OCR Studio\`) |
| Настройки + профили | `%LOCALAPPDATA%\OCRStudio\config\` |
| Логи (rotating 5×5 MB) | `%LOCALAPPDATA%\OCRStudio\logs\` |
| Временные PNG | `%LOCALAPPDATA%\OCRStudio\temp\` (очистка старше 24 ч при старте) |
| Recovery-снапшоты | `%LOCALAPPDATA%\OCRStudio\recovery\` |
| OCR-cache (LRU 2 GB) | `%LOCALAPPDATA%\OCRStudio\ocr-cache\` |
| HTR-модель (docачка) | `%LOCALAPPDATA%\OCRStudio\models\got_ocr2\` |

## Производительность

Эмпирические значения на **4-ядерном Intel i5-1135G7, 16 GB RAM, SSD**
(профиль `default`, 300 DPI, 1 воркер):

| Задание | Скорость |
| --- | --- |
| 1-страничный PDF | 2–4 сек |
| Книга 50 страниц, печать | 1.5–2 мин |
| Книга 500 страниц, печать | 12–18 мин |
| Повторный прогон того же PDF (OCR-cache hit) | < 1 сек (только копирование) |

Профиль `universal_accurate` (600 DPI, полный pipeline) — примерно
**в 4 раза медленнее** `default` за счёт растеризации и
adaptive-бинаризации.

Параллельная обработка: `Настройки → Число параллельных воркеров`
(по умолчанию 2, рекомендовано 2–4 в зависимости от CPU).

## Что работает через CUDA

GOT-OCR 2.0 автоматически использует CUDA, если доступна:
- **CUDA 11.8+** и NVIDIA драйвер версии ≥ 520
- **5 GB VRAM** для inference в fp16
- При отсутствии CUDA — работает на CPU (в 5–10 раз медленнее,
  но функционально идентично)

Tesseract/OCRmyPDF — CPU-only.

## Известные ограничения

- PDF **с паролем** не обрабатываются (приложение покажет ошибку).
- PDF с **уже встроенным текстовым слоем** по умолчанию пропускаются
  (`skip_text=true` в профиле) — можно переопределить.
- Файлы > 2 GB могут не влезать в память при 600 DPI — для таких
  используйте профиль `default` (300 DPI) или уменьшите `max_pages`.
- Экран < 1280×720 может обрезать часть UI даже при включённых
  QScrollArea-обёртках.
