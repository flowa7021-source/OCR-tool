# Changelog

Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).
Версионирование: [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

### Added
- **Pluggable OCR engines + GOT-OCR 2.0 для рукописного текста**:
  - Абстрактный `OCREngine` интерфейс (`src/application/engines/`),
    через который пайплайн вызывает движок. Tesseract обёрнут в
    `TesseractEngine`, поведение по умолчанию не изменилось.
  - `GOTOCREngine` (Apache-2.0, Stepfun) — Transformer-OCR с поддержкой
    рукописного и печатного текста на 80+ языках включая русский.
    Зависимости (`torch`, `transformers`, `Pillow`, `tiktoken`)
    вынесены в `[htr]` extras: `pip install ocr-studio[htr]`.
  - `ModelManager` (`src/infrastructure/model_manager.py`) скачивает
    ~580 МБ весов с Hugging Face Hub в
    `%LOCALAPPDATA%/OCRStudio/models/got_ocr2/` со streaming-progress,
    retry, проверкой размера и SHA-256, корректной отменой.
  - Меню «Движок OCR» в MainWindow → «Скачать GOT-OCR 2.0» открывает
    `ModelDownloadDialog` с QProgressBar и worker-потоком.
  - Dropdown «OCR-движок» в `SettingsPanel`: показывает оба движка с
    их доступностью и пояснениями; недоступный вариант показывается
    серым с подсказкой как его включить.
  - `OCREngineKind` enum + `engine` поле на `OCRConfig` (default:
    TESSERACT). Сериализуется в JSON-профилях.
  - 5-й встроенный профиль `handwritten_mixed` с предустановленным
    GOT-OCR 2.0, отключённой бинаризацией и лёгким CLAHE.

### Tests
- 30 новых тестов (228 → 258 в текущей итерации):
  - `test_engines` (13): абстракция, регистр, Tesseract обёртка
  - `test_model_manager` (11): скачивание через локальный HTTP-сервер,
    отмена, проверки размера, idempotency
  - `test_got_ocr_engine` (6): availability probes без torch, с torch,
    с моделью; happy-path с замоканным fitz/PIL/torch
- **Автопубликация инсталлятора через GitHub Actions**:
  - `push` в `main`/dev-ветку → rolling prerelease с тегом `latest-dev`
    (перезаписывается на каждом успешном build, всегда свежий инсталлятор
    на странице Releases);
  - `push` тега `v*` → постоянный Release; `-rc`/`-beta`/`-alpha` →
    автоматически prerelease;
  - `pull_request` в `main` → только build без публикации;
  - `workflow_dispatch` → ручной запуск с опциональным флагом `publish=false`.
  - Кэширование Tesseract installer и tessdata между прогонами.
  - Retry-логика для скачиваний (4 попытки с backoff).
  - Sanity-check инсталлятора (запускается с `/?`, не виснет).
  - Actions Summary с версией / commit / размером.
- **GitHub Actions workflow** `build-installer.yml` — собирает полный
  Windows-инсталлятор. Скачивает Tesseract 5.5.0 (UB Mannheim),
  tessdata_best (rus + eng + osd), генерирует `app.ico` из SVG,
  запускает `pytest` + `compileall`, `PyInstaller --onedir`, затем
  Inno Setup 6. Артефакт: `OCRStudio-Setup-<version>.exe` + `.sha256`.
- `build.py` теперь подхватывает `resources/icons/app.ico`, если файл
  присутствует (иначе бинарник собирается без Windows-иконки).
- **CLI-режим** (`src/cli.py`): `python -m src.cli file.pdf`
  или `ocr-cli` после установки. Поддерживает:
  - одиночный файл с `-o`;
  - пакет из файлов и директорий (рекурсивный поиск `*.pdf`);
  - `--profile` для выбора профиля;
  - `--workers` для параллельной обработки;
  - `--txt` / `--docx` для дополнительных форматов;
  - `--list-profiles` / `--version` / `-v`/`-vv`.
- 20 unit-тестов CLI (парсинг, обнаружение файлов, dispatcher,
  process_single с mocked-пайплайном).
- Smoke-тесты `SettingsPanel` (4) и `ResultsPanel` (3).
- **CI-workflow** (GitHub Actions): lint (ruff), compileall + import
  smoke, pytest-матрица Ubuntu + Windows × Python 3.11/3.12.
- `ParallelProcessor.cancel_job` / `cancel_all` / `active_job_count`.
- Toolbar Pause (Ctrl+P) / Stop (Ctrl+.) — в MainWindow.
- Queue statistics live label в status bar.
- Типизированные исключения пайплайна для повреждённых / защищённых /
  пустых PDF.

### Added (предыдущие итерации)
- SVG-иконки для тулбара (Открыть, Старт, Сохранить) и приложения (`app.svg`).
- `src/ui/icons.py` — загрузчик тинтованных SVG через подмену `currentColor`.
- Интеграционные тесты `ImagePreprocessor` на синтетических изображениях
  (13 тестов: все режимы бинаризации, denoise-цепочки, CLAHE, фон,
  grayscale-вход, preview_step).
- Полный набор тестов `ExportManager` (TXT, DOCX, PDF passthrough,
  dispatcher, clipboard, 14 тестов).
- `dev_run.py` — запуск UI без Tesseract/OCRmyPDF для быстрой проверки
  разметки и поведения панелей в dev-окружении.
- `RecoveryManager` — снимки очереди в `%LOCALAPPDATA%/OCRStudio/recovery/`,
  автосохранение раз в 5 страниц, предложение возобновить при старте.
- Per-page прогресс-бридж из worker-процессов в UI через
  `multiprocessing.Manager.Queue()` + фоновый drain-поток.
- `LogViewer` — live-tail лог-файла с кнопкой очистки.
- `PreferencesDialog` — UI для `AppSettings` (воркеры, автосохранение, тема).
- Recent Files submenu + `Ctrl+,` на настройки.
- `PostprocessPanel` — редактирование правил постобработки и custom-regex.

### Fixed
- `ExportManager.copy_to_clipboard` ошибочно импортировал `PyQt6`; теперь
  правильно использует `PySide6`.
- Отсутствующий `objectName` на `QDockWidget` мешал `saveState()`.

### Changed
- `psutil` переведён из soft-optional в объявленные зависимости (RAM/CPU
  отображение в status-bar теперь гарантированно).
- `MainWindow._apply_job_result` пропагирует `JobResult.status` в очередь
  и удаляет recovery-снимок при завершении.

## [1.0.0] — initial skeleton

### Added
- Слоистая архитектура (UI / Application / Domain / Infrastructure) на
  Python 3.11 + PySide6 + OCRmyPDF + Tesseract 5.5 + OpenCV + PyMuPDF.
- OCR-пайплайн: PyMuPDF rasterize → предобработка (OpenCV, deskew lib,
  page-dewarp) → OCRmyPDF (встроенная предобработка отключена) →
  постобработка (автокоррекция rus/eng, NFC, regex-правила).
- 4 встроенных профиля: default, low_quality_scan, contracts_ru,
  english_text. CRUD пользовательских профилей с импортом/экспортом.
- Параллельная обработка (`ProcessPoolExecutor`, 1–4 воркера).
- Экспорт: searchable PDF, TXT (UTF-8 / CP1251), DOCX, буфер обмена.
- Тёмная тема (QSS), drag-and-drop, high-DPI, resizable panels.
- PyInstaller build-скрипт, Inno Setup installer.
- 93 unit-теста.
