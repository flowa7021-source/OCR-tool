# Changelog

Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).
Версионирование: [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

### Added
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
