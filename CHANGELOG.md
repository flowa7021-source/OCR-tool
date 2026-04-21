# Changelog

Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).
Версионирование: [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

### Changed — Profiles consolidation (декабрь 2026)
- **7 builtin профилей → 1.** Удалены `default`, `quick_reliable`,
  `low_quality_scan`, `contracts_ru`, `english_text` и `tn_upd`.
  Единственный builtin теперь — **`universal_accurate`**, в котором
  собран best-of-all из удалённых:
  * Sauvola binarisation + 400 DPI + deskew + CLAHE + border_removal
    (база, проверена на real scan корпусе);
  * полный word-level pipeline: drop_low_conf_words +
    soft_rescue_dropped_words + adaptive_confidence_threshold +
    per_word_script_disambiguation + per_word_clahe_rescue +
    per_word_upscale_rescue + user_words_fuzzy_rescue +
    per_block_psm_retry;
  * `skip_text=True` (для уже OCR'нутых PDF — экономит 30-60 с/документ),
    из удалённого `tn_upd`;
  * `load_freq_dawg=0` (отключает freq-DAWG, который биасит ИНН/КПП/ОГРН
    в похожие слова), из `contracts_ru` + `tn_upd`;
  * `validate_identifiers=True` (1-edit fix для ИД-полей через каталог
    `expected/*.json`), из `tn_upd` + `quick_reliable`;
  * `extract.enabled=True, kind="tn_upd", multi_document=True` —
    встроенный парсер ТН/УПД работает по умолчанию.

  `ProfileManager.initialize_builtins` теперь **перезаписывает**
  builtin при каждом запуске → апгрейд приложения автоматически
  даёт canonical-настройки без ручной чистки %APPDATA%/profiles.
  Кастомизация — через `duplicate` (профиль с `builtin=False` не
  трогается).

  CLI default `--profile`: `default` → `universal_accurate`.
  GUI dropdown preselect — `universal_accurate`. `last_profile` в
  settings.json при FileNotFoundError откатывается на
  `universal_accurate` через `ProfileManager.get_current`.

### Changed — Parser confidence ceiling (0.93 → 0.99)
- **Per-field validation paths** в 7 extractors. Раньше потолок
  per-field был 0.9 (декларированная шкала «1.0 = section + validated»
  не реализована); теперь возвращаем 1.0 при прохождении structural-
  валидации:
  * `shipper`, `reception` — ИНН проходит контрольную сумму ФНС;
  * `consignee` — результат начинается с ORG-маркера (ООО/АО/ИП);
  * `cargo` — содержит ед. изм. (шт/кг/т/м³) или типовой грузовой
    термин (блок/плита/материал/георешетка);
  * `volume` — full triplet (мест+нетто+объём) ИЛИ «N шт» из cargo;
  * `driver` — полное ФИО (Фамилия Имя Отчество);
  * `number` — паттерн «№ X от ДАТА» (двойной сигнал).

  Замер: avg parser conf на чистом sidecar-корпусе **0.93 → 0.99**
  (4 PDF, 16 rows × 9 полей).

### Added — Parser accuracy 35% → 91% on `inputs/*.txt`
- Структурный парсинг пост-OCR `[Грузоотправитель]` / `— ГРУЗ —` /
  `— ПОГРУЗКА —` секций; per-field accuracy на golden-датасете:
  number 0% → 100%, shipper 0% → 100%, consignee 0% → 100%,
  cargo 25% → 100%, volume 0% → 25%* (`«N шт»` — должный вид),
  driver 75% → 100%, reception 0% → 100%.

### Added — End-to-end OCR + parser workflow
- **`.github/workflows/e2e-tn-pipeline.yml`** — Windows-runner
  workflow с полным production стеком (Tesseract 5.5 + tessdata_best
  rus/eng/osd + Ghostscript) и 4 независимыми гейтами:
  1. parser-only golden (`run_golden.py`, ≥ 70% accuracy);
  2. parser ceiling (avg parser conf ≥ 95% на clean sidecar);
  3. fast E2E (`scripts/e2e_tn_pipeline.py`, pytesseract+parser);
  4. full E2E (`scripts/e2e_tn_pipeline_full.py`, production
     `OCRPipeline` + `universal_accurate` + parser).
- **`scripts/e2e_tn_pipeline_full.py`** — production-pipeline e2e
  тест с combined confidence (geometric mean of OCR_conf × parser_conf),
  fail-fast на отсутствующую среду (exit 4, без silent skip).
- **`scripts/run_golden.py --min-accuracy`** флаг → exit 2 при
  просадке.

### Added
- **Извлечение полей ТН / УПД как пост-OCR шаг** — парсер
  `src.tn_parser` (ранее отдельный `OCR parser/`) интегрирован в
  основной конвейер и доступен через профиль, UI-вкладку и CLI:
  - Встроенный профиль **`tn_upd`** (schema v11): PSM=SINGLE_BLOCK,
    OTSU, `border_removal=True`, `load_freq_dawg=0`,
    `validate_identifiers/entities=True` + `extract.enabled=True`.
    Сигналит пайплайну запускать парсер после OCR.
    *(Декабрь 2026: `tn_upd` слит в `universal_accurate` — см. раздел
    «Profiles consolidation» выше.)*
  - Новая секция **`ProfileData.extract`** (`ExtractConfig` +
    `LlmFallbackConfig`) в JSON-профилях. Миграция v10→v11
    прозрачна: profile без extract → `enabled=False`.
  - **`JobResult.parsed: ParsedDocument | None`** — строки
    (`ParsedRow.to_json_dict()`) + `overall_confidence` +
    `snapshot_path`. Core-слой не зависит от `src.tn_parser` в
    compile-time (dict-форма).
  - **Hook в `OCRPipeline`**: `_maybe_run_parser()` вызывается из
    всех трёх completion-путей (cache-hit, text-layer-bypass,
    OCR-main). Orchestrator `src.application.parsers.tn_orchestrator`
    никогда не бросает — падение парсера не валит OCR-задачу.
  - **UI-вкладка «Парсер накладных»** (`InvoiceParserPanel`,
    `QTableView` 13 колонок, цвета confidence #F4CCCC/#FFF2CC/
    #D9EAD3 byte-exact с excel.py, empty-state placeholder,
    кнопка «Экспорт в Excel…»).
  - **`ExportFormat.EXCEL`** в `ExportManager.export()` →
    делегирует `tn_parser.excel.write_excel_safe` (xlsx + .log +
    .snapshot.json-сайдкар). Обрабатывает `PermissionError`
    (файл открыт в Excel), `ENOSPC`, fallback-имя при lock'е.
  - **CLI**: флаг `--excel` рядом с `--txt`/`--docx`; подкоманды
    `ocr-cli parser golden / update-golden / collect-feedback /
    feedback-stats` (обёртки над `scripts/*.py`, диспатчер через
    `importlib.import_module` не тянет OCR-движок).
  - **LLM-fallback opt-in** (`[project.optional-dependencies.llm]`
    — `anthropic`, `pydantic`). Ключ хранится в
    `AppSettings.anthropic_api_key` (settings.json, НЕ в профиле —
    экспорт профиля не утекает credentials). UI-поле в
    `PreferencesDialog` (PasswordEchoOnEdit, trim whitespace).
    `src.infrastructure.llm_credentials.apply_to_environment`
    синхронизирует ключ в `ANTHROPIC_API_KEY` с sentinel-защитой
    shell-set env'а. Оффлайн-билд Windows-installer'а остаётся
    полностью локальным — `anthropic` не попадает в бандл.
  - **Installer / CI**: `build.py` +4 флага PyInstaller
    (`--collect-all=rapidfuzz`, `--collect-all=scripts`,
    `--collect-submodules=src.tn_parser`,
    `--hidden-import=openpyxl`); новый smoke-шаг
    `build-installer.yml` поднимает установленный `OCRStudio.exe`
    на реальном TN PDF и проверяет наличие `rapidfuzz/*.pyd`,
    `openpyxl/`, `src/tn_parser/fields.py`, `scripts/run_golden.py`,
    `profiles/tn_upd.json` в распакованном бандле. Отдельная
    job `parser-tests` в `ci.yml` — Ubuntu, ~30 сек, минимальный
    dep-set `pymupdf openpyxl rapidfuzz`.
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
