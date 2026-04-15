# OCR Studio

Профессиональное десктопное OCR-приложение для Windows 10/11 с расширенной предобработкой сканов и генерацией searchable PDF.

## Возможности

- **Searchable PDF** — невидимый текстовый слой поверх сканов через [OCRmyPDF](https://ocrmypdf.readthedocs.io/) и Tesseract 5.5.0 (LSTM, best quality).
- **Русский + английский** — полностью оффлайн, языковые данные вшиты в сборку.
- **Расширенная предобработка** на OpenCV: бинаризация (Otsu / адаптивная / Sauvola), шумоподавление (median / gaussian / morphology / NLM), коррекция контраста (CLAHE), удаление теней, выпрямление наклона ([deskew](https://pypi.org/project/deskew/)) и физических деформаций ([page-dewarp](https://pypi.org/project/page-dewarp/)).
- **Профили настроек** — 4 предустановленных (по умолчанию, низкое качество скана, договоры, английский текст) + пользовательские.
- **Пакетная обработка** с параллельным выполнением 2–4 файлов (`ProcessPoolExecutor`).
- **Встроенный PDF-просмотрщик** с миниатюрами, масштабированием, сравнением «до/после».
- **Экспорт** в searchable PDF, TXT (UTF-8 / CP1251), DOCX (python-docx), буфер обмена.
- **Постобработка текста** — автокоррекция типичных OCR-ошибок (rus/eng), пользовательские regex-правила.
- **Тёмная тема** (QSS, PySide6 / Qt 6).

## Документация

- [Руководство пользователя](docs/USER_GUIDE.md) — установка, первый OCR, профили, HTR.
- [Системные требования](docs/SYSTEM_REQUIREMENTS.md) — ОС, CPU/RAM/GPU, дисковые пути.
- [CLI](docs/CLI.md) — запуск из командной строки, опции, коды возврата.
- [Горячие клавиши](docs/KEYBOARD_SHORTCUTS.md) — полный список сочетаний.
- [Архитектура](docs/ARCHITECTURE.md) — схема слоёв и поток данных.
- [Privacy & data storage](docs/PRIVACY.md) — что хранится локально, что уходит в сеть.

## Архитектура

Слоистая архитектура (Presentation / Application / Domain / Infrastructure). Подробнее — в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
src/
├── main.py                  # точка входа
├── app.py                   # bootstrap QApplication
├── core/                    # Domain: preprocessor, deskew, dewarp, postprocessor, models
├── application/             # Pipeline, OCRmyPDF, Queue, Parallel, Profiles, Export
├── ui/                      # PySide6: MainWindow, PDFViewer, панели
├── infrastructure/          # Tesseract, ConfigStorage, File utils, Logger
└── shared/                  # types, constants, validators
```

## Установка для разработки

```bash
python3.11 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS (для разработки):
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"
```

Нужен Tesseract 5.5.0:

- **Windows (runtime):** поместите `tesseract.exe` и DLL в `resources/tesseract/`, а `rus.traineddata` + `eng.traineddata` — в `resources/tessdata/`.
- **Dev на Linux/macOS:** установите системный `tesseract` (`apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng`).

Запуск GUI:
```bash
python -m src.main
# или, после pip install -e .
ocr-studio
```

Запуск из командной строки (без GUI):
```bash
# Одиночный файл
python -m src.cli document.pdf

# Пакет, 4 параллельных процесса
python -m src.cli --workers 4 ~/scans/

# С выбором профиля и экспортом в TXT/DOCX
python -m src.cli -p low_quality_scan --txt --docx -o out.pdf in.pdf

# Список доступных профилей
python -m src.cli --list-profiles
```

## Сборка Windows-инсталлятора

### Вариант 1 — через GitHub Actions (рекомендуется)

В репозитории есть workflow `.github/workflows/build-installer.yml`, который
собирает полностью готовый инсталлятор на `windows-latest` **автоматически**:

1. **Rolling dev-билд**: любой `push` в `main` или в активную dev-ветку
   запускает сборку и публикует её как prerelease с тегом `latest-dev`.
   Этот релиз **перезаписывается** на каждом успешном build — всегда на
   странице Releases лежит свежий инсталлятор с последнего коммита.
2. **Релиз по тегу**: `git tag v1.0.0 && git push --tags` → создаётся
   постоянный Release `v1.0.0` с `OCRStudio-Setup-1.0.0.exe` + `.sha256`.
   Теги с суффиксами `-rc`, `-beta`, `-alpha` автоматически помечаются
   prerelease.
3. **Ручной запуск**: Actions → *Build Windows Installer* → *Run workflow*.
   Можно переопределить версию Tesseract, версию инсталлятора, и
   выключить публикацию релиза флагом `publish=false` (тогда только
   workflow-артефакт).
4. **PR-билд**: pull request'ы в `main` также прогоняют всю сборку для
   проверки, но релиз не публикуется.

Что делает workflow (22 шага):
- скачивает **Tesseract 5.5.0** от UB Mannheim (с кэшированием инсталлятора
  между прогонами) и распаковывает `tesseract.exe` + DLL в
  `resources/tesseract/`;
- скачивает **tessdata_best** для `rus` + `eng` + `osd` (с кэшированием
  на уровне runner'а и проверкой минимального размера файлов);
- генерирует `app.ico` из `app.svg` для Windows-билда (multi-size);
- запускает unit-тесты и `compileall` как pre-build gate (если не
  прошли — Tesseract и инсталлятор не собираются вообще);
- запускает `PyInstaller --onedir`;
- верифицирует, что Tesseract и tessdata действительно попали в
  `dist/OCRStudio/` (проверяются оба возможных layout'а PyInstaller);
- компилирует `installer/setup.iss` через Inno Setup 6;
- sanity-check: инсталлятор запускается с флагом `/?` и возвращает
  корректный exit code (не висит);
- считает SHA-256;
- аплодит workflow-артефакт на 90 дней + публикует GitHub Release;
- пишет сводку в Actions Summary.

Артефакт workflow: `OCRStudio-Installer-<version>` (содержит `.exe` +
`.sha256`). Релиз — на странице Releases.

### Вариант 2 — локальная сборка на Windows

```bash
# 1. Скачайте Tesseract 5.5.0 и скопируйте tesseract.exe + DLL в
#    resources/tesseract/
# 2. Скачайте rus.traineddata и eng.traineddata в resources/tessdata/
# 3. (опционально) сгенерируйте resources/icons/app.ico из app.svg

python build.py               # PyInstaller onedir → dist/OCRStudio/
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
```

Результат: `installer/Output/OCRStudio-Setup-<ver>.exe` (~200–300 МБ).

## Тесты

```bash
pytest tests/
```

## Лицензия

MIT — см. [LICENSE](LICENSE).

### Интегрируемые библиотеки
| Библиотека | Лицензия | Роль |
|---|---|---|
| [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) | MPL-2.0 | Ядро наложения текстового слоя |
| [deskew](https://github.com/sbrunner/deskew) | MIT | Определение угла наклона |
| [page-dewarp](https://github.com/lmmx/page-dewarp) | MIT | Выпрямление деформаций |
| [python-docx](https://github.com/python-openxml/python-docx) | MIT | Экспорт в Word |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | OCR-движок |
