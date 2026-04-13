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

## Архитектура

Слоистая архитектура (Presentation / Application / Domain / Infrastructure). См. [AGENTS.md](AGENTS.md) или исходный код `src/`.

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

```bash
python build.py           # PyInstaller onedir → dist/OCRStudio/
```

Затем компилируйте `installer/setup.iss` через Inno Setup 6 → получите `installer/Output/OCRStudio-Setup-<ver>.exe` (~200–300 МБ).

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
