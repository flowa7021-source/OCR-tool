# Участие в разработке OCR Studio

## Настройка окружения

```bash
python3.11 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e ".[dev]"
```

Для полноценного OCR на Linux:
```bash
sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
```

На Windows — поместите `tesseract.exe` 5.5.0 и DLL в `resources/tesseract/`,
`rus.traineddata` + `eng.traineddata` в `resources/tessdata/`.

## Быстрый запуск UI без OCR

```bash
python dev_run.py               # пустое окружение
python dev_run.py путь/к/файлу.pdf   # открыть PDF при старте
```

`dev_run.py` подменяет `ParallelProcessor` и `TesseractWrapper` на заглушки,
возвращающие placeholder-результаты. Удобно для проверки разметки и
поведения панелей на любой машине.

## Запуск тестов

```bash
# Все юнит-тесты
QT_QPA_PLATFORM=offscreen pytest tests/ --ignore=tests/integration

# Интеграционные (требуют установленные OpenCV, PyMuPDF)
pytest tests/integration/
```

Coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

## Стандарты кода

- **Type hints** обязательны на всех публичных функциях и методах.
- **Docstrings** в Google-стиле на классах и публичных методах.
- **`logging`** (не `print`) везде, `logger = logging.getLogger(__name__)`.
- **`pathlib.Path`** для путей; никакого строкового склеивания.
- **UTF-8**: явно `encoding="utf-8"` во всех файловых операциях.
- **Исключения**: конкретные типы; никогда `except:` без уточнения.
- **Разделение слоёв**: UI не содержит бизнес-логики; бизнес-логика не
  импортирует Qt (кроме `ExportManager.copy_to_clipboard` с lazy-import).
- Потокобезопасность UI: обновления виджетов — только через Qt signals или
  `QTimer.singleShot` с main-thread marshaling.

Линтеры и форматеры настроены в `pyproject.toml`:
```bash
black src tests
ruff check src tests --fix
mypy src
```

## Стиль коммитов

Короткая шапка (до 72 символов) в настоящем времени, затем опциональный
развёрнутый блок через пустую строку. Пример:

```
Add RecoveryManager for crash-safe queue snapshots

Each active QueueItem is serialized as JSON under RECOVERY_DIR on submit and
updated every N pages. MainWindow.prompt_recovery offers to resume on
startup. Adds 8 unit tests.
```

Не делайте `git commit --amend` после `push`. Не используйте `--no-verify`.

## Сборка Windows-инсталлятора

```bash
python build.py                # PyInstaller onedir → dist/OCRStudio/
# затем в Inno Setup:
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
```

На выходе: `installer/Output/OCRStudio-Setup-<ver>.exe` (~200–300 МБ).

## Архитектурные принципы

- **Domain-слой (core/)** полностью независим от Qt и OCRmyPDF. Модели
  сериализуемы в JSON для transfer между процессами.
- **Infrastructure-слой** инкапсулирует файловую систему, Tesseract-пути,
  логгирование.
- **Application-слой** оркестрирует пайплайн, очередь, пул процессов.
- **UI-слой** — только презентация; никогда не вызывает пайплайн в UI-потоке.

Расширение функционала обычно идёт так:
1. Добавьте поля в `core/models.py`.
2. Реализуйте логику в `core/` или `application/`.
3. Добавьте unit-тесты.
4. Пропишите UI-редактор в `ui/*_panel.py`.
5. Свяжите через `MainWindow`.
