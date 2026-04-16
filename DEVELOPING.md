# Развёртывание для разработки

Документ объясняет, **как тестировать изменения без сборки installer'а**.
Цикл «код → CI → installer → ручная проверка» занимает 15+ минут; если
вы просто поменяли что-то в пайплайне или движке, вам этот цикл
не нужен.

## Минимальный dev-setup на Windows

Предполагаем, что OCR Studio у вас уже установлена через релизный
installer — это даёт нам готовый Tesseract + Ghostscript + tessdata
под `%LOCALAPPDATA%\Programs\OCR Studio\_internal\resources\`. Тогда:

```powershell
git clone <repo> OCR-tool
cd OCR-tool
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Этого достаточно. `src.infrastructure.installed_app` автоматически
подхватит бандленные бинарники из установленной копии, и всё
нижеперечисленное «просто работает».

Если OCR Studio у вас не установлена, поставьте Tesseract отдельно
(`choco install tesseract`) или экспортируйте `$env:TESSERACT_CMD`
с путём к `tesseract.exe`.

## Три способа воспроизвести баг без installer-цикла

### 1. CLI end-to-end (самое быстрое)

```powershell
python -m src.cli "путь\к\вашему.pdf" --profile quick_reliable -o out.pdf -vv
```

Это запускает **тот же самый пайплайн**, что и GUI, только без Qt.
Воспроизводит 90% багов, которые мы встречали: timeout-ы,
engine mismatches, Unicode-пути, preflight, постобработку.

Ключи:
- `--profile` — любое из имён в `python -m src.cli --list-profiles`
- `-vv` — DEBUG-логи (по умолчанию только WARNING)
- `--txt --docx` — выгрузить текст рядом

### 2. Smoke-test script (проверка самого процесса)

```powershell
python scripts/smoke_test_real_ocr.py
```

Скрипт:
1. Синтезирует одностраничный русский PDF
2. Гоняет его через пайплайн с `quick_reliable`
3. Проверяет, что OCR распознал один из ожидаемых триграмм

Сценарий использования — **перед каждым push'ем в ветку**. Если
smoke прошёл — можно коммитить. Если упал — баг воспроизведён
локально за 30 секунд, не нужен installer-цикл.

### 3. Unit + integration тесты

```powershell
pytest -q -x tests/unit/ tests/integration/ --timeout=60
```

423+ тестов, ~20 секунд. Покрывают все наши регрессии. Тесты
`test_e2e_real_ocr.py` используют настоящий Tesseract и
автоматически пропускаются, если его нет в системе.

## Правило «repro-first»

Когда приходит лог с ошибкой:

1. **Первый коммит — failing тест**, воспроизводящий проблему.
2. **Второй коммит — фикс**, при котором тест становится зелёным.

Без (1) регрессия может вернуться молча. Тесты — это контракт, что
ошибка больше не появится.

## Что делать, если CLI не находит Tesseract

Сообщение об ошибке перечисляет три варианта:

```
Tesseract executable not found. Searched: ... Варианты:
  1) установить OCR Studio (бандленный Tesseract подхватится
     автоматически);
  2) установить Tesseract в систему (choco install tesseract);
  3) задать путь явно через переменную окружения TESSERACT_CMD.
```

Выберите что удобнее.
