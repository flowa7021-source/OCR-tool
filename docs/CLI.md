# CLI — режим командной строки

OCR Studio можно запускать без GUI — полезно для пакетного скриптинга,
CI smoke-тестов и headless-серверов. Qt не стартует, рисуется только
прогресс-лог в stdout.

## Способы запуска

```bash
# Из исходников:
python -m src.cli <аргументы>

# Из установленного wheel / pip install -e .:
ocr-studio-cli <аргументы>

# Windows-инсталлятор (будущий релиз): OCRStudio.exe --cli ...
```

## Синтаксис

```
ocr-studio-cli [опции] <inputs...>
```

`inputs` — один или несколько PDF-файлов и/или директорий. Для
директорий выполняется **рекурсивный** поиск `*.pdf`.

## Опции

| Опция                  | По умолчанию  | Описание                                                    |
| ---------------------- | ------------- | ----------------------------------------------------------- |
| `-o PATH`, `--output`  | `<in>_ocr.pdf`| Путь для выходного PDF (только при одном входном файле)    |
| `-p NAME`, `--profile` | `default`     | Имя профиля OCR (см. `--list-profiles`)                    |
| `--workers N`          | `1`           | 1–4 параллельных процесса (`1` = в текущем процессе)        |
| `--txt`                | off           | Дополнительно сохранить TXT рядом с PDF                     |
| `--docx`               | off           | Дополнительно сохранить DOCX рядом с PDF                    |
| `--list-profiles`      | off           | Напечатать список профилей и выйти                         |
| `-v`, `-vv`            | warnings      | `-v` = INFO, `-vv` = DEBUG                                  |
| `--version`            | —             | Версия приложения                                           |

## Коды возврата

| Код | Значение                                                |
| --- | ------------------------------------------------------- |
| 0   | Все файлы обработаны успешно                            |
| 1   | Хотя бы один файл завершился ошибкой OCR                |
| 2   | Ошибка аргументов / не найдено входов / нет профиля     |
| 3   | Ошибка при экспорте TXT/DOCX                            |

## Примеры

### Один файл с дефолтным профилем

```bash
python -m src.cli document.pdf
# Создаёт document_ocr.pdf рядом с оригиналом
```

### Выбор профиля и кастомный путь вывода

```bash
python -m src.cli -p low_quality_scan -o final.pdf scan.pdf
```

### Дополнительные форматы

```bash
python -m src.cli --txt --docx scan.pdf
# Создаст scan_ocr.pdf, scan_ocr.txt, scan_ocr.docx
```

### Пакет + параллелизм

```bash
python -m src.cli --workers 4 ~/scans/
```

> Примечание: при `--workers > 1` флаги `--txt` и `--docx` игнорируются —
> параллельный пул не возвращает полноценный `JobResult` для экспорта
> вторичных форматов. Для TXT/DOCX используйте `--workers 1`.

### Список профилей

```bash
python -m src.cli --list-profiles
# ⭐ default — Универсальный профиль для офисных сканов
# ⭐ low_quality_scan — Агрессивная предобработка для плохих сканов
# ⭐ contracts — Договора: максимальное качество распознавания
# ⭐ english — Английский текст
#   my_profile — (пользовательский)
```

## Переменные окружения

| Переменная         | Назначение                                                   |
| ------------------ | ------------------------------------------------------------ |
| `TESSERACT_PATH`   | Переопределяет путь к `tesseract(.exe)`                      |
| `TESSDATA_PREFIX`  | Путь к `tessdata/` (обычно задаётся автоматически)           |
| `QT_QPA_PLATFORM`  | Не используется (CLI не стартует Qt), но применяется в тестах |
| `PYTHONUNBUFFERED` | Рекомендуется `1` в CI, чтобы не терять stdout при падении   |

Все остальные настройки (число воркеров по умолчанию, интервал
автосохранения) читаются из `settings.json` — CLI и GUI используют
одну и ту же конфигурацию.

## Скриптинг

### Exit-code в bash

```bash
if ! python -m src.cli -p contracts ~/inbox/; then
  echo "хотя бы один файл упал" >&2
  exit 1
fi
```

### Логирование в файл

```bash
python -m src.cli -vv --workers 4 ~/scans/ 2>&1 | tee ocr.log
```

### Systemd-таймер

```ini
# /etc/systemd/system/ocr-batch.service
[Service]
ExecStart=/opt/venv/bin/python -m src.cli -p contracts /srv/incoming/
WorkingDirectory=/srv/ocr-studio
Environment=TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
```

### Windows Task Scheduler

`schtasks /Create /TN "OCR nightly" /TR "ocr-studio-cli.exe -p default C:\scans" /SC DAILY /ST 02:00`

## Отличия CLI от GUI

| Функция                                           | GUI | CLI |
| ------------------------------------------------- | --- | --- |
| Searchable PDF (главный артефакт)                 | ✓   | ✓   |
| TXT / DOCX экспорт                                | ✓   | ✓ (однопоточно) |
| Встроенные и пользовательские профили             | ✓   | ✓   |
| Пакетная обработка                                | ✓   | ✓   |
| Параллелизм                                       | ✓   | ✓   |
| Drag-and-drop, миниатюры, live-превью             | ✓   | —   |
| Ctrl+F поиск по распознанному тексту              | ✓   | —   |
| HTR / GOT-OCR 2.0                                 | ✓   | ✓¹  |
| Проверка обновлений                               | ✓   | —   |
| Логирование в rotating-файл                       | ✓   | ²   |

¹ Только если веса уже скачаны (через GUI или вручную в
`%LOCALAPPDATA%\OCRStudio\models\GOT-OCR-2.0\`).

² CLI по умолчанию пишет в stdout. Опциональное подключение
`RotatingFileHandler` — через свой wrapper-скрипт.
