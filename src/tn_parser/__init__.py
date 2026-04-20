"""Пакет парсинга транспортных накладных (ТН).

Верхнеуровневый API:
    from src.tn_parser import process_one_pdf, parse_text, ParsedRow

Разбивка по модулям:
    layout.py      — извлечение текста из PDF с учётом геометрии блоков
    normalize.py   — нормализация текста (переносы, confusables, пробелы)
    sections.py    — разбиение текста на пронумерованные разделы ТН (1–17)
    validators.py  — валидаторы ГРЗ / ИНН / даты
    fields.py      — извлечение конкретных полей из разделов
    splitter.py    — разбиение одного PDF на несколько накладных
    core.py        — склейка всего конвейера
    models.py      — датаклассы результата
"""

from .core import (
    extract_raw_text,
    iter_pdfs,
    parse_text,
    process_batch,
    process_one_pdf,
)
from .excel import COLUMNS, write_excel, write_excel_safe
from .models import GARBAGE, MISSING, FieldConfidence, ParsedRow
from .report import build_log_lines, write_log, write_log_safe

__all__ = [
    "MISSING",
    "GARBAGE",
    "ParsedRow",
    "FieldConfidence",
    "process_one_pdf",
    "parse_text",
    "extract_raw_text",
    "iter_pdfs",
    "process_batch",
    "COLUMNS",
    "write_excel",
    "write_excel_safe",
    "build_log_lines",
    "write_log",
    "write_log_safe",
]
