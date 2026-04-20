# -*- coding: utf-8 -*-
"""Запись ParsedRow в .xlsx с форматированием и колонкой уверенности.

Модуль общий для GUI и CLI. Содержит единственную публичную функцию
`write_excel(rows, output_path)`.

Схема колонок — 12 штук. Последняя колонка — «Уверенность, %» — окрашивается
условным форматированием: красным для <50%, жёлтым для 50–70%, зелёным выше.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .models import ParsedRow


SHEET_NAME = "Extraction"

COLUMNS: List[Tuple[str, int]] = [
    ("Транспортная накладная", 30),
    ("Дата", 14),
    ("№", 15),
    ("Грузоотправитель", 35),
    ("Грузополучатель", 35),
    ("Груз", 35),
    ("Объём", 22),
    ("Водитель", 25),
    ("Транспортное средство", 22),
    ("Прием груза", 40),
    ("Источник файл", 25),
    ("Примечание", 18),
    ("Уверенность, %", 14),
]

# Палитра для колонки confidence.
_CONF_FILL_LOW = PatternFill("solid", fgColor="F4CCCC")    # бледно-красный
_CONF_FILL_MID = PatternFill("solid", fgColor="FFF2CC")    # бледно-жёлтый
_CONF_FILL_OK = PatternFill("solid", fgColor="D9EAD3")     # бледно-зелёный


def _row_to_tuple(row: ParsedRow) -> tuple:
    """Расширяет `ParsedRow.to_excel_tuple()` колонкой confidence."""
    base = row.to_excel_tuple()
    conf_pct = round(row.confidence.overall() * 100)
    return base + (conf_pct,)


def _conf_fill(pct: int) -> PatternFill:
    if pct < 50:
        return _CONF_FILL_LOW
    if pct < 70:
        return _CONF_FILL_MID
    return _CONF_FILL_OK


def write_excel(rows: List[ParsedRow], output_path: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET_NAME

    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    body_font = Font(name="Arial", size=10)
    body_align = Alignment(horizontal="left", vertical="top", wrap_text=True)
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=False)

    headers = [c[0] for c in COLUMNS]
    ws.append(headers)
    for col_idx, _ in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = border

    confidence_col = len(COLUMNS)  # последняя колонка

    for row in rows:
        ws.append(_row_to_tuple(row))

    last_row = ws.max_row
    last_col = len(COLUMNS)
    for r in range(2, last_row + 1):
        for c in range(1, last_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = body_font
            cell.border = border
            if c == confidence_col:
                pct = int(cell.value) if isinstance(cell.value, (int, float)) else 0
                cell.fill = _conf_fill(pct)
                cell.alignment = center_align
                cell.number_format = '0"%"'
            else:
                cell.alignment = body_align

    for idx, (_, width) in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(last_col)}{max(last_row, 1)}"

    wb.save(output_path)


def _timestamped(path: str) -> str:
    """`dir/out.xlsx` → `dir/out_20240414_230015.xlsx`."""
    base, ext = os.path.splitext(path)
    return f"{base}_{datetime.now():%Y%m%d_%H%M%S}{ext}"


def write_excel_safe(rows: List[ParsedRow], output_path: str) -> str:
    """Безопасная запись .xlsx: если основной файл заблокирован (обычно
    открыт в Excel), пишем под соседним именем с меткой времени.

    Возвращает фактический путь, куда удалось записать. Бросает оригинальное
    исключение, если и запасной вариант не сработал.

    Также пишет снапшот исходного вывода парсера в `<output>.snapshot.json`
    для последующей сверки с правками оператора (feedback-loop, см.
    `tools/collect_feedback.py`).
    """
    try:
        write_excel(rows, output_path)
        actual = output_path
    except PermissionError:
        alt = _timestamped(output_path)
        write_excel(rows, alt)
        actual = alt
    # Снапшот — рядом с Excel. Ошибки записи снапшота не фатальны.
    try:
        from pathlib import Path
        from .feedback import save_snapshot
        save_snapshot(rows, Path(actual + ".snapshot.json"))
    except Exception:  # noqa: BLE001
        pass
    return actual
