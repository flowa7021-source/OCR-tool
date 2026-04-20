# -*- coding: utf-8 -*-
"""Формирование человекочитаемого лога рядом с Excel-файлом.

В отличие от сырого `logging`, здесь всё подано так, чтобы оператору было
удобно пройтись глазами по файлу и быстро найти строки, требующие проверки:

- Цветных маркеров нет (файл .log, в блокноте без подсветки)
- Используем ASCII-префиксы [OK]/[WARN]/[ERR] + тире
- Для каждой строки печатается confidence и список подозрительных полей

Публичные функции:
    build_log_lines(run_info, rows_by_file)  -> List[str]
    write_log(path, lines)                    — просто записывает файл
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import GARBAGE, MISSING, ParsedRow


FIELD_LABELS = {
    "date": "Дата",
    "number": "№",
    "shipper": "Грузоотправитель",
    "consignee": "Грузополучатель",
    "cargo": "Груз",
    "volume": "Объём",
    "driver": "Водитель",
    "vehicle": "Транспортное средство",
    "reception": "Приём груза",
}


def _problem_fields(row: ParsedRow, threshold: float = 0.5) -> List[str]:
    """Список полей с уверенностью ниже порога (или с MISSING/GARBAGE)."""
    conf = row.confidence
    mapping = {
        "date": (conf.date, row.date),
        "number": (conf.number, row.number),
        "shipper": (conf.shipper, row.shipper),
        "consignee": (conf.consignee, row.consignee),
        "cargo": (conf.cargo, row.cargo),
        "volume": (conf.volume, row.volume),
        "driver": (conf.driver, row.driver),
        "vehicle": (conf.vehicle, row.vehicle),
        "reception": (conf.reception, row.reception),
    }
    bad = []
    for key, (c, val) in mapping.items():
        if c < threshold or val in (MISSING, GARBAGE):
            bad.append(FIELD_LABELS[key])
    return bad


def _row_level(row: ParsedRow) -> str:
    if row.note.startswith("ERROR:"):
        return "ERR "
    if row.confidence.overall() < 0.5 or "LOW" in row.note:
        return "WARN"
    return "OK  "


def build_log_lines(
    input_path: str,
    output_path: str,
    elapsed_s: float,
    rows_by_file: Dict[str, List[ParsedRow]],
) -> List[str]:
    total_files = len(rows_by_file)
    ok_files = sum(
        1 for rows in rows_by_file.values()
        if all(not r.note.startswith("ERROR:") for r in rows)
    )
    err_files = total_files - ok_files
    total_rows = sum(len(rs) for rs in rows_by_file.values())

    lines: List[str] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("=" * 70)
    lines.append("  Парсер транспортных накладных — отчёт")
    lines.append("=" * 70)
    lines.append(f"  Запуск:     {ts}")
    lines.append(f"  Источник:   {input_path}")
    lines.append(f"  Результат:  {output_path}")
    lines.append(f"  Файлов:     {total_files}  (OK: {ok_files}, ошибок: {err_files})")
    lines.append(f"  Строк:      {total_rows}")
    lines.append(f"  Время:      {elapsed_s:.1f} с")
    lines.append("")

    for fname, rows in rows_by_file.items():
        for idx, row in enumerate(rows):
            level = _row_level(row)
            pct = round(row.confidence.overall() * 100)
            suffix = f"#{idx + 1}" if len(rows) > 1 else ""
            lines.append(f"[{level}] {pct:>3}% | {fname}{suffix}")

            if row.note.startswith("ERROR:"):
                lines.append(f"       └─ {row.note}")
                continue

            lines.append(f"       ├─ {row.waybill} от {row.date}")
            problems = _problem_fields(row)
            if problems:
                lines.append(f"       └─ Проверить: {', '.join(problems)}")
            else:
                lines.append(f"       └─ Все поля распознаны.")

    lines.append("")
    lines.append("=" * 70)
    return lines


def write_log(path: str, lines: Iterable[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_log_safe(path: str, lines: Iterable[str]) -> str:
    """Аналог write_log с фоллбэком на имя с меткой времени при
    PermissionError.

    Полезно на Windows, где лог-файл может оказаться открытым в блокноте.
    """
    # Iterable может быть генератором — материализуем, чтобы пройтись дважды.
    lines = list(lines)
    try:
        write_log(path, lines)
        return path
    except PermissionError:
        import os
        from datetime import datetime
        base, ext = os.path.splitext(path)
        alt = f"{base}_{datetime.now():%Y%m%d_%H%M%S}{ext}"
        write_log(alt, lines)
        return alt
