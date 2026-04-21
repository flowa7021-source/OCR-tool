"""Form-aware cargo table parser (idea #7 top-10).

Решает Bug 4 класс — «cargo forgets label-words of adjacent
columns». Раздел 4 «Груз» ТН-формы имеет чёткую табличную
структуру::

    № п/п | Наименование груза | Кол-во мест | Масса нетто |
          |                    |             | / брутто    |
          |                    |             |             |
    1     | Георешетка TENSAR  | 66 рул      | 2805 кг     |
    2     | Георешетка TriAx   | 63 рул      | 4504 кг     |

OCR выдаёт flat-text где столбцы склеены табами / `|` / multi-
пробелами. Regex-approach (``fields.extract_cargo``) очищает
result через list of label-keywords, но не понимает структуры —
легко цепляет «неопасный» / «рулон» / «пп лентой» из соседних
колонок как часть наименования.

Этот модуль — структурный parser:

    1. :func:`detect_table_structure` — heuristic «это table?»
       по консистентным separator'ам в ≥ 2 строках.

    2. :func:`parse_cargo_table` — разбивает на rows[dict[header,
       value]]. Header нормализуется case-insensitive.

    3. :func:`extract_name_from_table` — удобная обёртка, join'ит
       значения колонки «Наименование» всех строк.

Интеграция в :func:`src.tn_parser.fields.extract_cargo` — сравнение
table-way vs regex-way: если regex-result содержит маркеры
label-leak'а соседних колонок, table-result побеждает.
"""

from __future__ import annotations

import re

#: Separator'ы колонок в OCR-flat'ном table-тексте. Pipe (``|``)
#: — классический grid-rendering; tab — структурный OCR; 3+
#: consecutive spaces — column-alignment fallback.
_COLUMN_SEPARATOR_RE = re.compile(r"\s*\|\s*|\t+|\s{3,}")

#: Минимум строк с consistent separator-count для детекции таблицы.
#: 1 строка — мог быть legitimate pipe'ом в тексте; 2+ — уже signal.
_MIN_TABLE_ROWS = 2

#: Keywords для обнаружения header'а колонки «Наименование».
_NAME_HEADER_KEYWORDS = ("наимен",)


def _split_row(line: str) -> list[str]:
    """Разбить строку на колонки по separator'у. Пустые cells
    сохраняются (поддерживают позиционный match header ↔ row).

    Если в строке 1 column — значит separator'а не было.
    """
    cells = _COLUMN_SEPARATOR_RE.split(line.strip())
    return [c.strip() for c in cells]


def detect_table_structure(body: str) -> bool:
    """True если body содержит ≥ :data:`_MIN_TABLE_ROWS` строк с
    консистентным числом separator'ов (одинаковое кол-во колонок).
    """
    if not body or not body.strip():
        return False
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < _MIN_TABLE_ROWS:
        return False

    multi_col_lines = [
        _split_row(ln) for ln in lines
        if len(_split_row(ln)) >= 2
    ]
    if len(multi_col_lines) < _MIN_TABLE_ROWS:
        return False

    # Модальное число колонок. Принимаем таблицу если ≥ _MIN_TABLE_ROWS
    # строк имеют одинаковое число колонок (≥ 2).
    from collections import Counter
    counter = Counter(len(row) for row in multi_col_lines)
    top_count, top_freq = counter.most_common(1)[0]
    return top_count >= 2 and top_freq >= _MIN_TABLE_ROWS


def parse_cargo_table(body: str) -> list[dict[str, str]]:
    """Parse body в list of rows, каждая row — dict[header, value].

    Первая строка трактуется как header. Если cell-count первой
    строки отличается от последующих — используется cell-count
    majority для header detection.

    Returns:
        List of row-dicts. Пустой список если структура не
        детектируется как таблица.
    """
    if not detect_table_structure(body):
        return []

    lines = [ln for ln in body.splitlines() if ln.strip()]
    all_rows = [_split_row(ln) for ln in lines]
    # Выбираем самое популярное cell-count — это «основной» shape
    # таблицы. Строки с другим cell-count отбрасываются (шапка,
    # ltitle, footer).
    from collections import Counter
    counter = Counter(len(r) for r in all_rows)
    main_cols = counter.most_common(1)[0][0]

    shaped = [r for r in all_rows if len(r) == main_cols]
    if len(shaped) < _MIN_TABLE_ROWS:
        return []

    header = shaped[0]
    data_rows = shaped[1:]
    return [
        {header[i]: row[i] for i in range(main_cols)}
        for row in data_rows
    ]


def _find_name_column_key(row: dict[str, str]) -> str | None:
    """Вернуть ключ который выглядит как «Наименование» (case-
    insensitive, stem match на «наимен»).
    """
    for key in row:
        k_low = (key or "").lower()
        if any(kw in k_low for kw in _NAME_HEADER_KEYWORDS):
            return key
    return None


def extract_name_from_table(body: str) -> str:
    """Извлечь join'ed значения колонки «Наименование» из table-body.

    Returns:
        Join'ed string (через ``; `` если ≥ 2 непустых значений),
        или пустая строка если table не детектирована или колонки
        «Наименование» нет.
    """
    rows = parse_cargo_table(body)
    if not rows:
        return ""
    name_key = _find_name_column_key(rows[0])
    if not name_key:
        return ""
    values = [
        r.get(name_key, "").strip()
        for r in rows
        if r.get(name_key, "").strip()
    ]
    if not values:
        return ""
    return "; ".join(values)


__all__ = [
    "detect_table_structure",
    "parse_cargo_table",
    "extract_name_from_table",
]
