"""Layout-aware извлечение текста с учётом координат блоков.

OCR-слой в PDF уже содержит bbox для каждого span'а. На двухколоночных
формах (типовая ТН, реверсивная сторона УПД) это позволяет отделить
текст левой колонки от правой — иначе парсер склеивает «Компания-
перевозчик» и «ФИО водителя» из разных колонок в одну строку.

Стратегия:
    1. Извлекаем все spans через `page.get_text("dict")`.
    2. Группируем в «строки» по y-координате (толерантность ~3pt).
    3. Проверяем двуколоночность: страница делится пополам по X, если
       >60% строк имеют разрыв шире 30pt — это двуколоночный layout.
    4. Для двуколоночных: собираем левую и правую колонки отдельно,
       внутри каждой — по y, потом по x. Join через «\\n».
    5. Для одноколоночных: стандартный flat stream (по y, потом x).

Этот модуль даёт текст в том же формате, что `page.get_text("text")`,
но с геометрически корректным порядком чтения.
"""

from __future__ import annotations

import fitz

_Y_TOLERANCE = 3.0        # pt, толерантность слипания спанов в одну строку
_COL_GAP_THRESHOLD = 30.0  # pt, минимальный разрыв между колонками
_COL_MIN_FRACTION = 0.5   # доля строк с разрывом для активации


def _collect_spans(page: fitz.Page) -> list[tuple[float, float, float, float, str]]:
    """Собирает все текстовые spans страницы как (x0, y0, x1, y1, text)."""
    spans: list[tuple[float, float, float, float, str]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                bb = span.get("bbox")
                if not bb or len(bb) < 4:
                    continue
                spans.append((bb[0], bb[1], bb[2], bb[3], text))
    return spans


def _group_into_lines(
    spans: list[tuple[float, float, float, float, str]],
) -> list[list[tuple[float, float, float, float, str]]]:
    """Группирует spans по y-координате в логические строки."""
    if not spans:
        return []
    # Сортируем по y, потом по x.
    ordered = sorted(spans, key=lambda s: (s[1], s[0]))
    lines: list[list[tuple[float, float, float, float, str]]] = []
    current: list[tuple[float, float, float, float, str]] = [ordered[0]]
    current_y = ordered[0][1]
    for sp in ordered[1:]:
        if abs(sp[1] - current_y) <= _Y_TOLERANCE:
            current.append(sp)
        else:
            lines.append(sorted(current, key=lambda s: s[0]))
            current = [sp]
            current_y = sp[1]
    if current:
        lines.append(sorted(current, key=lambda s: s[0]))
    return lines


def _is_two_column(
    lines: list[list[tuple[float, float, float, float, str]]],
    page_width: float,
) -> bool:
    """Страница двуколоночная, если в ≥ COL_MIN_FRACTION строк есть
    разрыв ≥ COL_GAP_THRESHOLD примерно по центру."""
    if not lines:
        return False
    mid = page_width / 2
    gap_lines = 0
    for line in lines:
        if len(line) < 2:
            continue
        # Ищем самый большой разрыв между соседними spans.
        max_gap = 0.0
        gap_x = 0.0
        for a, b in zip(line, line[1:], strict=False):
            gap = b[0] - a[2]
            if gap > max_gap:
                max_gap = gap
                gap_x = (a[2] + b[0]) / 2
        if max_gap >= _COL_GAP_THRESHOLD and abs(gap_x - mid) < page_width * 0.2:
            gap_lines += 1
    return gap_lines / max(len(lines), 1) >= _COL_MIN_FRACTION


def _render_two_column(
    lines: list[list[tuple[float, float, float, float, str]]],
    page_width: float,
) -> str:
    """Двуколоночный рендер: сначала весь левый столбец, потом правый."""
    mid = page_width / 2
    left_lines: list[str] = []
    right_lines: list[str] = []
    for line in lines:
        left = [s for s in line if (s[0] + s[2]) / 2 < mid]
        right = [s for s in line if (s[0] + s[2]) / 2 >= mid]
        if left:
            left_lines.append(" ".join(s[4] for s in left))
        if right:
            right_lines.append(" ".join(s[4] for s in right))
    parts = []
    if left_lines:
        parts.append("\n".join(left_lines))
    if right_lines:
        parts.append("\n".join(right_lines))
    # Разделитель между колонками — двойной \n, чтобы секционер
    # не слил «Грузоотправитель» из левой с «Заказчик услуг» из правой.
    return "\n\n".join(parts)


def _render_flat(
    lines: list[list[tuple[float, float, float, float, str]]],
) -> str:
    """Одноколоночный рендер: spans в каждой строке через пробел, строки \\n."""
    return "\n".join(" ".join(s[4] for s in line) for line in lines)


def render_page(page: fitz.Page) -> str:
    """Главная функция: возвращает текст страницы в геометрически
    корректном порядке. Двуколоночные — отдельно левая/правая, иначе
    стандартный flat stream."""
    spans = _collect_spans(page)
    if not spans:
        return ""
    lines = _group_into_lines(spans)
    if not lines:
        return ""
    if _is_two_column(lines, page.rect.width):
        return _render_two_column(lines, page.rect.width)
    return _render_flat(lines)


def extract_layout_text(pdf_path: str) -> str:
    """Извлекает текст из всех страниц PDF с учётом layout."""
    doc = fitz.open(pdf_path)
    try:
        return "\f".join(render_page(page) for page in doc)
    finally:
        doc.close()
