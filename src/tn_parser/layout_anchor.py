"""Layout-aware anchoring через bbox-координаты (idea #1 top-10).

Решает класс багов **«adjacent-cell leak»**: regex-якорь в
:mod:`src.tn_parser.sections` ищет «1. Грузоотправитель» и может
случайно подхватить соседнюю ячейку «1а Заказчик услуг» (обе
начинаются с цифры + Cyrillic-заголовка). Без геометрии текста это
неразличимо — OCR сбивает порядок строк при колонках.

На реальных ТН (Приложение № 4 к Постановлению № 2200) левая
колонка всегда содержит ответственные стороны (Грузоотправитель /
Грузополучатель / Перевозчик), правая — заказчиков услуг и
второстепенные реквизиты. Знание x-координаты токенов разрешает
неоднозначность.

**Источник bbox** — ``pytesseract.image_to_data(output_type=DICT)``
TSV формат. Он УЖЕ вычисляется в :class:`src.application.pipeline.
OCRPipeline._compute_confidences` для per-word confidence. Мы
переиспользуем его без дополнительных OCR-вызовов.

**Формат TSV** (dict[str, list] равной длины для каждой колонки):

    {
        "level":     [1, 2, 2, 3, 3, 5, 5, ...],
        "page_num":  [1, 1, 1, 1, 1, 1, 1, ...],
        "block_num": [0, 1, 1, 1, 1, 1, 1, ...],
        "par_num":   [0, 0, 0, 1, 1, 1, 1, ...],
        "line_num":  [0, 0, 0, 1, 1, 1, 2, ...],
        "word_num":  [0, 0, 0, 0, 0, 1, 2, 1, ...],
        "left":      [0, 100, 100, 100, 100, 100, 180, ...],
        "top":       [0, 50, 50, 50, 50, 50, 50, 80, ...],
        "width":     [page_width, 400, 400, 400, 400, 60, 120, ...],
        "height":    [page_height, 600, 600, 40, 40, 20, 20, ...],
        "conf":      [-1, -1, -1, -1, -1, 95, 87, ...],
        "text":      ["", "", "", "", "", "ООО", "Ромашка", ...],
    }

Только элементы с ``level=5`` (слово-уровень) и непустым text,
conf > 0 релевантны — уровни 1-4 это block/paragraph/line
контейнеры без содержимого.
"""

from __future__ import annotations

import re
from typing import Literal

Column = Literal["left", "right"]


def _is_word_level(tsv: dict[str, list], idx: int) -> bool:
    """TSV row индекса ``idx`` — текстовый word-токен.

    Предпочитаем level==5 (word) когда колонка ``level`` есть, но
    если её нет (упрощённый TSV в unit-тестах) — пропускаем по
    наличию непустого text + conf > 0.
    """
    text = str(tsv.get("text", [""])[idx] or "").strip()
    if not text:
        return False
    # conf=-1 / 0 — pytesseract использует для whitespace-только
    # токенов и для уровневых agregate-строк без содержимого.
    try:
        conf = float(tsv.get("conf", [0])[idx])
    except (ValueError, TypeError, IndexError):
        conf = 0
    if conf <= 0:
        return False
    # Если ``level`` колонка присутствует — уважаем её (настоящий
    # pytesseract TSV всегда имеет level, и уровни 1-4 это
    # block/paragraph/line aggregates без значимого text/conf).
    if "level" in tsv:
        try:
            level = int(tsv["level"][idx])
        except (ValueError, TypeError, IndexError):
            level = 5  # неизвестно — считаем word
        # level=5 = word; level<5 = aggregate (block/par/line).
        # Но aggregate-row обычно с conf=-1 или text="", так что
        # conf > 0 check выше уже отсеет. Level check — defensive
        # для редких OCR-случаев.
        if level not in (0, 5):
            # 0 = not set (наши моки иногда), 5 = word.
            return False
    return True


def _iter_word_tokens(tsv: dict[str, list]):
    """Yield (idx, dict-view) для каждого word-level токена."""
    n = len(tsv.get("text", []))
    for i in range(n):
        if not _is_word_level(tsv, i):
            continue
        yield i, {
            "left":   int(tsv["left"][i]),
            "top":    int(tsv["top"][i]),
            "width":  int(tsv["width"][i]),
            "height": int(tsv["height"][i]),
            "conf":   float(tsv["conf"][i]),
            "text":   str(tsv["text"][i]),
        }


def _center_x(token: dict) -> float:
    return token["left"] + token["width"] / 2.0


def find_tokens_by_column(
    tsv: dict[str, list],
    *,
    column: Column,
    page_width: int,
) -> list[dict]:
    """Вернуть word-level токены чьи x-центры попадают в ``column``.

    Сторона определяется по ``page_width / 2`` — всё что левее центра
    = left, всё правее = right.

    Args:
        tsv: ``pytesseract.image_to_data`` в DICT-формате. Пустой
            или не-dict → возвращается пустой список.
        column: ``"left"`` или ``"right"``.
        page_width: Ширина страницы в тех же координатах, что и
            ``left``/``width`` в TSV (обычно px при 400 DPI).

    Returns:
        Список dict'ов с ключами ``left, top, width, height, conf,
        text``. Порядок — как в TSV (сверху-вниз, слева-направо
        внутри строк — естественный порядок Tesseract'а).
    """
    if not tsv or "text" not in tsv:
        return []
    if page_width <= 0:
        return []

    mid = page_width / 2.0
    out: list[dict] = []
    for _idx, tok in _iter_word_tokens(tsv):
        cx = _center_x(tok)
        if (column == "left" and cx < mid) or (
            column == "right" and cx >= mid
        ):
            out.append(tok)
    return out


def find_section_region(
    tsv: dict[str, list],
    *,
    anchor_pattern: str,
    prefer_column: Column,
    page_width: int,
) -> tuple[int, int] | None:
    """Найти вертикальный диапазон секции по якорю в указанной колонке.

    Проходит по всем word-level токенам в ``prefer_column``, ищет
    первое совпадение с ``anchor_pattern`` (regex, IGNORECASE,
    unicode-aware). Возвращает ``(top_y, bottom_y)`` bbox строки где
    якорь найден — это вертикальный диапазон ТОЛЬКО строки якоря,
    caller сам решает сколько «вниз» брать за тело секции (обычно
    до следующего anchor'а).

    Args:
        tsv: TSV-dict.
        anchor_pattern: Regex для поиска (например, ``r"грузоотправ"``
            поймает «Грузоотправитель», «Грузоотправителя» и т.п.).
        prefer_column: В какой колонке искать. Если якоря нет в этой
            колонке — вернуть ``None`` (caller может решить перейти
            на regex-fallback на весь текст).
        page_width: Ширина страницы для column-определения.

    Returns:
        ``(top, bottom)`` или ``None``.
    """
    if not anchor_pattern:
        return None
    try:
        rx = re.compile(anchor_pattern, re.IGNORECASE)
    except re.error:
        return None

    column_tokens = find_tokens_by_column(
        tsv, column=prefer_column, page_width=page_width,
    )
    # Сортируем по y — ищем самое верхнее совпадение.
    column_tokens.sort(key=lambda t: (t["top"], t["left"]))

    for tok in column_tokens:
        if rx.search(tok["text"]):
            return (tok["top"], tok["top"] + tok["height"])
    return None


def extract_column_text_between(
    tsv: dict[str, list],
    *,
    y_top: int,
    y_bottom: int,
    column: Column,
    page_width: int,
) -> str:
    """Склеить текст из выбранной колонки в вертикальном диапазоне.

    Сохраняет читательский порядок: сверху-вниз, внутри строки
    слева-направо. Строки разделяет ``\\n``.

    Использование: после ``find_section_region`` вызываем эту функцию
    чтобы получить raw-текст секции, очищенный от правой колонки.

    Args:
        y_top, y_bottom: Вертикальный диапазон (inclusive).
        column: ``"left"`` или ``"right"``.
        page_width: Для column-определения.

    Returns:
        Собранная строка. Пустая если в диапазоне нет токенов.
    """
    tokens = find_tokens_by_column(tsv, column=column, page_width=page_width)
    in_range = [
        t for t in tokens
        if y_top <= t["top"] <= y_bottom
        or y_top <= t["top"] + t["height"] <= y_bottom
    ]
    if not in_range:
        return ""

    # Группируем по строкам — токены с близкими ``top`` (±половина
    # высоты) = одна строка. Tesseract сам на уровне TSV даёт
    # line_num группировку, но её здесь нет — собираем из
    # координат.
    # Сначала сортируем по y, потом кластеризуем.
    in_range.sort(key=lambda t: (t["top"], t["left"]))
    lines: list[list[dict]] = []
    for tok in in_range:
        if not lines:
            lines.append([tok])
            continue
        # Ширина строки — средняя высота токенов в последней строке
        # × 0.5 допуск. Гарантирует что токены с top=200 и top=210
        # (одна строка) схлопнутся.
        last_line = lines[-1]
        avg_h = sum(t["height"] for t in last_line) / len(last_line)
        baseline = last_line[0]["top"]
        if abs(tok["top"] - baseline) <= avg_h * 0.5:
            last_line.append(tok)
        else:
            lines.append([tok])

    # Внутри каждой строки — сортируем по x.
    out_lines = []
    for line in lines:
        line.sort(key=lambda t: t["left"])
        out_lines.append(" ".join(t["text"] for t in line))
    return "\n".join(out_lines)


__all__ = [
    "Column",
    "find_tokens_by_column",
    "find_section_region",
    "extract_column_text_between",
]
