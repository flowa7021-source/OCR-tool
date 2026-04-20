"""Извлечение текста из PDF с сохранением порядка чтения.

Пробуем две стратегии и выбираем ту, что даёт больше содержательного
текста:

    1. `page.get_text("text")` — собственный порядок чтения PDF. Именно эту
       текстовую выдачу получает Acrobat при копировании. Для большинства
       машиночитаемых и OCR-прослоечных PDF это лучший вариант.
    2. `page.get_text("blocks", sort=True)` — блочная выдача с внутренней
       сортировкой PyMuPDF. Иногда даёт более чистый результат для табличных
       форм без нормального порядка чтения.

Результат разделяет страницы символом `\f`.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from .layout_blocks import render_page as _render_layout_page


def _page_text_plain(page) -> str:
    return page.get_text("text") or ""


def _page_text_blocks(page) -> str:
    blocks = page.get_text("blocks", sort=True) or []
    text_blocks = [
        b for b in blocks
        if len(b) >= 7 and b[6] == 0 and b[4] and b[4].strip()
    ]
    return "\n\n".join(b[4].strip() for b in text_blocks)


def _page_text_layout(page) -> str:
    """Layout-aware рендер: двуколоночные страницы получают разделитель
    между левой и правой колонкой, что упрощает парсеру разделение
    «Грузоотправитель» / «Заказчик услуг» в двух колонках формы."""
    try:
        return _render_layout_page(page) or ""
    except Exception:  # pragma: no cover
        return ""


def _score_text(text: str) -> int:
    """Грубая мера «полезности» текста: количество букв+цифр."""
    return sum(1 for c in text if c.isalnum())


def extract_best_text(pdf_path: str) -> str:
    """Выбирает лучшую стратегию для каждой страницы.

    Приоритет: plain > blocks > layout-aware. Layout-aware подключаем
    ТОЛЬКО если он существенно богаче содержимым — иначе он ломает
    хорошо упорядоченные страницы (OCR-Tesseract уже сам строит
    линейный flow, разбиение по колонкам его рвёт).

    Порог: layout должен быть минимум на 30% богаче лучшего из
    plain/blocks, чтобы его выбрать.
    """
    doc = fitz.open(pdf_path)
    try:
        pages: list[str] = []
        for page in doc:
            plain = _page_text_plain(page)
            blocks = _page_text_blocks(page)
            best_score = max(_score_text(plain), _score_text(blocks))

            # Layout-aware рассматриваем только если базовые стратегии
            # дали мало: OCR плохо справился с линейным порядком.
            # Иначе предпочитаем blocks (при разнице ≥20%) или plain.
            layout = _page_text_layout(page) if best_score < 100 else ""
            layout_score = _score_text(layout)

            if layout_score >= best_score * 1.3 and layout_score > 50:
                pages.append(layout)
            elif _score_text(blocks) > _score_text(plain) * 1.2:
                pages.append(blocks)
            else:
                pages.append(plain)
        return "\f".join(pages)
    finally:
        doc.close()


# Сохраняем старые имена для совместимости и для CLI/GUI, где они могут
# явно вызываться как «резерв».
def extract_blocks_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        return "\f".join(_page_text_blocks(page) for page in doc)
    finally:
        doc.close()


def extract_plain_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        return "\f".join(_page_text_plain(page) for page in doc)
    finally:
        doc.close()
