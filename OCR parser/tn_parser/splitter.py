# -*- coding: utf-8 -*-
"""Разбиение текста на несколько накладных, если они в одном PDF.

Две стратегии:

1. **Page-aware** (если в тексте есть разделитель страниц «\\f»):
   классифицируем каждую страницу (TTN_START / TTN_CONT / UPD /
   REGISTRY / BLANK) и собираем в документы только TTN-страницы.
   Это правильный путь для сводных PDF вида «УПД + реестр + ТН × N».

2. **Anchor-based** (fallback, когда страницы не размечены):
   каждое вхождение заголовка «Транспортная накладная» считаем
   границей нового документа.
"""

from __future__ import annotations

import re
from typing import List


_DOC_HEADER = re.compile(
    r"(?mi)^\s*транспортн(?:ая|ой)\s+накладн(?:ая|ой)\b"
)

# Страничные признаки.
_TTN_HEADER_RE = re.compile(
    r"транспортн\w+\s+накладн\w+", re.IGNORECASE
)
# Признак «страница 1 ТН»: есть как минимум один раздел ТН из тех,
# что обычно помещаются на первой странице формы. OCR может съесть
# номер раздела, но название раздела часто выживает. Требуем слово
# длиной ≥ 8 букв — короткие «груз» ловят слишком много.
_TTN_FIRST_PAGE_MARKERS_RE = re.compile(
    # Только специфичные для ТН разделы. «Продавец» / «Покупатель»
    # НЕ входят — это шапка счёта-фактуры (UPD), а не ТН.
    r"(?i)грузоотправ|грузополуч|перевозчик|транспортн\w+\s+средств"
)
_UPD_MARKERS_RE = re.compile(
    r"счет[-\s]?фактура|универсал\w+\s+передаточн", re.IGNORECASE
)
_REGISTRY_MARKERS_RE = re.compile(
    r"\bреестр\b|отгрузочн\w+\s+специф|товарн\w+\s+реестр",
    re.IGNORECASE,
)


def _classify_page(text: str) -> str:
    """Возвращает: 'ttn_start' | 'ttn_cont' | 'upd' | 'registry' | 'blank'.

    Приоритет по шапке первых 500 символов: UPD/счёт-фактура и реестр
    узнаются первыми, т.к. у них в теле могут встречаться те же слова,
    что в ТН («Грузоотправитель» как поле УПД), и по ним их с TTN не
    различить. Если шапка — ТН, используем признаки TTN.
    """
    stripped = text.strip()
    if len(stripped) < 100:
        return "blank"
    head = text[:800]

    if _UPD_MARKERS_RE.search(head):
        return "upd"
    if _REGISTRY_MARKERS_RE.search(head):
        return "registry"

    has_header = bool(_TTN_HEADER_RE.search(text))
    has_ttn_marker = bool(_TTN_FIRST_PAGE_MARKERS_RE.search(text))
    # Страница 1 ТН: есть шапка + раздел ТН. OCR часто съедает номер
    # раздела, но название («Грузоотправитель», «Перевозчик») выживает.
    if has_header and has_ttn_marker:
        return "ttn_start"
    # Есть шапка ТН, но без распознаваемых разделов — страница 2
    # (оборотная, с разделами 9-12 + подписи).
    if has_header:
        return "ttn_cont"

    return "blank"


def _split_by_pages(text: str) -> List[str]:
    """Page-aware разбиение. Возвращает [] если не помогло.

    Применяется только к PDF, содержащим НЕ-TTN страницы (UPD /
    REGISTRY). На файлах с одним типом страниц page-aware отдаёт
    худший результат, чем anchor-based: в двустраничных ТН,
    подшитых «все лицевые → все оборотные», мы не сможем
    правильно связать лицевую и оборотную одной ТН.
    """
    if "\f" not in text:
        return []
    pages = text.split("\f")
    types = [_classify_page(p) for p in pages]
    if "ttn_start" not in types:
        return []
    has_non_ttn = any(t in ("upd", "registry") for t in types)
    if not has_non_ttn:
        return []

    docs: List[str] = []
    current: List[str] = []
    for page, t in zip(pages, types):
        if t == "ttn_start":
            if current:
                docs.append("\n".join(current).strip())
            current = [page]
        elif t == "ttn_cont" and current:
            current.append(page)
        # upd / registry / blank — пропускаем целиком.
    if current:
        docs.append("\n".join(current).strip())
    return [d for d in docs if len(d) >= 80]


def split_documents(text: str) -> List[str]:
    """Разбивает нормализованный текст на список отдельных накладных."""
    if not text:
        return []

    # 1) Попытка page-aware разбиения для сводных PDF (UPD + TTN × N).
    page_docs = _split_by_pages(text)
    if page_docs:
        return page_docs

    # 2) Fallback: anchor-based разбиение по заголовкам.
    matches = list(_DOC_HEADER.finditer(text))
    if len(matches) <= 1:
        return [text.strip()]

    starts = [m.start() for m in matches]
    starts.append(len(text))
    docs: List[str] = []
    for i in range(len(matches)):
        chunk = text[starts[i]:starts[i + 1]].strip()
        if len(chunk) >= 80:
            docs.append(chunk)
    return docs if docs else [text.strip()]
