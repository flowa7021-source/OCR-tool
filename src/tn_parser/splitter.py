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

# Граница документа ТН в тексте. Матчим строку, которая содержит
# «Транспортная накладная» как заголовок (именительный падеж —
# косвенные падежи «транспортной накладной», «транспортную накладную»
# встречаются в прозе/резюме и не должны быть границей документа).
# Поддерживаем два стиля:
#   1) строка начинается с «Транспортная накладная …» (классический OCR);
#   2) строка вида «ДОКУМЕНТ №1 — ТРАНСПОРТНАЯ НАКЛАДНАЯ — …»
#      (пост-OCR структурированный формат).
_DOC_HEADER = re.compile(
    r"(?mi)^[^\n]*?(?<![\w\-])"
    r"транспортная\s+накладная\b"
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
# Признак «страница 2 ТН» (оборотная сторона без шапки-формы): разделы
# 8-17 формы Постановления № 2200. OCR-мусорной подложки из разделов
# 1-7 здесь нет (они все на первой странице), зато есть приёмка груза,
# сдача, переадресовка, замечания, расчётная масса и подпись водителя.
# Пользователи часто сшивают двусторонние ТН «лицо + оборот», при
# page-aware-split эта страница должна быть ``ttn_cont`` а не
# ``blank`` — иначе reception / замечания выпадают из парсинга.
_TTN_BACK_PAGE_MARKERS_RE = re.compile(
    r"(?i)"
    r"при[её]м\s+груз"              # раздел 8
    r"|сдач[аи]\s+груз"             # раздел 9
    r"|расчетн\w+\s+масс"           # раздел 10 (расчётная масса)
    r"|переадресовк"                # раздел 12
    r"|подпис\w+\s+водител"         # раздел 17 (подпись водителя)
    r"|фактическ\w+.*?(?:врем|дат).*?приб"
    r"|стоимост\w+\s+услуг\s+перевозчик"  # раздел 15-16
    r"|оговорк\w+\s+и\s+замечан"    # раздел 14
)
_UPD_MARKERS_RE = re.compile(
    r"счет[-\s]?фактура|универсал\w+\s+передаточн", re.IGNORECASE
)
_REGISTRY_MARKERS_RE = re.compile(
    r"\bреестр\b|отгрузочн\w+\s+специф|товарн\w+\s+реестр",
    re.IGNORECASE,
)
# Attachment-типы: страницы которые физически в ТН-пакете, но не
# являются самой ТН. Классический пример — доверенность на получение
# груза по форме М-2 (приказ Минфина № 17 от 31.10.1997). На
# UPD_36 парсер раньше собирал реквизиты из доверенности (АО
# «Моспроект-3») как consignee самой ТН — регрессия по реальным
# данным. Теперь такие страницы классифицируются отдельно и
# исключаются из page-aware сборки документа.
_POA_MARKERS_RE = re.compile(
    r"(?i)\bдоверенност\w+\b|\bприложени\w+\s+к\s+(?:договору|упд|накладной)",
)


def _classify_page(text: str) -> str:
    """Возвращает: 'ttn_start' | 'ttn_cont' | 'upd' | 'registry' |
    'attachment' | 'blank'.

    Приоритет по шапке первых 800 символов: UPD/счёт-фактура,
    реестр и attachment-документы (доверенности / приложения)
    узнаются первыми, т.к. у них в теле могут встречаться те же
    слова что в ТН («Грузополучатель» как поле доверенности).
    Если шапка — ТН, используем признаки TTN.
    """
    stripped = text.strip()
    if len(stripped) < 100:
        return "blank"
    head = text[:800]

    if _UPD_MARKERS_RE.search(head):
        return "upd"
    if _REGISTRY_MARKERS_RE.search(head):
        return "registry"
    if _POA_MARKERS_RE.search(head):
        return "attachment"

    has_header = bool(_TTN_HEADER_RE.search(text))
    has_ttn_marker = bool(_TTN_FIRST_PAGE_MARKERS_RE.search(text))
    has_back_marker = bool(_TTN_BACK_PAGE_MARKERS_RE.search(text))
    # Страница 1 ТН: есть шапка + раздел ТН. OCR часто съедает номер
    # раздела, но название («Грузоотправитель», «Перевозчик») выживает.
    if has_header and has_ttn_marker:
        return "ttn_start"
    # Есть шапка ТН, но без распознаваемых разделов — страница 2
    # (оборотная, с разделами 9-12 + подписи).
    if has_header:
        return "ttn_cont"
    # Нет шапки «Транспортная накладная» но есть маркеры оборотной
    # страницы (приём груза, сдача, расчётная масса, переадресовка
    # и т.п.) — это Page 2 двустороннего экземпляра. Без этой ветки
    # page-aware split терял бы всю обратную сторону, reception /
    # замечания выпадали из Excel. Регрессия апреля 2026: после
    # добавления ``attachment`` в non-TTN список, has_non_ttn стал
    # True даже на простых ТН+доверенность пакетах, page-aware
    # включился — и без этой ветки p.2 дропалась.
    if has_back_marker:
        return "ttn_cont"

    return "blank"


def _split_by_pages(text: str) -> list[str]:
    """Page-aware разбиение. Возвращает [] если не помогло.

    Применяется к PDF, содержащим НЕ-TTN страницы (UPD / REGISTRY /
    ATTACHMENT). На файлах с одним типом страниц page-aware отдаёт
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
    has_non_ttn = any(
        t in ("upd", "registry", "attachment") for t in types
    )
    if not has_non_ttn:
        return []

    docs: list[str] = []
    current: list[str] = []
    for page, t in zip(pages, types, strict=False):
        if t == "ttn_start":
            if current:
                docs.append("\n".join(current).strip())
            current = [page]
        elif t == "ttn_cont" and current:
            current.append(page)
        # upd / registry / attachment / blank — пропускаем целиком.
    if current:
        docs.append("\n".join(current).strip())
    return [d for d in docs if len(d) >= 80]


def split_documents(text: str) -> list[str]:
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
    docs: list[str] = []
    for i in range(len(matches)):
        chunk = text[starts[i]:starts[i + 1]].strip()
        if len(chunk) >= 80:
            docs.append(chunk)
    return docs if docs else [text.strip()]
