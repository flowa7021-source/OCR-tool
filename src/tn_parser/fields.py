"""Извлечение конкретных полей из разделов ТН.

Принцип: сначала ищем поле в «своём» разделе (высокая уверенность), если
нет — падаем на эвристики по всему тексту (низкая уверенность). Каждый
экстрактор возвращает кортеж (value, confidence).

Сложности реального OCR:
- В таблице заголовок ячейки и её значение — на разных «логических» строках.
- Ячейки с двумя колонками (способ | ФИО, марка | ГРЗ) после извлечения
  разворачиваются в две соседние строки.
- Пояснения в скобках («(реквизиты, позволяющие идентифицировать…)») и
  формульные подсказки (« является экспедитором», « (а) … ») — мусор.

Экстракторы ниже стараются это учесть.
"""

from __future__ import annotations

import re

from .models import GARBAGE, MISSING
from .normalize import (
    clean_value,
    is_garbage,
    is_noise_line,
    strip_garbage_tokens,
)
from .validators import (
    GRZ_CANDIDATE,
    find_grz,
    is_valid_date,
    is_valid_grz,
    is_valid_inn,
)

_DATE_ANY = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")

# Даты, фигурирующие на бланках форм ТН / счёта-фактуры (даты
# Постановлений Правительства РФ, печатаются прямо на бланке).
# Это не дата ТН, а метаданные формы.
_FORM_METADATA_DATES = frozenset({
    "30.11.2021",  # ПП № 2116 — форма ТН (ред. 2022)
    "21.12.2020",  # ПП № 2200 — правила перевозок
    "26.12.2011",  # ПП № 1137 — форма счёта-фактуры
    "02.04.2021",  # изменение в ПП № 1137
    "02.04.2024",  # ред. ПП № 1117
    "11.12.2023",  # ред. правил перевозок
})

# Номер: не захватываем "Экземпляр №" (подпись у графы экземпляра).
# N[º°]? убран — голая латинская «N» слишком широкий маркер (матчит «RENAULT» и т.п.).
# Для «No» обязательна граница слова (\b) — иначе ловит «No» внутри OCR-мусора:
# «Tpyronoyaren» → «No» + «yaren» ≠ номер.
_NUMBER_AFTER_SYMBOL = re.compile(
    r"(?:№|\bNo\.?)\s*[:\-–—]?\s*"
    r"([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-_/.]{0,48})",
    re.IGNORECASE,
)
# Запасной: номер вплотную к "№" без пробела («№7145/Б»)
_NUMBER_STICKY = re.compile(
    r"(?:№|\bNo\.?)\s*\n?\s*([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-_/.]{0,48})",
    re.IGNORECASE,
)
# Терпимый к OCR-шуму между «№» и значением: «№ — |7145/Б» (форма с
# разделительной колонкой). \n намеренно НЕ включён — перенос строки
# означает, что значение поля пустое.
_NUMBER_LAX = re.compile(
    r"(?:№|\bNo\.?)[ \t\|:\-–—_.]{0,8}"
    r"([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-_/.]{1,48})",
    re.IGNORECASE,
)

# Шаблон высокой точности: «№ <значение> от YYYY-MM-DD» или
# «№ <значение> от DD.MM.YYYY». Встречается в пост-OCR формате
# (inputs/TN_k_UPD_*.txt) сразу под заголовком ТН. Срабатывает раньше
# остальных (_NUMBER_STICKY и Co.), потому что структура «№ … от <дата>»
# однозначно идентифицирует пару (номер ТН, дата ТН) — в отличие от
# одиночных «№ N» внутри прозы или в «ДОКУМЕНТ №1».
_NUMBER_WITH_DATE = re.compile(
    r"(?:№|\bNo\.?)\s*"
    r"([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-_/.]{1,48})"
    r"\s+от\s+(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})",
    re.IGNORECASE,
)

# Якорь для поиска номера и даты ТН — «Транспортная накладная» в
# именительном падеже. Косвенные падежи («транспортной накладной»,
# «транспортную накладную») типичны для резюмирующей прозы
# («Скан содержит один экземпляр транспортной накладной …») и
# для ссылок на форму; в них номер ТН не следует.
_WAYBILL_HEADER = re.compile(
    r"транспортная\s+накладная", re.IGNORECASE
)

# Строки-служебки, которые надо пропустить в начале раздела контрагента.
_SERVICE_LINE_RE = re.compile(
    r"^(?:"
    r"является\s+(?:экспедитором|грузоотправителем)"
    r"|\(\s*реквизиты\b"
    r"|полное\s+наименование"
    r"|сокращ\w*\s+наименование"
    r"|наименование\s+(?:юр|лица)"
    r"|фио\b|ф\.?и\.?о\.?"
    r"|экземпляр\s*№?"
    r"|реквизиты\s+документа"
    r"|да\b|нет\b"
    r"|заказчик\s+услуг\b"                 # «Заказчик услуг по организации…»
    r"|при\s+наличи[ии]\b"                 # автономная пометка «(при наличии)»
    r"|[\[\(][\s xх×✓✔][\]\)]"   # чекбоксы
    r"|[\-–—=_\s]{3,}"            # разделители из дефисов
    r"|\([а-яА-Я]+\)"             # короткие пометки в скобках: «(а)», «(б)»
    r")",
    re.IGNORECASE,
)


def _is_service_or_empty(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Целиком в скобках: «(реквизиты, позволяющие…)»
    if re.match(r"^\(.+\)\s*$", s):
        return True
    return bool(_SERVICE_LINE_RE.match(s))


# Маркеры «после этой строки реквизиты контрагента закончились».
# Нужны как end-anchor в extract_org: любой из этих начальных токенов
# означает, что дальше — подпись/печать/контакт/служебная разметка, и это
# НЕ должно попадать в поле грузоотправителя/грузополучателя/перевозчика.
_ORG_END_MARKERS = re.compile(
    r"^(?:"
    r"подпис[ьиея]"                        # Подпись, Подписью
    r"|м\.?\s*п\.?\b"                      # МП, М.П.
    r"|печат[ьи]\b"
    r"|контактн(?:ое|ый|ого)\s+лиц"        # Контактное лицо
    r"|дата\s+составлен"
    r"|ответственн[оы]й\s+за"
    r"|должност[ьи]\b"
    r"|доверенност"
    r"|ф\.?\s*и\.?\s*о\.?\s+(?:водител|ответствен|предста)"
    r")",
    re.IGNORECASE,
)

# Маркеры «после этой строки наименование груза закончилось». Следующие
# атрибуты (класс опасности, упаковка, тара, способ погрузки, условия
# хранения) — отдельные графы, не часть названия.
_CARGO_END_MARKERS = re.compile(
    r"^(?:"
    r"класс\s+опасност"
    r"|упаковк"
    r"|тара\b"
    r"|способ\s+(?:погрузк|упаковк)"
    r"|условия\s+(?:хранен|перевозк)"
    r"|маркировк"
    r"|номер\s+контейнер"
    r")",
    re.IGNORECASE,
)

# Ключевые слова соседних граф — используется для обрезки fallback-regex:
# когда одно поле и его «сосед» оказались на одной строке OCR.
_ADJACENT_FIELD_CUTOFF = re.compile(
    r"\b(?:"
    r"грузоотправител"
    r"|грузополучател"
    r"|перевозчик"
    r"|транспортн(?:ое|ого)\s+средств"
    r"|при[её]м\s+груз"
    r"|выдач\w*\s+груз"
    r"|переадресовк"
    r"|сопроводительн"
    r"|стоимость\s+(?:услуг|перевозк)"
    r"|оговорк"
    r"|отметк\w+\s+грузо"
    r"|указан\w+\s+грузоотправ"
    r")",
    re.IGNORECASE,
)


# Новый контракт извлечения.
_ORG_PREFIX_RE = re.compile(
    r"(?:"
    r"\b(?:ООО|ОАО|АО|ЗАО|ПАО|НКО|ПБОЮЛ|ИП|ТОО|КФХ|АНО|ЧУ|ФГУП|ГУП|МУП|ФГБУ|ГБУ|НОУ|АНПО)\b"
    # OCR часто пишет «000» (три нуля) вместо «ООО» — опознаём только
    # перед кавычкой или заглавной буквой, чтобы не путать с «000 руб».
    r"|\b000(?=\s*[«\"'“”„А-ЯЁ])"
    r")",
    re.IGNORECASE,
)
_INN_INCLUSIVE_RE = re.compile(r"\bИНН[:\s]*\d{10,12}", re.IGNORECASE)
_FINANCIAL_MARKER_RE = re.compile(
    r"\b(?:ИНН|КПП|ОГРН|ОКПО|ОКВЭД|БИК)\b", re.IGNORECASE
)
_FIO_RE = re.compile(
    # «Иванов И.И.» / «Иванов И. И.» / «И.И. Иванов»
    r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\."
    r"|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+"
    # OCR иногда даёт один инициал: «Кузибеков И.», «Кулоков Ш.»
    r"|\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.(?!\s?[А-ЯЁ])"
)
# Полное ФИО — Фамилия Имя Отчество (без инициалов). Все три слова
# ОБЯЗАНЫ начинаться с заглавной буквы (русской или латинской —
# первое слово может быть mixed-script после OCR'а: «Fentes
# Александр Николаснич»). Без требования заглавной первой буквы
# регекс ловит мусор: «pea Иохтановления Правитеглатва» в OCR'е
# UPD_36 — лидером там фрагмент «ред pea …», и «pea» — не фамилия.
_FIO_FULL_RE = re.compile(
    r"\b[A-ZА-ЯЁ][A-Za-zА-ЯЁа-яё]{2,13}[ \t]+"
    r"[А-ЯЁ][а-яё]{2,14}[ \t]+"
    r"[А-ЯЁ][а-яё]{2,14}\b"
)

# Слот-2/слот-3 blacklist (substring-match — OCR может мангнуть
# любую часть слова). Если 2-й или 3-й токен содержит ВНУТРИ
# себя один из этих stem'ов — это типовая government / legal
# формулировка («Постановления Правительства», «Министерство
# Финансов», «Российской Федерации»), не имя.
#
# OCR-tolerant: используем стрипинг гласных и подобных, чтобы
# одинаково ловить «Постановления» / «Иохтановления» (П→И, о→ох…).
# Конкретно «тановлен» / «новлени» — стабильные хвосты слова
# «постановления», переживают любой OCR-шум на префиксе.
_FAKE_FIO_SUBSTRINGS = (
    "тановлен", "новлени", "становле",     # постановление
    "вительств", "ительств", "теглатв",    # правительство
    "инистер", "инстер",                    # министерство
    "едераци", "ёдераци",                   # федерация
    "оссийск", "оссийс", "оссийс",          # российск
    "оложени", "ложени",                    # положение
    "риложени", "ложени",                   # приложение
    "ерзавозк", "перевозк",                 # перевозки
    "транспорт",
    "общество",
    "компани",
    "предприяти",
    "налоговой", "налоговая",
)


def _looks_like_real_fio(matched: str) -> bool:
    """Защита от ложных _FIO_FULL_RE matches на gov/legal-формулировках.

    OCR на bold-шапках бланка часто выдаёт три Capital-Cased
    токена, не относящиеся к ФИО («Постановления Правительства
    Российской», «Министерство Финансов России», «ред Постановления
    Правительства» — последнее в UPD_36 OCR). Защищаемся substring-
    match'ем по стабильным хвостам слов: даже если OCR мангнул
    «Постановления» → «Иохтановления», stem «тановлен» сохраняется.

    Возвращает False если match похож на gov/legal-формулировку
    (т.е. НЕ имя водителя).
    """
    parts = matched.split()
    if len(parts) < 3:
        return True
    low = matched.lower()
    return all(stem not in low for stem in _FAKE_FIO_SUBSTRINGS)
# OCR регулярно искажает «шт»: «нтт», «штт», «шт.». Допускаем
# 2–3 буквы, последняя — обязательно «т». НЕ включаем «ит» (слишком
# часто ловит «Итого», «5 из 10» и подобные артефакты).
_SHT_OCR = r"(?:шт|штт|нтт|нт)"
_CARGO_QTY_SHT_RE = re.compile(
    rf",?\s*\d+(?:[,.]\d+)?\s*{_SHT_OCR}\.?\s*$", re.IGNORECASE
)
_CARGO_KOL_VO_MEST_RE = re.compile(
    r"\bкол[-\s]?во\s+мест\b|\bколичество\s+мест\b", re.IGNORECASE
)
_QTY_SHT_INLINE_RE = re.compile(
    rf"\b(\d+(?:[,.]\d+)?\s*{_SHT_OCR}\.?)", re.IGNORECASE
)
_NETTO_BRUTTO_RE = re.compile(
    r"нетто[^\n]*брутто[^\n]*(?:объ[её]м|м[³3])[^\n]*", re.IGNORECASE
)

# Компоненты тройки (мест, нетто, объём), встречающиеся в пост-OCR
# структурированном формате отдельными строками:
#   Количество мест: 129
#   Масса нетто: 7.3095 т
#   Объем: 57.948 м³
# Допускаем «Кол-во мест», «Количество мест», «Мест:», OCR-варианты
# единиц («т.», «Т», «тонн», «м3», «м³», «куб»), десятичный
# разделитель — запятая или точка.
_PLACES_LINE_RE = re.compile(
    r"(?im)^\s*(?:кол[-\s]?во|количество)?\s*мест(?:а)?\s*[:\-–—]\s*(\d+)\b"
)
_NETTO_LINE_RE = re.compile(
    r"(?im)(?:масса\s+)?нетто\s*[:\-–—]\s*(\d+(?:[,.]\d+)?)\s*(?:т|тн|тонн)\b"
)
_VOLUME_LINE_RE = re.compile(
    r"(?im)об[ъь][её]м\s*[:\-–—]\s*(\d+(?:[,.]\d+)?)\s*(?:м[³3]|куб)"
)
# Split-точка для cargo — слова-label'ы соседних колонок формы ТН
# («класс опасности», «упаковка», «способ погрузки», «маркировка»,
# «бирка», «ярлык», «паспорт», «штрих-код», «сертификат»). OCR
# склеивает соседние ячейки таблицы в одну строку, и без этого
# регекса cargo превращается в «ACTUAL_NAME, бирка, ярлык; способ».
# Паттерн консервативный — требует начало слова (\b) и legitimate
# русский корень, иначе поймаем false-positive'ы вроде «бирки»
# внутри наименования упаковки.
_CARGO_ATTR_SPLIT_RE = re.compile(
    r"(?i)\b(?:класс\s+опасност|упаковк|тара\b"
    r"|способ\s+(?:погрузк|упаковк|перевозк)"
    r"|условия\s+(?:хранен|перевозк)"
    r"|маркировк|номер\s+контейнер"
    r"|бирк[аи]|ярлык\w*|паспорт\w*|сертификат\w*"
    r"|штрих[-\s]?код\w*|штриховой\s+код)"
)


def _meaningful_lines(body: str) -> list[str]:
    """Разбиваем тело секции на строки, выкидываем служебные."""
    if not body:
        return []
    out = []
    for raw in body.splitlines():
        line = raw.strip()
        if _is_service_or_empty(line):
            continue
        # Строки с инструкцией-пояснением в стиле «(…текст…)» после значения:
        # «Самовывоз  (реквизиты, позволяющие…)» — чистим хвост.
        line = re.sub(r"\s*\([^)]*реквизиты[^)]*\)\s*$", "", line, flags=re.IGNORECASE)
        # Чистим «Заказчик услуг … (при наличии)» — он может прилипнуть
        # к строке с именем организации как левый префикс:
        # «Га Заказчик услуг по организации, перевозки груза (при наличии), ООО …»
        # Реальный OCR часто ставит запятую внутри фразы, поэтому [^,]* недостаточно.
        # Решение: при наличии «заказчик услуг» на строке — ищем первую org-метку
        # (ООО/АО/ИП…) и берём текст начиная с неё; если org-метки нет — вся строка
        # является служебной пометкой и отбрасывается.
        if re.search(r"заказчик\s+услуг", line, re.IGNORECASE):
            m_org = re.search(r"\b(ООО|АО|ЗАО|ПАО|ИП|ПБОЮЛ)\b", line, re.IGNORECASE)
            if m_org:
                line = line[m_org.start():]
            else:
                continue
        line = line.strip(" \t,;")
        # Слишком короткие строки — почти наверняка OCR-мусор (одиночные
        # символы/слоги): «Г», «а.», «ГЕР», «ав4'». Значимых данных не несут.
        if len(line) <= 3:
            continue
        # Строка без единого содержательного слова (≥ 4 букв подряд), даты или
        # длинного числа (ИНН/индекс) — OCR-мусор вроде «/ Й /», «11 [3] Т».
        if (not re.search(r"[А-Яа-яЁёA-Za-z]{4,}", line)
                and not re.search(r"\d{2}\.\d{2}\.\d{4}", line)
                and not re.search(r"\d{5,}", line)):
            continue
        # Строка-шум по токен-статистике: большая доля мусорных токенов
        # («іі-і», «Ц:і», «Бекам'тбд»).
        if is_noise_line(line):
            continue
        # Чистим одиночные мусорные токены в строке (украинские буквы,
        # апострофы-в-середине, 3+ переключения скриптов).
        line = strip_garbage_tokens(line)
        line = line.strip(" \t,;")
        if not line or len(line) <= 3:
            continue
        if not is_garbage(line):
            out.append(line)
    return out


def _find_valid_inn_match(s: str) -> re.Match | None:
    """Первый валидный (по контрольной сумме ФНС) ИНН в строке."""
    for m in re.finditer(r"\b(\d{10}|\d{12})\b", s):
        if is_valid_inn(m.group(1)):
            return m
    return None


def _cut_at_inn_inclusive(s: str) -> str | None:
    """Если в строке есть «ИНН + 10–12 цифр» — вернуть срез до конца ИНН.

    Толерантен к OCR-искажениям префикса «ИНН»: «ИНИ», «HHH», «ИНI»,
    «Инн», «И Н Н» — любые 2–4 буквы непосредственно перед группой из
    10–12 цифр считаем квази-ИНН и обрезаем включительно.
    """
    m = _INN_INCLUSIVE_RE.search(s)
    if m:
        return s[: m.end()].strip(" ,;")
    # Валидный ИНН (по контр-сумме) даже без префикса «ИНН» — сильный
    # сигнал. Лучше него: ищем «буквы + пробел + 10/12 цифр».
    m = re.search(r"\b[A-Za-zА-Яа-яЁё]{2,4}\.?\s+(\d{10}|\d{12})\b", s)
    if m and is_valid_inn(m.group(1)):
        return s[: m.end()].strip(" ,;")
    # Фолбэк: просто первый валидный ИНН.
    m = _find_valid_inn_match(s)
    if m:
        return s[: m.end()].strip(" ,;")
    return None


def _cut_before_financial(s: str) -> str:
    """Обрезать перед первым финансовым маркером.

    Маркеры:
        1) _FINANCIAL_MARKER_RE (ИНН/КПП/ОГРН/ОКПО/ОКВЭД/БИК);
        2) 10–12 цифр подряд — почти гарантированно ИНН, даже если
           OCR исказил сам префикс «ИНН» → «ИНИ», «HHH», «ИНI» и т.п.

    Берём ту позицию, что встретилась РАНЬШЕ. Без этого после
    «…, ИНИ 7707820850, КПП 770701001» обрезка по КПП оставляла
    искажённый ИНН внутри строки.
    """
    positions = []
    m = _FINANCIAL_MARKER_RE.search(s)
    if m:
        positions.append(m.start())
    # Ищем ВАЛИДНЫЙ ИНН (по контрольной сумме). Случайные 10–12 цифр
    # (например, телефон «89306796587») не пройдут.
    m = _find_valid_inn_match(s)
    if m:
        positions.append(max(0, m.start() - 4))
    if positions:
        return s[: min(positions)].strip(" ,;-–—")
    return s


_BANK_NAMES_RE = re.compile(
    r"\b(?:СБЕРБАНК|ВТБ|АЛЬФА[-\s]?БАНК|ГАЗПРОМБАНК|РОССЕЛЬХОЗБАНК"
    r"|ОТКРЫТИЕ|РОСБАНК|ТИНЬКОФФ|УРАЛСИБ|ПСБ|СОВКОМБАНК|БИН[-\s]?БАНК"
    r"|МКБ|МОСКОВСКИЙ\s+КРЕДИТНЫЙ|РАЙФФАЙЗЕН|ЮНИКРЕДИТ)\b",
    re.IGNORECASE,
)


def _trim_to_org(s: str) -> str:
    """Отрезать левый префикс до первого ORG-маркера (ООО/АО/ИП/...).

    ORG-маркеры перед названием банка (например, «ПАО СБЕРБАНК»,
    «АО АЛЬФА-БАНК») пропускаем — это реквизиты расчётного счёта,
    не сама компания. Ищем следующий ORG после банка.
    """
    pos = 0
    while True:
        m = _ORG_PREFIX_RE.search(s, pos)
        if not m:
            return s
        # Смотрим на 60 символов вперёд от ORG — если встретилось имя
        # банка, пропускаем этот ORG и ищем следующий.
        tail = s[m.end(): m.end() + 60]
        if _BANK_NAMES_RE.search(tail):
            pos = m.end()
            continue
        return s[m.start():]


# ---------------------------------------------------------------------------


def extract_number_and_date(
    head: str, full_text: str
) -> tuple[str, float, str, float]:
    """Возвращает (number, conf_number, date, conf_date).

    Поиск номера:
        1) После «Транспортная накладная» ищем № + значение в 400 символах.
        2) Если не вышло — сканируем весь текст, пропуская «Экземпляр №».
    """
    number = MISSING
    conf_num = 0.0
    date = MISSING
    conf_date = 0.0

    def _pick_number(region: str, base_conf: float) -> tuple[str, float]:
        # Собираем все кандидаты и выбираем первый непустой/осмысленный.
        for rx in (_NUMBER_STICKY, _NUMBER_AFTER_SYMBOL, _NUMBER_LAX):
            for m in rx.finditer(region):
                # Проверяем, не «Экземпляр №» ли это.
                # Стратегия: смотрим на ближайшие 20 символов слева, но НЕ
                # пересекаем границу строки. Это обрабатывает два случая:
                #   а) «Экземпляр №\n№ 7145/Б» — переносы между строками:
                #      second № не видит «экземпляр» из предыдущей строки.
                #   б) «Экземпляр №  Дата 23.07.2022  № 7145/Б» — одна строка:
                #      second № смотрит лишь 20 симв. назад → «23.07.2022  »,
                #      «экземпляр» не попадает в окно.
                line_start = region.rfind("\n", 0, m.start())
                line_start = 0 if line_start < 0 else line_start + 1
                # Узкое окно (20 симв.) — для коротких маркеров, чтобы
                # «Экземпляр №  Дата 23.07.2022  № 7145/Б» на одной строке
                # не блокировал второй (настоящий) № по слову «экземпляр».
                near_start = max(line_start, m.start() - 20)
                near_ctx = region[near_start: m.start()].lower()
                if "экземпляр" in near_ctx or "экз." in near_ctx:
                    continue
                # Полное начало строки — для длинных заголовков
                # («Приложение No 4», «Постановление Правительства», «Договор No …»).
                # «ДОКУМЕНТ №N — …» — нумерация документа внутри пост-OCR
                # пакета (inputs/UPD_* / TN_k_UPD_*). N — это порядковый
                # индекс, а не номер ТН.
                line_ctx = region[line_start: m.start()].lower()
                if ("приложение" in line_ctx or "прил." in line_ctx
                        or "постановлен" in line_ctx or "договор" in line_ctx
                        or "к правилам" in line_ctx
                        or "документ" in line_ctx):
                    continue
                # Широкое окно (300 симв. включая предыдущие строки) —
                # OCR часто переносит «(в ред. Постановления … № 2116)»
                # на несколько строк, и маркер «постановлен» уезжает
                # из текущей строки.
                wide_ctx = region[max(0, m.start() - 300): m.start()].lower()
                if ("постановлен" in wide_ctx
                        or "правительств" in wide_ctx and "рф" in wide_ctx
                        or "в ред." in wide_ctx or "к правилам" in wide_ctx
                        or "приложени" in wide_ctx and ("1137" in wide_ctx
                                                        or "2116" in wide_ctx
                                                        or "2200" in wide_ctx
                                                        or "534" in wide_ctx)):
                    continue
                candidate = m.group(1).strip(" .,:;")
                if not candidate:
                    continue
                if candidate.lower() in ("экземпляр", "экз"):
                    continue
                # Номер накладной всегда содержит цифру. Одинокие буквы
                # («й», «yaren») — это OCR-мусор после потерянного №.
                if not re.search(r"\d", candidate):
                    continue
                # Известные номера постановлений / приложений, с которыми
                # печатают бланки ТН. Если парсер ловит один из них, почти
                # гарантированно это служебный маркер формы, а не номер ТН.
                # OCR часто уничтожает слово «Постановление»/«Приложение»
                # целиком, поэтому контекстный blacklist срабатывает не
                # всегда — этот список спасает.
                if candidate in ("1137", "2116", "2200", "2311", "272",
                                 "534", "1117"):
                    continue
                if is_garbage(candidate):
                    continue
                return candidate, base_conf
        return MISSING, 0.0

    source = head if head else full_text
    if source:
        anchor = _WAYBILL_HEADER.search(source)
        if anchor:
            tail = source[anchor.end(): anchor.end() + 500]
            # Высокоприоритетный паттерн «№ X от YYYY-MM-DD» — если
            # сработал, даёт сразу и номер, и дату ТН. Это устойчивее
            # к ложным «№» в шапке («ДОКУМЕНТ №1», «Приложение № 4»)
            # и в самой строке «(в ред. … № 2116)».
            m_nd = _NUMBER_WITH_DATE.search(tail)
            if m_nd:
                cand = m_nd.group(1).strip(" .,:;")
                if cand and re.search(r"\d", cand) and cand not in (
                    "1137", "2116", "2200", "2311", "272", "534", "1117",
                ) and not is_garbage(cand):
                    number = cand
                    # 1.0: номер ТН + дата ТН на одной строке — двойной
                    # сигнал, парсер уверен. 0.7: только из head.
                    conf_num = 1.0 if head else 0.7
                    date_raw = m_nd.group(2)
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_raw):
                        y, mo, d = date_raw.split("-")
                        cand_date = f"{d}.{mo}.{y}"
                    else:
                        cand_date = date_raw
                    if is_valid_date(cand_date) and cand_date not in _FORM_METADATA_DATES:
                        date = cand_date
                        conf_date = 1.0 if head else 0.7
            if number == MISSING:
                number, conf_num = _pick_number(tail, 0.9 if head else 0.6)
            for date_m in _DATE_ANY.finditer(tail):
                cand = date_m.group(1)
                if not is_valid_date(cand):
                    continue
                if cand in _FORM_METADATA_DATES:
                    continue  # дата с бланка формы, не из содержимого ТН
                date = cand
                conf_date = 1.0 if head else 0.7
                break
            # OCR иногда теряет символ «№» — тогда после «Транспортная
            # накладная» идёт <дата>\n<номер>. Ловим номер как первую
            # осмысленную строку после заголовка, которая не похожа на
            # дату / служебную метку / заголовок раздела.
            if number == MISSING:
                for raw in tail.splitlines()[:10]:
                    ln = raw.strip(" \t|—–-[]()")
                    if not ln or len(ln) < 2 or len(ln) > 50:
                        continue
                    low = ln.lower()
                    if (low.startswith(("транспортн", "заказ", "дата",
                                        "экземпляр", "№", "no", "приложение"))
                            or "накладн" in low):
                        continue
                    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", ln):
                        continue
                    if not re.fullmatch(
                        r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-_/.]{1,48}", ln
                    ):
                        continue
                    if not re.search(r"\d", ln):
                        continue
                    if is_garbage(ln):
                        continue
                    number = ln.strip(" .,:;")
                    conf_num = 0.8 if head else 0.55
                    break

    # Резерв: ищем по всему тексту.
    if number == MISSING and full_text:
        number, c = _pick_number(full_text[:1200], 0.5)
        conf_num = c if number != MISSING else 0.0
    if date == MISSING and full_text:
        for m in _DATE_ANY.finditer(full_text):
            cand = m.group(1)
            if is_valid_date(cand) and cand not in _FORM_METADATA_DATES:
                date = cand
                conf_date = 0.5
                break

    return number, conf_num, date, conf_date


# ---------------------------------------------------------------------------


def _collect_org_lines(section_body: str, max_lines: int = 4) -> str:
    """Склеить первые значимые строки контрагента через запятую.

    End-anchors: _ORG_END_MARKERS обрывают, _ADJACENT_FIELD_CUTOFF режет
    строку и прекращает сбор (защита от склейки двухколоночной формы).
    """
    lines = _meaningful_lines(section_body)
    cut_lines: list[str] = []
    for ln in lines:
        if _ORG_END_MARKERS.match(ln):
            break
        cut = _ADJACENT_FIELD_CUTOFF.search(ln)
        if cut and cut.start() > 0:
            head = ln[: cut.start()].strip(" ,;:-–—")
            if head:
                cut_lines.append(head)
            break
        cut_lines.append(ln)
    return ", ".join(ln.rstrip(",") for ln in cut_lines[:max_lines])


def _fallback_org_line(full_text: str, fallback_kw: str) -> str | None:
    pat = re.compile(
        rf"{fallback_kw}\s*[:\-–—]?\s*\n?\s*([^\n\r]{{2,500}})",
        re.IGNORECASE,
    )
    m = pat.search(full_text)
    if not m:
        return None
    candidate = m.group(1).strip()
    cut = _ADJACENT_FIELD_CUTOFF.search(candidate)
    if cut and cut.start() > 0:
        candidate = candidate[: cut.start()].strip(" ,;:-–—")
    if not candidate or _SERVICE_LINE_RE.match(candidate):
        return None
    return candidate


def _has_validated_inn(text: str) -> bool:
    """True если в строке есть ИНН-кандидат (10/12 цифр), прошедший
    контрольную сумму ФНС. Сигнал «поле прошло валидацию» → conf 1.0."""
    return _find_valid_inn_match(text) is not None


def extract_shipper(section_body: str, full_text: str) -> tuple[str, float]:
    """Грузоотправитель: от ORG-префикса до «ИНН \\d+» включительно.

    КПП/ОГРН/ОКПО всегда обрезаем. Если ИНН отсутствует — обрезать
    перед первым финансовым маркером.

    Confidence:
        1.0 — секция найдена + извлечённая строка содержит валидный ИНН
              (контрольная сумма ФНС прошла).
        0.9 — секция найдена, ИНН не валидируем (отсутствует или
              OCR-искажение цифр).
        0.5 — fallback по «грузоотправитель» в полном тексте.
    """
    if section_body:
        joined = _collect_org_lines(section_body, max_lines=4)
        if joined:
            joined = _trim_to_org(joined)
            cut_inn = _cut_at_inn_inclusive(joined)
            joined = cut_inn if cut_inn else _cut_before_financial(joined)
            joined = joined[:500].strip(" ,;")
            if joined and not is_garbage(joined):
                return joined, (1.0 if _has_validated_inn(joined) else 0.9)

    if full_text:
        candidate = _fallback_org_line(full_text, "грузоотправитель")
        if candidate:
            candidate = _trim_to_org(candidate)
            cut_inn = _cut_at_inn_inclusive(candidate)
            candidate = cut_inn if cut_inn else _cut_before_financial(candidate)
            candidate = candidate[:500].strip(" ,;")
            if candidate and not is_garbage(candidate):
                return candidate, 0.5

    return MISSING, 0.0


def _starts_with_org_prefix(text: str) -> bool:
    """True если строка начинается с одного из ORG-маркеров
    (ООО / АО / ИП / ...). Используется как «структурная валидация»
    consignee/cargo, где ИНН по design не включён в поле, но сама
    форма «ORG имя» — сильный сигнал, что извлечение состоялось."""
    return bool(_ORG_PREFIX_RE.match(text.strip()))


def extract_consignee(section_body: str, full_text: str) -> tuple[str, float]:
    """Грузополучатель: ORG-префикс → перед первым ИНН/КПП/ОГРН/ОКПО.

    Получатель всегда без ИНН и КПП (тот, кто принимает груз, не
    обязан раскрывать налоговые реквизиты в теле поля).

    Confidence:
        1.0 — секция найдена + результат начинается с ORG-маркера
              (структурная валидация: «АО Х», «ООО Y» — настоящее
              имя организации).
        0.9 — секция найдена, ORG-маркер не на старте (OCR-искажение
              префикса).
        0.5 — fallback по «грузополучатель» в полном тексте.
    """
    if section_body:
        joined = _collect_org_lines(section_body, max_lines=4)
        if joined:
            joined = _trim_to_org(joined)
            joined = _cut_before_financial(joined)
            joined = joined[:500].strip(" ,;")
            if joined and not is_garbage(joined):
                return joined, (1.0 if _starts_with_org_prefix(joined) else 0.9)

    if full_text:
        candidate = _fallback_org_line(full_text, "грузополучатель")
        if candidate:
            candidate = _trim_to_org(candidate)
            candidate = _cut_before_financial(candidate)
            candidate = candidate[:500].strip(" ,;")
            if candidate and not is_garbage(candidate):
                return candidate, 0.5

    return MISSING, 0.0


# ---------------------------------------------------------------------------


_CARGO_NAME_LINE_RE = re.compile(
    # Допускаем до 8 мусорных символов в начале строки: OCR часто ставит
    # «„-«», «|»», «з—^'» и т.п. перед словом «наименование».
    r"^[^\nа-яёА-ЯЁa-zA-Z]{0,8}(?:\d+[.)]\s*)?"
    r"[а-яА-Я\-\^'\"«»„]{0,3}?н.{1,5}мен\w*"
    r"\s*[:\-–—\u2010-\u2015\u2212]+\s*(.*)$",
    re.IGNORECASE,
)
_CARGO_MEASURE_LINE_RE = re.compile(
    r"^(ед\.\s*изм|кол-во|количес|масс|объ[её]м|нетто|брутто|в том числе)\b",
    re.IGNORECASE,
)


_CARGO_LEADING_JUNK_RE = re.compile(
    # OCR-артефакты в начале cargo-строки от соседних колонок формы:
    # ``]29 мест. тм. TENSAR…`` — цифры + единицы измерения + шум до
    # реального имени груза. Удаляем ведущую последовательность из
    # брекета/тире/цифр/единиц + пунктуации, если дальше идёт
    # легитимное содержательное слово.
    r"^[\]\[\)\(\-–—\s\d.,;:/]+(?:мест|шт|кг|т|тн|тонн|м[³3])\.?[\s,;.:]*",
    re.IGNORECASE,
)


def _clean_cargo_name(name: str) -> str:
    """Очистить наименование груза.

    Удаляются:
      * leading-артефакты от соседних колонок (``]29 мест. тм.``);
      * хвост ``«кол-во мест»`` и всё после него;
      * хвост ``«N шт»``;
      * label'ы атрибутов упаковки/маркировки/класса опасности
        (``бирка``, ``ярлык``, ``способ погрузки`` и т.п.) и всё
        после них.
    """
    # Leading-мусор: срезаем даже если дальше пусто (лучше пустая
    # строка чем фальшивые данные).
    name = _CARGO_LEADING_JUNK_RE.sub("", name).strip()
    m_kvm = _CARGO_KOL_VO_MEST_RE.search(name)
    if m_kvm:
        name = name[: m_kvm.start()].strip(" ,;-–—")
    name = _CARGO_QTY_SHT_RE.sub("", name).strip(" ,;-–—")
    parts = _CARGO_ATTR_SPLIT_RE.split(name, maxsplit=1)
    if parts:
        name = parts[0].strip(" ,;-–—")
    return name[:400].strip(" ,;")


def _first_cargo_name_from_lines(lines: list[str]) -> str | None:
    pending_label = False
    for _idx, ln in enumerate(lines):
        if _CARGO_END_MARKERS.match(ln):
            break
        m = _CARGO_NAME_LINE_RE.match(ln)
        if m:
            val = m.group(1).strip()
            if val:
                return val
            pending_label = True
            continue
        # OCR может съесть «на» в «наименование» — тогда строка выглядит
        # как «„-«именование — X» без начального «н». Ловим корень
        # «менование». Значение — только то, что идёт ПОСЛЕ этого корня
        # (иначе мы бы цепляли дефис из OCR-мусора ДО слова).
        m_word = re.search(r"мен[оае]ван\w*", ln, re.IGNORECASE)
        if m_word:
            after = ln[m_word.end():]
            m2 = re.match(
                r"[\s:\-–—\u2010-\u2015\u2212]+\s*(.+)",
                after,
            )
            if m2 and m2.group(1).strip():
                return m2.group(1).strip()
            pending_label = True
            continue
        low = ln.lower()
        if low.startswith("груз:") or low.startswith("груз —") or low.startswith("груз -"):
            parts = re.split(r"[:\-–—]", ln, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
            continue
        if pending_label:
            if _CARGO_MEASURE_LINE_RE.match(low):
                continue
            if _CARGO_KOL_VO_MEST_RE.search(low):
                continue
            if not re.search(r"[А-Яа-яЁёA-Za-z]{3,}", ln):
                continue
            # Снимаем начальные тире/двоеточия — они часть разделителя
            # от заголовка «Наименование» на предыдущей строке.
            return re.sub(
                r"^[\s:\-–—\u2010-\u2015\u2212]+", "", ln,
            ).strip()
    # Не нашли по меткам — первая «содержательная» строка.
    for ln in lines:
        if _CARGO_END_MARKERS.match(ln):
            break
        low = ln.lower()
        if _CARGO_MEASURE_LINE_RE.match(low):
            continue
        if not re.search(r"[А-Яа-яЁёA-Za-z]{3,}", ln):
            continue
        return ln
    return None


_CARGO_VALIDATION_RE = re.compile(
    r"\b(шт|штт|нтт|кг|т|тонн|м3|м³|куб|"
    r"рул|рулон|мешк|пач|короб|паллет|поддон|места?"
    r"|блок|плит[аы]|труб[аы]|кабел|сырь|материал|товар"
    r"|изделие|оборудовани|георешетк|георешётк|конструкци)\b",
    re.IGNORECASE,
)


def _is_validated_cargo(name: str) -> bool:
    """True если в наименовании груза есть structural-сигнал —
    единица измерения / упаковочный термин / типовой грузовой
    предмет. Защищает 1.0-conf от ложных срабатываний."""
    return bool(_CARGO_VALIDATION_RE.search(name))


def _is_regex_cargo_contaminated(name: str) -> bool:
    """True если regex-извлечённое имя содержит явные label-слова
    соседних колонок формы («класс опасности», «упаковка», «бирка»
    и т.п.) — значит split недочистил, и table-parser даст лучший
    результат."""
    if not name:
        return False
    low = name.lower()
    markers = (
        "класс опасн", "упаковк", "тара", "бирк", "ярлык",
        "способ погруз", "способ упаков", "штрих",
    )
    return any(m in low for m in markers)


def extract_cargo(section_body: str, full_text: str) -> tuple[str, float]:
    """Наименование груза. Без 'Кол-во мест', без хвоста 'N шт',
    без 'Класс опасности' / 'Упаковка' / 'Тара'.

    Confidence:
        1.0 — секция найдена + structural сигнал (ед. измерения /
              упаковка / типовой грузовой термин).
        0.9 — секция найдена, без structural сигнала.
        0.5 — fallback regex по «наименование груза».

    Табличная extraction (idea #7 top-10): если section_body имеет
    структуру таблицы (≥ 2 rows с consistent separator'ами),
    пробуем :mod:`src.tn_parser.cargo_table` в параллель к regex.
    Если regex-name загрязнён label'ами соседних колонок —
    table-name побеждает.
    """
    # Попытка 1: regex-way из секции.
    regex_name = ""
    regex_conf = 0.0
    if section_body:
        lines = _meaningful_lines(section_body)
        candidate = _first_cargo_name_from_lines(lines)
        if candidate:
            candidate = _clean_cargo_name(candidate)
            if candidate and not is_garbage(candidate):
                regex_name = candidate
                regex_conf = 1.0 if _is_validated_cargo(candidate) else 0.9

    # Попытка 2: table-way (form-aware, idea #7).
    table_name = ""
    if section_body:
        try:
            from .cargo_table import (
                detect_table_structure,
                extract_name_from_table,
            )

            if detect_table_structure(section_body):
                raw_table = extract_name_from_table(section_body)
                if raw_table:
                    cleaned = _clean_cargo_name(raw_table)
                    if cleaned and not is_garbage(cleaned):
                        table_name = cleaned
        except ImportError:
            pass

    # Table-way побеждает когда regex-name отсутствует или содержит
    # label-контамину соседних колонок. Table-name идёт из bare
    # «Наименование»-колонки и label'ами не грешит.
    if table_name and (
        not regex_name or _is_regex_cargo_contaminated(regex_name)
    ):
        return table_name, 1.0 if _is_validated_cargo(table_name) else 0.9

    if regex_name:
        return regex_name, regex_conf

    if full_text:
        for pat in (
            r"н.{1,5}мен\w*\s+груз\w*\s*[:\-–—\u2010-\u2015\u2212]+\s*([^\n\r]{2,400})",
            r"н.{1,5}мен\w*\s*[:\-–—\u2010-\u2015\u2212]+\s*([^\n\r]{2,400})",
        ):
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                candidate = _clean_cargo_name(m.group(1).strip()[:400])
                if candidate and not is_garbage(candidate):
                    return candidate, 0.5

    return MISSING, 0.0


_KOL_VO_MEST_VALUE_RE = re.compile(
    r"кол[-\s]?во\s+мест\s*[—:\-–\u2010-\u2015\u2212]?\s*(\d+)",
    re.IGNORECASE,
)


def _extract_triplet(source: str) -> list[str]:
    """Извлекает тройку (мест, нетто, объём) из отдельных строк секции.

    Порядок в выводе фиксированный: «N мест, X т, Y м³» — чтобы golden-
    проверка умела вычленять отдельные числа. Если какой-то из
    компонентов отсутствует — пропускаем его, не ставим плейсхолдер.
    """
    parts: list[str] = []
    m = _PLACES_LINE_RE.search(source)
    if m:
        parts.append(f"{m.group(1)} мест")
    m = _NETTO_LINE_RE.search(source)
    if m:
        parts.append(f"{m.group(1)} т")
    m = _VOLUME_LINE_RE.search(source)
    if m:
        parts.append(f"{m.group(1)} м³")
    return parts


def extract_volume(cargo_section: str, full_text: str) -> tuple[str, float]:
    """Объём / количество.

    Правило (согласовано с пользователем):
        1. Если наименование груза уже содержит «N шт» — это и есть
           «должный вид» ответа. Выводим «N шт» без лишнего шума.
        2. Иначе собираем тройку (мест, нетто, объём) из отдельных
           строк секции «— ГРУЗ —» / кол-во-мест блока.
        3. Если и тройка пуста, пробуем legacy-формы (однострочный
           «Нетто … Брутто … Объём …», «Кол-во мест — N»).
        4. Иначе MISSING.
    """
    for source, base_conf in ((cargo_section, 0.9), (full_text, 0.5)):
        if not source:
            continue

        # (1) «N шт» внутри наименования груза — приоритетно, как
        # просил пользователь. Матч ищем только в секции «Груз», чтобы
        # не подцепить «384 шт» из верхней сводной прозы
        # («комбинации из 8 мест / 384 шт / 15 мест / 720 шт»).
        sht = _QTY_SHT_INLINE_RE.search(source)
        if sht and source is cargo_section:
            # 1.0: число + единица «шт» — структурно валидно, парсер
            # уверен. Это не приближение «может быть штуки».
            return sht.group(1).strip(), 1.0 if base_conf >= 0.9 else base_conf

        # (2) Тройка по отдельным строкам — структурированный формат
        # «Количество мест: 129\n Масса нетто: 7.3095 т\n Объем: 57.948 м³».
        triplet = _extract_triplet(source)
        if triplet:
            # 1.0 если все 3 компонента + section-source: каждая строка
            # дала свой regex-hit, валидация структуры пройдена.
            full_triplet = (len(triplet) >= 3) and source is cargo_section
            return ", ".join(triplet), (
                1.0 if full_triplet else base_conf
            )

        # (3) Legacy: однострочный «Нетто … Брутто … Объём …».
        parts: list[str] = []
        m = _KOL_VO_MEST_VALUE_RE.search(source)
        if m:
            parts.append(f"{m.group(1)} мест")
        if sht:
            parts.append(sht.group(1).strip())
        m = _NETTO_BRUTTO_RE.search(source)
        if m:
            parts.append(m.group(0).strip(" ,;"))
        if parts:
            return ", ".join(parts), base_conf
    return MISSING, 0.0


# ---------------------------------------------------------------------------


def _compact_grz_search(region: str) -> str | None:
    """Схлопываем пробелы внутри буквенно-цифровых кластеров и ищем ГРЗ."""
    compact = re.sub(
        r"(?<=[А-ЯЁA-Z0-9])\s+(?=[А-ЯЁA-Z0-9])", "", region.upper()
    )
    grz = find_grz(compact)
    if grz and is_valid_grz(grz):
        return grz
    return None


def extract_driver(section_body: str, full_text: str) -> tuple[str, float]:
    """Водитель: ТОЛЬКО ФИО из правой колонки графы 6 «Перевозчик».

    Логика:
        1) _FIO_RE («Иванов И.И.» / «И.И. Иванов») в секции → conf 1.0
        2) _FIO_RE по всему тексту → conf 0.5
        3) MISSING — название компании-перевозчика сюда НЕ попадает.

    Если в графе 6 нет ФИО (только название компании или «Самовывоз»),
    поле остаётся пустым — это семантически правильно, водителя в форме
    не указали.
    """
    if section_body:
        m = _FIO_RE.search(section_body)
        if m:
            return m.group(0).strip(), 1.0
        pos = 0
        while True:
            m = _FIO_FULL_RE.search(section_body, pos)
            if not m:
                break
            pre = section_body[max(0, m.start() - 5): m.start()].lower()
            if (
                "ип " not in pre and "ип\n" not in pre
                and _looks_like_real_fio(m.group(0))
            ):
                # 1.0: полное ФИО (Фамилия Имя Отчество, не «И.И.
                # Иванов») в section_body — самая специфичная и
                # однозначная форма; короткие initials — менее
                # уникальны (могут совпасть с подписью представителя).
                # _looks_like_real_fio защищает от government/legal
                # формулировок («ред Постановления Правительства»).
                return m.group(0).strip(), 1.0
            pos = m.end()

    if full_text:
        # Ищем ФИО только в окне после слова «перевозчик» (допускаем
        # OCR-искажения: «перевозчи»/«персвояки»/«псревозчнк»…).
        # Иначе _FIO_RE по всему тексту ловит кладовщика / подпись
        # из раздела «Приём груза» — это НЕ водитель.
        anchor = re.search(r"п[ес][рст]евоз\w{0,5}", full_text, re.IGNORECASE)
        if anchor:
            window = full_text[anchor.start(): anchor.start() + 1500]
            m = _FIO_RE.search(window)
            if m:
                return m.group(0).strip(), 0.5
            # Полные ФИО в окне: пропускаем имена, которым предшествует
            # «ИП » — это руководитель компании-перевозчика, а не
            # водитель.
            pos = 0
            while True:
                m = _FIO_FULL_RE.search(window, pos)
                if not m:
                    break
                pre = window[max(0, m.start() - 5): m.start()].lower()
                if (
                    "ип " not in pre and "ип\n" not in pre
                    and _looks_like_real_fio(m.group(0))
                ):
                    return m.group(0).strip(), 0.4
                pos = m.end()

    return MISSING, 0.0


def _vehicle_marka_candidates(section_body: str, grz: str | None) -> list[str]:
    """Строки, из которых можно вытащить марку. Используем splitlines напрямую,
    чтобы «MAN TGS», «В 404 КМ» не отфильтровались _meaningful_lines."""
    out: list[str] = []
    for raw in section_body.splitlines():
        ln = raw.strip()
        if not ln or _is_service_or_empty(ln):
            continue
        low = ln.lower()
        if re.match(r"^\((тип|марка|модель|регистрационн|рег\.?)", low):
            continue
        if low.startswith(("марка", "модель", "тип", "т/с")):
            parts = re.split(r"[:\-–—]", ln, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                out.append(parts[1].strip())
            continue
        if grz:
            compact_ln = re.sub(
                r"(?<=[А-ЯЁA-Z0-9])\s+(?=[А-ЯЁA-Z0-9])", "", ln.upper()
            )
            m_grz = GRZ_CANDIDATE.search(compact_ln)
            if m_grz:
                if m_grz.start() > 0:
                    ns = 0
                    end_pos = len(ln)
                    for ci, ch in enumerate(ln):
                        if not ch.isspace():
                            if ns == m_grz.start():
                                end_pos = ci
                                break
                            ns += 1
                    brand_raw = ln[:end_pos].strip(" ,;()")
                    if brand_raw and len(brand_raw) >= 2:
                        out.append(brand_raw)
                continue
        if re.search(r"\b(инн|кпп|огрн|окпо)\b", low):
            continue
        if 2 <= len(ln) <= 60:
            out.append(ln)
    return out


def extract_vehicle(section_body: str, full_text: str) -> tuple[str, float]:
    """Транспортное средство: ТОЛЬКО ГРЗ (госномер) слитно без пробелов.

    Пример: «С782СК62», «Р814НР152». Марка автомобиля НЕ извлекается —
    пользователю нужен только идентификатор ТС (ГРЗ), по которому его
    можно однозначно найти.
    """
    if section_body:
        grz = _compact_grz_search(section_body)
        if grz:
            return grz.replace(" ", ""), 1.0

    if full_text:
        grz = _compact_grz_search(full_text)
        if grz:
            return grz.replace(" ", ""), 0.6
        for pat in (
            r"гос\.?\s*номер\s*[:\-–—]?\s*([^\n\r]{2,40})",
            r"рег\.?\s*знак\s*[:\-–—]?\s*([^\n\r]{2,40})",
            r"государственн\w*\s+регистрационн\w*\s+номер\s*[:\-–—]?\s*([^\n\r]{2,40})",
        ):
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                candidate = clean_value(m.group(1), MISSING, GARBAGE)
                if candidate not in (MISSING, GARBAGE):
                    return candidate, 0.4

    return MISSING, 0.0


# ---------------------------------------------------------------------------


_RECEPTION_STOP = re.compile(
    r"\n\s*(?:"
    r"\d{1,2}\s*[.)]\s*(?:выдача\s+груз|переадресовк|отметк|стоимость|прочие\s+условия)"
    r"|сдач\w*\s+груз"
    r"|выдач\w*\s+груз"
    r"|доставк\w*\s+груз"
    r"|переадресовк"
    r"|отметк\w+\s+грузоотпр"
    r")",
    re.IGNORECASE,
)

# Строка считается мусорной, если <30% символов — буквы/цифры, и в ней
# много «технических» символов (скобки/слэши/подчёркивания, длинные серии
# одиночных пробелов).
_RECEPTION_NOISE_CHARS = set("[](){}|\\/=_~^`<>*#$%")


def _looks_noisy(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Строки ≤ 3 символов — одиночные литеры/слоги вроде «Г», «а.», «ГЕР».
    if len(s) <= 3:
        return True
    # Украинские буквы/диакритика/встроенные апострофы по токенам.
    if is_noise_line(s):
        return True
    # Нет ни одного «осмысленного» слова (≥ 4 букв подряд), ни даты, ни
    # длинного числа (телефон/ИНН/индекс)? Значит строка — OCR-мусор вроде
    # «11 [3] Т» или «000 д».
    if (not re.search(r"[А-Яа-яЁёA-Za-z]{4,}", s)
            and not re.search(r"\d{2}\.\d{2}\.\d{4}", s)
            and not re.search(r"\d{5,}", s)):
        return True
    alnum = sum(1 for c in s if c.isalnum())
    if alnum == 0:
        return True
    if len(s) >= 4 and alnum / len(s) < 0.3:
        return True
    noise = sum(1 for c in s if c in _RECEPTION_NOISE_CHARS)
    if noise >= 3 and noise / max(len(s), 1) > 0.3:
        return True
    # Подряд идущие короткие «токены» из 1-2 символов — это почти всегда
    # OCR-мусор на границе ячеек таблицы.
    tokens = s.split()
    if len(tokens) >= 4:
        short = sum(1 for t in tokens if len(t) <= 2)
        if short / len(tokens) > 0.6:
            return True
    return False


def _reception_collect(lines: list[str]) -> str | None:
    """Собирает первую содержательную строку и ещё до 3 следующих,
    пока не появится ИНН / 10–12 цифр подряд (OCR мог потерять «ИНН»).
    """
    if not lines:
        return None
    start = 0
    for i, ln in enumerate(lines):
        if _ORG_PREFIX_RE.search(ln):
            start = i
            break
    collected: list[str] = []
    for ln in lines[start: start + 4]:
        collected.append(ln)
        if _INN_INCLUSIVE_RE.search(ln) or re.search(r"\b\d{10,12}\b", ln):
            break
    return ", ".join(collected)


def _reception_trim(target: str) -> str:
    target = _trim_to_org(target)
    # OCR-огрызок «, Ин,» / «, ИН,» перед 10–12 цифр → восстанавливаем «ИНН ».
    target = re.sub(
        r",\s*[ИИ]н{1,2}\.?\s*,?\s*(?=\d{10,12}\b)",
        ", ИНН ", target, flags=re.IGNORECASE,
    )
    cut_inn = _cut_at_inn_inclusive(target)
    if not cut_inn:
        m = re.search(r"\b\d{10,12}\b", target)
        if m:
            cut_inn = target[: m.end()].strip(" ,;")
    target = cut_inn if cut_inn else _cut_before_financial(target)
    # Огрызки «, Ин» / «, И» / «, Н» в самом конце.
    target = re.sub(r",\s*[А-Яа-яЁёA-Za-z]{1,3}\.?\s*$", "", target)
    return target[:500].strip(" ,;")


def extract_reception(section_body: str, full_text: str) -> tuple[str, float]:
    """Приём груза: только ПЕРВАЯ содержательная строка.

    От ORG-префикса до «ИНН \\d+» включительно; КПП/ОГРН/ОКПО обрезаются.
    Если ни в одной строке нет ORG-префикса — берём первую как есть.

    Confidence:
        1.0 — секция найдена + результат содержит валидный ИНН
              (контрольная сумма ФНС прошла) — структурно бесспорный
              сигнал, что мы извлекли реальную организацию-приёмщика.
        0.9 — секция найдена, ИНН не валидируется.
        0.5 — fallback по «приём груза» в полном тексте.
    """
    if section_body:
        body = _RECEPTION_STOP.split(section_body, maxsplit=1)[0]
        raw = [ln for ln in body.splitlines() if not _looks_noisy(ln)]
        cleaned = [strip_garbage_tokens(ln.strip()) for ln in raw]
        cleaned = [
            ln for ln in cleaned
            if ln and len(ln) > 3 and re.search(
                r"[А-Яа-яЁёA-Za-z]{4,}|\d{5,}|\d{2}\.\d{2}\.\d{4}", ln
            )
        ]
        target = _reception_collect(cleaned)
        if target:
            target = _reception_trim(target)
            if target and not is_garbage(target):
                return target, (1.0 if _has_validated_inn(target) else 0.9)

    if full_text:
        m = re.search(r"при[ёе]м\s+груз\w*", full_text, re.IGNORECASE)
        if m:
            chunk = full_text[m.end(): m.end() + 1500]
            chunk = _RECEPTION_STOP.split(chunk, maxsplit=1)[0]
            lines = [ln.strip() for ln in chunk.splitlines() if not _looks_noisy(ln)]
            lines = [ln for ln in lines if ln]
            target = _reception_collect(lines)
            if target:
                target = _reception_trim(target)
                if target and not is_garbage(target):
                    return target, 0.5

    return MISSING, 0.0


# ---------------------------------------------------------------------------


def extract_all(sections: dict[str, str], full_text: str) -> dict[str, tuple[str, float]]:
    """Единая точка: прогоняет все экстракторы и возвращает словарь."""
    head = sections.get("head", "")
    cargo_section = sections.get("cargo", "")

    number, c_num, date, c_date = extract_number_and_date(head, full_text)

    shipper, c_shipper = extract_shipper(sections.get("shipper", ""), full_text)
    consignee, c_consignee = extract_consignee(sections.get("consignee", ""), full_text)
    driver, c_driver = extract_driver(sections.get("carrier", ""), full_text)
    cargo, c_cargo = extract_cargo(cargo_section, full_text)
    volume, c_volume = extract_volume(cargo_section, full_text)
    vehicle, c_vehicle = extract_vehicle(sections.get("vehicle", ""), full_text)
    reception, c_reception = extract_reception(sections.get("reception", ""), full_text)

    return {
        "number": (number, c_num),
        "date": (date, c_date),
        "shipper": (shipper, c_shipper),
        "consignee": (consignee, c_consignee),
        "cargo": (cargo, c_cargo),
        "volume": (volume, c_volume),
        "driver": (driver, c_driver),
        "vehicle": (vehicle, c_vehicle),
        "reception": (reception, c_reception),
    }
