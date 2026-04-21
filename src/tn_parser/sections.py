"""Разбиение текста ТН на разделы по их семантической роли.

Формы ТН бывают разных редакций: нумерация разделов гуляет (например, в
форме из ПП №2200 «Перевозчик» — раздел 10, а в старой форме — раздел 6).
Поэтому вместо ключей-номеров мы возвращаем `Dict[role, content]`.

Ключи:
    "head"      — всё до первого опознанного раздела
    "shipper"   — грузоотправитель
    "consignee" — грузополучатель
    "cargo"     — груз
    "carrier"   — перевозчик
    "vehicle"   — транспортное средство
    "reception" — приём груза
"""

from __future__ import annotations

import re

try:
    from rapidfuzz import fuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False


# Канонические начала заголовков. Более длинные/специфичные — первыми,
# чтобы «Приём груза» не перекрывалось «Груз».
# Для shipper/consignee — несколько синонимов: в ТН «Грузоотправитель» /
# «Грузополучатель», в счёте-фактуре / УПД — «Продавец» / «Покупатель»,
# иногда «Поставщик».
# «Погрузка» даёт секцию reception в пост-OCR структурированном формате —
# там «Владелец инфраструктуры: ООО X, ИНН N» совпадает с infrastructure_owner
# из экспертной разметки. «Лицо, принимающее груз» — отдельный блок,
# игнорируется (в inputs это обычно не сторона с ИНН, а ФИО кладовщиков).
_ROLE_TITLES: list[tuple[str, tuple[str, ...]]] = [
    ("reception", (
        "прием груза", "приём груза", "погрузка груза", "погрузка",
    )),
    ("consignee", ("грузополучатель", "покупатель")),
    ("shipper", ("грузоотправитель", "продавец", "поставщик")),
    ("vehicle", ("транспортное средство",)),
    ("carrier", ("перевозчик",)),
    ("cargo", ("груз",)),
]

# Заголовки, которые мы узнаём как стоп-маркеры, но контент не забираем.
# В пост-OCR структурированном формате добавлены «— СТОРОНЫ —»,
# «— ОСНОВАНИЕ —», «— РАЗГРУЗКА —», «— ПОДПИСИ И ПЕЧАТИ —»,
# «— ПРИМЕЧАНИЯ О СКАНЕ —», «заказчик перевозки», «разгрузка»,
# «подписи и печати», «примечания о скане» — их контент не надо
# выгружать в поля, но они служат границами между секциями.
_IGNORED_TITLES: tuple[str, ...] = (
    "сопроводительные документы",
    "указания грузоотправителя",
    "условия перевозки",
    "информация о принятии",
    "оговорки и замечания",
    "прочие условия",
    "переадресовк",
    "стоимость услуг",
    "стоимость перевозки",
    "дата составления",
    "отметки",
    "выдача груза",
    "сдача груза",
    "стороны",
    "основание",
    "разгрузка",
    "подписи и печати",
    "примечания о скане",
    "заказчик перевозки",
    "доверенность",
    "объект строительства",
    "лицо, принимающее груз",
    "лицо принимающее груз",
)


# Заголовок с номером: «1.», «2)», «6 .», нестрогий разделитель.
# После разделителя — первое «букво-содержательное» значение; допускаем
# латиницу (OCR часто превращает кириллицу в её латинский визуальный
# дубликат: «3.Tpys» вместо «3. Груз»).
_NUMBERED = re.compile(
    r"(?m)^\s*(\d{1,2})\s*[.)\u00a0]\s*([А-ЯЁа-яёA-Za-z][^\n]{0,80})"
)

# OCR часто теряет точку после номера графы: «6 Перевозчик» вместо «6. Перевозчик».
# Эту форму принимаем только для ИЗВЕСТНЫХ ролевых названий — иначе любая
# строка «N слово» (например «1 Наименование» внутри секции «Груз») была бы
# ошибочно интерпретирована как заголовок новой графы.
_KNOWN_TITLES = [name for _, names in _ROLE_TITLES for name in names] + list(_IGNORED_TITLES)
_NUMBERED_NODOT = re.compile(
    r"(?mi)^\s*(\d{1,2})\s+("
    + "|".join(re.escape(name) for name in _KNOWN_TITLES)
    + r")\b[^\n]{0,80}$"
)

# Заголовок без номера — отдельной строкой.
_BARE = re.compile(
    r"(?mi)^\s*("
    + "|".join(re.escape(name) for name in _KNOWN_TITLES)
    + r")\b[^\n]{0,80}$"
)

# Заголовок в квадратных скобках: «[Грузоотправитель]», «[Перевозчик]»,
# «[Лицо, принимающее груз]». Используется в пост-OCR структурированном
# формате (inputs/TN_k_UPD_*.txt).
_BRACKETED = re.compile(
    r"(?mi)^\s*\[\s*("
    + "|".join(re.escape(name) for name in _KNOWN_TITLES)
    + r")[^\]\n]{0,80}\]\s*$"
)

# Заголовок в em-dash обрамлении: «— ГРУЗ —», «— ТРАНСПОРТНОЕ СРЕДСТВО —»,
# «— ПОГРУЗКА —», «— СТОРОНЫ —». Дефисы могут быть разных юникодных форм
# (\u2014 «—», \u2013 «–», ASCII «-»). Используется в пост-OCR формате.
_EMDASH_WRAPPED = re.compile(
    r"(?mi)^\s*[\u2014\u2013\u2015\-]{1,2}\s*("
    + "|".join(re.escape(name) for name in _KNOWN_TITLES)
    + r")\b[^\n]{0,80}?[\u2014\u2013\u2015\-]{1,2}\s*$"
)


# Эталоны для нечёткого сопоставления (≥ 6 букв — короче брать опасно,
# слишком много ложных срабатываний).
_FUZZY_TITLES: list[tuple[str, tuple[str, ...]]] = [
    ("reception", ("приём груза", "прием груза", "погрузка груза")),
    ("consignee", ("грузополучатель", "покупатель")),
    ("shipper", ("грузоотправитель", "продавец", "поставщик")),
    ("vehicle", ("транспортное средство",)),
    ("carrier", ("перевозчик",)),
]

_FUZZY_THRESHOLD = 82      # минимальный score для принятия
_FUZZY_MARGIN = 6          # разрыв между лучшим и вторым кандидатом

# Form-comment anchors. На бланках ТН после каждого header'а идёт
# explanatory text в скобках стандартного образца:
#
#   1. Грузоотправитель
#   ООО "..."
#   (реквизиты, позволяющие идентифицировать Грузоотправителя)
#   2. Грузополучатель
#   АО "..."
#   (реквизиты, позволяющие идентифицировать Грузополучателя)
#
# Когда OCR съедает сам header (Tesseract промахивается на жирном
# шрифте small-caps), эти comment-anchors часто выживают — они
# набраны мелким светлым шрифтом, OCR с ним справляется лучше.
# Используем закрывающую часть «идентифицировать <Роль>» как
# DETECTION-сигнал того, что мы только что прошли границу секции.
#
# Декабрь 2026: на UPD_36 OCR превратил «2. Грузополучатель» в
# мусор, но «(реквизиты, позволяющие идентифицировать
# Гручополучателя)» сохранился — без этого anchor'а consignee
# извлечь нельзя.
# Form-comment anchors по STEM'ам с толерантностью к OCR-mangling.
# OCR на бланках ТН систематически портит длинные слова: «(реквизиты»
# → «Зреквизить», «идентифицировать» → «идентийнииривать»,
# «Грузополучателя» → «Гручополучателя». Поэтому здесь даём широкие
# regex'ы по STEM'у роли — корень обычно сохраняется даже когда
# суффиксы исковерканы.
#
# Каждый anchor — пара (role, stem-regex). Anchor сработал =
# секция <role> существует где-то рядом, даже если её header'а
# в тексте нет. Применяется как fallback ТОЛЬКО когда обычные
# header-based маркеры не нашли соответствующую роль.
# OCR systematically mangles 'з' → 'ч'/'3'/'c' / 'е' → 'с' / 'о' → '0' / 'а' → 'а'.
# Stem-классы для двух «грузо*» ролей принимают наиболее частые
# OCR-замены первого характерного chars'а после «гру»:
#   3 — Latin/Cyr digit-like
#   ч/щ/Ц — частая mis-recognition бoldface 'з'
#   c — Latin c (mistaken for 'с')
_GRUZO_CHARS = "[з3cчщЦzc]"
_FORM_COMMENT_ANCHORS: list[tuple[str, str]] = [
    # «грузоотправ» / «гручоотправ» / «гру3оотправ» — толерант к 'з'/'ч'/'3'.
    ("shipper", rf"гру{_GRUZO_CHARS}[ао]?[оа]?тправ"),
    # «грузополуч» / «гручополуч» / «гру3опалуч».
    ("consignee", rf"гру{_GRUZO_CHARS}[аоуeс]?п[оа]?луч"),
    # «перевозчик» / «перевозщик» / «перевозчнк» / «псрсвозчик» —
    # ключ здесь СУФФИКС «-чик/-щик/-чнк/-щнк», чтобы не матчить
    # «перевозки» (genitive of «перевозка» = транспортная операция,
    # не role-слово). OCR редко ломает суффикс этой длины.
    ("carrier", r"п[еc][рп]?[еc]во[зc][чщ][ин][кчн]"),
    # «принимающее груз» / «приним...груз»
    ("reception", rf"приним\w{{0,4}}\s+гру{_GRUZO_CHARS}"),
]


def _find_form_comment_anchors(text: str) -> list[tuple[int, str]]:
    """Ищет stem-токены ролей внутри explanatory-комментариев формы.

    Стратегия: для каждой роли ищем stem-regex anywhere в тексте,
    но требуем чтобы СРАЗУ ПЕРЕД anchor'ом (в ±60 чар) был
    typical form-comment маркер — «идентифицировать», «позволяющие»,
    «(реквизиты». Это сигнал что мы попали в explanatory text
    стандартной формы ТН (где после комментария идёт следующая
    секция), а не в произвольную прозу с упоминанием роли.

    Без узкого контекста ловятся false-positives: «перевоз» в
    form-header «Правил перевозок грузов автомобильным транспортом»,
    «Грузоотправитель» внутри длинных explanatory комментариев
    о маршруте перевозки и т.п.

    Результат: позиция начала строки ПЕРЕД anchor'ом — там лежит
    содержимое искомой секции (ORG-line).
    """
    out: list[tuple[int, str]] = []
    # Маркер «explanatory comment стандартной формы ТН». OCR
    # систематически ломает «идентифицировать» (ф↔й, ц↔н, в→б),
    # «позволяющие» (л↔п, ю→1, ц→с). Принимаем очень разношёрстно
    # по первым 5-6 буквам + стабильный хвост «ировать/ировдть/
    # иривать»). Маркер должен быть в 80-char окне ПЕРЕД stem'ом.
    context_re = re.compile(
        r"(?:идент[ийнсв][фйнс][ниио][нцт]?\w*"
        r"|[пи]озволя\w*|[зс]воляющ\w*|зволяют|"
        r"реквизит\w*|Зреквизит)",
        re.IGNORECASE,
    )
    for role, stem_pattern in _FORM_COMMENT_ANCHORS:
        for m in re.finditer(stem_pattern, text, re.IGNORECASE):
            anchor_pos = m.start()
            # Узкое окно — 60 чар СЛЕВА от anchor'а. Маркер должен
            # быть прямо перед stem'ом, как в «идентифицировать
            # Грузополучателя» — не «идентифицировать (любая прочая
            # проза 200 chars) Грузополучателя».
            window_start = max(0, anchor_pos - 60)
            left_window = text[window_start:anchor_pos]
            if not context_re.search(left_window):
                continue
            # Содержимое секции расположено ПЕРЕД anchor'ом и его
            # explanatory comment'ом: backwalk до начала последней
            # значимой строки ВЫШЕ открытой скобки.
            paren_start = text.rfind("(", window_start, anchor_pos)
            scan_until = paren_start if paren_start >= 0 else anchor_pos
            backwalk_pos = max(0, scan_until - 250)
            line_starts = [
                i + 1 for i in range(backwalk_pos, scan_until)
                if text[i] == "\n"
            ]
            section_start = line_starts[0] if line_starts else backwalk_pos
            out.append((section_start, role))
    return out

_HEAD_WORD_SPLIT_RE = re.compile(r"[\s:.,–—\-]+")


def _fuzzy_classify(low: str) -> str | None:
    """Нечёткая классификация через rapidfuzz.

    Берём первое слово заголовка (head_word) и сравниваем его с эталонами:
        * fuzz.ratio — для сопоставления одиночного слова;
        * fuzz.partial_ratio — для составных («транспортное средство»);
    Чтобы не путать shipper/consignee с близким префиксом «грузо…»,
    требуем разрыв (_FUZZY_MARGIN) между лучшим и вторым кандидатом.
    """
    if not _HAS_RAPIDFUZZ:
        return None
    head_word = _HEAD_WORD_SPLIT_RE.split(low, maxsplit=1)[0]
    if len(head_word) < 5:
        return None
    candidate_full = low[:40]
    scores: list[tuple[str, int]] = []
    for role, names in _FUZZY_TITLES:
        best = 0
        for name in names:
            s1 = fuzz.ratio(head_word, name.split()[0])
            s2 = fuzz.partial_ratio(candidate_full, name)
            best = max(best, s1, s2)
        scores.append((role, best))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores or scores[0][1] < _FUZZY_THRESHOLD:
        return None
    if len(scores) > 1 and scores[0][1] - scores[1][1] < _FUZZY_MARGIN:
        return None
    return scores[0][0]


def _classify_title(title: str) -> str | None:
    """По тексту заголовка определяет роль или "__ignored__".

    Стратегия (в порядке): __ignored__ → точный startswith по
    каноническим именам → специальная обработка «груз…» → fuzzy-match
    через rapidfuzz (порог 82, margin 6).
    """
    low = title.lower().strip()
    for ign in _IGNORED_TITLES:
        if low.startswith(ign):
            return "__ignored__"
    # Точное совпадение для длинных эталонов (быстрый путь).
    for role, names in _ROLE_TITLES:
        if role == "cargo":
            continue  # «груз» обрабатываем ниже отдельно
        for name in names:
            if low.startswith(name):
                return role
    # «груз…» — особый случай (см. блок ниже).
    if low.startswith("груз"):
        rest = low[4:]
        if not rest or rest[0] in " :.,-–—\t":
            return "cargo"
        # Дальше — однозначные OCR-варианты shipper/consignee
        # (быстрый путь до fuzzy):
        if re.match(r"[ово]{0,2}(?:получ|пелуч|палуч|опуч)", rest):
            return "consignee"
        if re.match(r"[ов]{0,2}(?:отправ|тправ|втир|стправ|тир|тпр)", rest):
            return "shipper"
        # Если ничего не подошло — пускаем fuzzy для shipper/consignee
        # (он ловит произвольные OCR-искажения, не зашитые выше).
        fz = _fuzzy_classify(low)
        if fz in ("shipper", "consignee"):
            return fz
        return None
    # Не-«груз» заголовки: пробуем fuzzy для остальных ролей.
    return _fuzzy_classify(low)


# Маппинг по номеру раздела для новой формы ТН (Приложение № 4 к ПП РФ
# № 2200 от 30.11.2021). Применяется, если _classify_title не опознал
# название графы — например, OCR превратил «Груз» в «Tpys» или
# «Перевозчик» в «П!рсвохчик».
_POSITIONAL_ROLE = {
    "3": "cargo",
    "6": "carrier",
    "7": "vehicle",
    "8": "reception",
}


def _find_markers(text: str) -> list[tuple[int, str]]:
    """Возвращает отсортированный список (offset, role|"__ignored__").

    Стратегия: собираем все возможные маркеры (нумерованные + bare), затем
    для каждой роли оставляем ПЕРВОЕ вхождение. Ignored-маркеры оставляем
    все — они нужны как границы.
    """
    candidates: list[tuple[int, str]] = []

    # 1) Нумерованные заголовки с точкой/скобкой.
    for m in _NUMBERED.finditer(text):
        role = _classify_title(m.group(2))
        if role is None:
            # OCR мог полностью исказить название, но цифра раздела
            # обычно распознаётся. Fallback по номеру — только для
            # cargo/carrier/vehicle/reception (shipper/consignee
            # занимают много места и их важно не путать).
            role = _POSITIONAL_ROLE.get(m.group(1))
        if role is not None:
            candidates.append((m.start(), role))

    # 2) Нумерованные заголовки БЕЗ точки («6 Перевозчик») — только для
    #    известных ролевых заголовков, чтобы не ловить ложные срабатывания.
    for m in _NUMBERED_NODOT.finditer(text):
        role = _classify_title(m.group(2))
        if role is not None:
            candidates.append((m.start(), role))

    # 3) Голые заголовки (на отдельной строке).
    for m in _BARE.finditer(text):
        role = _classify_title(m.group(1))
        if role is not None:
            candidates.append((m.start(), role))

    # 4) Заголовки в квадратных скобках «[Грузоотправитель]».
    for m in _BRACKETED.finditer(text):
        role = _classify_title(m.group(1))
        if role is not None:
            candidates.append((m.start(), role))

    # 5) Заголовки в em-dash обрамлении «— ГРУЗ —».
    for m in _EMDASH_WRAPPED.finditer(text):
        role = _classify_title(m.group(1))
        if role is not None:
            candidates.append((m.start(), role))

    # 6) Form-comment anchors: «(реквизиты, …идентифицировать <Роль>)»
    #    дают слабые маркеры — только для тех ролей, у которых ИНАЧЕ
    #    мы вообще не нашли границу. Это спасает шапки, полностью
    #    съеденные OCR (см. UPD_36 — «2. Грузополучатель» исчез,
    #    но explanatory comment «(реквизиты, позволяющие
    #    идентифицировать Гручополучателя)» сохранился).
    primary_roles = {role for _, role in candidates if role != "__ignored__"}
    for pos, role in _find_form_comment_anchors(text):
        if role not in primary_roles:
            candidates.append((pos, role))

    # Первое вхождение каждой роли.
    seen_roles: set[str] = set()
    markers: list[tuple[int, str]] = []
    # Сортируем сначала по позиции.
    candidates.sort(key=lambda x: x[0])
    for pos, role in candidates:
        if role == "__ignored__":
            markers.append((pos, role))
            continue
        if role in seen_roles:
            continue
        seen_roles.add(role)
        markers.append((pos, role))

    return markers


def split_sections(text: str) -> dict[str, str]:
    """Делит нормализованный текст на разделы по ролям."""
    if not text:
        return {}

    markers = _find_markers(text)
    result: dict[str, str] = {}

    if not markers:
        result["head"] = text.strip()
        return result

    head = text[: markers[0][0]].strip()
    if head:
        result["head"] = head

    for i, (pos, role) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        chunk = text[pos:end].strip()

        nl = chunk.find("\n")
        header_line = chunk if nl < 0 else chunk[:nl]
        rest = "" if nl < 0 else chunk[nl + 1 :].strip()

        # Заголовки в em-dash / квадратных скобках — «[Грузоотправитель]»,
        # «— ГРУЗ —» — не имеют inline-значения. Снимаем обрамление
        # (скобки и em-dash'и по краям), чтобы inline-split не засосал
        # хвостовой «—» или «]» как значение поля.
        stripped_header = header_line.strip(
            " \t\u2014\u2013\u2015\u2010\u2011\u2012-[]"
        )
        inline = ""
        inline_split = re.split(r"[:\-–—]", stripped_header, maxsplit=1)
        if len(inline_split) == 2 and inline_split[1].strip():
            inline = inline_split[1].strip()

        body = inline + "\n" + rest if inline and rest else inline or rest

        if role == "__ignored__":
            continue
        if role not in result:
            result[role] = body

    return result
