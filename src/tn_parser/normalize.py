"""Нормализация текста перед парсингом.

Операции:
    1. Убираем невидимые символы (soft hyphen, zero-width, BOM).
    2. Склеиваем слова, разорванные переносом: "при-\\nём" → "приём".
    3. Схлопываем подряд идущие горизонтальные пробелы (но не переводы
       строк — они нужны сегментатору).
    4. Склеиваем пустые строки (2+ \\n в 1).
    5. Чиним латинские confusables внутри кириллических слов (A→А, P→Р и т. п.).

ВАЖНО: мы НЕ применяем `unicodedata.normalize("NFKC", ...)` — NFKC превращает
`№` в две буквы `No`, а также ломает другие спец-символы ТН. Для нормализации
confusables этого и не нужно.
"""

from __future__ import annotations

import re

# Невидимые символы, встречающиеся в машиночитаемых PDF.
_INVISIBLE = {
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM
    "\u202a",  # LRE
    "\u202c",  # PDF
}

# Все юникодные тире/дефисы, которые встречаются в ТН.
DASHES = "\u002d\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

# Перенос слова через дефис на новой строке.
_HYPHEN_BREAK = re.compile(
    rf"([А-Яа-яЁёA-Za-z])[{DASHES}]\s*\n\s*([А-Яа-яЁёA-Za-z])"
)

# Пробелы/табы подряд (не \n).
_HORIZ_WS = re.compile(r"[ \t\u00a0\u2009\u202f]+")

# 2+ \n → один \n.
_MULTI_NL = re.compile(r"\n{2,}")

# Содержательный символ: кириллица, латиница или цифра. Латиницу считаем
# валидной, т.к. в ТН встречаются брэнды ТС (RENAULT, VOLVO, MAN), артикулы
# товаров (B30F300) и т. п.
_MEANINGFUL = re.compile(r"[А-Яа-яЁёA-Za-z0-9]")

# Латиница, визуально совпадающая с кириллицей (регистрозависимо).
_LAT_TO_CYR = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у",
}
_MIXED_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def _is_cyr(c: str) -> bool:
    return bool(c) and ("\u0400" <= c <= "\u04FF")


def _strip_invisible(text: str) -> str:
    for ch in _INVISIBLE:
        if ch in text:
            text = text.replace(ch, "")
    return text


def _fix_confusables(text: str) -> str:
    """Замена латинских двойников на кириллицу только при кириллических
    соседях (см. test_mixed_script_product_name_preserved)."""

    def replace(m: re.Match[str]) -> str:
        chars = list(m.group(0))
        for i, c in enumerate(chars):
            if c in _LAT_TO_CYR:
                left = chars[i - 1] if i > 0 else ""
                right = chars[i + 1] if i + 1 < len(chars) else ""
                if _is_cyr(left) or _is_cyr(right):
                    chars[i] = _LAT_TO_CYR[c]
        return "".join(chars)

    return _MIXED_WORD.sub(replace, text)


def normalize_for_sections(text: str) -> str:
    """Мягкая нормализация с сохранением переводов строк.

    Шаги:
        1. Strip невидимых символов (soft hyphen, zero-width, BOM).
        2. CRLF/CR → LF.
        3. Склеить переносы «сло-\\nво» → «слово».
        4. Схлопнуть подряд идущие горизонтальные пробелы.
        5. 2+ \\n → один \\n.
        6. Confusables: латинские двойники → кириллица в кириллических
           словах.
        7. Lexicon correction (декабрь 2026): замена OCR-mangled
           вариантов критичных ТН/УПД терминов на canonical-формы
           («Грузаатправитель» → «Грузоотправитель», etc.). Это
           даёт парсеру canonical-headers по которым matchятся
           section-регексы; без lex-коррекции section detection
           работал бы только через form-comment anchors (более
           хрупкие).
    """
    if not text:
        return ""
    text = _strip_invisible(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _HORIZ_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n", text)
    text = _fix_confusables(text)
    # Lex-коррекция — импортируем лениво, чтобы src.tn_parser остался
    # независимым от src.core при отсутствии последнего (минимальные
    # инсталляции парсерного CLI без OCR-стека).
    try:
        from src.core.lexicon_corrector import correct as _lex_correct
        text = _lex_correct(text)
    except ImportError:  # pragma: no cover — src.core отсутствует в parser-only билдах
        pass
    # Широкий fuzzy-корректор (по ~17k словоформ русского языка).
    # Тоже ленивый; не требует pymorphy3 в runtime (словарь
    # сгенерирован build-time в resources/ru_lexicon.txt).
    try:
        from src.core.fuzzy_corrector import correct as _fuzzy_correct
        text = _fuzzy_correct(text)
    except ImportError:  # pragma: no cover
        pass
    return text.strip()


def collapse(text: str) -> str:
    """Жёсткая нормализация в одну строку."""
    text = normalize_for_sections(text)
    return text.replace("\n", " ").strip()


def is_garbage(s: str) -> bool:
    return not _MEANINGFUL.search(s or "")


# --- Мусорные OCR-токены --------------------------------------------------

# Уникальные украинские буквы — в русскоязычных ТН не встречаются. Когда
# OCR их выдаёт, это всегда путаница с «и/й/е/г».
_UKRAINIAN_LETTERS = re.compile(r"[іїєґІЇЄҐ]")

# Латиница с диакритикой — для ТН такая же аномалия, как и украинские буквы.
_EXOTIC_DIACRITIC = re.compile(
    r"[àáâãäåèéêëìíîïòóôõöùúûüýÿñç"
    r"ÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ]"
)

# Апостроф/кавычка/обратный апостроф ВНУТРИ слова между буквами —
# типичная OCR-артефакт на плохом скане: «Бекам'тбд», «'Г'чп», «ав4'і».
_EMBEDDED_QUOTE = re.compile(
    r"[А-ЯЁа-яёA-Za-z0-9]"
    r"[\"'`\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F]+"
    r"[А-ЯЁа-яёA-Za-z]"
)

# Токен, заканчивающийся апострофом/кавычкой после буквы/цифры —
# «ав4'», «помощ'», «да'» — фрагменты OCR-разорванных слов.
_TRAILING_QUOTE = re.compile(
    r"[А-ЯЁа-яёA-Za-z0-9]+"
    r"[\"'`\u2018\u2019\u201A\u201B\u201C\u201D]+$"
)

# Открывающие/закрывающие кавычки — для исключения парно-окружённых
# токенов (« "Моспроект-3" ») из правила «цифра + хвостовая кавычка».
# Без этого легитимные имена в двойных кавычках считаются мусорными.
_OPEN_QUOTES = "\"'`\u2018\u201A\u201B\u201C\u201E\u201F«„"
_CLOSE_QUOTES = "\"'`\u2019\u201D«»"


def _script_of(ch: str) -> str:
    """'c' — кириллица (включая ё), 'l' — латиница, '' — другое."""
    if not ch:
        return ""
    if "А" <= ch <= "я" or ch in "Ёё":
        return "c"
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return "l"
    return ""


def _script_runs(word: str) -> int:
    """Число блоков одного скрипта (подряд c/l) в слове."""
    prev, runs = "", 0
    for c in word:
        s = _script_of(c)
        if s and s != prev:
            runs += 1
        if s:
            prev = s
    return runs


def is_ocr_garbage_token(token: str) -> bool:
    """Токен — OCR-мусор?

    Примеры мусора: 'іі-і', 'Бекам\\'тбд', 'ав4\\'', 'Ц:і', 'іЁт', 'РРДВО'.
    Легитимные токены: 'Тенsar', 'RENAULT', 'ООО', 'B30F300', '7743553262'.
    """
    if not token:
        return True
    # Проверки на RAW-токене ДО стрипинга: встроенные кавычки и «цифра+кавычка»
    # в конце. Стрипать нельзя, иначе потеряем сигнал.
    if _EMBEDDED_QUOTE.search(token):
        return True
    # «цифра + кавычка в конце» — артефакт вида «ав4'», «помощ'». Но
    # если токен ОКРУЖЁН парными кавычками (« "Моспроект-3" » —
    # легитимное имя в кавычках), пропускаем — это не OCR-мусор.
    if (
        re.search(r"[0-9][\"'`\u2018\u2019]+$", token)
        and not (token[:1] in _OPEN_QUOTES and token[-1:] in _CLOSE_QUOTES)
    ):
        return True
    # Стрипаем обрамление.
    core = token.strip(
        " \t\r\n.,;:!?\"'«»()[]{}|\\/=_~^`<>*#$%+"
        "\u2010\u2011\u2012\u2013\u2014\u2015"
        "\u2018\u2019\u201C\u201D"
    )
    if not core:
        # Токен из одной пунктуации. Одиночный разделитель или повтор
        # одного знака — не мусор: «—», «-», «...», «———».
        # Разнородная пунктуация на ≥ 2 символа — мусор: «'|_'», «'*!"'"».
        return not (len(token) <= 1 or len(set(token)) <= 1)
    # Украинская буква в русском документе = OCR-ошибка.
    if _UKRAINIAN_LETTERS.search(core):
        return True
    # Экзотическая диакритика (é, ï, ñ…) — в ТН не бывает.
    if _EXOTIC_DIACRITIC.search(core):
        return True
    # Апостроф внутри слова — «Бекам'тбд», «'Г'чп».
    if _EMBEDDED_QUOTE.search(core):
        return True
    if _TRAILING_QUOTE.search(core):
        return True
    # Слишком много переключений скриптов в одном слове: «іЁт» (после
    # того как confusables уже починили отдельные буквы). Допускаем до
    # 2 блоков — это покрывает легитимные «Тенsar» (кирилл + латин).
    return _script_runs(core) >= 3


def is_noise_line(line: str, min_garbage_ratio: float = 0.5) -> bool:
    """Строка считается шумом, если ≥ половины токенов — OCR-мусор.

    Параметр `min_garbage_ratio` — доля токенов, при которой линия
    отбраковывается полностью. 0.5 — консервативное значение.
    """
    s = (line or "").strip()
    if not s:
        return True
    tokens = re.split(r"\s+", s)
    if not tokens:
        return True
    garbage = sum(1 for t in tokens if is_ocr_garbage_token(t))
    return (garbage / len(tokens)) >= min_garbage_ratio


def strip_garbage_tokens(s: str) -> str:
    """Удаляет из строки OCR-мусорные токены, сохраняя порядок."""
    if not s:
        return s
    # Делим по whitespace, сохраняя разделители отдельно.
    parts = re.split(r"(\s+)", s)
    out = []
    for p in parts:
        if p.strip() and is_ocr_garbage_token(p.strip()):
            continue
        out.append(p)
    result = "".join(out)
    # Схлопываем двойные пробелы, которые могли образоваться после удаления.
    return re.sub(r"\s{2,}", " ", result).strip()


def clean_value(value: str, missing: str, garbage: str) -> str:
    if not value:
        return missing
    v = value.strip(" \t\r\n:;,.-–—|")
    if not v:
        return missing
    if is_garbage(v):
        return garbage
    return v
