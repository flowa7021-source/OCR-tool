"""Валидаторы для извлечённых полей."""

from __future__ import annotations

import datetime as _dt
import re

# --- Дата -----------------------------------------------------------------

_DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")


def is_valid_date(s: str) -> bool:
    """Проверка: дата валидна И попадает в правдоподобный диапазон для ТН.

    ТН — действующие документы, не архивные: принимаем даты за последние
    ~10 лет и на ближайший год вперёд. Даты типа «30.11.2021» (дата
    Постановления Правительства РФ № 2116, часто печатается на бланке
    формы) этот чек не отличит от реальных, но совсем дикие («1999» или
    «2099») отсечёт.
    """
    m = _DATE_RE.fullmatch(s.strip()) if s else None
    if not m:
        return False
    dd, mm, yyyy = map(int, m.groups())
    current_year = _dt.date.today().year
    if not (current_year - 10 <= yyyy <= current_year + 1):
        return False
    try:
        _dt.date(yyyy, mm, dd)
    except ValueError:
        return False
    return True


# --- ГРЗ (государственный регистрационный знак) --------------------------

# 1) Легковой: А123АА777 (3 цифры региона) или А123АА77 (2 цифры).
# 2) Прицеп/такси: АА1234 77 или АА123456.
# Разрешённые буквы (ГОСТ Р 50577): АВЕКМНОРСТУХ.
_GRZ_LETTERS = "АВЕКМНОРСТУХ"

# Латинские визуальные двойники для OCR-толерантного поиска.
_LAT_TO_CYR_GRZ = {
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
    "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
    "Y": "У", "X": "Х",
}
# Класс символов, который допускает и кириллицу, и её латинских двойников.
_GRZ_CHAR = f"[{_GRZ_LETTERS}{''.join(_LAT_TO_CYR_GRZ)}]"

_GRZ_MAIN = re.compile(
    rf"^[{_GRZ_LETTERS}]\d{{3}}[{_GRZ_LETTERS}]{{2}}\s?\d{{2,3}}$"
)
_GRZ_TRAILER = re.compile(
    rf"^[{_GRZ_LETTERS}]{{2}}\d{{4}}\s?\d{{2,3}}$"
)


def _cyrillic_grz(raw: str) -> str:
    """Приводит кандидат ГРЗ к кириллице (A→А, P→Р и т. п.)."""
    return "".join(_LAT_TO_CYR_GRZ.get(c, c) for c in raw.upper())


def is_valid_grz(s: str) -> bool:
    if not s:
        return False
    candidate = _cyrillic_grz(s).replace("  ", " ").strip()
    return bool(_GRZ_MAIN.match(candidate) or _GRZ_TRAILER.match(candidate))


# Поиск кандидата ГРЗ в «сыром» тексте — допускаем и латинские двойники,
# позже нормализуем в кириллицу.
GRZ_CANDIDATE = re.compile(
    # (?!\d) prevents matching inside longer digit sequences (e.g. ИНН7743553262
    # where НН7743553 would otherwise look like a valid trailer plate).
    rf"(?:(?:{_GRZ_CHAR}\d{{3}}{_GRZ_CHAR}{{2}}\s?\d{{2,3}}|"
    rf"{_GRZ_CHAR}{{2}}\d{{4}}\s?\d{{2,3}}))(?!\d)"
)


def find_grz(text: str) -> str | None:
    """Находит ГРЗ в тексте. Возвращает кандидат в кириллице.

    Если в OCR проскочили латинские двойники (`P 814 HP 152`), приводим
    их к кириллице перед возвратом (`Р 814 НР 152`).
    """
    if not text:
        return None
    m = GRZ_CANDIDATE.search(text.upper())
    if not m:
        return None
    return _cyrillic_grz(m.group(0)).strip()


_GRZ_MAIN_PARTS = re.compile(
    rf"^([{_GRZ_LETTERS}])(\d{{3}})([{_GRZ_LETTERS}]{{2}})\s?(\d{{2,3}})$"
)
_GRZ_TRAILER_PARTS = re.compile(
    rf"^([{_GRZ_LETTERS}]{{2}})(\d{{4}})\s?(\d{{2,3}})$"
)


def format_grz(grz: str) -> str:
    """Канонический вид ГРЗ с пробелами: «Р 814 НР 152»."""
    if not grz:
        return grz
    s = grz.upper().replace(" ", "")
    m = _GRZ_MAIN_PARTS.match(s)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
    m = _GRZ_TRAILER_PARTS.match(s)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    return grz


# --- ИНН -------------------------------------------------------------------


def is_valid_inn(s: str) -> bool:
    """Проверка контрольной суммы ИНН (10 или 12 цифр)."""
    if not s or not s.isdigit():
        return False
    digits = [int(c) for c in s]
    if len(digits) == 10:
        weights = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        checksum = sum(d * w for d, w in zip(digits[:9], weights, strict=False)) % 11 % 10
        return checksum == digits[9]
    if len(digits) == 12:
        w1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        w2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        c1 = sum(d * w for d, w in zip(digits[:10], w1, strict=False)) % 11 % 10
        c2 = sum(d * w for d, w in zip(digits[:11], w2, strict=False)) % 11 % 10
        return c1 == digits[10] and c2 == digits[11]
    return False


_INN_CANDIDATE = re.compile(r"\b(\d{10}|\d{12})\b")


def find_inn(text: str) -> str | None:
    if not text:
        return None
    for m in _INN_CANDIDATE.finditer(text):
        if is_valid_inn(m.group(1)):
            return m.group(1)
    return None


# --- КПП -------------------------------------------------------------------
#
# КПП (код причины постановки на учёт) — 9 цифр. Строгой контрольной
# суммы у КПП нет (в отличие от ИНН), формат регламентирован приказом
# ФНС № ММВ-7-6/435@: 4 цифры региона/ИФНС + 2 цифры причины + 3 цифры
# порядкового номера. Валидация сводится к "9 цифр в правильной
# позиции после слова КПП".
_KPP_RE = re.compile(r"\bКПП[:\s]*(\d{9})\b", re.IGNORECASE)


def find_kpp(text: str) -> str | None:
    """Первый 9-значный КПП в тексте, следующий за префиксом ``КПП``.

    Возвращает ``None``, если префикс отсутствует или цифр меньше 9.
    Специально НЕ ищем голые 9-значные числа — это могли бы быть
    номера телефонов / индексы / остатки ИНН, что дало бы ложные
    срабатывания.
    """
    if not text:
        return None
    m = _KPP_RE.search(text)
    return m.group(1) if m else None


# --- ОГРН ------------------------------------------------------------------
#
# ОГРН (основной государственный регистрационный номер) — 13 цифр для
# юрлица, 15 цифр для ИП (ОГРНИП). Контрольная сумма — остаток от
# деления на 11/13 соответственно. Проверяем обе длины.
_OGRN_CANDIDATE = re.compile(r"\bОГРН(?:ИП)?[:\s]*(\d{13}|\d{15})\b", re.IGNORECASE)


def is_valid_ogrn(s: str) -> bool:
    """Контрольная сумма ОГРН (13 цифр) или ОГРНИП (15 цифр)."""
    if not s or not s.isdigit():
        return False
    if len(s) == 13:
        # Контрольная цифра = остаток от деления первых 12 цифр (как
        # 12-значное число) на 11, младший разряд результата.
        return int(s[:12]) % 11 % 10 == int(s[12])
    if len(s) == 15:
        return int(s[:14]) % 13 % 10 == int(s[14])
    return False


def find_ogrn(text: str) -> str | None:
    """Первый валидный (по контр-сумме) ОГРН/ОГРНИП в тексте."""
    if not text:
        return None
    for m in _OGRN_CANDIDATE.finditer(text):
        if is_valid_ogrn(m.group(1)):
            return m.group(1)
    return None
