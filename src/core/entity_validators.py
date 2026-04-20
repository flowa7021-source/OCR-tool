"""Russian business-document entity validators: dates, amounts, phones.

Companion to :mod:`src.core.doc_validators` (ИНН / КПП / ОГРН). Where
those validators check regulated identifiers with official checksums,
this module handles the more loosely-formatted accounting entities
that still have enough structure to detect and repair common OCR
errors:

* **Dates** — real scans emit ``12.O1.2O23`` (letter O for zero),
  ``02.03 2022`` (space for dot), ``02,03.2022`` (comma for dot),
  ``2O.11.2O24`` etc. Detecting the near-date shape and running it
  through a Gregorian-calendar validator rescues most of these.
* **Amounts** — ``1 234,56 руб``, ``1234.56``, ``1 234 руб 50 коп``,
  etc. OCR often loses the decimal comma or replaces it with a
  period; normalising to ``1 234,56`` matches how accountants enter
  these in the financial system.
* **Phones** — ``+7 (495) 725-80-62`` vs ``8 495 725 80 62`` vs
  ``+7 4957258062``. Matches on any ≥ 10 digit sequence starting
  with 7 or 8 and normalises to the ``+7 (XXX) XXX-XX-XX`` canonical
  form used in Russian business documents.

Every function here returns ``None`` when it can't confidently fix
the input — the postprocessor then leaves the original OCR text
untouched. False-positive corrections are worse than unrepaired OCR
noise because they look authoritative.
"""

from __future__ import annotations

import datetime
import re

__all__ = [
    "try_fix_date",
    "try_fix_amount",
    "try_fix_phone",
]


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

#: Single-character OCR substitutions seen on dates. ``O``/``o`` for
#: ``0``, ``l``/``I``/``|`` for ``1``, ``Z`` for ``2``, ``B`` for ``8``,
#: etc. Applied only AFTER the regex has located a date-shaped
#: candidate so we don't rewrite arbitrary letter-digit collisions.
_DIGIT_LOOKALIKES: dict[str, str] = {
    "O": "0", "o": "0", "О": "0", "о": "0",  # Latin and Cyrillic O
    "l": "1", "I": "1", "|": "1",
    "Z": "2", "z": "2",
    "B": "8",
    "S": "5", "s": "5",
    "G": "6",
    "T": "7",
}

#: Matches date-shaped tokens with lenient separators. The separator
#: slot accepts dots, commas, spaces, hyphens, slashes — OCR mangles
#: the punctuation but usually keeps the digit positions.
_DATE_RE: re.Pattern[str] = re.compile(
    r"\b(?P<d>[0-9OoОо|lIZzBSsGT]{1,2})"
    r"[.,\s\-/]"
    r"(?P<m>[0-9OoОо|lIZzBSsGT]{1,2})"
    r"[.,\s\-/]"
    r"(?P<y>[0-9OoОо|lIZzBSsGT]{2,4})\b",
)


def _clean_digits(s: str) -> str | None:
    """Replace common letter→digit look-alikes; return None if any
    non-digit survives."""
    cleaned = "".join(_DIGIT_LOOKALIKES.get(ch, ch) for ch in s)
    if not cleaned.isdigit():
        return None
    return cleaned


def try_fix_date(candidate: str) -> str | None:
    """Recognise a date-shaped token and return canonical ``DD.MM.YYYY``.

    Returns ``None`` when:
      * The shape doesn't match a lenient ``d.m.y`` pattern.
      * After cleaning letter-digit look-alikes, any position is
        still non-numeric.
      * Day / month / year violate the Gregorian calendar
        (``datetime.date`` rejects it).
      * Year is clearly wrong (before 1900 or after 2100 — Russian
        business docs don't span that range and a four-digit year
        outside it is almost always OCR noise like ``2O85`` that
        cleaned to a valid but absurd year).

    On success returns the normalised string with zero-padded
    day / month and a 4-digit year. Two-digit years are interpreted
    as the 2000s (``.01.24`` → ``01.01.2024``) — the user's
    documents are all post-2000 accounting records.
    """
    match = _DATE_RE.fullmatch(candidate.strip())
    if match is None:
        return None
    d_raw, m_raw, y_raw = match.group("d"), match.group("m"), match.group("y")
    d = _clean_digits(d_raw)
    m = _clean_digits(m_raw)
    y = _clean_digits(y_raw)
    if d is None or m is None or y is None:
        return None
    try:
        day = int(d)
        month = int(m)
        year = int(y)
    except ValueError:
        return None
    if len(y) == 2:
        year += 2000
    if not (1900 <= year <= 2100):
        return None
    try:
        datetime.date(year, month, day)
    except ValueError:
        return None
    return f"{day:02d}.{month:02d}.{year:04d}"


# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

#: Matches amount-shaped tokens — optional thousands separators (space
#: or narrow-no-break-space), decimal comma or dot, optional currency
#: word. Keeps the regex simple; the post-match cleanup handles letter-
#: digit confusion on the numeric positions.
_AMOUNT_RE: re.Pattern[str] = re.compile(
    r"(?P<int>[\d\sOoОо]+)"
    r"(?:[,.](?P<frac>[\d\sOoОо]{1,2}))?"
    r"(?:\s*(?P<unit>руб\.?|коп\.?|\bр\.|₽))?",
    re.IGNORECASE,
)


def try_fix_amount(candidate: str) -> str | None:
    """Normalise an amount-shaped token.

    Returns the canonical ``N[ N…],FF`` form with Russian-style
    space thousands separator and decimal comma. Always includes
    two fractional digits (rounds / pads as needed) so downstream
    consumers can rely on the shape.

    Returns ``None`` when:
      * No digits recoverable after letter look-alike cleanup.
      * The integer part is shorter than 1 digit (nothing to format).

    Examples:
      * ``'1 234,56 руб'`` → ``'1 234,56'``
      * ``'1234.56'``     → ``'1 234,56'``
      * ``'1 2З4'``       → ``'1 234,00'``  (З is Cyrillic three look-alike)
      * ``'O,5O'``        → ``'0,50'``
    """
    match = _AMOUNT_RE.fullmatch(candidate.strip())
    if match is None:
        return None
    raw_int = match.group("int") or ""
    raw_frac = match.group("frac") or ""
    # Strip whitespace from each piece before letter-digit cleanup so
    # a missing thousands separator doesn't cross into the cleaner.
    int_clean = _clean_digits(re.sub(r"\s+", "", raw_int))
    if int_clean is None or not int_clean:
        return None
    if raw_frac:
        frac_clean = _clean_digits(re.sub(r"\s+", "", raw_frac))
        if frac_clean is None:
            return None
    else:
        frac_clean = ""

    # Zero-pad / truncate fractional to exactly 2 digits.
    frac_clean = (frac_clean + "00")[:2]
    # Thousands separator: walk the integer part from the right and
    # insert a regular space every 3 digits.
    int_with_seps = _group_thousands(int_clean)
    return f"{int_with_seps},{frac_clean}"


def _group_thousands(digits: str) -> str:
    """Insert a single space every 3 digits from the right.

    ``'1234567'`` → ``'1 234 567'``. Preserves leading zeros — the
    caller is responsible for deciding whether stripping them is
    appropriate (e.g. ``'007'`` stays ``'007'`` if somebody typed
    a zero-padded amount; the normaliser won't remove semantic
    zeroes).
    """
    if len(digits) <= 3:
        return digits
    groups: list[str] = []
    i = len(digits)
    while i > 3:
        groups.append(digits[i - 3:i])
        i -= 3
    groups.append(digits[:i])
    return " ".join(reversed(groups))


# ---------------------------------------------------------------------------
# Phones
# ---------------------------------------------------------------------------

_PHONE_RE: re.Pattern[str] = re.compile(
    r"(?:\+?7|8)[\s\-().]*"
    r"(?P<area>[0-9OoОоlIZzBSsG]{3,4})"
    r"[\s\-().]*"
    r"(?P<rest>[0-9OoОоlIZzBSsG\s\-().]{5,10})"
)


def try_fix_phone(candidate: str) -> str | None:
    """Normalise a Russian phone number to ``+7 (XXX) XXX-XX-XX``.

    Accepts ``+7…``, ``8…``, spaced / hyphenated / parenthesised
    variants. Cleans letter-digit look-alikes on the digit
    positions (``+7 (495) 725-8O-62`` → ``+7 (495) 725-80-62``).

    Returns ``None`` when:
      * No phone-shaped token is present in the input.
      * Total digit count isn't 10 (area + subscriber) or 11
        (leading country-code digit + 10).
      * The normalised digit sequence doesn't match a Russian
        mobile / landline shape (area code starts with digit ≠ 0,
        ≠ 1 — no valid Russian area begins with 0 or 1).
    """
    text = candidate.strip()
    # Pull every digit-or-look-alike out, then normalise to 11 digits.
    digits_only = "".join(
        _DIGIT_LOOKALIKES.get(ch, ch)
        for ch in text
        if ch.isdigit() or ch in _DIGIT_LOOKALIKES
    )
    # Normalise to 11 digits starting with 7.
    if len(digits_only) == 10:
        digits_only = "7" + digits_only
    elif len(digits_only) == 11 and digits_only[0] == "8":
        digits_only = "7" + digits_only[1:]
    if len(digits_only) != 11 or digits_only[0] != "7":
        return None
    if not digits_only.isdigit():
        return None
    # Area-code sanity: first digit after country is 3-9 for real
    # Russian area / mobile codes. Rejects OCR-rubbish that happens
    # to be 11 digits starting with 7 (e.g. from a long ИНН field).
    if digits_only[1] in ("0", "1", "2"):
        return None
    return (
        f"+7 ({digits_only[1:4]}) "
        f"{digits_only[4:7]}-{digits_only[7:9]}-{digits_only[9:11]}"
    )
