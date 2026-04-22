"""Validators and OCR-aware correctors for Russian tax/legal requisites.

Implements checksum checks for ИНН (10/12 digits), ОГРН/ОГРНИП
(13/15 digits), and structural validation for КПП (9 digits).

The correctors try typical OCR confusions (``0`` ↔ ``O`` ↔ ``О``,
``1`` ↔ ``l`` ↔ ``I``, ``3`` ↔ ``З``, ``5`` ↔ ``Ы``, ``8`` ↔ ``В``, …)
and accept the first substitution that passes the checksum. They're
cheap — validator is O(n), corrector is O(n · k) where k is the small
alphabet of confusable pairs.

Usage::

    from src.shared.requisite_validators import (
        validate_inn, correct_inn, validate_ogrn, correct_ogrn,
    )

    validate_inn("7813266190")            # True — real company ИНН
    validate_inn("781326619O")            # False — trailing O, not 0
    correct_inn("781326619O")             # "7813266190" — fixed by O→0
    validate_ogrn("5137746157490")        # True
"""

from __future__ import annotations

# OCR confusion pairs — chars that look similar in scanned text. Each tuple
# is (digit_variant, non_digit_variant). The corrector tries substitutions
# in BOTH directions (digit ↔ letter), so both characters must be listed.
_OCR_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("O", "o", "О", "о", "Q"),
    "1": ("I", "l", "i", "|", "!"),
    "2": ("Z", "z"),
    "3": ("З", "з", "E"),
    "4": ("Ч", "ч", "A", "А"),
    "5": ("Ы", "S", "s"),
    "6": ("б",),
    "7": ("T", "Т", "т"),
    "8": ("В", "в", "B"),
    "9": ("g", "q", "Ч"),
}

# Reverse map — for each non-digit char, the digit it's likely confused with.
_TO_DIGIT: dict[str, str] = {}
for _digit, _variants in _OCR_CONFUSIONS.items():
    for _v in _variants:
        _TO_DIGIT[_v] = _digit


def _digits_only(s: str) -> str:
    """Strip whitespace and non-digit characters."""
    return "".join(ch for ch in s if ch.isdigit())


def _candidate_digit_strings(
    s: str, max_substitutions: int = 3,
) -> list[str]:
    """Enumerate plausible all-digit readings of ``s`` by swapping confused
    chars for their digit counterparts. Limits to ``max_substitutions`` to
    keep combinatorial blow-up bounded.
    """
    positions: list[tuple[int, str]] = []  # (index, forced_digit)
    chars = list(s.strip())
    for i, ch in enumerate(chars):
        if ch.isdigit():
            continue
        if ch in _TO_DIGIT:
            positions.append((i, _TO_DIGIT[ch]))
        else:
            # Non-digit, non-confused char — can't be rescued via this pass.
            return []
    if len(positions) > max_substitutions:
        return []
    # Apply every subset of substitutions (we want the minimal-edit solution
    # first, so enumerate from smallest to largest).
    candidates: list[str] = []
    for k in range(len(positions) + 1):
        for subset in _subsets(positions, k):
            out = chars.copy()
            for i, d in subset:
                out[i] = d
            # If some non-digit chars are still present, skip (we only
            # want all-digit candidates).
            if any(not c.isdigit() for c in out):
                continue
            candidates.append("".join(out))
    return candidates


def _subsets(items: list, k: int):
    """Yield all k-element subsets of a list (deterministic order)."""
    if k == 0:
        yield []
        return
    if k > len(items):
        return
    for i in range(len(items) - k + 1):
        for rest in _subsets(items[i + 1:], k - 1):
            yield [items[i]] + rest


# ── ИНН ────────────────────────────────────────────────────────────────────

_INN_10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8, 0)
_INN_12_WEIGHTS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0)
_INN_12_WEIGHTS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0)


def validate_inn(s: str) -> bool:
    """True iff ``s`` is a syntactically + checksum-valid ИНН (10 or 12 digits)."""
    d = _digits_only(s)
    if len(d) == 10:
        return _inn10_control(d) == int(d[9])
    if len(d) == 12:
        nd = [int(c) for c in d]
        c11 = sum(nd[i] * _INN_12_WEIGHTS_11[i] for i in range(11)) % 11 % 10
        c12 = sum(nd[i] * _INN_12_WEIGHTS_12[i] for i in range(12)) % 11 % 10
        return c11 == nd[10] and c12 == nd[11]
    return False


def _inn10_control(digits: str) -> int:
    nd = [int(c) for c in digits]
    return sum(nd[i] * _INN_10_WEIGHTS[i] for i in range(10)) % 11 % 10


def correct_inn(s: str, max_substitutions: int = 3) -> str | None:
    """Try OCR confusion substitutions; return the first checksum-valid
    ИНН, or ``None`` if nothing works within the budget.
    """
    if validate_inn(s):
        return _digits_only(s)
    for candidate in _candidate_digit_strings(s, max_substitutions):
        if validate_inn(candidate):
            return candidate
    return None


# ── ОГРН ───────────────────────────────────────────────────────────────────

def validate_ogrn(s: str) -> bool:
    """True iff ``s`` is a valid ОГРН (13 digits, company) or ОГРНИП (15, IP)."""
    d = _digits_only(s)
    if len(d) == 13:
        # control = (first_12_digits as int) % 11, then last digit of that
        control = (int(d[:12]) % 11) % 10
        return control == int(d[12])
    if len(d) == 15:
        control = (int(d[:14]) % 13) % 10
        return control == int(d[14])
    return False


def correct_ogrn(s: str, max_substitutions: int = 3) -> str | None:
    """Same approach as :func:`correct_inn` but for ОГРН/ОГРНИП."""
    if validate_ogrn(s):
        return _digits_only(s)
    for candidate in _candidate_digit_strings(s, max_substitutions):
        if validate_ogrn(candidate):
            return candidate
    return None


# ── КПП ────────────────────────────────────────────────────────────────────

def validate_kpp(s: str) -> bool:
    """Structural check for КПП: 9 digits. No checksum exists; we verify
    the 5th-6th position (region/reason code) is one of the known patterns
    (``\\d{2}`` digits or ``[A-Z0-9]{2}`` for foreign branches).
    """
    d = s.strip()
    if len(d) != 9:
        return False
    if not d[:4].isdigit() or not d[6:].isdigit():
        return False
    # Reason code (positions 4-5) must be either 2 digits or 2 chars A-Z/0-9.
    reason = d[4:6]
    return reason.isalnum() and reason.upper() == reason


def correct_kpp(s: str, max_substitutions: int = 3) -> str | None:
    """Try to coerce ``s`` into a valid 9-digit КПП by fixing confused digits.
    Since КПП has no checksum, we only verify structure after substitution.
    """
    if validate_kpp(s):
        return s.strip()
    # Only substitute the positions that must be digits (0-3 and 6-8).
    stripped = s.strip()
    if len(stripped) != 9:
        return None
    # Build candidates where positions 0-3, 6-8 become digits.
    mandatory_digit_positions = [0, 1, 2, 3, 6, 7, 8]
    subs: list[tuple[int, str]] = []
    for i in mandatory_digit_positions:
        ch = stripped[i]
        if ch.isdigit():
            continue
        if ch in _TO_DIGIT:
            subs.append((i, _TO_DIGIT[ch]))
        else:
            return None
    if len(subs) > max_substitutions:
        return None
    for k in range(len(subs) + 1):
        for subset in _subsets(subs, k):
            out = list(stripped)
            for i, d in subset:
                out[i] = d
            cand = "".join(out)
            if validate_kpp(cand):
                return cand
    return None


__all__ = [
    "validate_inn", "correct_inn",
    "validate_ogrn", "correct_ogrn",
    "validate_kpp", "correct_kpp",
]
