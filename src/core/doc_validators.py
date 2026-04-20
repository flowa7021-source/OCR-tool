"""Russian business-document identifier validators.

OCR on scanned forms regularly returns a correct-LOOKING ИНН / КПП /
ОГРН with a single wrong digit — Tesseract's LSTM confuses ``8``
with ``6``, ``1`` with ``7``, or inserts a space that splits a
single token into two. For regulated identifiers, the correct
number is structurally constrained (length + checksum), so we can
DETECT the corruption and in many cases FIX IT via a single-edit
search.

Validators in this module return ``True`` only when the string
passes its official checksum algorithm. The ``try_fix_*`` helpers
search the 1-edit neighbourhood (one substitution or one deletion)
for a valid identifier — returning ``None`` when no close-enough
valid version exists. Callers shouldn't accept a fix that's
ambiguous (multiple valid 1-edit neighbours) — the helper
returns ``None`` in that case too, so the OCR output stays
unchanged rather than getting rewritten to a wrong number.

The validators implement the official algorithms published by
the Federal Tax Service:

* ИНН — INN-10 (legal entity, 10 digits) weighted-sum / mod 11
  algorithm; INN-12 (individual entrepreneur, 12 digits) uses
  two sequential weighted sums.
* ОГРН — OGRN-13 (legal entity) is first-12-digits % 11, last
  digit of result matches position 13. OGRN-15 (individual
  entrepreneur, aka ОГРНИП) is first-14-digits % 13, last digit
  of result matches position 15.
* КПП — no checksum, purely a 9-char format check (``DDDD[AA|DD]DDD``).

Used by the postprocessor when ``PostprocessConfig.validate_identifiers``
is on; used by the doc-catalog loader to sanity-check ground-
truth JSONs for typos.
"""

from __future__ import annotations

import re

__all__ = [
    "validate_inn",
    "validate_kpp",
    "validate_ogrn",
    "try_fix_inn",
    "try_fix_ogrn",
    "catalog_assisted_fix",
]


# ---------------------------------------------------------------------------
# ИНН — Taxpayer Identification Number
# ---------------------------------------------------------------------------

_INN10_WEIGHTS: tuple[int, ...] = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_11: tuple[int, ...] = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_12: tuple[int, ...] = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _inn_checksum_digit(digits: tuple[int, ...], weights: tuple[int, ...]) -> int:
    """Return the Federal-Tax-Service weighted checksum digit.

    Applies the official formula ``(Σ digit_i * weight_i) mod 11``
    with a ``>= 10 → 0`` fold-down for the final digit.
    """
    total = sum(d * w for d, w in zip(digits, weights, strict=True))
    remainder = total % 11
    return remainder if remainder < 10 else 0


def validate_inn(s: str) -> bool:
    """Return True iff ``s`` is a digit-only ИНН that passes checksum.

    Accepts 10-digit legal-entity or 12-digit individual/IE formats.
    Anything else (wrong length, contains non-digits) returns False
    without raising — callers can feed raw OCR strings straight in.
    """
    if not isinstance(s, str) or not s.isdigit():
        return False
    if len(s) == 10:
        digits = tuple(int(c) for c in s)
        return _inn_checksum_digit(digits[:9], _INN10_WEIGHTS) == digits[9]
    if len(s) == 12:
        digits = tuple(int(c) for c in s)
        check11 = _inn_checksum_digit(digits[:10], _INN12_WEIGHTS_11)
        check12 = _inn_checksum_digit(digits[:11], _INN12_WEIGHTS_12)
        return check11 == digits[10] and check12 == digits[11]
    return False


# ---------------------------------------------------------------------------
# КПП — Tax Registration Reason Code
# ---------------------------------------------------------------------------

# КПП has no checksum; it's 9 chars: 4-digit tax office + 2-char
# reason (digits OR letters A-Z) + 3-digit serial. The middle pair
# is where letters appear — e.g. ``01A`` for foreign-org branches.
_KPP_RE = re.compile(r"^\d{4}[\dA-ZА-Я]{2}\d{3}$")


def validate_kpp(s: str) -> bool:
    """Return True iff ``s`` is a well-formed КПП. Format only — no
    checksum exists, so a format-valid value might still be a typo
    on a real company's КПП."""
    if not isinstance(s, str):
        return False
    return bool(_KPP_RE.fullmatch(s))


# ---------------------------------------------------------------------------
# ОГРН / ОГРНИП — State Registration Number
# ---------------------------------------------------------------------------


def validate_ogrn(s: str) -> bool:
    """Return True iff ``s`` is a digit-only ОГРН / ОГРНИП with valid
    checksum.

    OGRN-13 (legal entity, 13 digits): first 12 digits as integer
    modulo 11 — last digit of the result matches position 13.
    OGRN-15 (individual entrepreneur, 15 digits, aka ОГРНИП): first
    14 digits as integer modulo 13 — last digit of the result
    matches position 15.
    """
    if not isinstance(s, str) or not s.isdigit():
        return False
    if len(s) == 13:
        prefix = int(s[:12])
        return prefix % 11 % 10 == int(s[12])
    if len(s) == 15:
        prefix = int(s[:14])
        return prefix % 13 % 10 == int(s[14])
    return False


# ---------------------------------------------------------------------------
# Fixup helpers — try the 1-edit neighbourhood for a valid ID
# ---------------------------------------------------------------------------


def _single_substitution_neighbours(s: str) -> set[str]:
    """Yield every string one digit-substitution away from ``s``.

    Fixes the common Tesseract confusion: ``8→6``, ``1→7``, ``0→O``
    gets pre-normalised outside. We don't cap the search to a
    specific confusion map on purpose — any digit for any other
    digit is valid OCR noise, and checking all 9 × len(s) candidates
    is ~90 cheap integer-math hits on a 10-digit ИНН.
    """
    if not s.isdigit():
        return set()
    out: set[str] = set()
    for i in range(len(s)):
        original = s[i]
        for d in "0123456789":
            if d == original:
                continue
            out.add(s[:i] + d + s[i + 1 :])
    return out


def _single_deletion_neighbours(s: str) -> set[str]:
    """Yield every string with ONE character removed.

    Tesseract occasionally splits one digit into two glyphs at high
    DPI — e.g. a narrow ``1`` reported as ``1I`` → 11 digits where
    the document has 10. Trying the deletion neighbourhood
    recovers the original.
    """
    return {s[:i] + s[i + 1 :] for i in range(len(s))}


def _try_fix(candidate: str, *, validator) -> str | None:
    """Return the unique valid 1-edit correction, or ``None``.

    Multiple valid corrections are treated as ambiguous — we refuse
    to pick one. Zero valid corrections return ``None`` so the
    caller can leave the OCR string untouched.
    """
    valid = {
        n for n in _single_substitution_neighbours(candidate)
        if validator(n)
    }
    valid |= {
        n for n in _single_deletion_neighbours(candidate)
        if validator(n)
    }
    if len(valid) == 1:
        return valid.pop()
    return None


def try_fix_inn(s: str) -> str | None:
    """Return the unique 1-edit correction of ``s`` that validates as
    an ИНН, or ``None`` if already valid / no unique correction.

    Use like::

        if not validate_inn(token):
            corrected = try_fix_inn(token)
            if corrected is not None:
                replace token with corrected
    """
    if not isinstance(s, str) or not s.isdigit():
        return None
    if validate_inn(s):
        return None  # Already valid; nothing to fix.
    if len(s) not in (10, 11, 12, 13):
        # Deletion of 11 → 10 and 13 → 12 are the only
        # realistic over-count patterns; substitution keeps length.
        return None
    return _try_fix(s, validator=validate_inn)


def try_fix_ogrn(s: str) -> str | None:
    """Same as :func:`try_fix_inn`, for ОГРН / ОГРНИП."""
    if not isinstance(s, str) or not s.isdigit():
        return None
    if validate_ogrn(s):
        return None
    if len(s) not in (13, 14, 15, 16):
        return None
    return _try_fix(s, validator=validate_ogrn)


def catalog_assisted_fix(
    s: str,
    *,
    known_valid: frozenset[str] | set[str],
) -> str | None:
    """Return a catalog entry that's one edit away from ``s``, or None.

    Structural 1-edit fix (``try_fix_inn`` / ``try_fix_ogrn``) is
    too permissive for identifier fields because any single-digit
    substitution at a non-check-digit position PLUS a compensating
    change at the check position produces another valid identifier
    — there are usually 10+ distinct valid 1-edit neighbours.
    Structural fix therefore refuses to pick one.

    When the caller has a ground-truth catalog (e.g. the user's
    known counterparties), we can resolve the ambiguity: if exactly
    ONE catalog entry is a 1-edit neighbour of the OCR string,
    that entry wins. The catalog narrows the checksum-valid
    population from thousands down to a handful, and picking the
    unique match among those is the high-confidence correction.

    Args:
        s: OCR-extracted identifier (may be corrupt or truncated).
        known_valid: Set of ground-truth identifiers — ИНН / ОГРН
            values that actually appear in the user's document
            corpus. Typically loaded via
            :func:`src.core.doc_catalog.load_catalog`.

    Returns:
        The unique catalog entry that is a 1-edit substitution OR
        deletion away from ``s``, OR ``s`` itself if it's already
        in ``known_valid``, OR ``None`` when no such unique match
        exists. Never returns the original ``s`` when it was a
        typo; never returns an ambiguous pick.
    """
    if not isinstance(s, str) or not s:
        return None
    if s in known_valid:
        return None  # already canonical; nothing to fix
    neighbours = _single_substitution_neighbours(s) | _single_deletion_neighbours(s)
    matches = neighbours & set(known_valid)
    if len(matches) == 1:
        return next(iter(matches))
    return None
