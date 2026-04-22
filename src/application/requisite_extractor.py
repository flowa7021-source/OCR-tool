"""Engine-agnostic extractor: scans OCR text, finds Russian-requisite
patterns, validates checksums, and OCR-corrects where possible.

Complements the full-blown ``tn_parser`` (which understands TN/UPD
document structure): this module operates on raw OCR output without
caring about document layout. It's the right tool when you need to
yank ИНН/ОГРН/КПП + dates out of arbitrary Russian business text —
export receipts, free-form invoices, email footers.

Design:
    - Each field has a Pattern (regex for candidate strings),
      a validator (checksum / structural), and an optional corrector
      (OCR confusion substitution).
    - Candidates are scored by (validator passes) + (length match) +
      (local context keyword like "ИНН" / "КПП" nearby).
    - The extractor returns all candidates sorted by score, so the
      caller can pick "all" (for a header with multiple parties) or
      "best" (for a single-field case).

Use together with the ``DomainLM`` from :mod:`src.shared.domain_lm`
for free-text fields (party names, addresses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.shared.requisite_validators import (
    correct_inn,
    correct_kpp,
    correct_ogrn,
)


@dataclass(frozen=True)
class Candidate:
    """A single extracted value with provenance + quality score."""
    field: str              # "inn" / "ogrn" / "kpp" / "date"
    value: str              # canonical form (checksum-fixed if possible)
    raw_match: str          # original substring in the OCR text
    start: int              # char offset in the input text
    end: int
    score: float            # 0..1, higher = more confident
    was_corrected: bool = False
    context_keyword: str | None = None  # nearby label like "ИНН"


# ── Regex patterns — tolerant of OCR confusions ────────────────────────────
# We keep the regexes loose (letters that could be mis-OCR'd digits) and
# rely on the per-field validator to reject false positives.

_DIGITISH = r"[0-9OoОоIlI|ЗзВв]"
# 10 digits for company ИНН, 12 for individual.
_INN10_RE = re.compile(rf"\b({_DIGITISH}{{10}})\b")
_INN12_RE = re.compile(rf"\b({_DIGITISH}{{12}})\b")
# 13 (company) or 15 (individual entrepreneur) digits for ОГРН/ОГРНИП.
_OGRN_RE = re.compile(rf"\b({_DIGITISH}{{13}})\b|\b({_DIGITISH}{{15}})\b")
_KPP_RE = re.compile(rf"\b({_DIGITISH}{{9}})\b")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[./\- ]\d{1,2}[./\- ]\d{2,4})\b|\b(\d{4}-\d{2}-\d{2})\b"
)

# Context keywords that boost a candidate's score when they appear in a
# small window before the match. OCR often drops punctuation so we match
# case-insensitively and don't require a colon.
_CONTEXT_WINDOW = 25  # chars
_CONTEXT_KEYWORDS = {
    "inn":  ["инн", "инн:", "inn"],
    "ogrn": ["огрн", "огрнип"],
    "kpp":  ["кпп", "kpp"],
    "date": ["от", "дата", "date"],
}


def _context_keyword(text: str, start: int, field: str) -> str | None:
    """Return the first label keyword in the ``CONTEXT_WINDOW`` chars before
    the match position, or None.
    """
    lo = max(0, start - _CONTEXT_WINDOW)
    snippet = text[lo:start].lower()
    for kw in _CONTEXT_KEYWORDS.get(field, ()):
        if kw in snippet:
            return kw
    return None


# ── Per-field scoring + correction ─────────────────────────────────────────

def _score_inn(raw: str, corrected: str | None, ctx_kw: str | None) -> float:
    if corrected is None:
        return 0.0
    # Base score = 0.7 (checksum passed).
    score = 0.7
    if ctx_kw is not None:
        score += 0.2  # label nearby → very likely this IS an ИНН
    if corrected == raw:
        score += 0.1  # no OCR correction needed — cleaner match
    return min(1.0, score)


def _score_ogrn(raw: str, corrected: str | None, ctx_kw: str | None) -> float:
    if corrected is None:
        return 0.0
    score = 0.7
    if ctx_kw is not None:
        score += 0.2
    if corrected == raw:
        score += 0.1
    return min(1.0, score)


def _score_kpp(raw: str, corrected: str | None, ctx_kw: str | None) -> float:
    # КПП has no checksum; weight context heavier.
    if corrected is None:
        return 0.0
    score = 0.4
    if ctx_kw is not None:
        score += 0.4
    if corrected == raw:
        score += 0.1
    return min(1.0, score)


# ── Main entry points ──────────────────────────────────────────────────────

def extract_inn(text: str) -> list[Candidate]:
    """Find all ИНН (10 or 12 digit, checksum-valid) in ``text``."""
    candidates: list[Candidate] = []
    for pattern in (_INN10_RE, _INN12_RE):
        for m in pattern.finditer(text):
            raw = m.group(1)
            fixed = correct_inn(raw)
            ctx = _context_keyword(text, m.start(), "inn")
            score = _score_inn(raw, fixed, ctx)
            if score <= 0:
                continue
            candidates.append(Candidate(
                field="inn",
                value=fixed,  # type: ignore[arg-type]
                raw_match=raw,
                start=m.start(),
                end=m.end(),
                score=score,
                was_corrected=(fixed != raw),
                context_keyword=ctx,
            ))
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def extract_ogrn(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for m in _OGRN_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        fixed = correct_ogrn(raw)
        ctx = _context_keyword(text, m.start(), "ogrn")
        score = _score_ogrn(raw, fixed, ctx)
        if score <= 0:
            continue
        candidates.append(Candidate(
            field="ogrn", value=fixed,  # type: ignore[arg-type]
            raw_match=raw, start=m.start(), end=m.end(),
            score=score, was_corrected=(fixed != raw),
            context_keyword=ctx,
        ))
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def extract_kpp(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    # Avoid matching 9-digit strings that are already part of an ИНН
    # (12-digit individual ИНН contains 9-digit substrings at offset 0).
    consumed_ranges = [
        (m.start(), m.end())
        for p in (_INN10_RE, _INN12_RE, _OGRN_RE)
        for m in p.finditer(text)
    ]
    for m in _KPP_RE.finditer(text):
        if any(m.start() >= a and m.end() <= b for a, b in consumed_ranges):
            continue
        raw = m.group(1)
        fixed = correct_kpp(raw)
        ctx = _context_keyword(text, m.start(), "kpp")
        score = _score_kpp(raw, fixed, ctx)
        if score <= 0:
            continue
        candidates.append(Candidate(
            field="kpp", value=fixed,  # type: ignore[arg-type]
            raw_match=raw, start=m.start(), end=m.end(),
            score=score, was_corrected=(fixed != raw),
            context_keyword=ctx,
        ))
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def extract_dates(text: str) -> list[Candidate]:
    """Find date-shaped strings (dd.mm.yyyy, yyyy-mm-dd, etc.).
    No checksum; validates structurally via :mod:`datetime.strptime`.
    """
    import datetime
    candidates: list[Candidate] = []
    for m in _DATE_RE.finditer(text):
        raw = (m.group(1) or m.group(2)).strip()
        canonical = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d-%m-%Y",
                    "%Y-%m-%d", "%d %m %Y"):
            try:
                parsed = datetime.datetime.strptime(raw, fmt).date()
                canonical = parsed.isoformat()
                break
            except ValueError:
                continue
        if canonical is None:
            continue
        ctx = _context_keyword(text, m.start(), "date")
        score = 0.7 + (0.2 if ctx else 0)
        candidates.append(Candidate(
            field="date", value=canonical, raw_match=raw,
            start=m.start(), end=m.end(), score=score,
            was_corrected=(canonical != raw),
            context_keyword=ctx,
        ))
    return sorted(candidates, key=lambda c: c.score, reverse=True)


@dataclass
class ExtractionResult:
    inns: list[Candidate] = field(default_factory=list)
    ogrns: list[Candidate] = field(default_factory=list)
    kpps: list[Candidate] = field(default_factory=list)
    dates: list[Candidate] = field(default_factory=list)


def extract_all(text: str) -> ExtractionResult:
    """Run all extractors, return a structured result."""
    return ExtractionResult(
        inns=extract_inn(text),
        ogrns=extract_ogrn(text),
        kpps=extract_kpp(text),
        dates=extract_dates(text),
    )


__all__ = [
    "Candidate", "ExtractionResult",
    "extract_inn", "extract_ogrn", "extract_kpp", "extract_dates",
    "extract_all",
]
