"""Structured-validation-driven OCR retry.

When a field-shaped OCR output fails its domain validator (ИНН/ОГРН
checksum, date parse, amount format), the `low_conf_retry` module
already covers low-confidence cases. This module covers the opposite
failure mode: the word came back with HIGH confidence but is actually
wrong — the model saw ``781326619O`` (letter O) and was sure about it,
and domain_lm didn't catch it because O-to-0 isn't word-level fuzzy.

Strategy:
    For each word that smells like a requisite (matches a pattern)
    but fails its validator:
      1. Try OCR-confusion correction (cheap, in-memory).
      2. If that fails, re-run the OCR on the crop with alt
         preprocessing via the `low_conf_retry` machinery.
      3. If the retried output validates, use it.
      4. Otherwise, keep the original — never silently write a
         suspect value.

This complements (doesn't replace) the postprocess-level
``_validate_identifiers`` in :mod:`src.core.text_postprocessor` —
that runs on concatenated text after all word-level passes. Here we
act per-word at the engine level where the crop is still available
for re-OCR.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.application.low_conf_retry import retry_low_confidence
from src.shared.requisite_validators import (
    correct_inn,
    correct_kpp,
    correct_ogrn,
    validate_inn,
    validate_kpp,
    validate_ogrn,
)

logger = logging.getLogger(__name__)

# Requisite-shaped tokens. Loose — we want candidates, the validator decides.
_INN_SHAPE = re.compile(r"^[0-9OoОоIlI|ЗзВвЧч!]{10}$|^[0-9OoОоIlI|ЗзВвЧч!]{12}$")
_OGRN_SHAPE = re.compile(r"^[0-9OoОоIlI|ЗзВвЧч!]{13}$|^[0-9OoОоIlI|ЗзВвЧч!]{15}$")
_KPP_SHAPE = re.compile(r"^[0-9OoОоIlI|ЗзВвЧч!]{9}$")
_DATE_SHAPE = re.compile(
    r"^\d{1,2}[./\-,]\d{1,2}[./\-,]\d{2,4}$"
)


@dataclass
class ValidationRetryStats:
    checked: int = 0
    fixed_offline: int = 0
    fixed_via_retry: int = 0
    unfixable: int = 0
    non_requisite: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "checked": self.checked,
            "fixed_offline": self.fixed_offline,
            "fixed_via_retry": self.fixed_via_retry,
            "unfixable": self.unfixable,
            "non_requisite": self.non_requisite,
        }


def _classify_field(token: str) -> str | None:
    """Return the field type a token smells like, or None."""
    if _INN_SHAPE.match(token):
        return "inn"
    if _OGRN_SHAPE.match(token):
        return "ogrn"
    if _KPP_SHAPE.match(token):
        return "kpp"
    if _DATE_SHAPE.match(token):
        return "date"
    return None


def _offline_fix(token: str, field: str) -> str | None:
    """Pure-Python fix via OCR-confusion substitution + checksum. Does
    not touch pixels — cheapest possible rescue.
    """
    if field == "inn":
        return correct_inn(token)
    if field == "ogrn":
        return correct_ogrn(token)
    if field == "kpp":
        return correct_kpp(token)
    if field == "date":
        # Dates: try replacing commas with dots, revalidate via datetime.
        import datetime
        cleaned = token.replace(",", ".").replace("-", ".").replace(" ", ".")
        for fmt in ("%d.%m.%Y", "%d.%m.%y"):
            try:
                datetime.datetime.strptime(cleaned, fmt)
                return cleaned
            except ValueError:
                continue
        return None
    return None


def _is_valid(token: str, field: str) -> bool:
    if field == "inn":
        return validate_inn(token)
    if field == "ogrn":
        return validate_ogrn(token)
    if field == "kpp":
        return validate_kpp(token)
    if field == "date":
        import datetime
        for fmt in ("%d.%m.%Y", "%d.%m.%y"):
            try:
                datetime.datetime.strptime(token, fmt)
                return True
            except ValueError:
                continue
        return False
    return False


def validate_and_retry(
    page: np.ndarray,
    raw_results: list[tuple[list[list[float]], str, float]],
    recognise: Callable[[np.ndarray], list[tuple[list[list[float]], str, float]]],
) -> tuple[list[tuple[list[list[float]], str, float]], ValidationRetryStats]:
    """Scan ``raw_results`` for requisite-shaped words and rescue
    failing ones.

    Args:
        page: full-page grayscale array (for crop-level re-OCR).
        raw_results: original OCR output as (bbox, text, conf) tuples.
        recognise: callable that runs OCR on a crop — same signature as
            used by :func:`src.application.low_conf_retry.retry_low_confidence`.

    Returns:
        (patched_results, stats).
    """
    stats = ValidationRetryStats()
    patched: list[tuple[list[list[float]], str, float]] = []
    # Collect indices + bbox+text+conf for words that need a crop-level retry.
    need_retry: list[tuple[int, list[list[float]], str, float]] = []
    for i, (bbox, text, conf) in enumerate(raw_results):
        field = _classify_field(text)
        if field is None:
            stats.non_requisite += 1
            patched.append((bbox, text, conf))
            continue
        stats.checked += 1
        if _is_valid(text, field):
            patched.append((bbox, text, conf))
            continue
        fixed = _offline_fix(text, field)
        if fixed is not None and _is_valid(fixed, field):
            logger.info(
                "structured_retry: %s %r → %r (offline fix)",
                field, text, fixed,
            )
            patched.append((bbox, fixed, min(1.0, conf + 0.1)))
            stats.fixed_offline += 1
            continue
        # Need crop-level retry to rescue.
        need_retry.append((i, bbox, text, conf))
        patched.append((bbox, text, conf))  # placeholder, overwritten below

    if need_retry:
        subset = [(bbox, text, conf) for _, bbox, text, conf in need_retry]
        # Force retry (threshold=1.0 = retry everything) on this subset.
        retried, _ = retry_low_confidence(
            page, subset, recognise,
            threshold=1.0, min_improvement=0.0,
        )
        for (orig_idx, bbox, orig_text, orig_conf), (_, new_text, new_conf) in zip(
            need_retry, retried, strict=True,
        ):
            field = _classify_field(orig_text)
            if field is None:
                continue
            # If retry produced a valid requisite — take it.
            for candidate in (new_text, _offline_fix(new_text, field)):
                if candidate and _is_valid(candidate, field):
                    logger.info(
                        "structured_retry: %s %r → %r (crop retry%s)",
                        field, orig_text, candidate,
                        ", then offline" if candidate != new_text else "",
                    )
                    patched[orig_idx] = (bbox, candidate, max(new_conf, orig_conf))
                    stats.fixed_via_retry += 1
                    break
            else:
                stats.unfixable += 1
    return patched, stats


__all__ = ["validate_and_retry", "ValidationRetryStats"]
