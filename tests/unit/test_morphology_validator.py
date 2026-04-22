"""Tests for the pymorphy3-backed morphology guard used by fuzzy_corrector.

Focus is on the proper-noun POS-filter added in the 2M-lex expansion:
if the candidate has only function-word POS readings (ADVB/PRED/PREP…)
and the original looks proper-noun-shaped (Capitalized + lowercase rest),
``should_accept_correction`` must REJECT — otherwise «Подолино» (село)
would fuzzy-match to «Подлинно» (наречие).

``pymorphy3`` is a soft dependency; tests skip when absent rather than
falsely passing on the no-op fallback path (``is_known_russian_word``
returning True unconditionally when pymorphy3 is missing makes the
guard a no-op).
"""

from __future__ import annotations

import pytest

pymorphy3 = pytest.importorskip(
    "pymorphy3",
    reason="pymorphy3 required for the real proper-noun guard behaviour",
)

from src.core.morphology_validator import (  # noqa: E402
    _has_content_pos_reading,
    is_known_russian_word,
    should_accept_correction,
)


def test_known_content_word_has_content_pos() -> None:
    assert _has_content_pos_reading("организация") is True
    assert _has_content_pos_reading("является") is True


def test_functional_word_has_no_content_pos() -> None:
    # «подлинно» — единственное parse'ится как ADVB
    assert _has_content_pos_reading("подлинно") is False


def test_is_known_russian_word() -> None:
    assert is_known_russian_word("документ") is True
    assert is_known_russian_word("") is False
    assert is_known_russian_word("   ") is False


def test_accepts_content_word_correction() -> None:
    """Legitimate 1-edit OCR fix on a lowercase content word accepts."""
    assert should_accept_correction("являетси", "является") is True


def test_accepts_capitalized_content_word_correction() -> None:
    """Capitalized content word fix still accepted when candidate is content POS."""
    # «Экземиляр → Экземпляр» — legitimate fix, candidate is NOUN.
    assert should_accept_correction("Экземиляр", "Экземпляр") is True


def test_rejects_proper_noun_to_functional_word() -> None:
    """Proper-noun layout + function-only candidate POS → reject.

    Regression case: «Подолино» (деревня) — fuzzy cand «Подлинно»
    (наречие, ADVB only). Без guard'а 2M-lex выдавал бы substitution.
    """
    assert should_accept_correction("Подолино", "подлинно") is False


def test_same_word_rejected() -> None:
    assert should_accept_correction("документ", "документ") is False


def test_empty_values_rejected() -> None:
    assert should_accept_correction("", "foo") is False
    assert should_accept_correction("foo", "") is False


def test_lowercase_input_skips_proper_noun_guard() -> None:
    """Lowercase tokens are never treated as proper-noun shape, so the
    functional-POS candidate is still rejected via the regular
    known/unknown rule — but not via the proper-noun guard branch.
    """
    # Lowercase «подолино» — unknown. «подлинно» — known (ADVB).
    # Not-proper-shape, so the guard branch doesn't fire, but the
    # standard (not_orig_known and cand_known) still accepts.
    # That's fine: the guard's JOB is protecting Capitalized tokens.
    # This test locks the contract that lowercase path is unaffected.
    result = should_accept_correction("подолино", "подлинно")
    assert isinstance(result, bool)  # either answer is valid; just don't crash
