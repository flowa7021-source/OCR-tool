"""Unit tests for :mod:`src.core.doc_validators`.

The module validates Russian business identifiers and tries to fix
corrupt OCR'd versions via a 1-edit search. These tests pin:

  * Every ground-truth identifier from ``expected/*.json`` passes
    its respective validator (real-data integration baked into a
    unit test).
  * Checksum rejection of common corruption patterns.
  * ``try_fix_*`` refuses to guess when the 1-edit neighbourhood
    contains multiple valid identifiers.
  * ``catalog_assisted_fix`` picks the unique catalog match when
    one exists, returns ``None`` for zero / multiple matches, and
    never overrides an input that's already canonical.
"""

from __future__ import annotations

import pytest

from src.core.doc_validators import (
    catalog_assisted_fix,
    try_fix_inn,
    try_fix_ogrn,
    validate_inn,
    validate_kpp,
    validate_ogrn,
)

# All identifiers below come from the user's hand-validated ground-
# truth corpus in ``expected/*.json``. Having real values here means
# a regression that breaks validator math shows up as a test failure
# with business semantics, not just abstract checksum logic.
_REAL_INN_10 = [
    "5003131890", "7707820890", "7720396310", "7743553262",
    "7743775240", "7811757210", "7813266190", "9702027120",
]
_REAL_INN_12 = ["524606438004"]
_REAL_OGRN_13 = ["5137746157490"]
_REAL_KPP = [
    "504445001", "770701001", "772501001", "774301001", "781101001",
]


# ---------------------------------------------------------------------------
# validate_inn
# ---------------------------------------------------------------------------


class TestValidateInn:
    @pytest.mark.parametrize("inn", _REAL_INN_10 + _REAL_INN_12)
    def test_real_ground_truth_passes(self, inn: str) -> None:
        """Every real ИНН the user has captured validates cleanly."""
        assert validate_inn(inn), f"{inn!r} should validate"

    def test_wrong_last_digit_fails(self) -> None:
        # 7813266190 is real → flip the last digit to 9.
        assert not validate_inn("7813266199")

    def test_wrong_middle_digit_fails(self) -> None:
        assert not validate_inn("7813266100")  # 4→1 in middle

    @pytest.mark.parametrize(
        "bad",
        ["", "abc", "12345", "12345678901234567890", "781326619O"],
    )
    def test_malformed_rejected(self, bad: str) -> None:
        """Non-digit input / wrong length → False, never raise."""
        assert not validate_inn(bad)

    def test_non_string_input_rejected(self) -> None:
        assert not validate_inn(7813266190)  # type: ignore[arg-type]
        assert not validate_inn(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_ogrn
# ---------------------------------------------------------------------------


class TestValidateOgrn:
    @pytest.mark.parametrize("ogrn", _REAL_OGRN_13)
    def test_real_13digit_passes(self, ogrn: str) -> None:
        assert validate_ogrn(ogrn)

    def test_wrong_check_digit_fails(self) -> None:
        # 5137746157490 → ...491 corrupts the check digit.
        assert not validate_ogrn("5137746157491")

    @pytest.mark.parametrize(
        "bad",
        ["", "12", "abc", "513774615749", "51377461574900"],
    )
    def test_malformed_rejected(self, bad: str) -> None:
        assert not validate_ogrn(bad)


# ---------------------------------------------------------------------------
# validate_kpp
# ---------------------------------------------------------------------------


class TestValidateKpp:
    @pytest.mark.parametrize("kpp", _REAL_KPP)
    def test_real_kpp_passes(self, kpp: str) -> None:
        assert validate_kpp(kpp)

    @pytest.mark.parametrize(
        "bad",
        ["", "12345", "1234567890", "ABC123456", "77070100l"],
    )
    def test_malformed_rejected(self, bad: str) -> None:
        assert not validate_kpp(bad)

    def test_foreign_branch_reason_code_accepted(self) -> None:
        """КПП for foreign-org branches has letters in positions
        5-6 (e.g. ``01A``); the regex permits that."""
        assert validate_kpp("7707Z1001")


# ---------------------------------------------------------------------------
# try_fix_inn / try_fix_ogrn — structural-only, no catalog
# ---------------------------------------------------------------------------


class TestStructuralFix:
    """Without a catalog the fix is usually ambiguous — the helper
    returns ``None`` rather than guess."""

    def test_valid_input_returns_none(self) -> None:
        """Nothing to fix — the caller should leave it alone."""
        assert try_fix_inn("7813266190") is None
        assert try_fix_ogrn("5137746157490") is None

    def test_ambiguous_fix_returns_none(self) -> None:
        """10-digit ИНН with a single-digit typo has ~10 valid 1-
        edit neighbours — refuse to pick."""
        # 7813266199 is 1 edit from real 7813266190 but the 10
        # position-0..position-8 substitution pairs each yield a
        # valid new ИНН too.
        assert try_fix_inn("7813266199") is None

    def test_malformed_input_returns_none(self) -> None:
        assert try_fix_inn("abc") is None
        assert try_fix_inn("") is None
        assert try_fix_ogrn("123") is None


# ---------------------------------------------------------------------------
# catalog_assisted_fix — the production path
# ---------------------------------------------------------------------------


class TestCatalogAssistedFix:
    def test_unique_catalog_match_is_picked(self) -> None:
        cat = frozenset({"7813266190"})
        assert catalog_assisted_fix(
            "7813266199", known_valid=cat,
        ) == "7813266190"

    def test_already_canonical_returns_none(self) -> None:
        """Exact match already in catalog → nothing to fix."""
        cat = frozenset({"7813266190"})
        assert catalog_assisted_fix("7813266190", known_valid=cat) is None

    def test_no_catalog_match_returns_none(self) -> None:
        """Corrupt token AND no 1-edit catalog entry → leave alone."""
        cat = frozenset({"9702027120"})
        assert catalog_assisted_fix(
            "7813266199", known_valid=cat,
        ) is None

    def test_multiple_catalog_1edit_matches_refused(self) -> None:
        """When two catalog entries are both 1-edit from the input,
        we can't unambiguously pick one → None."""
        # Both "7700000001" and "7800000001" are 1 sub from "7Ж00000001"
        cat = frozenset({"7700000001", "7800000001"})
        # Input is equidistant; no unique match.
        assert catalog_assisted_fix(
            "7900000001", known_valid=cat,
        ) is None

    def test_deletion_neighbour_recovered(self) -> None:
        """11-digit input → 10-digit catalog entry via deletion."""
        cat = frozenset({"7813266190"})
        # Extra "1" inserted at position 3 (OCR duplicating a digit).
        assert catalog_assisted_fix(
            "78113266190", known_valid=cat,
        ) == "7813266190"

    def test_empty_catalog_returns_none(self) -> None:
        assert catalog_assisted_fix("7813266199", known_valid=frozenset()) is None

    def test_empty_input_returns_none(self) -> None:
        cat = frozenset({"7813266190"})
        assert catalog_assisted_fix("", known_valid=cat) is None
