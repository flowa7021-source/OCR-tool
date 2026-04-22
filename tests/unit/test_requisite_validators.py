"""Tests for ИНН/ОГРН/КПП checksum validators and OCR correctors."""

from __future__ import annotations

import pytest

from src.shared.requisite_validators import (
    correct_inn,
    correct_kpp,
    correct_ogrn,
    validate_inn,
    validate_kpp,
    validate_ogrn,
)

# Real requisites scraped from the expected/*.json golden corpus.
REAL_INN_10 = [
    "7813266190",   # ГЕКСАФОРМ СПБ, ООО
    "7707820890",   # Моспроект-3, АО
]
REAL_OGRN_13 = [
    "5137746157490",  # Моспроект-3 ОГРН
]
REAL_KPP = [
    "770701001",  # Моспроект-3 КПП
    "781301001",  # Hypothetical Piter КПП
]


class TestValidateInn:
    @pytest.mark.parametrize("inn", REAL_INN_10)
    def test_real_company_inns_validate(self, inn):
        assert validate_inn(inn) is True

    def test_broken_checksum(self):
        # Last digit ±1 → invalid checksum
        assert validate_inn("7813266191") is False
        assert validate_inn("7813266199") is False

    def test_wrong_length(self):
        assert validate_inn("781326619") is False   # 9 digits
        assert validate_inn("78132661900") is False  # 11 digits

    def test_non_digits_fail(self):
        # String containing letters can't be a syntactically valid ИНН.
        assert validate_inn("78132661XX") is False

    def test_12_digit_individual_inn(self):
        # Well-known test 12-digit ИНН with valid checksum
        assert validate_inn("500100732259") is True

    def test_empty_string(self):
        assert validate_inn("") is False


class TestCorrectInn:
    def test_noop_on_valid(self):
        assert correct_inn("7813266190") == "7813266190"

    @pytest.mark.parametrize("bad,fixed", [
        ("781326619O", "7813266190"),  # trailing O (letter) → 0
        ("781З266190", "7813266190"),  # Cyrillic З → 3
        ("78I3266190", "7813266190"),  # I → 1
        ("7813266l90", "7813266190"),  # lowercase l → 1
    ])
    def test_single_ocr_confusion(self, bad, fixed):
        assert correct_inn(bad) == fixed

    def test_multiple_substitutions(self):
        # Two confusions: O→0 and З→3
        assert correct_inn("781З26619O") == "7813266190"

    def test_no_valid_correction(self):
        # Random digit string with no rescue within budget
        assert correct_inn("ABCDEFGHIJ") is None

    def test_budget_exhausted(self):
        # 4 substitutions needed — default budget is 3, should return None
        assert correct_inn("78I326619O", max_substitutions=1) is None


class TestValidateOgrn:
    @pytest.mark.parametrize("ogrn", REAL_OGRN_13)
    def test_real_company_ogrn(self, ogrn):
        assert validate_ogrn(ogrn) is True

    def test_broken_checksum(self):
        assert validate_ogrn("5137746157491") is False

    def test_wrong_length(self):
        assert validate_ogrn("513774615749") is False     # 12 digits
        assert validate_ogrn("51377461574900") is False   # 14 digits


class TestCorrectOgrn:
    def test_noop_on_valid(self):
        assert correct_ogrn("5137746157490") == "5137746157490"

    def test_single_confusion(self):
        # Replace last 0 with letter O
        assert correct_ogrn("513774615749O") == "5137746157490"


class TestValidateKpp:
    @pytest.mark.parametrize("kpp", REAL_KPP)
    def test_real_kpp(self, kpp):
        assert validate_kpp(kpp) is True

    def test_wrong_length(self):
        assert validate_kpp("77070100") is False
        assert validate_kpp("7707010011") is False

    def test_non_digit_in_mandatory_position(self):
        # Position 0 must be digit; letter O should fail.
        assert validate_kpp("O70701001") is False

    def test_foreign_branch_reason_code(self):
        # 5-6 positions can be alphanumeric uppercase (foreign branch).
        assert validate_kpp("7707AB001") is True


class TestCorrectKpp:
    def test_noop(self):
        assert correct_kpp("770701001") == "770701001"

    def test_trailing_l_to_1(self):
        assert correct_kpp("77070100l") == "770701001"

    def test_no_rescue_for_totally_wrong_length(self):
        assert correct_kpp("77070") is None
