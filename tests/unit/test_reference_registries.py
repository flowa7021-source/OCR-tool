"""Tests for snap-to-nearest reference registries."""

from __future__ import annotations

import pytest

from src.shared.reference_registries import (
    snap_bank_by_bik,
    snap_known_org,
    snap_legal_form,
)


class TestSnapLegalForm:
    @pytest.mark.parametrize("raw,expected", [
        ("ООО", "ООО"),
        ("ооо", "ООО"),
        ('"ооо"', "ООО"),
        ("ИП", "ИП"),
        ("АО", "АО"),
    ])
    def test_exact_match(self, raw, expected):
        assert snap_legal_form(raw) == expected

    def test_one_char_off(self):
        # "ОО0" → "ООО" within Lev-1
        assert snap_legal_form("ОО0") == "ООО"

    def test_unrecognisable_returns_none(self):
        assert snap_legal_form("XYZ123") is None
        assert snap_legal_form("") is None

    def test_pao_recognised(self):
        assert snap_legal_form("пао") == "ПАО"


class TestSnapBankByBik:
    def test_known_bik(self):
        assert snap_bank_by_bik("044525225") == "ПАО Сбербанк"

    def test_unknown_bik(self):
        assert snap_bank_by_bik("000000000") is None

    def test_empty(self):
        assert snap_bank_by_bik("") is None


class TestSnapKnownOrg:
    def test_exact_case(self):
        assert snap_known_org("ГЕКСАФОРМ") == "ГЕКСАФОРМ"

    def test_lowercase_normalised(self):
        assert snap_known_org("гексаформ") == "ГЕКСАФОРМ"

    def test_one_edit(self):
        assert snap_known_org("ГЕКСАФРРМ") == "ГЕКСАФОРМ"

    def test_unknown(self):
        assert snap_known_org("XYZ") is None
