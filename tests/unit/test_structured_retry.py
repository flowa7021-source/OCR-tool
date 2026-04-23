"""Tests for structured-validator OCR retry."""

from __future__ import annotations

import numpy as np

from src.application.structured_retry import (
    ValidationRetryStats,
    _classify_field,
    _is_valid,
    _offline_fix,
    validate_and_retry,
)


def _bbox(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class TestClassifyField:
    def test_inn_10_shape(self):
        assert _classify_field("7813266190") == "inn"

    def test_inn_12_shape(self):
        assert _classify_field("500100732259") == "inn"

    def test_ogrn_13_shape(self):
        assert _classify_field("5137746157490") == "ogrn"

    def test_kpp_9_shape(self):
        assert _classify_field("770701001") == "kpp"

    def test_date_dd_mm_yyyy(self):
        assert _classify_field("02.09.2022") == "date"

    def test_non_requisite_none(self):
        assert _classify_field("гексаформ") is None
        assert _classify_field("") is None


class TestIsValid:
    def test_valid_inn(self):
        assert _is_valid("7813266190", "inn") is True

    def test_invalid_inn_checksum(self):
        assert _is_valid("7813266199", "inn") is False

    def test_valid_ogrn(self):
        assert _is_valid("5137746157490", "ogrn") is True

    def test_valid_kpp(self):
        assert _is_valid("770701001", "kpp") is True

    def test_valid_date(self):
        assert _is_valid("02.09.2022", "date") is True


class TestOfflineFix:
    def test_inn_with_letter_o(self):
        # 781326619 + letter O → should be rescued to digit 0
        assert _offline_fix("781326619O", "inn") == "7813266190"

    def test_date_comma_normalised(self):
        assert _offline_fix("01,09,2022", "date") == "01.09.2022"

    def test_no_rescue_for_nonsense(self):
        assert _offline_fix("abcdefghij", "inn") is None


class TestValidateAndRetry:
    def test_skips_non_requisites(self):
        page = np.full((50, 100), 255, dtype=np.uint8)
        raw = [(_bbox(0, 0, 30, 20), "гексаформ", 0.9)]

        def never_called(arr):
            raise AssertionError("should not call retry on non-requisite")

        patched, stats = validate_and_retry(page, raw, never_called)
        assert patched == raw
        assert stats.non_requisite == 1
        assert stats.checked == 0

    def test_already_valid_passes_through(self):
        page = np.full((50, 100), 255, dtype=np.uint8)
        raw = [(_bbox(0, 0, 30, 20), "7813266190", 0.95)]

        def never_called(arr):
            raise AssertionError("should not call retry on valid INN")

        patched, stats = validate_and_retry(page, raw, never_called)
        assert patched == raw
        assert stats.checked == 1
        assert stats.fixed_offline == 0

    def test_offline_fix_applied(self):
        page = np.full((50, 100), 255, dtype=np.uint8)
        raw = [(_bbox(0, 0, 30, 20), "781326619O", 0.9)]

        def never_called(arr):
            raise AssertionError("should not retry — offline fix works")

        patched, stats = validate_and_retry(page, raw, never_called)
        assert patched[0][1] == "7813266190"
        assert stats.fixed_offline == 1

    def test_unfixable_marked(self):
        page = np.full((50, 100), 255, dtype=np.uint8)
        raw = [(_bbox(0, 0, 30, 20), "9999999999", 0.8)]  # invalid checksum

        def no_help(arr):
            return []  # retry yields nothing

        patched, stats = validate_and_retry(page, raw, no_help)
        assert patched[0][1] == "9999999999"  # unchanged
        assert stats.unfixable == 1


def test_stats_serialisation():
    s = ValidationRetryStats(
        checked=5, fixed_offline=2, fixed_via_retry=1, unfixable=1,
        non_requisite=10,
    )
    assert s.to_dict() == {
        "checked": 5, "fixed_offline": 2, "fixed_via_retry": 1,
        "unfixable": 1, "non_requisite": 10,
    }
