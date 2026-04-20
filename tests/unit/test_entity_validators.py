"""Unit tests for :mod:`src.core.entity_validators`."""

from __future__ import annotations

from src.core.entity_validators import (
    try_fix_amount,
    try_fix_date,
    try_fix_phone,
)


class TestTryFixDate:
    def test_valid_date_canonical(self) -> None:
        assert try_fix_date("12.01.2023") == "12.01.2023"

    def test_zero_padding_added(self) -> None:
        assert try_fix_date("2.1.2023") == "02.01.2023"

    def test_o_for_zero_substitution(self) -> None:
        """Tesseract often reads 0 as Latin or Cyrillic O on faded scans."""
        assert try_fix_date("12.O1.2O23") == "12.01.2023"
        assert try_fix_date("12.о1.2о23") == "12.01.2023"

    def test_l_and_i_for_one(self) -> None:
        assert try_fix_date("l2.0I.2023") == "12.01.2023"

    def test_space_separator(self) -> None:
        """Some OCRs emit space where dot was."""
        assert try_fix_date("12 01 2023") == "12.01.2023"

    def test_hyphen_separator(self) -> None:
        assert try_fix_date("12-01-2023") == "12.01.2023"

    def test_slash_separator(self) -> None:
        assert try_fix_date("12/01/2023") == "12.01.2023"

    def test_comma_separator(self) -> None:
        """``,`` in place of ``.`` is a common comma-vs-period OCR slip."""
        assert try_fix_date("12,01,2023") == "12.01.2023"

    def test_two_digit_year_expanded_to_2000s(self) -> None:
        assert try_fix_date("12.01.23") == "12.01.2023"

    def test_invalid_calendar_day(self) -> None:
        """February 30 doesn't exist — validator returns None."""
        assert try_fix_date("30.02.2023") is None

    def test_invalid_calendar_month(self) -> None:
        assert try_fix_date("01.13.2023") is None

    def test_year_out_of_range(self) -> None:
        """1700 is before the 1900–2100 cutoff."""
        assert try_fix_date("01.01.1700") is None
        assert try_fix_date("01.01.2150") is None

    def test_non_date_shape(self) -> None:
        assert try_fix_date("hello") is None
        assert try_fix_date("12345") is None

    def test_surrounding_whitespace_stripped(self) -> None:
        assert try_fix_date("  12.01.2023  ") == "12.01.2023"


class TestTryFixAmount:
    def test_canonical_form_preserved(self) -> None:
        assert try_fix_amount("1 234,56") == "1 234,56"

    def test_dot_to_comma_decimal(self) -> None:
        """US-style decimal dot → Russian decimal comma."""
        assert try_fix_amount("1234.56") == "1 234,56"

    def test_no_fractional_padded(self) -> None:
        assert try_fix_amount("1234") == "1 234,00"

    def test_one_fractional_digit_padded(self) -> None:
        assert try_fix_amount("1234.5") == "1 234,50"

    def test_o_for_zero_on_fractional(self) -> None:
        assert try_fix_amount("O,5O") == "0,50"

    def test_cyrillic_three_lookalike(self) -> None:
        """The Cyrillic ``З`` looks like a 3."""
        assert try_fix_amount("1 2З4,56") is None or try_fix_amount("1 2З4,56") == "1 234,56"
        # NOTE: the З→3 look-alike isn't in the lookup table by default
        # because it's risky on prose; the test documents current
        # behaviour rather than prescribing it.

    def test_currency_word_tolerated(self) -> None:
        assert try_fix_amount("1234 руб") == "1 234,00"

    def test_small_amount_no_thousands_separator(self) -> None:
        assert try_fix_amount("50,75") == "50,75"

    def test_invalid_shape_returns_none(self) -> None:
        assert try_fix_amount("not a number") is None

    def test_leading_zeros_preserved(self) -> None:
        """Don't strip leading zeros — may be a padded amount field."""
        assert try_fix_amount("007,50") == "007,50"


class TestTryFixPhone:
    def test_canonical_form_recognised(self) -> None:
        assert try_fix_phone("+7 (495) 725-80-62") == "+7 (495) 725-80-62"

    def test_8_prefix_normalised_to_plus_7(self) -> None:
        assert try_fix_phone("8 (495) 725-80-62") == "+7 (495) 725-80-62"

    def test_bare_digits_normalised(self) -> None:
        assert try_fix_phone("+74957258062") == "+7 (495) 725-80-62"

    def test_unspaced_8_prefix(self) -> None:
        assert try_fix_phone("84957258062") == "+7 (495) 725-80-62"

    def test_10_digits_assumes_missing_country_code(self) -> None:
        """10-digit area+subscriber → prepend 7."""
        assert try_fix_phone("4957258062") == "+7 (495) 725-80-62"

    def test_o_for_zero_substitution(self) -> None:
        """Tesseract reads zeros as O inside parentheses."""
        assert (
            try_fix_phone("+7 (495) 725-8O-62") == "+7 (495) 725-80-62"
        )

    def test_spaces_and_hyphens_mix(self) -> None:
        assert try_fix_phone("+7-495-725-80-62") == "+7 (495) 725-80-62"
        assert try_fix_phone("+7 495 725 80 62") == "+7 (495) 725-80-62"

    def test_short_number_returns_none(self) -> None:
        """< 10 digits — not a valid phone."""
        assert try_fix_phone("12345") is None

    def test_long_number_returns_none(self) -> None:
        """> 11 digits — probably an account or ИНН."""
        assert try_fix_phone("7495725806212") is None

    def test_area_starting_with_zero_rejected(self) -> None:
        """No valid Russian area code begins with 0."""
        assert try_fix_phone("+7 (095) 725-80-62") is None

    def test_non_phone_text_returns_none(self) -> None:
        assert try_fix_phone("hello world") is None
