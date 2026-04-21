"""Unit tests for :mod:`src.core.entity_normalizers`."""

from __future__ import annotations

import pytest

from src.core.entity_normalizers import (
    normalize_addresses,
    normalize_all,
    normalize_dates,
    normalize_phones,
)

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,expected", [
    ("29.08.2022", "29.08.2022"),          # already canonical
    ("29/08/2022", "29.08.2022"),
    ("29-08-2022", "29.08.2022"),
    ("1.8.2022", "01.08.2022"),             # zero-padding
    ("01.08.22", "01.08.2022"),             # 2-digit year → 2000s
    ("01.08.99", "01.08.1999"),             # 2-digit year ≥ 70 → 1900s
])
def test_numeric_date_formats(s: str, expected: str) -> None:
    assert normalize_dates(s) == expected


@pytest.mark.parametrize("s,expected", [
    ("29 августа 2022", "29.08.2022"),
    ("29 Августа 2022 года", "29.08.2022"),
    # «г.» в конце consumed regex'ом как optional-суффикс.
    ("1 января 2023 г.", "01.01.2023"),
    ("15 марта 2024", "15.03.2024"),
    ("31 декабря 1999", "31.12.1999"),
])
def test_wordy_date_formats(s: str, expected: str) -> None:
    assert normalize_dates(s) == expected


def test_invalid_date_unchanged() -> None:
    """32.13.2022 — невозможная дата, не трогаем."""
    assert normalize_dates("32.13.2022") == "32.13.2022"


def test_non_date_numbers_unchanged() -> None:
    """ИНН «7707820890» не должен восприниматься как дата."""
    assert normalize_dates("ИНН 7707820890") == "ИНН 7707820890"


def test_multiple_dates_in_text() -> None:
    s = "от 29/08/2022 до 15.03.2023 г."
    assert normalize_dates(s) == "от 29.08.2022 до 15.03.2023 г."


# ---------------------------------------------------------------------------
# Phones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,expected", [
    ("+7(812)441-30-24", "+7 (812) 441-30-24"),
    ("8(812)4413024", "+7 (812) 441-30-24"),
    ("8 (812) 441 30 24", "+7 (812) 441-30-24"),
    ("8-812-441-30-24", "+7 (812) 441-30-24"),
    ("+7 495 725-80-62", "+7 (495) 725-80-62"),
])
def test_phone_formats(s: str, expected: str) -> None:
    assert normalize_phones(s) == expected


def test_short_non_phone_unchanged() -> None:
    """Короткая последовательность цифр — не телефон."""
    assert normalize_phones("1234") == "1234"


def test_embedded_phone_in_text() -> None:
    s = "тел.: +7(812)441-30-24, доб. 123"
    out = normalize_phones(s)
    assert "+7 (812) 441-30-24" in out


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------


def test_comma_adds_space() -> None:
    s = "Москва,ул.Ленина,д.1"
    out = normalize_addresses(s)
    assert ", " in out
    assert "Москва, ул." in out


def test_prefix_adds_space() -> None:
    """г.Москва → г. Москва (space after period)."""
    out = normalize_addresses("г.Москва")
    assert out == "г. Москва"


def test_full_address_normalization() -> None:
    s = "125212,г.Москва,ул.Адмирала Макарова,д.6,стр.13"
    expected = (
        "125212, г. Москва, ул. Адмирала Макарова, д. 6, стр. 13"
    )
    assert normalize_addresses(s) == expected


def test_already_normalized_unchanged() -> None:
    s = "125212, г. Москва, ул. Ленина, д. 1"
    assert normalize_addresses(s) == s


# ---------------------------------------------------------------------------
# normalize_all
# ---------------------------------------------------------------------------


def test_normalize_all_empty() -> None:
    assert normalize_all("") == ""
    assert normalize_all(None) is None  # type: ignore[arg-type]


def test_normalize_all_idempotent() -> None:
    s = "от 29/08/2022, +7(812)441-30-24, г.Москва,ул.Ленина,д.1"
    once = normalize_all(s)
    twice = normalize_all(once)
    assert once == twice


def test_normalize_all_composes_all_three() -> None:
    s = "29/08/2022 по адресу г.Москва,ул.Ленина,д.1, тел.+7(812)441-30-24"
    out = normalize_all(s)
    assert "29.08.2022" in out
    assert "г. Москва, ул. Ленина, д. 1" in out
    assert "+7 (812) 441-30-24" in out
