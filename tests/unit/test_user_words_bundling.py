"""Tests that prove user-words and user-patterns resource files are bundled.

Stage D of Initiative 1: Russian-tuned accuracy. These tests verify that
the ``resources/tessdata/user-words.{lang}`` and
``resources/tessdata/user-patterns.{lang}`` files exist, have the right
encoding / line endings, and contain the sentinel terms / patterns we
rely on for downstream OCR quality (ИНН, КПП, ОГРН, months, the
DD.MM.YYYY date pattern, etc.).

The files live under ``resources/tessdata/`` because that is the same
directory PyInstaller already bundles via
``--add-data=resources/tessdata`` — so shipping them requires no
additional bundling hook, just a guard in ``build.ensure_resources`` to
fail loudly if someone deletes them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESOURCES_TESSDATA = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "tessdata"
)


# ---------------------------------------------------------------------------
# File existence + basic integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "user-words.rus",
        "user-words.eng",
        "user-patterns.rus",
        "user-patterns.eng",
    ],
)
def test_resource_file_exists(filename: str) -> None:
    """All four user-dict / user-pattern files must be present."""
    path = RESOURCES_TESSDATA / filename
    assert path.is_file(), (
        f"Missing bundled resource: {path}. This file is required for "
        "Tesseract user-words / user-patterns support (Stage D of "
        "accuracy initiative)."
    )


@pytest.mark.parametrize(
    "filename",
    [
        "user-words.rus",
        "user-words.eng",
        "user-patterns.rus",
        "user-patterns.eng",
    ],
)
def test_resource_file_is_non_empty(filename: str) -> None:
    path = RESOURCES_TESSDATA / filename
    assert path.stat().st_size > 0, f"{filename} is empty"


@pytest.mark.parametrize(
    "filename",
    [
        "user-words.rus",
        "user-words.eng",
        "user-patterns.rus",
        "user-patterns.eng",
    ],
)
def test_resource_file_is_valid_utf8(filename: str) -> None:
    """Tesseract requires UTF-8 encoded user-words / user-patterns files."""
    path = RESOURCES_TESSDATA / filename
    data = path.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{filename} is not valid UTF-8: {exc}")


@pytest.mark.parametrize(
    "filename",
    [
        "user-words.rus",
        "user-words.eng",
        "user-patterns.rus",
        "user-patterns.eng",
    ],
)
def test_resource_file_uses_lf_line_endings(filename: str) -> None:
    """CRLF line endings break Tesseract's line-based parsers on some platforms.

    We keep these files as LF-only so they behave identically on
    Linux CI and on Windows end-user machines.
    """
    path = RESOURCES_TESSDATA / filename
    raw = path.read_bytes()
    assert b"\r\n" not in raw, (
        f"{filename} contains CRLF line endings; keep it LF-only so "
        "Tesseract's line parser sees the same content on every OS."
    )


# ---------------------------------------------------------------------------
# Russian user-words sentinel check
# ---------------------------------------------------------------------------


def _read_words(filename: str) -> list[str]:
    """Return stripped non-empty lines from a user-words file."""
    path = RESOURCES_TESSDATA / filename
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestRussianUserWordsContents:
    """Sentinel checks: specific high-signal terms must be present."""

    @pytest.mark.parametrize(
        "term",
        ["руб", "шт", "ИНН", "КПП", "ОГРН", "ООО", "января"],
    )
    def test_contains_sentinel_term(self, term: str) -> None:
        words = _read_words("user-words.rus")
        assert term in words, (
            f"Expected {term!r} in user-words.rus — this is a high-signal "
            "term for Russian business documents (units, tax codes, "
            "entity types, months)."
        )

    def test_has_at_least_100_words(self) -> None:
        words = _read_words("user-words.rus")
        assert len(words) >= 100, (
            f"user-words.rus has only {len(words)} words; expected ≥100 "
            "for meaningful Tesseract dictionary coverage."
        )


class TestEnglishUserWordsContents:
    def test_contains_common_business_terms(self) -> None:
        words = _read_words("user-words.eng")
        for expected in ("invoice", "total", "LLC", "January"):
            assert expected in words, (
                f"Expected {expected!r} in user-words.eng"
            )

    def test_has_meaningful_size(self) -> None:
        words = _read_words("user-words.eng")
        assert len(words) >= 50, (
            f"user-words.eng has only {len(words)} words; too small to "
            "help Tesseract."
        )


# ---------------------------------------------------------------------------
# Russian user-patterns sentinel check
# ---------------------------------------------------------------------------


def _read_patterns(filename: str) -> list[str]:
    path = RESOURCES_TESSDATA / filename
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestRussianUserPatternsContents:
    """Sentinel checks for user-patterns.rus."""

    def test_contains_inn_10_digit_pattern(self) -> None:
        """ИНН (Russian tax ID, legal entity form) is exactly 10 digits."""
        patterns = _read_patterns("user-patterns.rus")
        expected = r"\d\d\d\d\d\d\d\d\d\d"
        assert expected in patterns, (
            f"Expected ИНН pattern {expected!r} in user-patterns.rus"
        )

    def test_contains_date_ddmmyyyy_pattern(self) -> None:
        """DD.MM.YYYY is the dominant Russian date format on documents.

        Апрель 2026: pattern записан как ``\\d\\d.\\d\\d.\\d\\d\\d\\d``
        — точка НЕ экранируется. Tesseract user-patterns-parser
        принимает escape только для класс-токенов (``\\d`` / ``\\A`` /
        ``\\c`` / ``\\n`` / ``\\p`` / ``\\*``); попытка экранировать
        обычный punctuation (``\\.`` / ``\\+`` / ``\\-``) приводит к
        «[tesseract] Invalid user pattern …» на каждом OCR-вызове.
        """
        patterns = _read_patterns("user-patterns.rus")
        expected = r"\d\d.\d\d.\d\d\d\d"
        assert expected in patterns, (
            f"Expected date pattern {expected!r} in user-patterns.rus"
        )

    def test_file_ends_with_blank_line(self) -> None:
        """Tesseract's user-patterns parser requires a trailing newline.

        See https://tesseract-ocr.github.io/tessdoc/APIExample-user_patterns.html
        — the reference parser in ``user_words.cpp`` / ``user_patterns.cpp``
        tokenizes on newline and if the file does NOT end with one, the
        last pattern is silently dropped.
        """
        raw = (RESOURCES_TESSDATA / "user-patterns.rus").read_bytes()
        assert raw.endswith(b"\n"), (
            "user-patterns.rus must end with a newline; otherwise "
            "Tesseract silently drops the last pattern."
        )


class TestEnglishUserPatternsContents:
    def test_contains_date_pattern(self) -> None:
        patterns = _read_patterns("user-patterns.eng")
        # US date format DD/MM/YYYY or MM/DD/YYYY — both shapes collapse
        # to the same regex; we just want ONE slash-separated date form.
        expected = r"\d\d/\d\d/\d\d\d\d"
        assert expected in patterns, (
            f"Expected date pattern {expected!r} in user-patterns.eng"
        )

    def test_file_ends_with_newline(self) -> None:
        raw = (RESOURCES_TESSDATA / "user-patterns.eng").read_bytes()
        assert raw.endswith(b"\n"), (
            "user-patterns.eng must end with a newline."
        )
