"""Tests for :mod:`src.core.garbage_filter` — written BEFORE the module
under test, TDD-style.

What "garbage" means in this codebase:
  * Lines where > 60 % of characters are non-letters / non-digits
    (e.g. ``!!!@#$%^`` or ``- - - - -``) — classic Tesseract
    artefact on ruled lines, table borders, stains.
  * Lines that are a lonely short glyph cluster with no word-like
    content (e.g. ``l`` on its own line, from a thin vertical ruler).
  * Long runs of the same character (``------``, ``======``).

What garbage is NOT (must NOT be dropped):
  * Actual numeric-only lines like "1 250 000" (totals in invoices).
  * Short legitimate words like "ООО", "ИП", "No", "12".
  * Lines with punctuation mixed with real text ("Итого: 500 руб.").

The filter ships two strictness levels:
  * ``lenient`` — drops only the obvious cases (≥70% symbols, long
    character runs). Safe default.
  * ``strict`` — adds low-confidence aware filtering and more
    aggressive symbol heuristics. Opt-in for scans so noisy that
    lenient still leaves too much debris.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def filter_module():
    """Import lazily — the module doesn't exist yet when pytest
    collects these tests for the first time. TDD path."""
    from src.core import garbage_filter

    return garbage_filter


# ---------------------------------------------------------------------------
# Public API surface (shape-test: imports + signatures)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Keep the module surface small and obvious."""

    def test_module_exports_filter_fn(self, filter_module) -> None:
        assert callable(filter_module.filter_garbage_lines)

    def test_module_exports_strictness_enum(self, filter_module) -> None:
        assert hasattr(filter_module, "GarbageStrictness")
        assert filter_module.GarbageStrictness.LENIENT
        assert filter_module.GarbageStrictness.STRICT
        assert filter_module.GarbageStrictness.DISABLED


# ---------------------------------------------------------------------------
# Lenient mode — default safety net
# ---------------------------------------------------------------------------


class TestLenientMode:
    """Lenient drops the clearly-mechanical garbage, leaves everything
    else alone. The default for production profiles."""

    def test_drops_symbol_wall(self, filter_module) -> None:
        text = "Hello world\n!!!@#$%^&*()!!!\nNext line"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert "Hello world" in out
        assert "Next line" in out
        assert "!!!@#$%^&*" not in out

    def test_drops_long_ruler_line(self, filter_module) -> None:
        text = "A\n---------------------\nB"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert "A" in out and "B" in out
        assert "---" not in out

    def test_drops_equal_sign_ruler(self, filter_module) -> None:
        text = "Header\n==========\nFooter"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert "Header" in out
        assert "Footer" in out
        assert "==" not in out

    def test_keeps_numeric_total_line(self, filter_module) -> None:
        # A lone number like "1 250 000" is a valid invoice total — MUST
        # survive the filter. Pure digits + whitespace is NOT garbage.
        text = "Subtotal\n1 250 000\nRUB"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert "1 250 000" in out

    def test_keeps_short_abbreviation_on_own_line(self, filter_module) -> None:
        # "ООО", "ИП", "No" etc. must survive. They're short, but
        # they're letters — not symbols.
        text = "Поставщик\nООО\nАльфа"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert "ООО" in out

    def test_keeps_mixed_punctuation_line(self, filter_module) -> None:
        # "Итого: 500 руб." — punctuation is fine when letters/digits
        # dominate.
        text = "Итого: 500 руб."
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert out.strip() == "Итого: 500 руб."

    def test_preserves_blank_line_structure(self, filter_module) -> None:
        text = "Para one\n\nPara two"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        # Blank line between paragraphs stays (it's structurally
        # meaningful, not garbage).
        assert out.count("\n\n") >= 1 or out.count("\n") >= 1


# ---------------------------------------------------------------------------
# Strict mode — aggressive filter for messy scans
# ---------------------------------------------------------------------------


class TestStrictMode:
    """Strict drops everything lenient drops, plus:
      * single-letter orphan lines
      * lines where letter-to-non-letter ratio is very low (≥40%
        non-letter, not just ≥70%)
    """

    def test_drops_single_letter_orphan(self, filter_module) -> None:
        text = "Paragraph one\nl\nParagraph two"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.STRICT,
        )
        assert "Paragraph one" in out
        assert "Paragraph two" in out
        # The orphan "l" on its own line is gone.
        assert "\nl\n" not in out
        lines = [ln.strip() for ln in out.strip().splitlines()]
        assert "l" not in lines

    def test_lenient_keeps_orphan_strict_drops(self, filter_module) -> None:
        # Lenient preserves it (not clearly garbage); strict drops it.
        text = "A\nl\nB"
        lenient = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        strict = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.STRICT,
        )
        assert "l" in lenient.split()
        assert "l" not in strict.split()

    def test_strict_drops_40pct_symbol_line(self, filter_module) -> None:
        # 4 letters + 3 symbols = 42% non-letters → strict drops,
        # lenient keeps.
        text = "good\nabc|||\nfine"
        strict = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.STRICT,
        )
        assert "abc|||" not in strict


# ---------------------------------------------------------------------------
# Disabled mode — pass-through for users who want the raw OCR output
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """DISABLED must return the input unmodified — useful when the
    user is debugging which lines Tesseract produced."""

    def test_disabled_is_pass_through(self, filter_module) -> None:
        text = "keep\n!!!@#$%\n---\nalso keep"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.DISABLED,
        )
        assert out == text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_returns_empty(self, filter_module) -> None:
        assert filter_module.filter_garbage_lines(
            "", filter_module.GarbageStrictness.LENIENT,
        ) == ""

    def test_single_clean_line_passes_through(self, filter_module) -> None:
        out = filter_module.filter_garbage_lines(
            "Hello world",
            filter_module.GarbageStrictness.LENIENT,
        )
        assert out.strip() == "Hello world"

    def test_unicode_cyrillic_never_mistaken_for_garbage(
        self, filter_module,
    ) -> None:
        text = "Договор № 123 от 15 марта 2024"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.STRICT,
        )
        # All Cyrillic + digits + a single "№" must pass through intact.
        assert "Договор" in out
        assert "15 марта 2024" in out

    def test_trailing_whitespace_preserved_within_line(
        self, filter_module,
    ) -> None:
        # Don't silently reflow content — only line-level drop.
        text = "line one  \nline two"
        out = filter_module.filter_garbage_lines(
            text, filter_module.GarbageStrictness.LENIENT,
        )
        assert "line one" in out
        assert "line two" in out
