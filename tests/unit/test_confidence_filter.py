"""Unit tests for :mod:`src.core.confidence_filter`.

The module rebuilds OCR text from Tesseract's TSV-style
``image_to_data`` output, dropping low-confidence words. These tests
exercise the shape of the input dict (which varies slightly across
pytesseract / Tesseract versions), the line-grouping logic and the
empty-input fallback that callers rely on to keep existing text.
"""

from __future__ import annotations

import pytest

from src.core.confidence_filter import reconstruct_text_from_tsv


def _tsv(
    *rows: tuple[str, float, int, int, int, int],
) -> dict[str, list]:
    """Build an ``image_to_data``-shaped dict from ``(text, conf, block,
    par, line, word_num)`` tuples.

    Real pytesseract output also includes ``level`` / ``left`` / ``top``
    / ``width`` / ``height`` / ``page_num``, but the filter ignores
    those — the test fixture keeps the dict minimal.
    """
    return {
        "text":      [r[0] for r in rows],
        "conf":      [r[1] for r in rows],
        "block_num": [r[2] for r in rows],
        "par_num":   [r[3] for r in rows],
        "line_num":  [r[4] for r in rows],
        "word_num":  [r[5] for r in rows],
    }


class TestThresholdFiltering:
    """Words below ``min_confidence`` are dropped; those at-or-above survive."""

    def test_drops_words_below_threshold(self) -> None:
        data = _tsv(
            ("hello", 95.0, 1, 1, 1, 1),
            ("noise", 15.0, 1, 1, 1, 2),  # sub-threshold → drop
            ("world", 92.0, 1, 1, 1, 3),
        )
        out = reconstruct_text_from_tsv(data, min_confidence=60.0)
        assert out == "hello world"

    def test_boundary_word_at_threshold_kept(self) -> None:
        """Word with exactly threshold conf is KEPT, not dropped."""
        data = _tsv(
            ("edge", 60.0, 1, 1, 1, 1),
            ("noise", 59.9, 1, 1, 1, 2),
        )
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == "edge"

    def test_negative_conf_placeholders_dropped(self) -> None:
        """Tesseract emits conf=-1 for level/page/block/par/line marker
        rows that carry no word text. Those must not pollute the output
        even when the row's ``text`` field happens to be non-empty
        (some Tesseract versions put the paragraph's first word there
        AND emit a separate word row — we drop the marker and keep the
        word row)."""
        data = _tsv(
            ("marker", -1.0, 1, 1, 1, 0),
            ("real",   80.0, 1, 1, 1, 1),
        )
        assert reconstruct_text_from_tsv(data, min_confidence=50.0) == "real"


class TestLineStructure:
    """Line breaks come from the TSV's ``(block, par, line)`` hierarchy."""

    def test_separate_lines_produce_newlines(self) -> None:
        data = _tsv(
            ("first",  80.0, 1, 1, 1, 1),
            ("line",   80.0, 1, 1, 1, 2),
            ("second", 80.0, 1, 1, 2, 1),  # line 2
            ("line",   80.0, 1, 1, 2, 2),
        )
        out = reconstruct_text_from_tsv(data, min_confidence=60.0)
        assert out == "first line\nsecond line"

    def test_different_blocks_still_newline_separated(self) -> None:
        """Block boundaries also trigger newlines — a table cell and a
        body paragraph must not be glued together just because Tesseract
        put them in the same paragraph number within their own blocks."""
        data = _tsv(
            ("celltext",  80.0, 1, 1, 1, 1),
            ("paratext",  80.0, 2, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(data, min_confidence=60.0)
        assert out == "celltext\nparatext"

    def test_word_order_preserved_within_line(self) -> None:
        """Reading order follows ``word_num`` ascending, regardless of
        the order rows appear in the TSV dict."""
        data = _tsv(
            ("middle", 80.0, 1, 1, 1, 2),
            ("last",   80.0, 1, 1, 1, 3),
            ("first",  80.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(data, min_confidence=60.0)
        assert out == "first middle last"


class TestEmptyInput:
    """Empty / all-dropped inputs return empty string (caller falls back)."""

    def test_empty_dict_returns_empty_string(self) -> None:
        assert reconstruct_text_from_tsv({}, min_confidence=60.0) == ""

    def test_all_low_conf_returns_empty(self) -> None:
        """Contract: an ALL-low-conf page yields empty output so the
        caller keeps whatever ``pr.text`` it already had — blanking
        out a legitimate OCR result just because confidence scoring
        itself was flaky would be a worse failure mode than "noise
        stays"."""
        data = _tsv(
            ("junk", 5.0, 1, 1, 1, 1),
            ("more", 10.0, 1, 1, 1, 2),
        )
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == ""

    def test_blank_words_and_whitespace_ignored(self) -> None:
        data = _tsv(
            ("", 90.0, 1, 1, 1, 1),   # blank
            ("   ", 90.0, 1, 1, 1, 2),  # whitespace-only
            ("kept", 90.0, 1, 1, 1, 3),
        )
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == "kept"


class TestRobustness:
    """Malformed / missing fields shouldn't crash the filter."""

    def test_missing_conf_key_drops_all(self) -> None:
        """Without ``conf`` the filter can't know which words to keep —
        dropping everything (→ empty) is the safe caller-falls-back
        behaviour."""
        data = {"text": ["hello", "world"]}
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == ""

    def test_missing_block_par_line_defaults_to_zero(self) -> None:
        """Without the hierarchy we get one output line — predictable
        and non-crashy, which is what real-world garbled TSVs need."""
        data = {"text": ["a", "b"], "conf": [90.0, 90.0]}
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == "a b"

    def test_non_string_text_entries_skipped(self) -> None:
        """Some TSV dumps contain ``None`` or numeric ``text`` rows;
        those must be skipped rather than crash ``.strip()``."""
        data = {
            "text": ["real", None, 42, "alsoreal"],
            "conf": [90.0, 90.0, 90.0, 90.0],
            "block_num": [1, 1, 1, 1],
            "par_num":   [1, 1, 1, 1],
            "line_num":  [1, 1, 1, 1],
            "word_num":  [1, 2, 3, 4],
        }
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == "real alsoreal"

    def test_non_numeric_conf_skipped(self) -> None:
        """A literal ``"N/A"`` or ``""`` in the conf column must not
        crash ``float()``."""
        data = _tsv(("real", 90.0, 1, 1, 1, 1))
        # Swap in a bad conf without breaking the row count.
        data["conf"] = ["N/A"]
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == ""


class TestRealisticGarbagePage:
    """End-to-end shape: a messy scan with real words + table ticks +
    stamp fragments. The filter should keep the readable text and
    drop the low-conf noise."""

    def test_drops_stamp_and_signature_noise_keeps_body_text(self) -> None:
        """Simulates page 4 of the user's transport invoice: a single
        paragraph of readable body text plus low-conf stamp / signature
        fragments floating around it. Filtering at 60 % yields ONLY
        the body text — the perceptual win the user asked for."""
        data = _tsv(
            # Body paragraph at high confidence
            ("ООО",           92.0, 1, 1, 1, 1),
            ("ГЕКСАФОРМ",     88.0, 1, 1, 1, 2),
            ("СПБ",           90.0, 1, 1, 1, 3),
            # Signature scribble row, low conf
            ("нe",            22.0, 2, 1, 1, 1),
            ("Taw",           18.0, 2, 1, 1, 2),
            ("Fam",           15.0, 2, 1, 1, 3),
            # Stamp fragment on another block, low conf
            ("ба",            30.0, 3, 1, 1, 1),
            ("к.",            12.0, 3, 1, 1, 2),
            # Second body line
            ("ИНН",           95.0, 1, 1, 2, 1),
            ("7813266190",    97.0, 1, 1, 2, 2),
        )
        out = reconstruct_text_from_tsv(data, min_confidence=60.0)
        assert out == "ООО ГЕКСАФОРМ СПБ\nИНН 7813266190"
        # Sanity: none of the low-conf junk words leaked into the output.
        for junk in ("нe", "Taw", "Fam", "ба", "к."):
            assert junk not in out


def test_module_exports_reconstruct_only() -> None:
    """The module deliberately exports one public name — this test is a
    tripwire: if someone adds another function without updating the
    module docstring, they'll see this test fail and get prompted to
    document the new surface."""
    import src.core.confidence_filter as cf

    assert cf.__all__ == ["reconstruct_text_from_tsv"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
