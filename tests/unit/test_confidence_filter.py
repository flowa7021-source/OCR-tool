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


class TestCapsCyrillicPreservation:
    """All-caps Cyrillic 3-7 char tokens (company / agency names)
    survive the confidence filter even below the threshold, provided
    the word's own confidence clears the ``_CAPS_COMPANY_MIN_CONFIDENCE``
    anti-noise floor.

    Targets real-world transport-invoice misses: ``БЕКАМ``, ``ДСК``,
    ``АВТОРЕСУРС`` etc. land on stamp-overlay lines where the
    per-word confidence comes back in the 30-55 range — above the
    anti-noise floor but below the nominal 60 % threshold. Preserving
    this shape of token lifts company-name recall without
    re-introducing stamp garbage.
    """

    def test_caps_cyrillic_company_kept_below_threshold(self) -> None:
        # "БЕКАМ" at 40 % conf — normally dropped by threshold=60,
        # preserved by the caps-company rule.
        tsv = _tsv(
            ("Компания", 85.0, 1, 1, 1, 1),
            ("БЕКАМ",    40.0, 1, 1, 1, 2),
            ("филиал",   85.0, 1, 1, 1, 3),
        )
        text = reconstruct_text_from_tsv(tsv, min_confidence=60.0)
        assert text == "Компания БЕКАМ филиал"

    def test_caps_cyrillic_dropped_if_below_anti_noise_floor(
        self,
    ) -> None:
        # Below 25 % — treated as noise, dropped even though the
        # pattern matches. This is the safety net against the caps-
        # pattern firing on pure-noise regions.
        tsv = _tsv(
            ("body",  80.0, 1, 1, 1, 1),
            ("БЕКАМ", 15.0, 1, 1, 1, 2),
        )
        text = reconstruct_text_from_tsv(tsv, min_confidence=60.0)
        assert text == "body"

    def test_mixed_case_not_preserved(self) -> None:
        # "Бекам" (title case) doesn't match the pattern — drops at
        # low conf like any other word.
        tsv = _tsv(
            ("Бекам", 40.0, 1, 1, 1, 1),
        )
        text = reconstruct_text_from_tsv(tsv, min_confidence=60.0)
        assert text == ""

    def test_lowercase_cyrillic_not_preserved(self) -> None:
        tsv = _tsv(
            ("бекам", 40.0, 1, 1, 1, 1),
        )
        assert reconstruct_text_from_tsv(tsv, min_confidence=60.0) == ""

    def test_longer_than_seven_chars_not_preserved(self) -> None:
        # "АВТОРЕСУРСПЛЮС" (>7) — too long to be a compact company
        # abbreviation; drop at low conf. Regular words that happen
        # to be all-caps (e.g. a screamed full word) shouldn't bypass
        # the filter.
        tsv = _tsv(
            ("АВТОРЕСУРСПЛЮС", 40.0, 1, 1, 1, 1),
        )
        assert reconstruct_text_from_tsv(tsv, min_confidence=60.0) == ""

    def test_two_chars_not_preserved(self) -> None:
        # 2-char caps could be junk (PSM artifacts). Pattern requires ≥3.
        tsv = _tsv(
            ("БК", 40.0, 1, 1, 1, 1),
        )
        assert reconstruct_text_from_tsv(tsv, min_confidence=60.0) == ""

    def test_short_caps_with_digits_not_preserved(self) -> None:
        # "UPD1" — contains digits, and Latin letters, skipping
        # the company-name rule.
        tsv = _tsv(
            ("UPD1", 40.0, 1, 1, 1, 1),
        )
        assert reconstruct_text_from_tsv(tsv, min_confidence=60.0) == ""

    def test_latin_caps_not_preserved(self) -> None:
        # "INV" is Latin caps — the rule is Cyrillic-only (different
        # character class with its own abundance patterns). A Latin
        # 3-letter token at low conf is more likely noise than a
        # business name in Russian documents.
        tsv = _tsv(
            ("INV", 40.0, 1, 1, 1, 1),
        )
        assert reconstruct_text_from_tsv(tsv, min_confidence=60.0) == ""

    def test_caps_cyrillic_above_threshold_still_kept(self) -> None:
        # Baseline — regular threshold path still works for caps
        # tokens at normal conf.
        tsv = _tsv(
            ("БЕКАМ", 82.0, 1, 1, 1, 1),
        )
        text = reconstruct_text_from_tsv(tsv, min_confidence=60.0)
        assert text == "БЕКАМ"

    def test_yo_letter_accepted(self) -> None:
        # "ЁЛКА" contains Ё (U+0401); rule accepts Ё as a caps
        # Cyrillic letter.
        tsv = _tsv(
            ("ЁЛКА", 35.0, 1, 1, 1, 1),
        )
        text = reconstruct_text_from_tsv(tsv, min_confidence=60.0)
        assert text == "ЁЛКА"


class TestDetectHandwrittenBlocks:
    """Block-level handwriting detection: blocks whose mean per-word
    confidence is below the cap AND contain ≥ N words are flagged
    as suspect, so the reconstructor can swap them out for a marker.
    """

    def test_low_conf_block_with_enough_words_flagged(self) -> None:
        from src.core.confidence_filter import detect_handwritten_blocks

        # Block 2 has 4 low-conf words (mean ~25 %) — should flag.
        # Block 1 has 3 high-conf words — should not.
        tsv = _tsv(
            ("Печатный", 90.0, 1, 1, 1, 1),
            ("текст",    92.0, 1, 1, 1, 2),
            ("здесь",    88.0, 1, 1, 1, 3),
            ("cquiglu",  25.0, 2, 1, 1, 1),
            ("nxoro",    28.0, 2, 1, 1, 2),
            ("ahalw",    20.0, 2, 1, 1, 3),
            ("xgoror",   27.0, 2, 1, 1, 4),
        )
        suspects = detect_handwritten_blocks(
            tsv, max_mean_confidence=40.0, min_words=3,
        )
        assert suspects == {(2, 1)}

    def test_low_conf_block_below_min_words_not_flagged(self) -> None:
        from src.core.confidence_filter import detect_handwritten_blocks

        # 2 low-conf words — too few to be confidently handwritten.
        # Stamp-overlay fragments fall here; let the word-level filter
        # drop them rather than swallowing with the marker.
        tsv = _tsv(
            ("bt", 20.0, 1, 1, 1, 1),
            ("zk", 25.0, 1, 1, 1, 2),
        )
        suspects = detect_handwritten_blocks(
            tsv, max_mean_confidence=40.0, min_words=3,
        )
        assert suspects == set()

    def test_high_conf_block_not_flagged_even_if_faded(self) -> None:
        from src.core.confidence_filter import detect_handwritten_blocks

        # Faded but printed block at mean 55 % — above the 40 % cap,
        # DON'T flag. These are the "light scan" pages where the
        # content is real but confidence sags.
        tsv = _tsv(
            ("Бледный", 55.0, 1, 1, 1, 1),
            ("текст",   58.0, 1, 1, 1, 2),
            ("здесь",   52.0, 1, 1, 1, 3),
        )
        suspects = detect_handwritten_blocks(
            tsv, max_mean_confidence=40.0, min_words=3,
        )
        assert suspects == set()

    def test_multiple_blocks_mixed(self) -> None:
        from src.core.confidence_filter import detect_handwritten_blocks

        tsv = _tsv(
            ("Good",      90.0, 1, 1, 1, 1),
            ("text",      88.0, 1, 1, 1, 2),
            ("here",      85.0, 1, 1, 1, 3),
            ("trdsdx",    18.0, 2, 1, 1, 1),
            ("qsdfkj",    22.0, 2, 1, 1, 2),
            ("pwern",     15.0, 2, 1, 1, 3),
            ("Another",   91.0, 3, 1, 1, 1),
            ("good",      90.0, 3, 1, 1, 2),
            ("block",     89.0, 3, 1, 1, 3),
            ("hndwrtn",   30.0, 4, 1, 1, 1),
            ("blocktoo",  25.0, 4, 1, 1, 2),
            ("scribble",  33.0, 4, 1, 1, 3),
            ("garbg",     28.0, 4, 1, 1, 4),
        )
        suspects = detect_handwritten_blocks(
            tsv, max_mean_confidence=40.0, min_words=3,
        )
        assert suspects == {(2, 1), (4, 1)}

    def test_empty_tsv_returns_empty_set(self) -> None:
        from src.core.confidence_filter import detect_handwritten_blocks

        assert detect_handwritten_blocks(
            {"text": [], "conf": []},
            max_mean_confidence=40.0,
            min_words=3,
        ) == set()

    def test_negative_conf_rows_ignored(self) -> None:
        """Placeholder rows (conf = -1) should not count toward the
        block's word count OR its mean."""
        from src.core.confidence_filter import detect_handwritten_blocks

        tsv = _tsv(
            ("", -1.0, 1, 1, 1, 0),  # placeholder
            ("", -1.0, 1, 1, 1, 0),  # placeholder
            ("real", 25.0, 1, 1, 1, 1),
            ("real", 22.0, 1, 1, 1, 2),
        )
        suspects = detect_handwritten_blocks(
            tsv, max_mean_confidence=40.0, min_words=3,
        )
        # Only 2 real words — below the 3 min — so not flagged.
        assert suspects == set()


class TestHandwrittenBlockReconstruction:
    """``reconstruct_text_from_tsv(handwritten_blocks=...)`` replaces
    flagged blocks with a single marker line, keeping non-flagged
    blocks intact."""

    def test_flagged_block_replaced_with_marker(self) -> None:
        tsv = _tsv(
            ("Printed",  90.0, 1, 1, 1, 1),
            ("text",     88.0, 1, 1, 1, 2),
            ("scribble", 20.0, 2, 1, 1, 1),
            ("garbage",  25.0, 2, 1, 1, 2),
            ("noise",    18.0, 2, 1, 1, 3),
        )
        out = reconstruct_text_from_tsv(
            tsv, min_confidence=0.0,
            handwritten_blocks={(2, 1)},
        )
        assert "Printed text" in out
        assert "⟨рукописный текст⟩" in out
        # The individual scribble words should NOT appear.
        assert "scribble" not in out
        assert "garbage" not in out

    def test_marker_appears_once_per_block(self) -> None:
        """Multi-line handwritten block → single marker."""
        tsv = _tsv(
            ("hw", 20.0, 1, 1, 1, 1),
            ("hw", 22.0, 1, 1, 1, 2),
            ("hw", 18.0, 1, 1, 2, 1),
            ("hw", 25.0, 1, 1, 2, 2),
        )
        out = reconstruct_text_from_tsv(
            tsv, min_confidence=0.0,
            handwritten_blocks={(1, 1)},
        )
        assert out.count("⟨рукописный текст⟩") == 1

    def test_no_hw_blocks_behaves_as_before(self) -> None:
        tsv = _tsv(
            ("Hello", 90.0, 1, 1, 1, 1),
            ("world", 85.0, 1, 1, 1, 2),
        )
        out = reconstruct_text_from_tsv(
            tsv, min_confidence=0.0, handwritten_blocks=None,
        )
        assert out == "Hello world"

    def test_custom_marker(self) -> None:
        tsv = _tsv(
            ("hw", 20.0, 1, 1, 1, 1),
            ("hw", 18.0, 1, 1, 1, 2),
            ("hw", 22.0, 1, 1, 1, 3),
        )
        out = reconstruct_text_from_tsv(
            tsv, min_confidence=0.0,
            handwritten_blocks={(1, 1)},
            handwritten_marker="[HW]",
        )
        assert out == "[HW]"

    def test_hw_block_with_no_surviving_words_still_emits_marker(
        self,
    ) -> None:
        """Even when ALL words in the flagged block are below the
        threshold and would drop, the marker still surfaces — the
        whole point is to signal "there was content here you may
        want to transcribe"."""
        tsv = _tsv(
            ("prose",    90.0, 1, 1, 1, 1),
            ("keeps",    88.0, 1, 1, 1, 2),
            ("junk1",    10.0, 2, 1, 1, 1),
            ("junk2",    15.0, 2, 1, 1, 2),
            ("junk3",    20.0, 2, 1, 1, 3),
        )
        out = reconstruct_text_from_tsv(
            tsv, min_confidence=60.0,
            handwritten_blocks={(2, 1)},
        )
        assert "prose keeps" in out
        assert "⟨рукописный текст⟩" in out


def test_module_exports() -> None:
    """Pin the public API of the confidence-filter module so adding a
    new symbol forces a deliberate docstring update."""
    import src.core.confidence_filter as cf

    assert cf.__all__ == [
        "HANDWRITTEN_MARKER",
        "detect_handwritten_blocks",
        "reconstruct_text_from_tsv",
    ]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
