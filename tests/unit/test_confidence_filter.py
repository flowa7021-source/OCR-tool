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


class TestSoftRescue:
    """Soft-rescue keeps borderline-conf tokens that look lexically clean.

    The goal is to recover the Tesseract under-confidence pattern on
    short digit runs (ИНН, суммы, даты) and all-caps acronyms without
    letting stamp / signature scribble back in. Every test here pairs
    what-rescues with what-still-drops so regressions in either
    direction get caught.
    """

    def test_disabled_by_default_preserves_legacy_behavior(self) -> None:
        """``soft_rescue`` defaults to False — every legacy caller is
        unaffected. Same input, same output as the strict-drop path."""
        data = _tsv(
            ("kept", 80.0, 1, 1, 1, 1),
            ("borderline", 55.0, 1, 1, 1, 2),
        )
        assert reconstruct_text_from_tsv(data, min_confidence=60.0) == "kept"

    def test_rescues_short_digit_run_in_band(self) -> None:
        """An ИНН-like digit run at 55% with threshold 60 gets rescued —
        this is the primary reason soft-rescue exists."""
        data = _tsv(
            ("ГЕКСАФОРМ", 88.0, 1, 1, 1, 1),
            ("7813266190", 55.0, 1, 1, 1, 2),  # real content, underweighted
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == "ГЕКСАФОРМ 7813266190"

    def test_rescues_all_cyrillic_acronym_in_band(self) -> None:
        """``ООО`` at 58% with threshold 60 is credible — single-script,
        length 3, rescue should keep it."""
        data = _tsv(
            ("ООО", 58.0, 1, 1, 1, 1),
            ("ГЕКСАФОРМ", 90.0, 1, 1, 1, 2),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == "ООО ГЕКСАФОРМ"

    def test_rescues_date_with_dot_separators(self) -> None:
        """``12.08.2024`` at 50% with threshold 60 — digit+sep token
        should rescue."""
        data = _tsv(
            ("12.08.2024", 50.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == "12.08.2024"

    def test_rescues_amount_with_comma_and_dot(self) -> None:
        """Russian-formatted amount ``1,200.50`` at 50% rescues."""
        data = _tsv(
            ("1,200.50", 50.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == "1,200.50"

    def test_does_not_rescue_below_absolute_floor(self) -> None:
        """A lexically-clean digit run at 40% is BELOW the 45% absolute
        floor — must still drop even though the shape looks fine. This
        protects against wild thresholds from misconfigured profiles."""
        data = _tsv(
            ("1234567890", 40.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == ""

    def test_does_not_rescue_outside_soft_margin(self) -> None:
        """A lexically-clean token at 30% is well outside the 15%
        margin below the threshold. Drop."""
        data = _tsv(
            ("ГЕКСАФОРМ", 30.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == ""

    def test_does_not_rescue_mixed_script_short_token(self) -> None:
        """``нe`` — 2 chars, Cyrillic н + Latin e — is exactly the
        signature-scribble failure mode rescue must AVOID, even at
        50% conf which is inside the band."""
        data = _tsv(
            ("нe", 50.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == ""

    def test_does_not_rescue_two_char_cyrillic(self) -> None:
        """``ба`` at 50% — single-script but only 2 chars, below the
        length-3 minimum. Stamp / noise territory."""
        data = _tsv(
            ("ба", 50.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == ""

    def test_does_not_rescue_punctuation_tail(self) -> None:
        """``к.`` at 50% — letter+punct, fails the single-script check
        because the dot isn't a letter AND the token isn't digit-shaped.
        Drop."""
        data = _tsv(
            ("к.", 50.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == ""

    def test_high_conf_always_kept_regardless_of_shape(self) -> None:
        """Above-threshold words are accepted untouched — soft-rescue
        only affects the sub-threshold band. A mixed-script 90-conf
        token (shouldn't happen in practice, but some Tesseract
        versions do emit them) is kept."""
        data = _tsv(
            ("Скаnia", 90.0, 1, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        assert out == "Скаnia"

    def test_rescue_band_respects_threshold(self) -> None:
        """At threshold=80, band = [65, 80). A 70% digit token should
        rescue; a 60% one should not."""
        data = _tsv(
            ("1234567890", 70.0, 1, 1, 1, 1),  # rescued
            ("9876543210", 60.0, 1, 1, 1, 2),  # dropped (below 65)
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=80.0, soft_rescue=True,
        )
        assert out == "1234567890"

    def test_realistic_form_page_preserves_content_and_drops_noise(
        self,
    ) -> None:
        """Combined scenario: body text, ИНН, and signature scribble all
        on the same page. Soft-rescue recovers the ИНН that would
        otherwise be lost to the strict threshold AND still drops the
        scribble."""
        data = _tsv(
            # Body paragraph (high conf)
            ("ООО",           92.0, 1, 1, 1, 1),
            ("ГЕКСАФОРМ",     88.0, 1, 1, 1, 2),
            ("СПБ",           90.0, 1, 1, 1, 3),
            # Underweighted real content in rescue band
            ("7813266190",    55.0, 1, 1, 2, 1),  # ИНН — rescue
            ("12.08.2024",    58.0, 1, 1, 2, 2),  # date — rescue
            # Signature / stamp scribble — stays out
            ("нe",            22.0, 2, 1, 1, 1),
            ("Taw",           18.0, 2, 1, 1, 2),
            ("ба",            30.0, 3, 1, 1, 1),
        )
        out = reconstruct_text_from_tsv(
            data, min_confidence=60.0, soft_rescue=True,
        )
        lines = out.splitlines()
        # First line: body text
        assert lines[0] == "ООО ГЕКСАФОРМ СПБ"
        # Second line: rescued ИНН + date
        assert lines[1] == "7813266190 12.08.2024"
        # Noise stayed out
        for junk in ("нe", "Taw", "ба"):
            assert junk not in out


class TestIsLexicallyValidRescue:
    """Unit coverage for the rescue predicate itself — cheaper than
    going through the full TSV path for boundary-case sweeps."""

    def test_pure_digit_token_valid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert _is_lexically_valid_rescue("1234567890")

    def test_digit_with_separators_valid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert _is_lexically_valid_rescue("12.08.2024")
        assert _is_lexically_valid_rescue("1,200.50")
        assert _is_lexically_valid_rescue("7813-2661-90")

    def test_all_cyrillic_letters_valid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert _is_lexically_valid_rescue("ГЕКСАФОРМ")
        assert _is_lexically_valid_rescue("ООО")
        assert _is_lexically_valid_rescue("Ёлка")

    def test_all_latin_letters_valid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert _is_lexically_valid_rescue("hello")
        assert _is_lexically_valid_rescue("SCANIA")

    def test_mixed_script_invalid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert not _is_lexically_valid_rescue("Скаnia")  # cyr+lat
        assert not _is_lexically_valid_rescue("нe")       # cyr+lat

    def test_short_tokens_invalid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert not _is_lexically_valid_rescue("ба")   # 2 chars
        assert not _is_lexically_valid_rescue("a")    # 1 char
        assert not _is_lexically_valid_rescue("12")   # 2 chars digit

    def test_letter_with_punctuation_invalid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert not _is_lexically_valid_rescue("к.")
        assert not _is_lexically_valid_rescue("hello!")

    def test_empty_invalid(self) -> None:
        from src.core.confidence_filter import _is_lexically_valid_rescue

        assert not _is_lexically_valid_rescue("")
        assert not _is_lexically_valid_rescue("   ")


def test_module_exports_reconstruct_only() -> None:
    """The module deliberately exports one public name — this test is a
    tripwire: if someone adds another function without updating the
    module docstring, they'll see this test fail and get prompted to
    document the new surface."""
    import src.core.confidence_filter as cf

    assert cf.__all__ == ["reconstruct_text_from_tsv"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
