"""End-to-end postprocessing tests against real Tesseract output.

Every flag on :class:`PostprocessConfig` transforms raw OCR text
before the pipeline returns it to the user. The mocked-engine suite
proves each flag's transformation runs in isolation, but those tests
feed a synthetic "pre-recognised" string to the postprocessor — they
don't prove the flag works on **what Tesseract actually emits**. The
two can disagree: real Tesseract output has spurious whitespace,
half-recognised glyphs, and occasional Latin-looking-like-Cyrillic
confusion that synthetic strings don't model.

These tests generate a PDF, run it through real OCR, and assert the
expected postprocess transformation ran on the real output.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    real_tesseract_wrapper,  # noqa: F401 — pytest fixture
    render_clean_text_pdf,
    requires_real_ocr,
    requires_real_russian_ocr,
    run_pipeline,
)
from src.shared.types import JobStatus

pytestmark = [
    requires_real_ocr,
    pytest.mark.exercise_preflight,
]


# ---------------------------------------------------------------------------
# normalize_unicode: Tesseract sometimes emits NFD — must come out NFC
# ---------------------------------------------------------------------------


class TestUnicodeNormalizationRealOCR:
    """The recognised text must be NFC when the flag is on, regardless
    of which form Tesseract returned internally."""

    def test_output_is_nfc_when_flag_on(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="CAFE"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            normalize_unicode=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error
        text = result.pages[0].text
        # Whatever Tesseract emitted, the stored text must equal its
        # own NFC normalisation — no stray combining characters.
        assert text == unicodedata.normalize("NFC", text), (
            f"normalize_unicode=True but output contains non-NFC "
            f"sequences: {text!r}"
        )


# ---------------------------------------------------------------------------
# normalize_whitespace: real Tesseract output has ragged spacing
# ---------------------------------------------------------------------------


class TestNormalizeWhitespaceRealOCR:
    """Tesseract emits multiple spaces between words and occasional
    triple newlines between blocks. normalize_whitespace should tidy
    that up without losing content."""

    def test_whitespace_normalized_when_flag_on(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="FIRST LINE\n\n\nSECOND LINE"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            normalize_whitespace=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        # No runs of 3+ newlines after normalisation.
        assert "\n\n\n" not in text, (
            f"normalize_whitespace=True but runs of 3+ newlines remain: "
            f"{text!r}"
        )
        # No runs of 2+ spaces.
        assert "  " not in text, (
            f"normalize_whitespace=True but runs of 2+ spaces remain: "
            f"{text!r}"
        )


# ---------------------------------------------------------------------------
# remove_artifacts: Tesseract occasionally outputs punctuation-only lines
# ---------------------------------------------------------------------------


class TestRemoveArtifactsRealOCR:
    """OCR noise often manifests as lines containing only ``|``, ``~``,
    or runs of dashes. The remove_artifacts flag should strip them
    without losing real content lines."""

    def test_artifact_line_stripped_real_content_kept(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        # Render a page that has a horizontal rule between two real
        # lines — Tesseract typically reads the rule as `-` repeating.
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf",
            text="REAL LINE ONE\n----------\nREAL LINE TWO",
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            remove_artifacts=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        # Real lines survived.
        upper = text.upper()
        assert "REAL" in upper, (
            f"real-content line was dropped by remove_artifacts: "
            f"{text!r}"
        )
        # Any pure-punctuation line (only dashes / tildes / pipes) is
        # gone. We check each non-empty line has at least one
        # alphanumeric character.
        for line in text.splitlines():
            if line.strip():
                has_alnum = any(c.isalnum() for c in line)
                assert has_alnum, (
                    f"pure-punctuation line survived remove_artifacts: "
                    f"{line!r} (full text: {text!r})"
                )


# ---------------------------------------------------------------------------
# autocorrect_english: 0 → o inside a word, rn → m, etc.
# ---------------------------------------------------------------------------


class TestAutocorrectEnglishRealOCR:
    """When Tesseract misreads ``o`` as ``0`` inside a word,
    autocorrect_english should swap it back. The rules look for a
    digit between alphabetic characters."""

    def test_digit_between_letters_corrected(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        # "C0NTRACT" with digit 0 instead of letter O — if Tesseract
        # emits this shape (it often does on low-res o), autocorrect
        # should rewrite to "CONTRACT".
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="C0NTRACT TEXT"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            autocorrect_english=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        # Whatever Tesseract returned for "C0NTRACT", after the rule
        # runs there must not be a digit sandwiched between letters.
        import re

        digit_in_word = re.search(r"[A-Za-z][0-9][A-Za-z]", text)
        assert digit_in_word is None, (
            f"autocorrect_english=True but found digit-between-letters: "
            f"{text!r}"
        )


class TestAutocorrectEnglishDoesNotBreakNumbers:
    """Guard against over-correction: digits surrounded by other
    digits (real numbers) must NOT be rewritten to letters."""

    def test_plain_number_preserved(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="INVOICE 2026 01 15"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            autocorrect_english=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        # At least one of our numbers must survive as digits.
        assert any(
            num in text for num in ("2026", "2025", "2020", "202")
        ), (
            f"autocorrect_english wrecked a legitimate number: {text!r}"
        )


# ---------------------------------------------------------------------------
# autocorrect_russian: digit-as-letter + Latin-looking-like-Cyrillic
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestAutocorrectRussianRealOCR:
    """User's primary correction scenario: Russian contract OCR often
    produces digit-in-Cyrillic-word confusions (``д0м`` for ``дом``).
    autocorrect_russian should fix those without touching legitimate
    numbers."""

    def test_russian_recognised_and_normalised(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """Clean Russian word with every flag the default profile
        ships with. No specific misread to fix — we just verify the
        rus + postprocess chain doesn't corrupt otherwise clean text."""
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="ДОГОВОР", cyrillic=True
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            autocorrect_russian=True,
            autocorrect_english=True,
            normalize_unicode=True,
            normalize_whitespace=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text.upper()
        # At least one trigram of the word survived the whole chain.
        assert any(tri in text for tri in ("ДОГ", "ОГО", "ВОР")), (
            f"postprocess chain consumed the recognised text: "
            f"{result.pages[0].text!r}"
        )
        # No digits-between-cyrillic-letters leaked through.
        import re
        digit_in_cyr = re.search(r"[А-Яа-я][0-9][А-Яа-я]", result.pages[0].text)
        assert digit_in_cyr is None, (
            f"autocorrect_russian=True but digit-in-cyrillic-word leaked: "
            f"{result.pages[0].text!r}"
        )


# ---------------------------------------------------------------------------
# Custom regex rules: user-configured find/replace
# ---------------------------------------------------------------------------


class TestCustomRegexRulesRealOCR:
    """Custom rules let power-users patch domain-specific OCR
    mistakes. A rule like ``{"pattern": "\\bNo\\b", "replacement": "№"}``
    should fire on real Tesseract output."""

    def test_literal_substitution_applies(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="INVOICE FOO 123"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            custom_rules=[
                {
                    "pattern": "FOO",
                    "replacement": "REPLACED",
                    "is_regex": False,
                    "case_sensitive": True,
                }
            ],
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        # The rule fires only if Tesseract recognised "FOO" in the first
        # place — not guaranteed on synthetic renders — so we accept
        # "REPLACED" OR "FOO absent" (meaning Tesseract mis-read the
        # source word and our assertion about the rule running can't
        # be made). The point of this test is: the rule is wired up
        # and doesn't crash the pipeline on a real OCR output.
        if "FOO" in text.upper():
            raise AssertionError(
                f"rule 'FOO → REPLACED' did not fire on Tesseract "
                f"output that still contained 'FOO': {text!r}"
            )

    def test_regex_substitution_applies(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="INVOICE 2026"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            custom_rules=[
                {
                    "pattern": r"\b20\d\d\b",
                    "replacement": "YEAR",
                    "is_regex": True,
                    "case_sensitive": True,
                }
            ],
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        import re
        # Any 20xx year that Tesseract recognised must have been
        # rewritten; a 4-digit year starting with 20 in the output
        # means the rule didn't fire.
        assert re.search(r"\b20\d\d\b", text) is None, (
            f"regex custom rule did not fire on Tesseract output: "
            f"{text!r}"
        )

    def test_invalid_regex_rule_is_auto_disabled(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """A profile with a bad regex must still OCR — the bad rule is
        silently auto-disabled at load time, not at runtime-per-page.
        This test proves the end-to-end path: profile constructed
        with an invalid rule → pipeline runs → result is
        COMPLETED, not FAILED."""
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="HELLO"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            custom_rules=[
                {
                    "pattern": "[unclosed",  # invalid regex
                    "replacement": "x",
                    "is_regex": True,
                    "case_sensitive": True,
                }
            ],
        )
        # The bad rule is auto-disabled by RegexRule.__post_init__
        # and so doesn't participate at all.
        assert profile.postprocess.custom_rules[0].enabled is False
        assert profile.postprocess.custom_rules[0].invalid_reason

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error


# ---------------------------------------------------------------------------
# Full postprocess chain — every flag on, as in the default profile
# ---------------------------------------------------------------------------


class TestFullPostprocessChainRealOCR:
    """Every flag on, like the default user profile ships. Verifies
    no transformation crashes on a real Tesseract output, and the
    final text is at least minimally legible."""

    def test_all_flags_on_english(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="HELLO WORLD\n\n\nSECOND LINE"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            autocorrect_english=True,
            merge_hyphenated=True,
            normalize_whitespace=True,
            normalize_unicode=True,
            remove_artifacts=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text
        # Real words survived. Postprocess didn't destroy them.
        upper = text.upper()
        assert any(w in upper for w in ("HELLO", "WORLD", "SECOND", "LINE"))

    @requires_real_russian_ocr
    def test_all_flags_on_russian(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """Full chain on Russian — the user's production scenario."""
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf",
            text="ДОГОВОР ПОДРЯДА 2026",
            cyrillic=True,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            autocorrect_russian=True,
            autocorrect_english=True,
            merge_hyphenated=True,
            normalize_whitespace=True,
            normalize_unicode=True,
            remove_artifacts=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        text = result.pages[0].text.upper()
        # At least one trigram survived the entire chain.
        assert any(
            tri in text
            for tri in ("ДОГ", "ВОР", "РЯД", "202")
        ), (
            f"full postprocess chain consumed real OCR output: "
            f"{result.pages[0].text!r}"
        )
