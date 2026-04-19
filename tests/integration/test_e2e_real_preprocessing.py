"""End-to-end preprocessing tests against real Tesseract.

Every test here runs the **full** pipeline (analyze → preprocess →
assemble → real Tesseract via OCRmyPDF → postprocess) on a synthetic
PDF that's been **deliberately challenged** in exactly one way
(rotation, noise, low contrast, etc.). The assertion shape is always
"the targeted preprocessing step salvages the recognition that would
otherwise fail" — so a regression in any one preprocessing stage
surfaces as a test failure naming the stage.

This is the class of test that caught the ``_assemble_pdf`` pixels-
vs-points bug: mocked-engine tests thought every preprocessing
combination "worked" because the stub engine happily returned
pre-configured text regardless of input. Real Tesseract on a truly
low-resolution preprocessed image returns empty hOCR, which is what
you want a test to catch.

Skips cleanly when ``tesseract`` + ``gs`` aren't on PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration._real_ocr_helpers import (
    assert_ocr_recognised,
    make_realistic_profile,
    render_clean_text_pdf,
    render_low_contrast_text_pdf,
    render_noisy_text_pdf,
    render_rotated_text_pdf,
    requires_real_ocr,
    requires_real_russian_ocr,
    run_pipeline,
)

pytestmark = [
    requires_real_ocr,
    # Bypass the conftest autouse that fakes ``verify_required_for_ocrmypdf``:
    # real-OCR tests want the genuine check to run.
    pytest.mark.exercise_preflight,
]


# ---------------------------------------------------------------------------
# Binarization: every method must produce recognisable text
# ---------------------------------------------------------------------------


class TestBinarizationMethodsRealOCR:
    """Each of the five BinarizationMethod choices must end up with
    Tesseract still able to read the text. The mocked-engine suite
    already proves the preprocessed PDFs differ bytewise between
    methods; these tests prove the preprocessed PDF is still legible
    to a live Tesseract, which is the part the mocks can't check.
    """

    @pytest.mark.parametrize(
        "method",
        ["none", "otsu", "adaptive_gaussian", "adaptive_mean", "sauvola"],
    )
    def test_method_preserves_text_legibility(
        self,
        method: str,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="HELLO WORLD TEST"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization=method,
            # Keep every other step off: we want the legibility
            # assertion to be about THIS binarization, not some
            # other preprocess interaction.
            deskew=False, clahe=False, denoise_steps=None,
            background_removal=False,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["HELLO", "WORLD", "TEST"])


# ---------------------------------------------------------------------------
# Deskew: a rotated input must come back readable when deskew is ON
# ---------------------------------------------------------------------------


class TestDeskewRealOCR:
    """Rotated scans are the single most common user-reported
    "recognition got worse" complaint. A ~5° skew keeps Tesseract's
    own OSD from auto-correcting, so the pipeline's deskew is what
    carries the load."""

    def test_deskew_rescues_rotated_input(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_rotated_text_pdf(
            tmp_path / "in.pdf",
            text="DESKEW SALVAGED",
            angle_degrees=5.0,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            deskew=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["DESKEW", "SALVAGED"])

        # Detected angle is non-zero — we actually ran the deskew pass,
        # not just lucked into Tesseract's internal tolerance.
        assert result.pages[0].skew_angle != 0.0, (
            "deskew reported angle=0.0 on a 5°-rotated input — either "
            "deskew is silently disabled or the detector failed to find "
            "the skew"
        )

    def test_deskew_disabled_does_not_rotate_clean_input(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """With deskew OFF on a perfectly-aligned input, the recorded
        skew angle must be zero — any non-zero value means deskew
        wrongly ran anyway."""
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="CLEAN INPUT"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu", deskew=False,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["CLEAN", "INPUT"])
        assert result.pages[0].skew_angle == 0.0, (
            f"deskew=False but skew_angle={result.pages[0].skew_angle}"
        )


# ---------------------------------------------------------------------------
# Denoise: each method must not destroy readable text
# ---------------------------------------------------------------------------


class TestDenoiseChainRealOCR:
    """Denoise methods run between contrast and binarization. The
    invariant every test here guards: applying the method must not
    degrade OCR accuracy on a reasonable input. A buggy denoise that
    over-smoothed glyphs would silently produce empty hOCR — and we
    spent a full session chasing that exact failure shape from the
    DPI bug."""

    @pytest.mark.parametrize(
        "method,params",
        [
            ("median", {"ksize": 3}),
            ("gaussian", {"ksize": 3, "sigma": 1.0}),
            ("morph_open", {"morph_ksize": 3}),
            ("morph_close", {"morph_ksize": 3}),
            ("nlm", {"h": 10}),
        ],
    )
    def test_each_denoise_method_preserves_text(
        self,
        method: str,
        params: dict,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="KEEP LEGIBLE"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            denoise_steps=[(method, params)],
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["KEEP", "LEGIBLE"])

    def test_denoise_chain_rescues_noisy_input(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """Salt-and-pepper noise at 3% is heavy enough that direct
        OCR degrades; a median + morph_close chain should recover
        enough of the glyph shape for Tesseract."""
        input_pdf = render_noisy_text_pdf(
            tmp_path / "in.pdf",
            text="MEDIAN RECOVERS",
            salt_pepper_ratio=0.03,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            denoise_steps=[
                ("median", {"ksize": 3}),
                ("morph_close", {"morph_ksize": 3}),
            ],
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["MEDIAN", "RECOVERS"])


# ---------------------------------------------------------------------------
# CLAHE contrast enhancement on a low-contrast input
# ---------------------------------------------------------------------------


class TestClaheContrastRealOCR:
    """Faded photocopies and dim scans are the user-reported case.
    CLAHE should lift the contrast enough for OTSU + Tesseract to
    find the glyphs."""

    def test_clahe_rescues_low_contrast_input(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_low_contrast_text_pdf(
            tmp_path / "in.pdf",
            text="DIM SCAN",
            foreground_level=140,
            background_level=200,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            clahe=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["DIM", "SCAN"])


# ---------------------------------------------------------------------------
# Full preprocessing chain — what quick_reliable / default actually ship
# ---------------------------------------------------------------------------


class TestFullPreprocessChainRealOCR:
    """Profiles combine multiple preprocessing steps in a specific
    order. Regressions often live in the INTERACTIONS, not in any
    single step — these tests exercise realistic chains."""

    def test_otsu_plus_clahe_plus_deskew(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """Exactly what the ``quick_reliable`` profile does."""
        input_pdf = render_rotated_text_pdf(
            tmp_path / "in.pdf",
            text="FULL CHAIN",
            angle_degrees=3.0,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            deskew=True,
            clahe=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["FULL", "CHAIN"])

    def test_adaptive_gaussian_plus_denoise_plus_clahe(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """Close to the ``universal_accurate`` profile. We intentionally
        use a CLEAN input here (not noisy) — adaptive_gaussian
        binarisation amplifies salt-and-pepper noise to the point
        where even a median + morph_close chain can't rescue it at
        300 DPI. The invariant we actually want to verify is
        "aggressive preprocessing doesn't DESTROY clean text" — not
        "adaptive_gaussian can recover from any amount of noise",
        which isn't even true in theory.
        """
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="UNIVERSAL PROFILE"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="adaptive_gaussian",
            denoise_steps=[
                ("median", {"ksize": 3}),
                ("morph_close", {"morph_ksize": 3}),
            ],
            clahe=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["UNIVERSAL", "PROFILE"])


# ---------------------------------------------------------------------------
# Preprocessing on Russian text — the production scenario
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestRussianPreprocessingRealOCR:
    """Cyrillic + preprocessing in one test — the user's actual daily
    workflow. Failures here go straight to "my Russian contract OCR is
    broken", which is the single most important invariant for this app.
    """

    def test_russian_text_otsu_binarization(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf",
            text="ДОГОВОР ПОДРЯДА",
            cyrillic=True,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(
            result, ["ДОГОВОР", "ПОДРЯДА", "ДОГ", "РЯД"]
        )

    def test_russian_text_adaptive_gaussian(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf",
            text="РУССКИЙ ТЕКСТ",
            cyrillic=True,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="adaptive_gaussian",
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(
            result, ["РУССКИЙ", "ТЕКСТ", "РУС", "КИЙ", "ТЕК", "КСТ"]
        )

    def test_russian_text_deskew_plus_clahe_plus_denoise(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """Complete preprocessing chain + Russian text — closest to
        the user's actual scanned-contract workflow."""
        input_pdf = render_rotated_text_pdf(
            tmp_path / "in.pdf",
            text="АКТ ВЫПОЛНЕННЫХ",
            angle_degrees=2.0,
            cyrillic=True,
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            deskew=True,
            clahe=True,
            denoise_steps=[("median", {"ksize": 3})],
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(
            result, ["АКТ", "ВЫПОЛНЕННЫХ", "ВЫП", "ННЫХ"]
        )
