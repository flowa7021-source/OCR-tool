"""Integration test: ``OCRConfig.drop_low_conf_words`` end-to-end.

Unit coverage for the line-grouping logic of
:func:`src.core.confidence_filter.reconstruct_text_from_tsv` lives in
``tests/unit/test_confidence_filter.py``. This module asserts the flag
actually propagates through the pipeline:

    profile.ocr.drop_low_conf_words = True
        → _compute_confidences invokes the filter
        → PageResult.text is replaced with the filtered reconstruction
        → PageResult.mean_confidence is recomputed over kept words only

A control case asserts the existing behaviour (no filter) is preserved
when the flag is off — the default path for every profile that didn't
opt in stays identical byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")
pytest.importorskip("cv2")

from src.application.engines.base import OCREngine, PageOCRResult
from src.application.engines.registry import reset_cache
from src.application.pipeline import OCRPipeline
from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import (
    OCRConfig,
    OCRJobConfig,
    PreprocessConfig,
    ProfileData,
)
from src.core.text_postprocessor import TextPostprocessor
from src.infrastructure.tesseract_wrapper import TesseractWrapper
from src.shared.types import (
    BinarizationMethod,
    JobStatus,
    OCREngineKind,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEngine(OCREngine):
    """Minimal OCR engine that copies the input PDF and returns the
    supplied text verbatim.

    The pipeline's Tesseract-branch post-processing
    (:meth:`_compute_confidences`) runs regardless of which engine
    produced the text — so even a stub engine exercises the
    ``drop_low_conf_words`` code path as long as the profile's
    ``ocr.engine`` is ``TESSERACT``.
    """

    kind = OCREngineKind.TESSERACT

    def __init__(self, page_texts: list[str]) -> None:
        self._page_texts = page_texts

    @property
    def name(self) -> str:
        return "Stub(conf-filter)"

    @property
    def description(self) -> str:
        return "test stub"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
        import shutil

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preprocessed_pdf, output_pdf)
        return [
            PageOCRResult(page_number=i + 1, text=t, mean_confidence=42.0)
            for i, t in enumerate(self._page_texts)
        ]


def _build_single_page_pdf(path: Path) -> Path:
    """Synthesize a 1-page PDF the pipeline can rasterise without errors."""
    import fitz

    doc = fitz.open()
    try:
        page = doc.new_page(width=300, height=200)
        page.insert_text((50, 100), "stub page 1")
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _profile(*, drop: bool) -> ProfileData:
    """Profile configured to trigger ``_compute_confidences``.

    * ``engine=TESSERACT`` — required gate on the confidence pass
    * ``confidence_threshold=60.0`` — the filter drops anything below
    * ``drop_low_conf_words`` — the knob under test
    """
    cfg = OCRConfig(
        engine=OCREngineKind.TESSERACT,
        dpi=150,
        confidence_threshold=60.0,
        drop_low_conf_words=drop,
    )
    pre = PreprocessConfig()
    # Keep preprocessing trivial — we're not testing the image pipeline.
    pre.binarization.method = BinarizationMethod.NONE
    pre.deskew.enabled = False
    return ProfileData(
        name=f"conf-filter-test(drop={drop})",
        ocr=cfg,
        preprocess=pre,
    )


# Shared fake TSV: mimics pytesseract.image_to_data dict output for a
# page containing two high-conf body words and two low-conf stamp /
# signature artefacts. After filtering at 60.0 only "hello" + "world"
# should remain, joined into a single line.
_FAKE_TSV = {
    "text":      ["hello", "Taw",  "world", "нe"],
    "conf":      [92.0,    18.0,   90.0,    22.0],
    "block_num": [1, 1, 1, 1],
    "par_num":   [1, 1, 1, 1],
    "line_num":  [1, 1, 1, 1],
    "word_num":  [1, 2, 3, 4],
}


@pytest.fixture(autouse=True)
def _reset_engines() -> None:
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDropLowConfWords:
    """``drop_low_conf_words=True`` rebuilds text from TSV; False doesn't."""

    def _run(
        self,
        tmp_path: Path,
        *,
        drop: bool,
        stub_text: str = "ORIGINAL STUB TEXT",
    ):
        input_pdf = _build_single_page_pdf(tmp_path / "in.pdf")
        output_pdf = tmp_path / "out" / "out.pdf"
        stub = _StubEngine(page_texts=[stub_text])

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=TesseractWrapper(),
            compute_confidence=True,  # required gate in pipeline
        )

        class _FakePyTesseract:
            """Stand-in for the ``pytesseract`` module imported inside
            ``_compute_confidences``. Only the functions actually called
            need to be present; attribute errors would bubble up and
            fail the test with a clear message."""

            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(*args, **kwargs):
                return _FAKE_TSV

        # The pipeline imports pytesseract lazily inside
        # ``_compute_confidences`` (to keep it optional on hosts without
        # the library). ``patch.dict(sys.modules, ...)`` slots our fake
        # module in so the lazy import grabs it instead of the real
        # pytesseract, which would try to call a system Tesseract
        # binary that CI may not expose.
        import sys

        with (
            patch("src.application.engines.get_engine", return_value=stub),
            patch.dict(sys.modules, {"pytesseract": _FakePyTesseract()}),
        ):
            return pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_profile(drop=drop),
                )
            )

    def test_filter_on_rewrites_text_to_high_conf_words_only(
        self, tmp_path: Path
    ) -> None:
        result = self._run(tmp_path, drop=True)
        assert result.status is JobStatus.COMPLETED, result.error
        assert len(result.pages) == 1
        text = result.pages[0].text or ""
        # Only the ≥60 % words survived.
        assert "hello" in text
        assert "world" in text
        # The low-conf noise tokens are GONE — this is the user-visible
        # win: garbage stamps / signatures no longer clutter the
        # results panel or TXT/DOCX export.
        assert "Taw" not in text
        assert "нe" not in text
        # Stub-provided text was fully replaced (not appended).
        assert "ORIGINAL STUB TEXT" not in text

    def test_filter_on_recomputes_mean_conf_over_kept_words(
        self, tmp_path: Path
    ) -> None:
        """Mean confidence must reflect the words the user actually sees.
        Without recomputation the UI would report 55 % (the mean of
        92, 18, 90, 22) while the visible text is only the 92+90 pair
        that averages 91 %. That gap is exactly the user perception
        bug this flag addresses."""
        result = self._run(tmp_path, drop=True)
        # Mean of (92, 90) = 91.0; allow small float slack.
        assert result.pages[0].mean_confidence == pytest.approx(91.0, abs=0.01)

    def test_filter_off_leaves_stub_text_intact(self, tmp_path: Path) -> None:
        """Control: default profiles (flag off) keep the original
        engine-produced text and the full-set mean_confidence. Guards
        against an accidental behaviour change to every other profile
        when universal_accurate flips the switch."""
        result = self._run(tmp_path, drop=False, stub_text="UNCHANGED")
        assert result.status is JobStatus.COMPLETED, result.error
        assert "UNCHANGED" in (result.pages[0].text or "")
        # Mean of (92, 18, 90, 22) = 55.5 — NOT the 91 the filter would
        # produce. If this ever reads ~91 the flag leaked globally.
        assert result.pages[0].mean_confidence == pytest.approx(55.5, abs=0.5)

    def test_filter_on_but_no_high_conf_keeps_original_text(
        self, tmp_path: Path
    ) -> None:
        """Safety rail: an all-below-threshold page means the filter
        returns empty, and the caller must keep whatever text it had —
        blanking out a result would be a worse failure mode than
        leaving noise in for one page. We achieve this by mocking a
        TSV with ONLY low-conf words and checking the stub text
        survives."""
        import sys

        all_low = {
            "text": ["junk", "more"],
            "conf": [5.0, 10.0],
            "block_num": [1, 1],
            "par_num":   [1, 1],
            "line_num":  [1, 1],
            "word_num":  [1, 2],
        }

        class _FakePT:
            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(*args, **kwargs):
                return all_low

        input_pdf = _build_single_page_pdf(tmp_path / "in.pdf")
        out_pdf = tmp_path / "out.pdf"
        stub = _StubEngine(page_texts=["KEEP THIS"])
        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=TesseractWrapper(),
            compute_confidence=True,
        )
        with (
            patch("src.application.engines.get_engine", return_value=stub),
            patch.dict(sys.modules, {"pytesseract": _FakePT()}),
        ):
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(out_pdf),
                    profile=_profile(drop=True),
                )
            )
        assert result.status is JobStatus.COMPLETED
        # Empty reconstruction → pr.text preserved (not blanked).
        assert "KEEP THIS" in (result.pages[0].text or "")
