"""Integration test: Step C's PDF text-layer filter reaches the pipeline.

Unit coverage for :func:`src.core.pdf_text_filter.filter_pdf_text_layer`
lives in ``tests/unit/test_pdf_text_filter.py``. This module asserts
the pipeline wiring — that when a profile opts into
``drop_low_conf_words`` and ``_compute_confidences`` observes a low-
confidence word on the preprocessed image, that word is ALSO stripped
from the output PDF's invisible text layer.

This is the piece that actually closes the "Ctrl-F in the output
PDF still finds garbage" gap the user reported.
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


class _EngineThatStampsTextLayer(OCREngine):
    """OCR engine that produces a PDF whose text layer simulates what
    OCRmyPDF would stamp — including both high-conf body text AND
    low-conf noise, so the filter has something to strip.

    Skips actual Tesseract; we don't need it for this test — we only
    want to verify the filter's text-layer edit reaches the final PDF.
    """

    kind = OCREngineKind.TESSERACT

    def __init__(self, words_per_page: list[list[tuple[str, float, float]]]) -> None:
        self._words_per_page = words_per_page

    @property
    def name(self) -> str:
        return "Stub(text-layer)"

    @property
    def description(self) -> str:
        return "test stub"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
        import fitz

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        try:
            for page_words in self._words_per_page:
                page = doc.new_page(width=612, height=792)
                for word, x, y in page_words:
                    page.insert_text((x, y), word, fontsize=12)
            doc.save(str(output_pdf))
        finally:
            doc.close()
        # Minimal PageOCRResult per page — pipeline doesn't read these
        # strings, it re-extracts via ``page.get_text("text")`` later.
        return [
            PageOCRResult(page_number=i + 1, text="", mean_confidence=50.0)
            for i in range(len(self._words_per_page))
        ]


def _build_single_page_input_pdf(path: Path) -> Path:
    """Minimal PDF the pipeline will rasterise + preprocess. Its actual
    contents don't matter — the stub engine above replaces the output
    unconditionally."""
    import fitz

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 100), "input placeholder", fontsize=12)
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _profile(*, drop: bool) -> ProfileData:
    cfg = OCRConfig(
        engine=OCREngineKind.TESSERACT,
        dpi=150,
        confidence_threshold=60.0,
        drop_low_conf_words=drop,
    )
    pre = PreprocessConfig()
    pre.binarization.method = BinarizationMethod.NONE
    pre.deskew.enabled = False
    return ProfileData(name=f"conf-pdf-filter(drop={drop})", ocr=cfg, preprocess=pre)


_FAKE_TSV = {
    "text":   ["keep", "trash", "also_keep"],
    "conf":   [95.0,    15.0,   92.0],
    # Pixel coords in a 612×792-pixel image so the pt⇄px transform
    # inside the filter becomes the identity. The three bboxes line
    # up with the three words the stub engine stamps below.
    "left":   [50,  200,  350],
    "top":    [90,  90,   90],
    "width":  [40,  40,   60],
    "height": [15,  15,   15],
    # Block/par/line grouping for the results-panel reconstruction.
    # The PDF filter itself doesn't use these columns.
    "block_num": [1, 1, 1],
    "par_num":   [1, 1, 1],
    "line_num":  [1, 1, 1],
    "word_num":  [1, 2, 3],
}


@pytest.fixture(autouse=True)
def _reset_engines() -> None:
    reset_cache()
    yield
    reset_cache()


def _run(tmp_path: Path, *, drop: bool) -> Path:
    """Helper: run the pipeline with the stub engine + fake TSV and
    return the output PDF path so the caller can inspect its text
    layer."""
    input_pdf = _build_single_page_input_pdf(tmp_path / "in.pdf")
    output_pdf = tmp_path / "out" / "out.pdf"

    # Stub engine stamps the exact three words whose bboxes the fake
    # TSV references — the filter should keep "keep"/"also_keep" and
    # redact "trash".
    stub = _EngineThatStampsTextLayer(
        words_per_page=[
            [
                ("keep",      50.0,  100.0),
                ("trash",     200.0, 100.0),
                ("also_keep", 350.0, 100.0),
            ]
        ]
    )

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
        compute_confidence=True,
    )

    # Fake pytesseract: returns our fixed TSV. Also stubs image size
    # so the filter's pixel→point transform is the identity. See the
    # sibling ``test_word_conf_filter.py`` module for why we patch via
    # ``sys.modules`` (pipeline imports lazily).
    class _FakePyTesseract:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(*args, **kwargs):
            return _FAKE_TSV

    # ``_compute_confidences`` calls ``cv2.imdecode`` on the
    # preprocessed PNGs; we patch only that ONE attribute so the
    # returned array has the expected 612×792 shape (identity
    # pt⇄px transform with the PDF pages). The rest of the pipeline
    # still uses the real cv2 for rasterisation and preprocessing.
    import sys

    import cv2 as real_cv2
    import numpy as np

    def _fake_imdecode(*args, **kwargs):
        # 612×792 grayscale array: shape (H, W) = (792, 612). The
        # filter reads img.shape[1] for width → 612, img.shape[0] for
        # height → 792 → identity pixel⇄point transform with the PDF.
        return np.zeros((792, 612), dtype=np.uint8)

    with (
        patch("src.application.engines.get_engine", return_value=stub),
        patch.dict(sys.modules, {"pytesseract": _FakePyTesseract()}),
        patch.object(real_cv2, "imdecode", _fake_imdecode),
    ):
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_profile(drop=drop),
            )
        )
    assert result.status is JobStatus.COMPLETED, result.error
    return output_pdf


def _pdf_text(path: Path) -> str:
    import fitz

    with fitz.open(str(path)) as doc:
        return doc.load_page(0).get_text("text") or ""


class TestStepCReachesPdfTextLayer:
    """``drop_low_conf_words=True`` removes low-conf words from the
    PDF's invisible text layer, not just the results panel."""

    def test_low_conf_word_stripped_from_pdf(self, tmp_path: Path) -> None:
        out_pdf = _run(tmp_path, drop=True)
        text = _pdf_text(out_pdf)
        # High-conf body words survive — they're what the user
        # downstream-consumer wants to find via Ctrl-F.
        assert "keep" in text
        assert "also_keep" in text
        # Low-conf noise word is GONE from the PDF's text layer.
        # Without Step C this assertion would fail: Ctrl-F in Adobe
        # Reader still finds "trash" even though the results panel
        # doesn't show it.
        assert "trash" not in text, (
            "Step C regression: low-conf word leaked into the PDF "
            f"text layer. Extracted: {text!r}"
        )

    def test_filter_off_leaves_pdf_text_layer_intact(
        self, tmp_path: Path
    ) -> None:
        """Control: when the profile hasn't opted in, the PDF text
        layer is the un-modified stub output. Guards against the
        filter accidentally running for every profile."""
        out_pdf = _run(tmp_path, drop=False)
        text = _pdf_text(out_pdf)
        assert "keep" in text
        assert "also_keep" in text
        # Noise stays — no filter applied.
        assert "trash" in text, (
            "Step C flag leaked: low-conf word was stripped even "
            "though the profile didn't opt in."
        )
