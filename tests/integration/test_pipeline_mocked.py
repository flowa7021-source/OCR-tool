"""Integration tests for :mod:`src.application.pipeline` with mocked OCR stack.

Real OCR (Tesseract/OCRmyPDF) cannot run in CI, so the external tools are
patched out. The test still exercises the full pipeline glue code end-to-end.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("ocrmypdf")
pytest.importorskip("fitz")

from src.application.pipeline import OCRPipeline  # noqa: E402
from src.core.models import (  # noqa: E402
    OCRJobConfig,
    PostprocessConfig,
    ProfileData,
)
from src.shared.types import JobStatus  # noqa: E402


class _StubPreprocessor:
    """ImagePreprocessor duck-type: returns image unchanged and angle=0."""

    def process(self, image: np.ndarray, cfg: Any) -> tuple[np.ndarray, float]:
        return image, 0.0


class _StubPostprocessor:
    """TextPostprocessor duck-type: returns text unchanged."""

    def process(self, text: str, cfg: PostprocessConfig) -> str:
        return text


class _StubTesseract:
    """TesseractWrapper duck-type that skips any real configuration."""

    _configured = True

    def configure_pytesseract(self) -> None:  # pragma: no cover - never called
        return None


def _fake_pixmap(width: int = 10, height: int = 10) -> MagicMock:
    pix = MagicMock()
    pix.width = width
    pix.height = height
    pix.n = 3
    pix.samples = (np.zeros((height, width, 3), dtype=np.uint8)).tobytes()
    return pix


def _fake_page(text: str = "sample text") -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    page.get_pixmap.return_value = _fake_pixmap()
    rect = MagicMock()
    page.rect = rect
    return page


def _fake_doc(pages: int = 1, text: str = "sample text") -> MagicMock:
    doc = MagicMock()
    doc.page_count = pages
    doc.load_page.side_effect = lambda i: _fake_page(text)
    doc.new_page.return_value = _fake_page(text)
    return doc


def test_pipeline_end_to_end_with_mocked_ocr(
    tmp_path: Path, fake_pdf_path: Path
) -> None:
    output_path = tmp_path / "out.pdf"

    # Track progress invocations.
    progress_calls: list[tuple[int, int, str]] = []

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_calls.append((current, total, stage))

    pipeline = OCRPipeline(
        preprocessor=_StubPreprocessor(),
        postprocessor=_StubPostprocessor(),
        tesseract=_StubTesseract(),
        progress_callback=on_progress,
        compute_confidence=False,
    )

    profile = ProfileData(name="test_profile")
    job = OCRJobConfig(
        input_path=str(fake_pdf_path),
        output_path=str(output_path),
        profile=profile,
    )

    def fake_ocrmypdf_run(options: Any) -> None:
        # Simulate OCRmyPDF by simply copying input → output.
        shutil.copy2(str(options.input_file), str(options.output_file))

    def fake_imwrite(path: str, img: Any) -> bool:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")
        return True

    with patch("src.application.pipeline.fitz") as fitz_mod, \
         patch("src.application.pipeline.run_ocrmypdf", side_effect=fake_ocrmypdf_run), \
         patch("src.application.pipeline.cv2") as cv2_mod:
        fitz_mod.open.side_effect = lambda *a, **kw: _fake_doc(pages=1)
        fitz_mod.Pixmap.side_effect = lambda *a, **kw: _fake_pixmap()
        cv2_mod.imwrite.side_effect = fake_imwrite
        cv2_mod.COLOR_RGB2BGR = 0
        cv2_mod.COLOR_RGBA2BGR = 0
        cv2_mod.cvtColor.side_effect = lambda arr, code: arr

        result = pipeline.run(job)

    assert result.status == JobStatus.COMPLETED
    assert result.page_count >= 1
    assert result.error is None
    # Progress callback must have been invoked at least once.
    assert progress_calls, "progress_callback was never invoked"
    # At least one call reports the 'ocr' stage.
    assert any(stage == "ocr" for _, _, stage in progress_calls)


def test_pipeline_failure_surface_as_job_result(
    tmp_path: Path, fake_pdf_path: Path
) -> None:
    """If OCRmyPDF errors out, the pipeline returns a FAILED JobResult."""
    from src.application.ocrmypdf_integration import OCRmyPDFError

    output_path = tmp_path / "out.pdf"
    pipeline = OCRPipeline(
        preprocessor=_StubPreprocessor(),
        postprocessor=_StubPostprocessor(),
        tesseract=_StubTesseract(),
        compute_confidence=False,
    )

    profile = ProfileData(name="test_profile")
    job = OCRJobConfig(
        input_path=str(fake_pdf_path),
        output_path=str(output_path),
        profile=profile,
    )

    def fake_imwrite(path: str, img: Any) -> bool:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")
        return True

    with patch("src.application.pipeline.fitz") as fitz_mod, \
         patch(
             "src.application.pipeline.run_ocrmypdf",
             side_effect=OCRmyPDFError("boom", exit_code=2),
         ), \
         patch("src.application.pipeline.cv2") as cv2_mod:
        fitz_mod.open.side_effect = lambda *a, **kw: _fake_doc(pages=1)
        fitz_mod.Pixmap.side_effect = lambda *a, **kw: _fake_pixmap()
        cv2_mod.imwrite.side_effect = fake_imwrite
        cv2_mod.COLOR_RGB2BGR = 0
        cv2_mod.COLOR_RGBA2BGR = 0
        cv2_mod.cvtColor.side_effect = lambda arr, code: arr

        result = pipeline.run(job)

    assert result.status == JobStatus.FAILED
    assert result.error is not None and "boom" in result.error
