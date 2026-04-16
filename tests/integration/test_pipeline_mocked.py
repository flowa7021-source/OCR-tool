"""Integration tests for :mod:`src.application.pipeline` with mocked OCR.

Real OCR (Tesseract/OCRmyPDF) is stubbed at the *engine* level rather
than by mocking ``fitz``/``cv2`` at the module level. Two reasons:

1. ``pipeline.py`` imports ``fitz`` and ``cv2`` lazily inside methods,
   not at module scope — ``patch("src.application.pipeline.fitz")``
   throws ``AttributeError: module has no attribute 'fitz'``.
2. Mocking those libraries wholesale misses real bugs in our own
   pipeline glue (the ``cv2.imwrite``-on-Unicode-path bug was in the
   glue, the mocks would've hidden it).

Instead we build a tiny real PDF with PyMuPDF, run the real pipeline
(real fitz, real cv2, real preprocessor, real postprocessor), and
only substitute the OCR engine with a stub. The stub mimics an OCR
engine that copies the preprocessed PDF to the output path and
returns a fixed text per page.

The autouse ``_bypass_external_tools_preflight`` fixture in
``tests/conftest.py`` keeps the pre-flight check from short-circuiting
when the CI agent lacks Tesseract/Ghostscript.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("ocrmypdf")
pytest.importorskip("fitz")
pytest.importorskip("cv2")

from src.application.engines.base import OCREngine, PageOCRResult  # noqa: E402
from src.application.engines.registry import reset_cache  # noqa: E402
from src.application.ocrmypdf_integration import OCRmyPDFError  # noqa: E402
from src.application.pipeline import OCRPipeline  # noqa: E402
from src.core.image_preprocessor import ImagePreprocessor  # noqa: E402
from src.core.models import (  # noqa: E402
    OCRConfig,
    OCRJobConfig,
    PreprocessConfig,
    ProfileData,
)
from src.core.text_postprocessor import TextPostprocessor  # noqa: E402
from src.infrastructure.tesseract_wrapper import TesseractWrapper  # noqa: E402
from src.shared.types import (  # noqa: E402
    BinarizationMethod,
    JobStatus,
    OCREngineKind,
)


def _build_test_pdf(path: Path, page_count: int = 1) -> Path:
    """Synthesize a tiny multi-page PDF with PyMuPDF."""
    import fitz

    doc = fitz.open()
    try:
        for i in range(page_count):
            page = doc.new_page(width=300, height=200)
            page.insert_text((50, 100), f"Page {i + 1} content")
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _fast_profile() -> ProfileData:
    """Profile tuned for fast, deterministic pipeline runs in tests."""
    cfg = OCRConfig(engine=OCREngineKind.TESSERACT, dpi=150)
    pre = PreprocessConfig()
    pre.binarization.method = BinarizationMethod.NONE
    pre.deskew.enabled = False
    return ProfileData(name="pipeline-test", ocr=cfg, preprocess=pre)


class _CopyEngine(OCREngine):
    """OCR engine that copies the preprocessed PDF to the output path.

    Stands in for real OCRmyPDF: our pipeline expects the engine to
    write a "searchable" PDF to ``output_pdf``. Copying keeps the file
    round-trip valid (pipeline's post-extract step opens it with fitz)
    while staying orders of magnitude faster than real Tesseract.
    """

    kind = OCREngineKind.TESSERACT

    def __init__(self, raise_error: Exception | None = None) -> None:
        self.raise_error = raise_error
        self.run_called = 0

    @property
    def name(self) -> str:
        return "copy-stub"

    @property
    def description(self) -> str:
        return "for tests"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
        self.run_called += 1
        if self.raise_error is not None:
            raise self.raise_error
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preprocessed_pdf, output_pdf)
        if progress_callback is not None:
            progress_callback(0, 1, "ocr")
            progress_callback(1, 1, "ocr")
        import fitz

        with fitz.open(str(output_pdf)) as doc:
            return [
                PageOCRResult(page_number=i + 1, text=f"stub page {i + 1}", mean_confidence=85.0)
                for i in range(doc.page_count)
            ]


@pytest.fixture(autouse=True)
def _reset_engines() -> None:
    reset_cache()
    yield
    reset_cache()


def _make_pipeline() -> OCRPipeline:
    return OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
    )


def test_pipeline_end_to_end_with_mocked_ocr(tmp_path: Path) -> None:
    """Full pipeline: real preprocess + real PyMuPDF + stubbed OCR.

    Asserts the end-to-end glue (analyze → preprocess → assemble →
    OCR → postprocess) actually wires together. Stubs only the OCR
    engine; everything else is the real implementation.
    """
    input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=2)
    output_pdf = tmp_path / "out" / "input_ocr.pdf"

    progress_calls: list[tuple[int, int, str]] = []

    def on_progress(current: int, total: int, stage: str) -> None:
        progress_calls.append((current, total, stage))

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
        progress_callback=on_progress,
        compute_confidence=False,
    )

    stub = _CopyEngine()
    with patch("src.application.engines.get_engine", return_value=stub):
        job = OCRJobConfig(
            input_path=str(input_pdf),
            output_path=str(output_pdf),
            profile=_fast_profile(),
        )
        result = pipeline.run(job)

    assert result.status is JobStatus.COMPLETED, result.error
    assert result.error is None
    assert output_pdf.exists()
    assert result.page_count == 2
    assert stub.run_called == 1
    # Progress fired for every major stage.
    stages = {s for _, _, s in progress_calls}
    assert "analyze" in stages
    assert "preprocess" in stages
    assert "assemble" in stages
    assert "ocr" in stages
    assert "postprocess" in stages


def test_pipeline_failure_surface_as_job_result(tmp_path: Path) -> None:
    """If the OCR engine raises, the pipeline returns a FAILED JobResult.

    Specifically tests the OCRmyPDFError path — our engine wrapper
    catches ocrmypdf exceptions and re-raises as OCRmyPDFError, which
    pipeline.run handles as a job-level failure (not a crash).
    """
    input_pdf = _build_test_pdf(tmp_path / "in.pdf", page_count=1)
    output_pdf = tmp_path / "out.pdf"

    pipeline = _make_pipeline()
    stub = _CopyEngine(raise_error=OCRmyPDFError("boom", exit_code=2))

    with patch("src.application.engines.get_engine", return_value=stub):
        job = OCRJobConfig(
            input_path=str(input_pdf),
            output_path=str(output_pdf),
            profile=_fast_profile(),
        )
        result = pipeline.run(job)

    assert result.status is JobStatus.FAILED
    assert result.error is not None
    assert "boom" in result.error
    # Engine was invoked but didn't produce output (as expected on error).
    assert stub.run_called == 1
    assert not output_pdf.exists()


def test_pipeline_corrupt_pdf_surfaces_typed_error(tmp_path: Path) -> None:
    """A corrupt input yields JobStatus.FAILED with CorruptPdfError text.

    Stubs the engine so the NEW pre-flight stage doesn't reject the job
    earlier on a CI runner without a system Tesseract install — the
    test's purpose is to exercise the ``_analyze_pdf`` failure path, not
    the engine-availability probe (which is covered in its own tests).
    """
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is not a PDF")

    pipeline = _make_pipeline()
    job = OCRJobConfig(
        input_path=str(bad),
        output_path=str(tmp_path / "out.pdf"),
        profile=_fast_profile(),
    )
    with patch("src.application.engines.get_engine", return_value=_CopyEngine()):
        result = pipeline.run(job)
    assert result.status is JobStatus.FAILED
    assert result.error is not None
    assert "повреждён" in result.error.lower() or "формат" in result.error.lower()


def test_pipeline_progress_callback_survives_handler_exceptions(
    tmp_path: Path,
) -> None:
    """Progress callback errors must not abort the job."""
    input_pdf = _build_test_pdf(tmp_path / "in.pdf", page_count=1)
    output_pdf = tmp_path / "out.pdf"

    calls = []

    def bad_progress(current, total, stage):
        calls.append(stage)
        if stage == "preprocess":
            raise RuntimeError("progress handler went boom")

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
        progress_callback=bad_progress,
        compute_confidence=False,
    )

    with patch(
        "src.application.engines.get_engine", return_value=_CopyEngine()
    ):
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_fast_profile(),
            )
        )

    # Pipeline completes despite the raising handler.
    assert result.status is JobStatus.COMPLETED, result.error
    assert "preprocess" in calls
