"""Pipeline orchestration tests with a fake OCR engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz

from src.application.engines.base import OCREngine, PageOCRResult
from src.application.pipeline import OCRPipeline, _word_boxes_to_tsv
from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import (
    JobResult,
    OCRConfig,
    OCRJobConfig,
    PostprocessConfig,
    PreprocessConfig,
    ProfileData,
)
from src.core.text_postprocessor import TextPostprocessor
from src.shared.types import JobStatus, OCREngineKind


class FakeEngine(OCREngine):
    kind = OCREngineKind.EASYOCR
    name = "fake"
    description = "test stub"

    def is_available(self):
        return True, "ok"

    def run(self, preprocessed_pdf, output_pdf, config,
            progress_callback=None, *, original_input_pdf=None):
        # Touch output: minimal one-page searchable PDF copy.
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 100), "Hello world", fontsize=12, render_mode=3)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_pdf))
        doc.close()
        return [PageOCRResult(
            page_number=1,
            text="Hello world",
            mean_confidence=92.0,
            word_boxes=[
                (10.0, 80.0, 60.0, 20.0, 95.0, "Hello"),
                (80.0, 80.0, 60.0, 20.0, 90.0, "world"),
            ],
        )]


def _make_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "in.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 100), "Hello world", fontsize=12)
    doc.save(str(p))
    doc.close()
    return p


def test_word_boxes_to_tsv_shape() -> None:
    boxes = [(1.0, 2.0, 3.0, 4.0, 88.0, "foo"),
             (5.0, 6.0, 7.0, 8.0, 50.0, "bar")]
    tsv = _word_boxes_to_tsv(boxes)
    assert tsv["text"] == ["foo", "bar"]
    assert tsv["conf"] == ["88.0", "50.0"]
    assert tsv["left"] == [1, 5]
    assert tsv["block_num"] == [1, 1]
    assert tsv["word_num"] == [1, 2]


def test_pipeline_runs_with_fake_engine(tmp_path: Path) -> None:
    src = _make_pdf(tmp_path)
    out = tmp_path / "out.pdf"
    profile = ProfileData(
        name="t", preprocess=PreprocessConfig(),
        ocr=OCRConfig(skip_text=False),
        postprocess=PostprocessConfig(),
    )
    job = OCRJobConfig(input_path=str(src), output_path=str(out),
                        profile=profile)
    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        compute_confidence=True,
    )
    fake = FakeEngine()
    with patch("src.application.engines.get_engine", return_value=fake):
        result = pipeline.run(job)
    assert isinstance(result, JobResult)
    assert result.status == JobStatus.COMPLETED, result.error
    assert out.exists()
    assert result.pages and result.pages[0].text


def test_pipeline_fails_when_engine_unavailable(tmp_path: Path) -> None:
    src = _make_pdf(tmp_path)
    out = tmp_path / "out.pdf"
    profile = ProfileData(
        name="t", preprocess=PreprocessConfig(),
        ocr=OCRConfig(skip_text=False), postprocess=PostprocessConfig(),
    )
    job = OCRJobConfig(input_path=str(src), output_path=str(out),
                        profile=profile)

    class DownEngine(FakeEngine):
        def is_available(self):
            return False, "no model installed"

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
    )
    with patch("src.application.engines.get_engine",
               return_value=DownEngine()):
        result = pipeline.run(job)
    assert result.status == JobStatus.FAILED
    assert "no model" in (result.error or "")
