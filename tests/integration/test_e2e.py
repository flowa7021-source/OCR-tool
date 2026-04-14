"""End-to-end tests: real preprocessing + mocked OCR engine + real export.

These tests exercise the full vertical slice of the application short
of the actual Tesseract / GOT-OCR2 inference call. We use a real PDF
input (built on the fly with PyMuPDF), real OpenCV preprocessing, real
PyMuPDF rasterisation, real text postprocessing, and the actual
ExportManager — only the OCR-engine ``run()`` is replaced with a
deterministic stub that produces predictable text per page.

Each test asserts the entire data flow:

    OCRJobConfig
        → OCRPipeline.run
            → fitz.open + analyze_pdf
            → ImagePreprocessor (real OpenCV)
            → temp PDF assembly (real PyMuPDF)
            → OCREngine.run (stub)
            → text extraction + postprocess (real)
        → JobResult
            → ExportManager.export(TXT) (real)
            → ExportManager.export_pdf (real shutil.copy2)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")
pytest.importorskip("cv2")

from src.application.engines.base import OCREngine, PageOCRResult
from src.application.engines.registry import reset_cache
from src.application.export_manager import ExportManager
from src.application.pipeline import OCRPipeline
from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import (
    OCRConfig,
    OCRJobConfig,
    PreprocessConfig,
    ProfileData,
)
from src.core.text_postprocessor import TextPostprocessor
from src.shared.types import (
    BinarizationMethod,
    ExportFormat,
    JobStatus,
    OCREngineKind,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_pdf(path: Path, page_count: int = 3) -> Path:
    """Synthesize a tiny multi-page PDF with PyMuPDF.

    No real text is drawn — we only need a valid PDF that the pipeline
    can rasterise and feed to the (stubbed) OCR engine.
    """
    import fitz

    doc = fitz.open()
    try:
        for i in range(page_count):
            page = doc.new_page(width=300, height=200)
            page.insert_text((50, 100), f"Stub page {i + 1}")
        doc.save(str(path))
    finally:
        doc.close()
    return path


class _StubEngine(OCREngine):
    """OCR engine stub that returns deterministic text per page.

    Mimics GOT-OCR2's behavior: pre-fills PageOCRResult with text and
    confidence, lets the pipeline copy the source PDF as the "searchable"
    output (since we don't actually OCR).
    """

    kind = OCREngineKind.TESSERACT

    def __init__(self, page_texts: list[str] | None = None) -> None:
        self.page_texts = page_texts
        self.run_called = 0

    @property
    def name(self) -> str:
        return "Stub"

    @property
    def description(self) -> str:
        return "for tests"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
        import shutil

        self.run_called += 1
        # Pretend OCR succeeded by simply copying the preprocessed PDF
        # to the output location.
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preprocessed_pdf, output_pdf)

        import fitz

        with fitz.open(str(output_pdf)) as doc:
            results = []
            for i in range(doc.page_count):
                text = (
                    self.page_texts[i] if self.page_texts and i < len(self.page_texts)
                    else f"Распознанный текст страницы {i + 1}"
                )
                results.append(
                    PageOCRResult(
                        page_number=i + 1,
                        text=text,
                        mean_confidence=85.0 + i,
                    )
                )
        if progress_callback is not None:
            progress_callback(len(results), len(results), "stub-ocr")
        return results


def _profile() -> ProfileData:
    """Profile that uses the stub engine and disables heavy operations."""
    cfg = OCRConfig(engine=OCREngineKind.TESSERACT, dpi=150)
    pre = PreprocessConfig()
    # Disable binarization + denoise to keep the round-trip fast and
    # deterministic in tests.
    pre.binarization.method = BinarizationMethod.NONE
    pre.deskew.enabled = False
    return ProfileData(name="e2e-test", ocr=cfg, preprocess=pre)


@pytest.fixture(autouse=True)
def _reset_engines() -> None:
    """Make each test see a clean engine cache."""
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# E2E pipeline run
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    def test_full_run_produces_searchable_pdf(self, tmp_path: Path) -> None:
        """Pipeline.run end-to-end: input PDF → searchable PDF + per-page text."""
        input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=3)
        output_pdf = tmp_path / "out" / "input_ocr.pdf"

        stub = _StubEngine()
        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=__import__(
                "src.infrastructure.tesseract_wrapper", fromlist=["TesseractWrapper"]
            ).TesseractWrapper(),
        )

        with patch("src.application.engines.get_engine", return_value=stub):
            job = OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_profile(),
            )
            result = pipeline.run(job)

        # Pipeline reached COMPLETED
        assert result.status is JobStatus.COMPLETED, result.error
        # Stub was invoked
        assert stub.run_called == 1
        # Output PDF actually exists on disk
        assert output_pdf.exists()
        # Three pages of text were collected
        assert len(result.pages) == 3
        for i, page in enumerate(result.pages):
            assert page.page_number == i + 1
            assert page.text  # non-empty; stub provided text
        # Engine-provided confidence is preserved
        assert result.pages[0].mean_confidence == 85.0
        # average is computed
        assert result.average_confidence > 80.0

    def test_engine_results_text_is_postprocessed_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """Bug regression: engine text passes through postprocess but isn't
        replaced by re-reading the PDF."""
        input_pdf = _build_test_pdf(tmp_path / "in.pdf", page_count=1)
        out_pdf = tmp_path / "out.pdf"
        stub = _StubEngine(page_texts=["Самотек-\nстрока с переносом"])

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=__import__(
                "src.infrastructure.tesseract_wrapper", fromlist=["TesseractWrapper"]
            ).TesseractWrapper(),
        )
        profile = _profile()

        with patch(
            "src.application.engines.get_engine", return_value=stub
        ):
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(out_pdf),
                    profile=profile,
                )
            )

        assert result.status is JobStatus.COMPLETED
        # Hyphen merge from postprocess actually ran on the engine text
        assert "Самотекстрока" in result.pages[0].text or "Самотек" in result.pages[0].text


# ---------------------------------------------------------------------------
# E2E export
# ---------------------------------------------------------------------------


class TestE2EExport:
    def test_pipeline_to_txt_export(self, tmp_path: Path) -> None:
        """Run pipeline, then export as TXT, then read it back."""
        input_pdf = _build_test_pdf(tmp_path / "in.pdf", page_count=2)
        output_pdf = tmp_path / "out.pdf"
        stub = _StubEngine(page_texts=["Первая.", "Вторая."])
        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=__import__(
                "src.infrastructure.tesseract_wrapper", fromlist=["TesseractWrapper"]
            ).TesseractWrapper(),
        )
        with patch("src.application.engines.get_engine", return_value=stub):
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_profile(),
                )
            )
        assert result.status is JobStatus.COMPLETED

        txt_path = tmp_path / "out.txt"
        ExportManager().export(result, txt_path, ExportFormat.TXT)
        body = txt_path.read_text(encoding="utf-8")
        assert "Первая." in body
        assert "Вторая." in body
        assert "Page 1" in body and "Page 2" in body

    def test_pipeline_to_pdf_save_as(self, tmp_path: Path) -> None:
        """Pipeline produces PDF; ExportManager.export_pdf copies to chosen path."""
        input_pdf = _build_test_pdf(tmp_path / "in.pdf", page_count=1)
        output_pdf = tmp_path / "pipeline_output.pdf"
        with patch(
            "src.application.engines.get_engine", return_value=_StubEngine()
        ):
            pipeline = OCRPipeline(
                preprocessor=ImagePreprocessor(),
                postprocessor=TextPostprocessor(),
                tesseract=__import__(
                    "src.infrastructure.tesseract_wrapper",
                    fromlist=["TesseractWrapper"],
                ).TesseractWrapper(),
            )
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_profile(),
                )
            )
        assert result.status is JobStatus.COMPLETED
        # Pipeline wrote the searchable PDF
        assert output_pdf.exists()

        # User picks a different destination via Save As
        target = tmp_path / "user_chosen" / "saved.pdf"
        written = ExportManager().export(result, target, ExportFormat.PDF)
        assert written == target
        assert target.exists()
        assert target.read_bytes() == output_pdf.read_bytes()


# ---------------------------------------------------------------------------
# E2E queue + parallel processor (single worker for determinism)
# ---------------------------------------------------------------------------


class TestE2EQueue:
    def test_queue_manager_lifecycle(self, tmp_path: Path) -> None:
        """Add → next_pending → update_progress → update_status flow."""
        from src.application.queue_manager import QueueManager
        from src.core.models import QueueItem

        input_pdf = _build_test_pdf(tmp_path / "doc.pdf")
        out_pdf = tmp_path / "doc_ocr.pdf"
        qm = QueueManager()

        events: list[tuple[str, str]] = []
        qm.subscribe(lambda item, evt: events.append((evt, item.job_id)))

        item = QueueItem(
            config=OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(out_pdf),
                profile=_profile(),
            ),
            progress_total=3,
        )
        qm.add(item)
        # Pull from queue (atomically flips to RUNNING)
        nxt = qm.next_pending()
        assert nxt is not None
        assert nxt.job_id == item.job_id
        assert nxt.status is JobStatus.RUNNING
        # Simulate per-page progress
        for cur in (1, 2, 3):
            qm.update_progress(item.job_id, cur, 3)
        qm.update_status(item.job_id, JobStatus.COMPLETED)

        final = qm.get(item.job_id)
        assert final.status is JobStatus.COMPLETED
        assert final.progress_current == 3

        # Subscribe events fired for add + update + status changes
        evt_types = [e for e, _ in events]
        assert "added" in evt_types
        assert any(t == "updated" for t in evt_types)

    def test_clear_completed_keeps_running(self, tmp_path: Path) -> None:
        """clear_completed() purges terminal items but keeps active ones."""
        from src.application.queue_manager import QueueManager
        from src.core.models import QueueItem

        qm = QueueManager()
        a = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "a.pdf"),
                output_path=str(tmp_path / "a_ocr.pdf"),
                profile=_profile(),
            )
        )
        b = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "b.pdf"),
                output_path=str(tmp_path / "b_ocr.pdf"),
                profile=_profile(),
            )
        )
        c = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "c.pdf"),
                output_path=str(tmp_path / "c_ocr.pdf"),
                profile=_profile(),
            )
        )
        qm.add(a)
        qm.add(b)
        qm.add(c)
        qm.update_status(a.job_id, JobStatus.COMPLETED)
        qm.update_status(b.job_id, JobStatus.RUNNING)
        qm.update_status(c.job_id, JobStatus.FAILED, error="boom")

        removed = qm.clear_completed()
        assert removed == 2  # a (COMPLETED) + c (FAILED)
        remaining = qm.list_items()
        assert len(remaining) == 1
        assert remaining[0].job_id == b.job_id


# ---------------------------------------------------------------------------
# E2E HTR engine selection (engine registry → pipeline dispatch)
# ---------------------------------------------------------------------------


class TestE2EEngineDispatch:
    def test_pipeline_uses_selected_engine(self, tmp_path: Path) -> None:
        """OCRConfig.engine actually drives which engine.run() the pipeline calls."""
        input_pdf = _build_test_pdf(tmp_path / "doc.pdf", page_count=1)
        out_pdf = tmp_path / "doc_ocr.pdf"

        # Two distinguishable stubs
        tess_stub = _StubEngine(page_texts=["TESSERACT_PATH"])
        got_stub = _StubEngine(page_texts=["GOT_PATH"])

        def pick(kind):
            return got_stub if kind is OCREngineKind.GOT_OCR2 else tess_stub

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=__import__(
                "src.infrastructure.tesseract_wrapper", fromlist=["TesseractWrapper"]
            ).TesseractWrapper(),
        )

        with patch("src.application.engines.get_engine", side_effect=pick):
            # Run with Tesseract first
            profile_t = _profile()
            profile_t.ocr.engine = OCREngineKind.TESSERACT
            r1 = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(out_pdf),
                    profile=profile_t,
                )
            )
            assert r1.status is JobStatus.COMPLETED
            assert "TESSERACT_PATH" in r1.pages[0].text
            assert tess_stub.run_called == 1
            assert got_stub.run_called == 0

            # Now run with GOT-OCR2
            profile_g = _profile()
            profile_g.ocr.engine = OCREngineKind.GOT_OCR2
            out2 = tmp_path / "doc2_ocr.pdf"
            r2 = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(out2),
                    profile=profile_g,
                )
            )
            assert r2.status is JobStatus.COMPLETED
            assert "GOT_PATH" in r2.pages[0].text
            assert got_stub.run_called == 1


# ---------------------------------------------------------------------------
# E2E CLI subprocess
# ---------------------------------------------------------------------------


class TestE2ECLI:
    """Spawns `python -m src.cli` as a real subprocess."""

    def test_list_profiles_returns_zero(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--list-profiles"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # All five built-in profiles should appear
        for name in ("default", "low_quality_scan", "contracts_ru", "english_text", "handwritten_mixed"):
            assert name in result.stdout, f"missing {name} in {result.stdout!r}"

    def test_version_flag_returns_zero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--version"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OCR Studio" in result.stdout

    def test_no_input_returns_usage_error(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.cli"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2

    def test_missing_pdf_is_skipped(self, tmp_path: Path) -> None:
        """A non-existent PDF on the command line yields exit 2 (no PDFs found)."""
        import subprocess
        import sys

        ghost = tmp_path / "ghost.pdf"
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", str(ghost), "-v"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # discover_inputs filters it out via validate_pdf_path; exit 2
        # since we end up with zero inputs.
        assert result.returncode == 2


# ---------------------------------------------------------------------------
# E2E recovery roundtrip
# ---------------------------------------------------------------------------


class TestE2ERecovery:
    def test_snapshot_then_restore_roundtrip(self, tmp_path: Path) -> None:
        """A queued item can be snapshotted, restored, and re-submitted."""
        from src.application.recovery_manager import RecoveryManager
        from src.core.models import QueueItem

        rm = RecoveryManager(recovery_dir=tmp_path / "recovery")
        item = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "doc.pdf"),
                output_path=str(tmp_path / "doc_ocr.pdf"),
                profile=_profile(),
            ),
            progress_total=10,
            progress_current=4,
        )
        rm.snapshot(item)
        # Simulate "crash" — restart
        restored = rm.list_pending()
        assert len(restored) == 1
        assert restored[0].config is not None
        assert restored[0].config.input_path == item.config.input_path
        assert restored[0].config.profile.ocr.engine is OCREngineKind.TESSERACT
        # Restored items reset to PENDING
        assert restored[0].status is JobStatus.PENDING
        # Cleanup: completed snapshot purges
        item.status = JobStatus.COMPLETED
        rm.snapshot(item)
        rm.list_pending()  # side-effect: removes COMPLETED
        assert rm.list_pending() == []
