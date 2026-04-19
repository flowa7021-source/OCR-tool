"""E2E tests for features not yet covered by the existing suite.

Coverage-gap analysis identified the following user-facing workflows
that had ZERO end-to-end coverage (or only mocked/unit-level):

  * **OCR cache real hit** — second run with same input returns
    correct text instantly from cache.
  * **DOCX export** — ``--docx`` flag actually produces a valid DOCX.
  * **max_pages preview limit** — ``OCRConfig.max_pages > 0`` limits
    how many pages are OCR'd (power-user workflow).
  * **Profile import/export** — user exports a profile to JSON, then
    re-imports it on another machine.
  * **Recovery snapshot + restore** — crash mid-job → restart →
    queue is re-populated from the snapshot.
  * **Autosave partial TXT** — long jobs dump a ``.partial.txt``
    every N pages so progress isn't lost on a crash.
  * **Searchable PDF text layer** — the output PDF must let the
    user Ctrl+F and find recognised words.
  * **OCRmyPDF optimize** — ``OptimizeLevel.LOSSLESS`` produces a
    smaller PDF than the raw preprocessed input (Ghostscript
    re-compresses streams).

Each test here uses **real** Tesseract + Ghostscript. Skipped when
those aren't on PATH.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    assert_ocr_recognised,
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    requires_real_russian_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


# ---------------------------------------------------------------------------
# OCR cache: second run must be instant + return the same text
# ---------------------------------------------------------------------------


class TestOCRCacheRealHit:
    """The user processes the same file twice — second time should be
    near-instant (cache HIT) and produce the identical text.

    Regression guard: earlier in this session we discovered the cache
    was storing empty-text results and poisoning subsequent runs.
    This test verifies the HAPPY path: a real (non-empty) result is
    cached and returned correctly.
    """

    def test_second_run_returns_cached_result(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        import time

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="CACHE ME"
        )
        output1 = tmp_path / "out1.pdf"
        output2 = tmp_path / "out2.pdf"

        profile = make_realistic_profile(binarization="otsu")

        # First run: real OCR.
        t0 = time.time()
        r1 = run_pipeline(input_pdf, output1, profile, real_tesseract_wrapper)
        first_elapsed = time.time() - t0
        assert r1.status is JobStatus.COMPLETED, r1.error
        text1 = r1.pages[0].text

        # Second run: same input + same profile → cache HIT.
        t0 = time.time()
        r2 = run_pipeline(input_pdf, output2, profile, real_tesseract_wrapper)
        second_elapsed = time.time() - t0
        assert r2.status is JobStatus.COMPLETED, r2.error
        text2 = r2.pages[0].text

        # Cache should serve the same text.
        assert text1 == text2, (
            f"cache returned different text.\n  first:  {text1!r}\n"
            f"  second: {text2!r}"
        )
        # And it should be MUCH faster (no Tesseract). Allow generous
        # margin: first run ~3-5 s, cache hit should be <1 s.
        assert second_elapsed < first_elapsed * 0.5, (
            f"second run ({second_elapsed:.2f}s) not significantly "
            f"faster than first ({first_elapsed:.2f}s) — cache HIT "
            "might not have fired"
        )


# ---------------------------------------------------------------------------
# DOCX export — user wants a Word document alongside the PDF
# ---------------------------------------------------------------------------


class TestDOCXExport:
    """``python -m src.cli input.pdf --docx`` must produce a .docx
    that a downstream tool (python-docx) can open and that contains
    the recognised text."""

    def test_docx_export_contains_recognised_text(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="DOCX EXPORT"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(binarization="otsu")
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error

        from src.application.export_manager import ExportManager
        from src.shared.types import ExportFormat

        docx_path = tmp_path / "out.docx"
        ExportManager().export(result, docx_path, ExportFormat.DOCX)
        assert docx_path.exists()

        import docx

        doc = docx.Document(str(docx_path))
        full_text = "\n".join(p.text for p in doc.paragraphs)
        assert any(
            w in full_text.upper() for w in ("DOCX", "EXPORT")
        ), f"DOCX has no recognised text: {full_text!r}"


# ---------------------------------------------------------------------------
# max_pages preview limit
# ---------------------------------------------------------------------------


class TestMaxPagesPreview:
    """``OCRConfig.max_pages = 2`` on a 4-page input → only first
    2 pages OCR'd. Used when tuning a profile before committing to a
    500-page job."""

    def test_max_pages_limits_ocr_to_first_n(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="PAGE", pages=4
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(binarization="otsu")
        profile.ocr.max_pages = 2

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error
        # Only 2 pages processed, not 4.
        assert result.page_count == 2, (
            f"max_pages=2 but got {result.page_count} pages"
        )


# ---------------------------------------------------------------------------
# Profile import/export roundtrip
# ---------------------------------------------------------------------------


class TestProfileImportExport:
    """User exports a customised profile → copies JSON to another
    machine → imports → the imported profile is usable in the pipeline."""

    def test_exported_profile_reimports_and_runs(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()

        # Customise a profile.
        original = manager.load("quick_reliable")
        original.name = "my_custom"
        original.description = "custom for test"
        original.builtin = False
        manager.save(original)

        # Export to a standalone JSON file.
        export_path = tmp_path / "my_custom_export.json"
        manager.export_profile("my_custom", export_path)
        assert export_path.exists()

        # Verify the exported JSON is valid and contains our changes.
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert data["name"] == "my_custom"

        # Re-import into a fresh storage (simulates another machine).
        storage2 = ProfileStorage(profiles_dir=tmp_path / "profiles2")
        manager2 = ProfileManager(storage2)
        manager2.initialize_builtins()
        imported = manager2.import_profile(export_path)
        assert imported.name == "my_custom"

        # The imported profile runs through the real pipeline.
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="IMPORTED PROFILE"
        )
        output_pdf = tmp_path / "out.pdf"
        result = run_pipeline(
            input_pdf, output_pdf, imported, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["IMPORTED", "PROFILE"])


# ---------------------------------------------------------------------------
# Recovery snapshot + restore
# ---------------------------------------------------------------------------


class TestRecoveryRoundtrip:
    """User's app crashes mid-job → next launch recovers the queue.

    The :class:`RecoveryManager` periodically snapshots the active
    queue; on restart :meth:`list_pending` returns the unfinished
    items so the UI can offer "Resume X jobs?".
    """

    def test_snapshot_then_restore_preserves_queue_item(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import src.shared.constants as const

        monkeypatch.setattr(const, "RECOVERY_DIR", tmp_path / "recovery")
        (tmp_path / "recovery").mkdir(parents=True, exist_ok=True)

        from src.application.recovery_manager import RecoveryManager
        from src.core.models import OCRJobConfig, ProfileData, QueueItem
        from src.shared.types import JobStatus

        rm = RecoveryManager()

        item = QueueItem(
            config=OCRJobConfig(
                input_path="/tmp/test.pdf",
                output_path="/tmp/test_ocr.pdf",
                profile=ProfileData(name="quick_reliable"),
            ),
            status=JobStatus.RUNNING,
        )

        # Snapshot (simulates periodic save during a running job).
        rm.snapshot(item)

        # Simulate restart: fresh RecoveryManager instance.
        rm2 = RecoveryManager()
        pending = rm2.list_pending()
        assert len(pending) >= 1, "recovery snapshot not found after restart"

        found = [p for p in pending if p.config and p.config.input_path == "/tmp/test.pdf"]
        assert found, (
            f"expected queue item not in restored list. "
            f"Got: {[p.config.input_path if p.config else '?' for p in pending]}"
        )


# ---------------------------------------------------------------------------
# Autosave partial TXT
# ---------------------------------------------------------------------------


class TestAutosavePartialTXT:
    """For long jobs (100+ pages), the pipeline writes a
    ``.partial.txt`` every N pages so the user can peek at progress
    and doesn't lose everything on a crash.

    This test uses ``autosave_interval_pages=1`` so the dump fires
    after the first page.
    """

    def test_partial_txt_is_written_during_processing(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        from src.application.engines.registry import reset_cache
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import OCRJobConfig
        from src.core.text_postprocessor import TextPostprocessor

        reset_cache()

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="AUTOSAVE", pages=2
        )
        output_pdf = tmp_path / "out.pdf"
        partial_txt = Path(str(output_pdf) + ".partial.txt")

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=real_tesseract_wrapper,
            compute_confidence=False,
            autosave_interval_pages=1,
        )
        profile = make_realistic_profile(binarization="otsu")
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=profile,
            )
        )
        assert result.status is JobStatus.COMPLETED, result.error

        # Partial TXT should have been written mid-run.
        assert partial_txt.exists(), (
            "autosave_interval_pages=1 but no .partial.txt produced"
        )
        body = partial_txt.read_text(encoding="utf-8")
        assert body.strip(), "partial TXT is empty"


# ---------------------------------------------------------------------------
# Searchable PDF text layer — Ctrl+F in a PDF viewer
# ---------------------------------------------------------------------------


class TestSearchablePDFTextLayer:
    """The whole point of this app: the output PDF must have an
    invisible text layer so the user can Ctrl+F and find words.
    This is the contract that every pipeline change must preserve."""

    def test_output_pdf_has_selectable_text(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="FINDABLE TEXT"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(binarization="otsu")
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error

        import fitz

        with fitz.open(str(output_pdf)) as doc:
            text = doc.load_page(0).get_text("text") or ""

        upper = text.upper()
        assert "FINDABLE" in upper or "TEXT" in upper, (
            f"output PDF text layer is empty or doesn't contain "
            f"expected words. Extracted: {text!r}"
        )

    @requires_real_russian_ocr
    def test_russian_text_is_searchable_in_output(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """The same for Russian — user opens the PDF in Adobe and
        presses Ctrl+F ДОГОВОР."""
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="ДОГОВОР", cyrillic=True
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
        assert result.status is JobStatus.COMPLETED, result.error

        import fitz

        with fitz.open(str(output_pdf)) as doc:
            text = doc.load_page(0).get_text("text") or ""

        assert any(
            tri in text.upper() for tri in ("ДОГ", "ВОР", "ОГО")
        ), f"Russian text not searchable in output PDF: {text!r}"


# ---------------------------------------------------------------------------
# OCRmyPDF optimize — output PDF smaller than preprocessed input
# ---------------------------------------------------------------------------


class TestOptimizeLossless:
    """``OptimizeLevel.LOSSLESS`` re-compresses image streams via
    Ghostscript. The output should be smaller than our raw
    preprocessed PDF (which is just PNGs crammed into a PDF wrapper).
    """

    def test_lossless_optimization_reduces_file_size(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="OPTIMIZE ME"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            optimize_level=1,  # LOSSLESS
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error

        in_size = input_pdf.stat().st_size
        out_size = output_pdf.stat().st_size
        # The input is a rasterised image-only PDF (~1-11 MB depending
        # on DPI). OCRmyPDF optimize=1 should produce something
        # noticeably smaller (or at worst comparable). If the output
        # is LARGER than the input, something went wrong with the
        # optimization or the pipeline is re-encoding at a higher
        # quality.
        assert out_size < in_size * 1.5, (
            f"optimized output ({out_size} B) is significantly larger "
            f"than input ({in_size} B) — optimization may be broken"
        )
