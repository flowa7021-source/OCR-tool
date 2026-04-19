"""Real-OCR → real export: TXT, DOCX, searchable PDF, clipboard.

After OCR, users export the recognised text in various formats.
Mocked tests in ``tests/unit/test_export_manager.py`` verify the
serialisation of a synthetic JobResult; these tests feed a REAL
JobResult from a REAL OCR run into the exporter and verify the
output file contains the recognised text.

Clipboard is tested only when a QApplication is available (it
isn't in the plain CI run, but ``QT_QPA_PLATFORM=offscreen`` in
the workflow provides one).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.export_manager import ExportManager
from src.shared.types import ExportFormat, JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


@pytest.fixture
def real_job_result(
    tmp_path: Path, real_tesseract_wrapper,
):
    """Run a real OCR pipeline once, yield the resulting JobResult."""
    input_pdf = render_clean_text_pdf(
        tmp_path / "in.pdf", "EXPORT TEST 12345", pages=1,
    )
    output_pdf = tmp_path / "ocr.pdf"
    profile = make_realistic_profile(binarization="otsu", dpi=200)
    result = run_pipeline(
        input_pdf, output_pdf, profile, real_tesseract_wrapper
    )
    assert result.status is JobStatus.COMPLETED, result.error
    return result


class TestExportTxtReal:
    """TXT export of a real JobResult must contain the recognised text."""

    def test_txt_contains_recognised_text(
        self, tmp_path: Path, real_job_result,
    ) -> None:
        dest = tmp_path / "export.txt"
        ExportManager().export(
            real_job_result, dest, ExportFormat.TXT, encoding="utf-8",
        )
        assert dest.exists()
        text = dest.read_text(encoding="utf-8")
        # Tesseract isn't byte-perfect on synthetic renders; accept
        # partial matches. The point is "the file is populated with
        # OCR output", not "the OCR is 100% accurate".
        recognised = (real_job_result.pages[0].text or "").strip()
        assert recognised, "real_job_result produced empty text"
        # At least one 4-character substring of what was recognised
        # must appear in the exported file — catches "TXT written
        # empty" regressions.
        assert any(
            recognised[i:i + 4] in text
            for i in range(len(recognised) - 3)
        ), (
            f"TXT export missing recognised content. "
            f"Recognised: {recognised!r}. File: {text!r}"
        )


class TestExportDocxReal:
    """DOCX export produces a valid .docx with the recognised text."""

    def test_docx_opens_and_contains_text(
        self, tmp_path: Path, real_job_result,
    ) -> None:
        dest = tmp_path / "export.docx"
        try:
            ExportManager().export(
                real_job_result, dest, ExportFormat.DOCX,
            )
        except Exception as exc:  # noqa: BLE001
            if "python-docx" in str(exc).lower():
                pytest.skip("python-docx not installed in this environment")
            raise

        assert dest.exists()
        from docx import Document  # noqa: WPS433 — lazy import

        doc = Document(str(dest))
        joined = "\n".join(p.text for p in doc.paragraphs)
        recognised = (real_job_result.pages[0].text or "").strip()
        assert recognised
        assert any(
            recognised[i:i + 4] in joined
            for i in range(len(recognised) - 3)
        ), f"DOCX missing OCR content: {joined!r}"


class TestExportSearchablePdfReal:
    """PDF re-export must produce a PDF with an extractable text layer."""

    def test_pdf_export_has_text_layer(
        self, tmp_path: Path, real_job_result,
    ) -> None:
        dest = tmp_path / "export.pdf"
        ExportManager().export(
            real_job_result, dest, ExportFormat.PDF,
        )
        assert dest.exists()

        import fitz

        with fitz.open(str(dest)) as doc:
            extracted = "\n".join(p.get_text("text") for p in doc)
        assert extracted.strip(), (
            "Exported PDF has no text layer — export path is writing "
            "an image-only PDF"
        )


class TestExportClipboardReal:
    """Clipboard export needs a QApplication; skip if headless bootstrapping
    couldn't start one."""

    def test_clipboard_receives_text(self, real_job_result) -> None:
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            pytest.skip("PySide6 not available")

        app = QApplication.instance() or QApplication([])
        assert app is not None

        ExportManager().export(
            real_job_result,
            Path("<unused for clipboard>"),
            ExportFormat.CLIPBOARD,
        )

        clipboard_text = app.clipboard().text()
        recognised = (real_job_result.pages[0].text or "").strip()
        assert recognised
        assert any(
            recognised[i:i + 4] in clipboard_text
            for i in range(len(recognised) - 3)
        ), (
            f"Clipboard missing OCR content. Got: {clipboard_text!r}"
        )
