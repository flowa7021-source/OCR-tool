"""Tests for OCR overlay, TXT encoding selector, and pipeline autosave.

These cover the three feature additions made during the final plan-compliance
pass: PDFViewer bounding-box overlay toggle, Windows-1251 export encoding
selector on ResultsPanel, and Pipeline._autosave_partial_txt.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.application.pipeline import OCRPipeline  # noqa: E402
from src.core.models import (  # noqa: E402
    OCRJobConfig,
    PageResult,
    ProfileData,
)
from src.ui.pdf_viewer import PDFViewer  # noqa: E402
from src.ui.results_panel import ResultsPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


class TestOverlay:
    def test_default_hidden(self, qapp: QApplication) -> None:
        viewer = PDFViewer()
        assert viewer.is_overlay_visible() is False

    def test_toggle_visibility(self, qapp: QApplication) -> None:
        viewer = PDFViewer()
        viewer.set_overlay_visible(True)
        assert viewer.is_overlay_visible() is True
        viewer.set_overlay_visible(False)
        assert viewer.is_overlay_visible() is False

    def test_set_word_boxes_stores(self, qapp: QApplication) -> None:
        viewer = PDFViewer()
        viewer.set_word_boxes(1, [(10, 20, 100, 30, 90.0), (5, 55, 50, 20, 40.0)])
        assert len(viewer._word_boxes[1]) == 2

    def test_set_word_boxes_replaces(self, qapp: QApplication) -> None:
        viewer = PDFViewer()
        viewer.set_word_boxes(1, [(10, 20, 100, 30, 90.0)])
        viewer.set_word_boxes(1, [(1, 2, 3, 4, 50.0)])
        assert len(viewer._word_boxes[1]) == 1

    def test_clear_word_boxes(self, qapp: QApplication) -> None:
        viewer = PDFViewer()
        viewer.set_word_boxes(1, [(10, 20, 100, 30, 90.0)])
        viewer.clear_word_boxes()
        assert viewer._word_boxes == {}

    def test_threshold_clamps_to_range(self, qapp: QApplication) -> None:
        viewer = PDFViewer()
        viewer.set_overlay_threshold(200.0)
        assert viewer._overlay_threshold == 100.0
        viewer.set_overlay_threshold(-50.0)
        assert viewer._overlay_threshold == 0.0


# ---------------------------------------------------------------------------
# Encoding selector
# ---------------------------------------------------------------------------


class TestEncodingSelector:
    def test_default_utf8(self, qapp: QApplication) -> None:
        panel = ResultsPanel()
        assert panel.txt_encoding() == "utf-8"

    def test_select_cp1251(self, qapp: QApplication) -> None:
        panel = ResultsPanel()
        idx = panel._cmb_txt_encoding.findData("cp1251")
        assert idx >= 0
        panel._cmb_txt_encoding.setCurrentIndex(idx)
        assert panel.txt_encoding() == "cp1251"

    def test_select_utf8_sig(self, qapp: QApplication) -> None:
        panel = ResultsPanel()
        idx = panel._cmb_txt_encoding.findData("utf-8-sig")
        assert idx >= 0
        panel._cmb_txt_encoding.setCurrentIndex(idx)
        assert panel.txt_encoding() == "utf-8-sig"


class TestExportManagerEncoding:
    def test_cp1251_roundtrip(self, tmp_path: Path) -> None:
        """export() with encoding=cp1251 produces a CP1251-readable file."""
        from src.application.export_manager import ExportManager
        from src.core.models import JobResult
        from src.shared.types import ExportFormat, JobStatus

        result = JobResult(
            job_id="j",
            status=JobStatus.COMPLETED,
            input_path="/tmp/in.pdf",
            output_path=str(tmp_path / "out.pdf"),
            pages=[PageResult(page_number=1, text="Тестовая строка")],
        )
        target = tmp_path / "out.txt"
        ExportManager().export(result, target, ExportFormat.TXT, encoding="cp1251")
        raw = target.read_bytes()
        # UTF-8 byte sequence for Cyrillic should NOT appear
        assert "Тестовая".encode("utf-8") not in raw
        # CP1251 decoding round-trips
        text = raw.decode("cp1251")
        assert "Тестовая строка" in text


# ---------------------------------------------------------------------------
# Autosave
# ---------------------------------------------------------------------------


class TestAutosave:
    def _pipeline(self, interval: int = 2) -> OCRPipeline:
        return OCRPipeline(
            preprocessor=MagicMock(),
            postprocessor=MagicMock(),
            tesseract=MagicMock(),
            autosave_interval_pages=interval,
        )

    def test_interval_stored(self) -> None:
        p = self._pipeline(5)
        assert p.autosave_interval_pages == 5

    def test_zero_disables(self) -> None:
        p = self._pipeline(0)
        assert p.autosave_interval_pages == 0

    def test_negative_clamped_to_zero(self) -> None:
        p = OCRPipeline(
            preprocessor=MagicMock(),
            postprocessor=MagicMock(),
            tesseract=MagicMock(),
            autosave_interval_pages=-4,
        )
        assert p.autosave_interval_pages == 0

    def test_autosave_writes_partial_file(self, tmp_path: Path) -> None:
        p = self._pipeline(2)
        p.autosave_path = tmp_path / "partial.txt"
        pages = [
            PageResult(page_number=1, text="Первая"),
            PageResult(page_number=2, text="Вторая", mean_confidence=80.0),
        ]
        job = OCRJobConfig(
            input_path=str(tmp_path / "in.pdf"),
            output_path=str(tmp_path / "out.pdf"),
            profile=ProfileData(name="default"),
        )
        p._autosave_partial_txt(pages, job)
        body = p.autosave_path.read_text(encoding="utf-8")
        assert "Первая" in body
        assert "Вторая" in body
        assert "Page 1" in body and "Page 2" in body

    def test_autosave_atomic_no_tmp_leftover(self, tmp_path: Path) -> None:
        p = self._pipeline(2)
        p.autosave_path = tmp_path / "partial.txt"
        pages = [PageResult(page_number=1, text="A")]
        job = OCRJobConfig(
            input_path=str(tmp_path / "in.pdf"),
            output_path=str(tmp_path / "out.pdf"),
            profile=ProfileData(name="default"),
        )
        p._autosave_partial_txt(pages, job)
        assert not list(tmp_path.glob("*.tmp"))

    def test_autosave_default_path(self, tmp_path: Path) -> None:
        p = self._pipeline(2)
        job = OCRJobConfig(
            input_path=str(tmp_path / "in.pdf"),
            output_path=str(tmp_path / "out.pdf"),
            profile=ProfileData(name="default"),
        )
        p._autosave_partial_txt([PageResult(page_number=1, text="hi")], job)
        default = tmp_path / "out.pdf.partial.txt"
        assert default.exists()
