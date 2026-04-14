"""Performance-regression tests for the MainWindow thread hygiene.

The user reported the app freezing:
  * at startup
  * after starting recognition on a file
  * while resizing windows

These tests guard the fixes that make the GUI thread responsive:

  * ``PDFViewer.set_word_boxes_bulk`` does one dict update + one
    re-render, not N signal round-trips.
  * ``_OverlayExtractorRunnable`` reads the PDF in a worker thread and
    delivers a single page→boxes dict via a signal.
  * ``_TesseractVerifyRunnable`` runs ``tesseract --version`` off the
    GUI thread; MainWindow receives a signal.
  * Window min-size fits a 1280×720 screen; panels live inside
    QScrollArea so resizing below their sizeHint doesn't reflow forever.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# Bulk word-boxes API
# --------------------------------------------------------------------------


class TestWordBoxesBulk:
    def test_bulk_replaces_all_pages_in_one_call(self, qtbot) -> None:
        from src.ui.pdf_viewer import PDFViewer

        v = PDFViewer()
        qtbot.addWidget(v)
        mapping = {i: [(0.0, 0.0, 10.0, 10.0, 90.0)] for i in range(1, 101)}
        # Must not raise and must populate every page.
        v.set_word_boxes_bulk(mapping)
        assert v._word_boxes[1] == [(0.0, 0.0, 10.0, 10.0, 90.0)]
        assert 100 in v._word_boxes

    def test_bulk_call_is_constant_renders(self, qtbot) -> None:
        """Bulk populate should call _render_current AT MOST once."""
        from unittest.mock import patch

        from src.ui.pdf_viewer import PDFViewer

        v = PDFViewer()
        qtbot.addWidget(v)
        v._overlay_visible = True  # would force render on each page otherwise
        big_mapping = {i: [] for i in range(1, 501)}
        with patch.object(v, "_render_current") as render:
            v.set_word_boxes_bulk(big_mapping)
        # Was N in the old per-page path; now at most 1.
        assert render.call_count <= 1


# --------------------------------------------------------------------------
# Overlay extractor runnable
# --------------------------------------------------------------------------


class TestOverlayRunnable:
    def test_missing_file_emits_nothing_but_does_not_crash(
        self, qtbot, tmp_path: Path
    ) -> None:
        from src.ui.main_window import _OverlayExtractorRunnable

        bogus = tmp_path / "nope.pdf"
        runnable = _OverlayExtractorRunnable(bogus, [])
        # Drive it synchronously on the test thread to assert behaviour.
        received: list[dict] = []
        failures: list[str] = []
        runnable.signals.ready.connect(received.append)
        runnable.signals.failed.connect(failures.append)
        runnable.run()
        # No ready, one failed.
        assert received == []
        assert failures

    def test_emits_mapping_for_real_pdf(self, qtbot, tmp_path: Path) -> None:
        import fitz

        from src.ui.main_window import _OverlayExtractorRunnable

        pdf = tmp_path / "doc.pdf"
        doc = fitz.open()
        try:
            for i in range(3):
                page = doc.new_page(width=200, height=100)
                page.insert_text((10, 50), f"hello {i}")
            doc.save(str(pdf))
        finally:
            doc.close()

        got: list[dict] = []
        runnable = _OverlayExtractorRunnable(pdf, [80.0, 80.0, 80.0])
        runnable.signals.ready.connect(got.append)
        runnable.run()
        assert len(got) == 1
        mapping = got[0]
        # Three pages, all keyed 1..3.
        assert set(mapping.keys()) == {1, 2, 3}


# --------------------------------------------------------------------------
# Tesseract verify runnable
# --------------------------------------------------------------------------


class TestTesseractVerifyRunnable:
    def test_runnable_emits_wrapper_result(self, qtbot, monkeypatch) -> None:
        from src.infrastructure.tesseract_wrapper import TesseractWrapper
        from src.ui.main_window import _TesseractVerifyRunnable

        monkeypatch.setattr(
            TesseractWrapper, "verify",
            lambda self: (True, "Tesseract 5.5.0"),
        )
        received: list[tuple[bool, str]] = []
        r = _TesseractVerifyRunnable()
        r.signals.result.connect(lambda ok, msg: received.append((ok, msg)))
        r.run()
        assert received == [(True, "Tesseract 5.5.0")]

    def test_runnable_swallows_wrapper_exception(
        self, qtbot, monkeypatch
    ) -> None:
        from src.infrastructure.tesseract_wrapper import TesseractWrapper
        from src.ui.main_window import _TesseractVerifyRunnable

        def _raise(self):
            raise RuntimeError("subprocess died")

        monkeypatch.setattr(TesseractWrapper, "verify", _raise)
        received: list[tuple[bool, str]] = []
        r = _TesseractVerifyRunnable()
        r.signals.result.connect(lambda ok, msg: received.append((ok, msg)))
        r.run()
        assert received[0][0] is False
        assert "subprocess died" in received[0][1]


# --------------------------------------------------------------------------
# Window geometry fits 1280×720
# --------------------------------------------------------------------------


class TestWindowGeometry:
    def _stub_heavy(self, monkeypatch, tmp_path: Path) -> None:
        import src.shared.constants as constants

        for name in ("USER_DATA_DIR", "CONFIG_DIR", "PROFILES_DIR", "TEMP_DIR",
                     "LOGS_DIR", "RECOVERY_DIR", "OCR_CACHE_DIR"):
            monkeypatch.setattr(constants, name, tmp_path / name.lower())
            (tmp_path / name.lower()).mkdir(exist_ok=True)

        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)

    def test_minimum_size_fits_1280x720(
        self, qtbot, monkeypatch, tmp_path: Path
    ) -> None:
        from PySide6.QtWidgets import QApplication

        self._stub_heavy(monkeypatch, tmp_path)
        QApplication.instance() or QApplication([])
        from src.app import create_application

        _, window = create_application([])
        try:
            m = window.minimumSize()
            assert m.width() <= 1280, f"min width {m.width()} wouldn't fit 1280-px screen"
            assert m.height() <= 720, f"min height {m.height()} wouldn't fit 720-px screen"
        finally:
            window.close()
            window.deleteLater()
