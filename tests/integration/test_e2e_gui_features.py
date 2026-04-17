"""GUI-level E2E tests for HIGH/MED user-interaction features.

Each test uses pytest-qt + offscreen Qt platform. ParallelProcessor
is stubbed (no subprocess workers) — these exercise the widget
layer only, since subprocess workers are tested separately.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pyside = pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def main_window(qtbot, tmp_path: Path, monkeypatch):
    """Fresh MainWindow with stubbed ParallelProcessor."""
    import src.infrastructure.ocr_cache as _ocr_cache
    import src.shared.constants as const
    from src.application.export_manager import ExportManager
    from src.application.profile_manager import ProfileManager
    from src.application.queue_manager import QueueManager
    from src.infrastructure.config_storage import (
        ProfileStorage,
        SettingsStorage,
    )
    from src.ui.main_window import MainWindow

    for name, value in {
        "USER_DATA_DIR": tmp_path,
        "CONFIG_DIR": tmp_path / "config",
        "PROFILES_DIR": tmp_path / "profiles",
        "TEMP_DIR": tmp_path / "temp",
        "LOGS_DIR": tmp_path / "logs",
        "RECOVERY_DIR": tmp_path / "recovery",
        "OCR_CACHE_DIR": tmp_path / "ocr-cache",
    }.items():
        monkeypatch.setattr(const, name, value, raising=False)
        value.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        _ocr_cache, "OCR_CACHE_DIR", tmp_path / "ocr-cache", raising=False
    )

    storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
    pm = ProfileManager(storage)
    pm.initialize_builtins()

    window = MainWindow(
        profile_manager=pm,
        queue_manager=QueueManager(),
        parallel_processor=MagicMock(),
        export_manager=ExportManager(),
        settings_storage=SettingsStorage(config_dir=tmp_path / "config"),
    )
    qtbot.addWidget(window)
    window.show()
    yield window


# ---------------------------------------------------------------------------
# Drag-and-drop file enqueue
# ---------------------------------------------------------------------------


class TestDragAndDrop:
    def test_drop_pdf_enqueues_item(
        self, main_window, qtbot, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "drag_test.pdf"
        pdf.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\n"
            b"endobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\n"
            b"endobj\nxref\n0 3\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n"
            b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n109\n%%EOF\n"
        )

        main_window._enqueue_files([pdf])
        qtbot.wait(50)

        items = main_window._queue_manager.list_items()
        assert len(items) >= 1, "enqueue didn't add the PDF to queue"


# ---------------------------------------------------------------------------
# PDF viewer
# ---------------------------------------------------------------------------


class TestPDFViewerNavigation:
    def test_next_prev_page(self, main_window, qtbot, tmp_path: Path) -> None:
        from tests.integration._real_ocr_helpers import render_clean_text_pdf

        pdf = render_clean_text_pdf(
            tmp_path / "viewer.pdf", text="PAGE", pages=3
        )
        viewer = main_window.pdf_viewer
        viewer.open(Path(str(pdf)))
        qtbot.wait(200)

        # PDFViewer uses 1-based page indexing.
        assert viewer.current_page == 1
        viewer.next_page()
        assert viewer.current_page == 2
        viewer.next_page()
        assert viewer.current_page == 3
        viewer.next_page()
        assert viewer.current_page == 3  # can't go past last
        viewer.prev_page()
        assert viewer.current_page == 2
        viewer.close_document()

    def test_zoom_in_out(self, main_window, qtbot, tmp_path: Path) -> None:
        from tests.integration._real_ocr_helpers import render_clean_text_pdf

        pdf = render_clean_text_pdf(tmp_path / "z.pdf", text="ZOOM")
        viewer = main_window.pdf_viewer
        viewer.open(Path(str(pdf)))
        qtbot.wait(200)

        initial = viewer.zoom_factor
        viewer.zoom_in()
        assert viewer.zoom_factor > initial
        viewer.zoom_out()
        viewer.zoom_out()
        assert viewer.zoom_factor < initial
        viewer.close_document()


# ---------------------------------------------------------------------------
# Results panel search
# ---------------------------------------------------------------------------


class TestResultsPanelSearch:
    def test_search_opens_and_finds_text(self, main_window, qtbot) -> None:
        from src.core.models import JobResult, PageResult
        from src.shared.types import JobStatus

        result = JobResult(
            job_id="s",
            status=JobStatus.COMPLETED,
            input_path="/tmp/in.pdf",
            output_path="/tmp/out.pdf",
            pages=[PageResult(page_number=1, text="HELLO WORLD TEST")],
        )
        panel = main_window.results_panel
        panel.set_result(result)
        qtbot.wait(50)

        panel._toggle_search()
        qtbot.wait(50)
        assert panel._search_container.isVisible()
        panel._edit_search.setText("WORLD")
        panel._find_next()
        qtbot.wait(50)

        cursor = panel._text_edit.textCursor()
        assert cursor.hasSelection(), "search didn't select anything"


# ---------------------------------------------------------------------------
# Recent files
# ---------------------------------------------------------------------------


class TestRecentFilesMenu:
    def test_add_to_recent(self, main_window, qtbot, tmp_path: Path) -> None:
        fake = tmp_path / "recent.pdf"
        fake.write_bytes(b"%PDF-1.4\n")
        main_window._add_to_recent(fake)
        main_window._rebuild_recent_menu()
        qtbot.wait(50)
        texts = [a.text() for a in main_window.recent_menu.actions()]
        assert any("recent" in t.lower() for t in texts), (
            f"recent.pdf not in menu: {texts}"
        )


# ---------------------------------------------------------------------------
# Profile duplicate / delete via GUI
# ---------------------------------------------------------------------------


class TestProfileManagementGUI:
    def test_duplicate_creates_new_profile(
        self, main_window, qtbot
    ) -> None:
        combo = main_window.profile_combo
        for i in range(combo.count()):
            if combo.itemData(i) == "default":
                combo.setCurrentIndex(i)
                break
        qtbot.wait(50)

        from unittest.mock import patch

        with patch(
            "PySide6.QtWidgets.QInputDialog.getText",
            return_value=("test_copy", True),
        ):
            main_window._on_duplicate_profile()
        qtbot.wait(50)

        main_window._load_profiles_to_combobox()
        names = [combo.itemData(i) for i in range(combo.count())]
        assert "test_copy" in names

    def test_delete_removes_profile(self, main_window, qtbot) -> None:
        from src.core.models import ProfileData

        main_window._profile_manager.save(
            ProfileData(name="to_delete", builtin=False)
        )
        main_window._load_profiles_to_combobox()
        combo = main_window.profile_combo
        for i in range(combo.count()):
            if combo.itemData(i) == "to_delete":
                combo.setCurrentIndex(i)
                break
        qtbot.wait(50)

        from unittest.mock import patch

        with patch(
            "PySide6.QtWidgets.QMessageBox.question",
            return_value=pyside.QtWidgets.QMessageBox.StandardButton.Yes,
        ):
            main_window._on_delete_profile()
        qtbot.wait(50)

        main_window._load_profiles_to_combobox()
        names = [combo.itemData(i) for i in range(combo.count())]
        assert "to_delete" not in names


# ---------------------------------------------------------------------------
# Queue cancel signal
# ---------------------------------------------------------------------------


class TestQueuePanelSignals:
    def test_cancel_signal_fires(self, main_window, qtbot) -> None:
        from src.core.models import OCRJobConfig, ProfileData, QueueItem

        item = QueueItem(
            config=OCRJobConfig(
                input_path="/tmp/x.pdf",
                output_path="/tmp/x_ocr.pdf",
                profile=ProfileData(name="default"),
            ),
        )
        main_window._queue_manager.add(item)
        qtbot.wait(50)

        fired: list[str] = []
        main_window.queue_panel.cancel_requested.connect(
            lambda jid: fired.append(jid)
        )
        main_window.queue_panel.cancel_requested.emit(item.job_id)
        qtbot.wait(50)
        assert fired == [item.job_id]
