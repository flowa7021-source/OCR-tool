"""Regression tests for the audit follow-up round.

Covers:

* ``QueuePanel._on_clear_completed_requested`` confirmation dialog
  (Yes/No behaviour, empty-queue short-circuit).
* ``QueuePanel.attach_queue`` now calls ``subscribe`` unconditionally —
  the defensive ``AttributeError`` branch is removed, so the fallback
  warning must never fire.
* Corrupt / malformed profile JSON doesn't crash the loader and doesn't
  get silently returned as a broken ``ProfileData``.
* ``max_pages`` boundary: ``pages == max_pages`` is NOT truncated
  (off-by-one guard on the pipeline's ``>`` comparison).
* ``_warn_if_max_pages_active`` fires once per session.
* ``PreferencesDialog`` accept/reject persistence for non-theme
  settings (workers, autosave).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# QueuePanel: confirm dialog for clear completed
# --------------------------------------------------------------------------


class TestClearCompletedConfirm:
    def _make_panel_with_jobs(self, qtbot, tmp_path: Path):
        from src.application.queue_manager import QueueManager
        from src.core.models import OCRJobConfig, ProfileData, QueueItem
        from src.shared.types import JobStatus
        from src.ui.queue_panel import QueuePanel

        qm = QueueManager()
        done = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "done.pdf"),
                output_path=str(tmp_path / "done_ocr.pdf"),
                profile=ProfileData(name="t"),
            )
        )
        running = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "run.pdf"),
                output_path=str(tmp_path / "run_ocr.pdf"),
                profile=ProfileData(name="t"),
            )
        )
        qm.add(done)
        qm.add(running)
        qm.update_status(done.job_id, JobStatus.COMPLETED)
        qm.update_status(running.job_id, JobStatus.RUNNING)

        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(qm)
        panel.show()
        return panel, qm

    def test_yes_removes_terminal_rows(self, qtbot, tmp_path: Path) -> None:
        from PySide6.QtWidgets import QMessageBox

        panel, _qm = self._make_panel_with_jobs(qtbot, tmp_path)
        assert panel.table.rowCount() == 2
        with patch.object(
            QMessageBox, "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            panel._on_clear_completed_requested()
        qtbot.waitUntil(lambda: panel.table.rowCount() == 1, timeout=1000)

    def test_no_keeps_everything(self, qtbot, tmp_path: Path) -> None:
        from PySide6.QtWidgets import QMessageBox

        panel, _qm = self._make_panel_with_jobs(qtbot, tmp_path)
        assert panel.table.rowCount() == 2
        with patch.object(
            QMessageBox, "question",
            return_value=QMessageBox.StandardButton.No,
        ) as mocked:
            panel._on_clear_completed_requested()
            assert mocked.called
        assert panel.table.rowCount() == 2

    def test_empty_queue_skips_dialog(self, qtbot, tmp_path: Path) -> None:
        """No terminal rows → no dialog pops, no work done."""
        from PySide6.QtWidgets import QMessageBox

        from src.application.queue_manager import QueueManager
        from src.ui.queue_panel import QueuePanel

        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(QueueManager())
        with patch.object(QMessageBox, "question") as mocked:
            panel._on_clear_completed_requested()
            assert not mocked.called


# --------------------------------------------------------------------------
# QueuePanel.attach_queue no longer silently swallows a missing subscribe()
# --------------------------------------------------------------------------


class TestSubscribeWiring:
    def test_subscribe_is_called(self, qtbot) -> None:
        from src.ui.queue_panel import QueuePanel

        panel = QueuePanel()
        qtbot.addWidget(panel)
        qm_mock = MagicMock()
        qm_mock.list_items.return_value = []
        panel.attach_queue(qm_mock)
        qm_mock.subscribe.assert_called_once()


# --------------------------------------------------------------------------
# Corrupt profile JSON
# --------------------------------------------------------------------------


class TestCorruptProfileJson:
    def test_list_profiles_skips_malformed_file(self, tmp_path: Path) -> None:
        """A garbage JSON file in the profiles dir is logged and skipped, not raised."""
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path)
        # Good profile.
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"name": "good"}), encoding="utf-8")
        # Broken JSON.
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json ,,,}", encoding="utf-8")
        # TypeError-inducing schema (list where dict expected).
        weird = tmp_path / "weird.json"
        weird.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        profiles = storage.list_profiles()
        names = [p.name for p in profiles]
        assert "good" in names
        # Nothing raised, and the two corrupt entries are silently skipped.
        assert all(n not in names for n in ("bad", "weird"))

    def test_load_raises_on_corrupt_named_profile(self, tmp_path: Path) -> None:
        """``load(name)`` propagates the decode error for the specific name —
        callers (ProfileManager.get_current) handle the fallback, not storage."""
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path)
        (tmp_path / "broken.json").write_text("{oops", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            storage.load("broken")

    def test_profile_manager_get_current_falls_back_on_missing(
        self, tmp_path: Path
    ) -> None:
        """If the saved ``current`` profile disappeared, get_current()
        returns the single builtin.

        Декабрь 2026: fallback profile name сменился с «default» на
        «universal_accurate» (см. profile_manager консолидацию 7→1).
        """
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path)
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        manager._current_name = "ghost"  # type: ignore[attr-defined]
        profile = manager.get_current()
        # Fallback rebinds to 'universal_accurate' (the only builtin).
        assert profile.name == "universal_accurate"
        assert manager._current_name == "default"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# max_pages boundary: equal should NOT truncate
# --------------------------------------------------------------------------


class TestMaxPagesBoundary:
    def test_pages_equal_max_is_not_truncated(self, tmp_path: Path) -> None:
        """Exactly 3 pages + max_pages=3 → no truncation (pipeline uses ``>``)."""
        import fitz

        from src.application.engines.base import PageOCRResult
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import (
            OCRConfig,
            OCRJobConfig,
            PreprocessConfig,
            ProfileData,
        )
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import BinarizationMethod, JobStatus

        pdf = tmp_path / "exact.pdf"
        doc = fitz.open()
        try:
            for i in range(3):
                p = doc.new_page(width=200, height=100)
                p.insert_text((10, 50), f"p{i}")
            doc.save(str(pdf))
        finally:
            doc.close()

        class _Stub:
            kind = None
            name = "stub"
            description = ""

            def is_available(self):
                return True, ""

            def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
                import shutil

                import fitz as _fitz

                shutil.copy2(preprocessed_pdf, output_pdf)
                with _fitz.open(str(output_pdf)) as d:
                    return [
                        PageOCRResult(page_number=i + 1, text=f"page {i + 1}")
                        for i in range(d.page_count)
                    ]

            def unload(self):
                pass

        pre = PreprocessConfig()
        pre.binarization.method = BinarizationMethod.NONE
        pre.deskew.enabled = False
        profile = ProfileData(
            name="equal",
            ocr=OCRConfig(max_pages=3, dpi=72),
            preprocess=pre,
        )
        out = tmp_path / "out.pdf"
        job = OCRJobConfig(
            input_path=str(pdf), output_path=str(out), profile=profile,
        )
        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=MagicMock(),
        )
        with patch("src.application.engines.get_engine", return_value=_Stub()):
            result = pipeline.run(job)
        assert result.status is JobStatus.COMPLETED
        assert result.page_count == 3  # all pages, no truncation


# --------------------------------------------------------------------------
# MainWindow: one-shot max_pages warning
# --------------------------------------------------------------------------


class TestMaxPagesWarningOnce:
    def test_warns_once_per_session(self, qtbot, tmp_path: Path) -> None:
        """Warn on first enqueue with max_pages>0, then stay silent."""
        from PySide6.QtWidgets import QMessageBox

        from src.application.export_manager import ExportManager
        from src.application.parallel_processor import ParallelProcessor
        from src.application.profile_manager import ProfileManager
        from src.application.queue_manager import QueueManager
        from src.core.models import OCRConfig, ProfileData
        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage
        from src.ui.main_window import MainWindow

        pm = ProfileManager(ProfileStorage(profiles_dir=tmp_path / "profiles"))
        pm.initialize_builtins()
        w = MainWindow(
            profile_manager=pm,
            queue_manager=QueueManager(),
            parallel_processor=ParallelProcessor(max_workers=1),
            export_manager=ExportManager(),
            settings_storage=SettingsStorage(config_dir=tmp_path / "cfg"),
        )
        qtbot.addWidget(w)

        profile = ProfileData(name="lim", ocr=OCRConfig(max_pages=5))
        with patch.object(QMessageBox, "information") as mocked:
            w._warn_if_max_pages_active(profile, 1)
            w._warn_if_max_pages_active(profile, 1)  # 2nd call silent
            assert mocked.call_count == 1

    def test_no_warning_when_max_pages_zero(self, qtbot, tmp_path: Path) -> None:
        from PySide6.QtWidgets import QMessageBox

        from src.application.export_manager import ExportManager
        from src.application.parallel_processor import ParallelProcessor
        from src.application.profile_manager import ProfileManager
        from src.application.queue_manager import QueueManager
        from src.core.models import OCRConfig, ProfileData
        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage
        from src.ui.main_window import MainWindow

        pm = ProfileManager(ProfileStorage(profiles_dir=tmp_path / "profiles"))
        pm.initialize_builtins()
        w = MainWindow(
            profile_manager=pm,
            queue_manager=QueueManager(),
            parallel_processor=ParallelProcessor(max_workers=1),
            export_manager=ExportManager(),
            settings_storage=SettingsStorage(config_dir=tmp_path / "cfg"),
        )
        qtbot.addWidget(w)

        profile = ProfileData(name="nolim", ocr=OCRConfig(max_pages=0))
        with patch.object(QMessageBox, "information") as mocked:
            w._warn_if_max_pages_active(profile, 3)
            assert not mocked.called


# --------------------------------------------------------------------------
# PreferencesDialog: non-theme accept/reject persistence
# --------------------------------------------------------------------------


class TestPreferencesPersistence:
    def test_accept_persists_workers_and_autosave(
        self, qtbot, tmp_path: Path
    ) -> None:
        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.preferences_dialog import PreferencesDialog

        storage = SettingsStorage(config_dir=tmp_path)
        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        dlg.workers.setValue(3)
        dlg.autosave.setValue(75)
        dlg._on_accept()

        fresh = SettingsStorage(config_dir=tmp_path).load()
        assert fresh.parallel_workers == 3
        assert fresh.autosave_interval_pages == 75

    def test_reject_discards_edits(self, qtbot, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.preferences_dialog import PreferencesDialog

        storage = SettingsStorage(config_dir=tmp_path)
        start = storage.load()
        start.parallel_workers = 2
        start.autosave_interval_pages = 25
        storage.save(start)

        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        dlg.workers.setValue(4)  # pending, not saved
        dlg._on_reject()

        fresh = SettingsStorage(config_dir=tmp_path).load()
        assert fresh.parallel_workers == 2  # unchanged

    def test_accept_rejects_invalid_workers(self, qtbot, tmp_path: Path) -> None:
        """Workers value below MIN_PARALLEL_WORKERS should not save."""
        from PySide6.QtWidgets import QMessageBox

        from src.infrastructure.config_storage import SettingsStorage
        from src.shared.constants import MIN_PARALLEL_WORKERS
        from src.ui.preferences_dialog import PreferencesDialog

        storage = SettingsStorage(config_dir=tmp_path)
        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        # Bypass the spinbox range to smuggle in an invalid value.
        dlg.workers.setRange(0, 999)
        dlg.workers.setValue(0)
        with patch.object(QMessageBox, "warning"):
            dlg._on_accept()
        assert dlg.workers.value() < MIN_PARALLEL_WORKERS
        # Dialog did NOT accept.
        assert dlg.result() != dlg.DialogCode.Accepted
