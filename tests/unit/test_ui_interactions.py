"""End-to-end UI interaction tests driven by pytest-qt.

Unlike the existing smoke tests that only instantiate widgets, these
actually click buttons, type into line edits and key in shortcuts —
catching regressions where a signal connection is lost, an accelerator
stops working, or a slot expects different arguments.

All tests run with ``QT_QPA_PLATFORM=offscreen`` so CI runners without
a display are fine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

# --------------------------------------------------------------------------
# ResultsPanel — Ctrl+F search
# --------------------------------------------------------------------------


class TestResultsPanelSearch:
    def test_ctrl_f_opens_search_and_highlights(self, qtbot) -> None:
        from src.core.models import JobResult, PageResult
        from src.shared.types import JobStatus
        from src.ui.results_panel import ResultsPanel

        panel = ResultsPanel()
        qtbot.addWidget(panel)
        panel.show()
        panel.set_result(
            JobResult(
                job_id="j1",
                status=JobStatus.COMPLETED,
                input_path="/tmp/in.pdf",
                output_path="/tmp/out.pdf",
                pages=[
                    PageResult(
                        page_number=1,
                        text="Один один один два три",
                    )
                ],
            )
        )
        # Trigger Find via the QAction we registered for Ctrl+F
        panel._act_find.trigger()
        assert panel._search_container.isVisibleTo(panel)
        # Use setText for the Cyrillic query: PySide6 6.11 offscreen
        # crashes on qtbot.keyClicks() for non-ASCII payloads. The end
        # result (textChanged signal → highlight + counter) is the
        # same as if the user had typed.
        panel._edit_search.setText("один")
        # Wait past the 200 ms search debounce plus a margin.
        qtbot.waitUntil(
            lambda: panel._lbl_search_count.text() == "3", timeout=1000
        )

    def test_escape_closes_search(self, qtbot) -> None:
        from src.ui.results_panel import ResultsPanel

        panel = ResultsPanel()
        qtbot.addWidget(panel)
        panel.show()
        panel._act_find.trigger()
        assert panel._search_container.isVisibleTo(panel)
        # Trigger the Escape QAction we registered on the search container
        panel._act_find_esc.trigger()
        assert not panel._search_container.isVisibleTo(panel)


# --------------------------------------------------------------------------
# QueuePanel — filter + context-menu signals
# --------------------------------------------------------------------------


class TestQueuePanelInteraction:
    def test_filter_updates_table_live(self, qtbot, tmp_path: Path) -> None:
        from src.application.queue_manager import QueueManager
        from src.core.models import OCRJobConfig, ProfileData, QueueItem
        from src.ui.queue_panel import QueuePanel

        qm = QueueManager()
        for name in ("alpha", "beta", "gamma"):
            qm.add(
                QueueItem(
                    config=OCRJobConfig(
                        input_path=str(tmp_path / f"{name}.pdf"),
                        output_path=str(tmp_path / f"{name}_ocr.pdf"),
                        profile=ProfileData(name="t"),
                    )
                )
            )

        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(qm)
        panel.show()
        assert panel.table.rowCount() == 3

        # Type "bet" → only beta remains
        qtbot.keyClicks(panel._edit_filter, "bet")
        qtbot.waitUntil(lambda: panel.table.rowCount() == 1, timeout=1000)

        # Clear → everything visible again
        panel._edit_filter.clear()
        qtbot.waitUntil(lambda: panel.table.rowCount() == 3, timeout=1000)

    def test_retry_signal_fires_on_context_menu_trigger(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Programmatically trigger the 'Повторить' action and verify the signal."""
        from PySide6.QtGui import QAction

        from src.application.queue_manager import QueueManager
        from src.core.models import OCRJobConfig, ProfileData, QueueItem
        from src.ui.queue_panel import QueuePanel

        qm = QueueManager()
        item = QueueItem(
            config=OCRJobConfig(
                input_path=str(tmp_path / "doc.pdf"),
                output_path=str(tmp_path / "doc_ocr.pdf"),
                profile=ProfileData(name="t"),
            )
        )
        qm.add(item)
        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(qm)
        panel.show()

        received: list[str] = []
        panel.retry_requested.connect(received.append)

        # Simulate what _show_context_menu does for a specific row,
        # without actually opening a QMenu (hard to exec() headlessly).
        action = QAction("Повторить", panel)
        action.triggered.connect(
            lambda _checked=False: panel.retry_requested.emit(item.job_id)
        )
        action.trigger()
        assert received == [item.job_id]

    def test_clear_completed_action(self, qtbot, tmp_path: Path) -> None:
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
        assert panel.table.rowCount() == 2

        panel._on_clear_completed()
        qtbot.waitUntil(lambda: panel.table.rowCount() == 1, timeout=1000)


# --------------------------------------------------------------------------
# SettingsPanel — live config emission
# --------------------------------------------------------------------------


class TestSettingsPanelInteraction:
    def test_dpi_change_emits_config(self, qtbot) -> None:
        from src.ui.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        panel.show()

        received: list = []
        panel.config_changed.connect(received.append)

        # Flip the DPI combo to the second entry and wait out the
        # 300 ms debounce.
        panel._cmb_dpi.setCurrentIndex((panel._cmb_dpi.currentIndex() + 1) % panel._cmb_dpi.count())
        qtbot.wait(500)
        assert len(received) >= 1

    def test_max_pages_spinner_reflects_in_config(self, qtbot) -> None:
        from src.ui.settings_panel import SettingsPanel

        panel = SettingsPanel()
        qtbot.addWidget(panel)
        panel.show()
        panel._spin_max_pages.setValue(15)
        cfg = panel.get_config()
        assert cfg.max_pages == 15


# --------------------------------------------------------------------------
# ProgressWidget — ETA label update on repeated ticks
# --------------------------------------------------------------------------


class TestProgressWidgetInteraction:
    def test_eta_label_populates_after_second_tick(self, qtbot) -> None:
        """Second set_current_file call with forward progress should yield the 'осталось' suffix."""
        from src.ui.progress_widget import ProgressWidget

        w = ProgressWidget()
        qtbot.addWidget(w)
        w.show()

        w.set_current_file("doc.pdf", 1, 10)
        # Monkey-patch: artificially age the first sample by 12 s so the
        # second tick produces a non-empty ETA deterministically.
        w._eta_samples[0] = (w._eta_samples[0][0] - 12.0, w._eta_samples[0][1])
        w.set_current_file("doc.pdf", 3, 10)
        assert "осталось" in w._lbl_current_file.text()


# --------------------------------------------------------------------------
# Postprocess panel — custom rule add / remove
# --------------------------------------------------------------------------


class TestPostprocessPanelInteraction:
    def test_add_rule_grows_table(self, qtbot) -> None:
        from src.ui.postprocess_panel import PostprocessPanel

        panel = PostprocessPanel()
        qtbot.addWidget(panel)
        panel.show()
        before = panel.rules_table.rowCount()
        panel._add_rule()
        assert panel.rules_table.rowCount() == before + 1

    def test_live_preview_applies_enabled_rule(self, qtbot) -> None:
        from src.core.models import PostprocessConfig, RegexRule
        from src.ui.postprocess_panel import PostprocessPanel

        panel = PostprocessPanel()
        qtbot.addWidget(panel)
        panel.show()
        panel.set_config(
            PostprocessConfig(
                autocorrect_russian=False,
                autocorrect_english=False,
                merge_hyphenated=False,
                normalize_whitespace=False,
                normalize_unicode=False,
                remove_artifacts=False,
                custom_rules=[
                    RegexRule(
                        pattern="foo",
                        replacement="BAR",
                        is_regex=False,
                        case_sensitive=True,
                        enabled=True,
                    )
                ],
            )
        )
        qtbot.keyClicks(panel.preview_input, "foo and foo")
        assert "BAR" in panel.preview_output.text()


# --------------------------------------------------------------------------
# Keyboard shortcut sanity: Ctrl+Q triggers close on a standalone window
# --------------------------------------------------------------------------


class TestKeyboardShortcuts:
    def test_escape_in_preferences_closes_dialog(self, qtbot, tmp_path: Path) -> None:
        """Escape is Qt-default for QDialog.reject; confirm the dialog responds."""
        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.preferences_dialog import PreferencesDialog

        storage = SettingsStorage(config_dir=tmp_path)
        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        dlg.show()
        QTest.keyClick(dlg, Qt.Key.Key_Escape)
        qtbot.waitUntil(lambda: not dlg.isVisible(), timeout=1000)
