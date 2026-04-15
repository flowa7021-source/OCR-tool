"""Tests locking in the deeper freeze-fix round.

Targets the concrete freeze sources found by tracing "Start OCR":

1. ``ParallelProcessor.prewarm`` exists and spins up the pool +
   progress bridge without the caller paying for it again later.
2. ``QueuePanel`` refresh is debounced for automatic events but
   immediate for direct user actions (attach / filter / clear).
3. ``MainWindow`` schedules recovery snapshots and temp cleanup on
   the global thread pool rather than inline on the GUI thread.

Named ``test_zzzz_`` so it sorts after the other order-sensitive
files (test_followup_audit, test_z_universal_preset …).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# ParallelProcessor.prewarm
# --------------------------------------------------------------------------


class TestPrewarm:
    def test_prewarm_calls_ensure_executor(self) -> None:
        from src.application.parallel_processor import ParallelProcessor

        pp = ParallelProcessor(max_workers=1)
        # Stub the executor so we don't actually spawn a subprocess.
        submitted: list = []

        class _FakeExecutor:
            def submit(self, fn):
                submitted.append(fn)
                class _F:
                    def result(self):
                        return "ready"
                return _F()

        with patch.object(pp, "_ensure_executor", return_value=_FakeExecutor()):
            pp.prewarm()
        assert len(submitted) == 1, "prewarm should submit exactly one warmup job"

    def test_prewarm_swallows_executor_failure(self) -> None:
        """A startup that can't spawn workers shouldn't prevent the UI from loading."""
        from src.application.parallel_processor import ParallelProcessor

        pp = ParallelProcessor(max_workers=1)
        with patch.object(
            pp, "_ensure_executor", side_effect=RuntimeError("spawn denied")
        ):
            pp.prewarm()  # must not raise

    def test_prewarm_worker_fn_is_importable(self) -> None:
        """The warmup worker must live at module scope so ProcessPoolExecutor
        can pickle it. A lambda or closure would be rejected."""
        # Round-trip through pickle.
        import pickle

        from src.application.parallel_processor import _prewarm_worker

        pickle.loads(pickle.dumps(_prewarm_worker))


# --------------------------------------------------------------------------
# QueuePanel debounce
# --------------------------------------------------------------------------


class TestQueuePanelDebounce:
    def test_refresh_is_coalesced(self, qtbot, tmp_path: Path) -> None:
        """Many rapid refresh() calls should result in one real rebuild."""
        from src.application.queue_manager import QueueManager
        from src.ui.queue_panel import QueuePanel

        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(QueueManager())
        # Tick 100 "queue events" rapid-fire.
        with patch.object(panel, "_do_refresh") as mock_do:
            for _ in range(100):
                panel.refresh()
            # No real refresh has happened yet — all calls coalesced.
            assert mock_do.call_count == 0
            # Advance the singleshot timer.
            qtbot.wait(150)
            # Exactly one refresh after the 100 ms debounce window.
            assert mock_do.call_count == 1

    def test_attach_queue_refreshes_immediately(
        self, qtbot, tmp_path: Path
    ) -> None:
        """attach_queue must synchronously populate the table."""
        from src.application.queue_manager import QueueManager
        from src.core.models import OCRJobConfig, ProfileData, QueueItem
        from src.ui.queue_panel import QueuePanel

        qm = QueueManager()
        for name in ("a", "b", "c"):
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
        # No waitUntil — must be set synchronously.
        assert panel.table.rowCount() == 3

    def test_clear_completed_refreshes_immediately(
        self, qtbot, tmp_path: Path
    ) -> None:
        """Direct user action should not wait for the 100 ms debounce."""
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
        qm.add(done)
        qm.update_status(done.job_id, JobStatus.COMPLETED)

        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(qm)
        assert panel.table.rowCount() == 1

        panel._on_clear_completed()
        # Immediate — not 100 ms later.
        assert panel.table.rowCount() == 0


# --------------------------------------------------------------------------
# Recovery snapshots off the GUI thread
# --------------------------------------------------------------------------


class TestRecoverySnapshotAsync:
    def test_schedule_uses_thread_pool(self, qtbot, tmp_path: Path) -> None:
        """_schedule_recovery_snapshot must NOT call snapshot on the GUI thread."""
        import src.shared.constants as constants

        for name in ("USER_DATA_DIR", "CONFIG_DIR", "PROFILES_DIR", "TEMP_DIR",
                     "LOGS_DIR", "RECOVERY_DIR", "OCR_CACHE_DIR"):
            setattr(constants, name, tmp_path / name.lower())
            (tmp_path / name.lower()).mkdir(exist_ok=True)

        # Avoid modal dialogs during test.
        from unittest.mock import MagicMock as _Mock

        from PySide6.QtWidgets import QMessageBox

        with patch.object(QMessageBox, "warning", return_value=0), patch.object(
            QMessageBox, "critical", return_value=0
        ):
            from src.infrastructure.tesseract_wrapper import TesseractWrapper

            with patch.object(
                TesseractWrapper, "verify", return_value=(True, "ok")
            ):
                from PySide6.QtWidgets import QApplication

                QApplication.instance() or QApplication([])
                from src.app import create_application

                _, window = create_application([])
                try:
                    # Swap recovery with a mock that records its thread.
                    import threading

                    calls: list[int] = []

                    def _snapshot(item):
                        calls.append(threading.get_ident())

                    window._recovery = _Mock()
                    window._recovery.snapshot.side_effect = _snapshot

                    main_thread_id = threading.get_ident()
                    item = _Mock()
                    window._schedule_recovery_snapshot(item)

                    qtbot.waitUntil(lambda: len(calls) == 1, timeout=2000)
                    # Must have run on a thread != main.
                    assert calls[0] != main_thread_id, (
                        "snapshot ran on the GUI thread — fix regressed"
                    )
                finally:
                    window.close()
                    window.deleteLater()


# --------------------------------------------------------------------------
# Static guard: no sync recovery.snapshot in _apply_job_progress
# --------------------------------------------------------------------------


class TestNoSyncDiskIoInProgressHandler:
    def test_apply_job_progress_does_not_call_snapshot_directly(self) -> None:
        """AST-level check that _apply_job_progress uses the scheduled
        helper, not a direct self._recovery.snapshot(...) call.

        Walks the AST and looks for ``Call(func=Attribute(attr='snapshot',
        value=Attribute(attr='_recovery', value=Name('self'))))`` —
        i.e. the exact pattern ``self._recovery.snapshot(...)``. Avoids
        false positives from helper names that contain 'snapshot'.
        """
        import ast

        src = (Path(__file__).parent.parent.parent / "src" / "ui" / "main_window.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)

        def _is_self_recovery_snapshot_call(node: ast.AST) -> bool:
            if not isinstance(node, ast.Call):
                return False
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "snapshot"):
                return False
            v = f.value
            if not (isinstance(v, ast.Attribute) and v.attr == "_recovery"):
                return False
            base = v.value
            return isinstance(base, ast.Name) and base.id == "self"

        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "_apply_job_progress"):
                continue
            for inner in ast.walk(node):
                assert not _is_self_recovery_snapshot_call(inner), (
                    "_apply_job_progress still calls self._recovery.snapshot(...) "
                    "directly — this was the GUI-thread freeze source. Use "
                    "_schedule_recovery_snapshot instead."
                )
            return
        import pytest as _pytest

        _pytest.fail("_apply_job_progress not found")
