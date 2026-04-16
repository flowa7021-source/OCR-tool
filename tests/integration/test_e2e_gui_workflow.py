"""GUI-level E2E: replicate the user's click-by-click experience.

Uses pytest-qt + Qt's offscreen platform to instantiate
:class:`MainWindow` directly — the same widget tree the user sees
when they launch OCR Studio. Drives it via the widgets' public API
(click the Start button, pick a profile, etc.) and verifies signals
land where the UI expects them.

What these tests add over the rest of the real-OCR suite: widget
wiring, Qt-thread-boundary bugs (QueuedConnection between worker
thread and GUI thread), and dropdown refresh races. Things that a
pipeline-level test can't see because it never touches a widget.

Note: we deliberately avoid spawning ``ParallelProcessor`` workers
from inside these tests. A subprocess-worker inside pytest-qt tends
to deadlock the Qt event loop when teardown order is even slightly
off, and the cost of debugging that outweighs the marginal extra
coverage — ``test_e2e_parallel_processor.py`` already exercises the
real-worker path independently. Here we focus on the GUI layer
itself with a fake in-process processor.

Requires PySide6 + pytest-qt + Qt platform plugin (already installed
on the CI + dev machines). Skipped cleanly otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


pyside = pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

# Force offscreen Qt before any widget construction.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def main_window(qtbot, tmp_path: Path, monkeypatch):
    """Build a MainWindow backed by per-test storage.

    We redirect every user-data directory into ``tmp_path`` so the
    test doesn't touch the real user's AppData. ``ParallelProcessor``
    is stubbed (no real subprocess spawn) — GUI tests exercise the
    widget layer, not the worker process.
    """
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

    # Redirect per-user dirs into tmp_path.
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
    profile_manager = ProfileManager(storage)
    profile_manager.initialize_builtins()
    queue_manager = QueueManager()
    # Stub the parallel processor — we don't want subprocess workers
    # hanging inside pytest-qt. GUI-layer invariants are what we test.
    from unittest.mock import MagicMock
    parallel_processor = MagicMock()
    parallel_processor.submit = MagicMock()
    parallel_processor.shutdown = MagicMock()
    export_manager = ExportManager()
    settings_storage = SettingsStorage(config_dir=tmp_path / "config")

    window = MainWindow(
        profile_manager=profile_manager,
        queue_manager=queue_manager,
        parallel_processor=parallel_processor,
        export_manager=export_manager,
        settings_storage=settings_storage,
    )
    qtbot.addWidget(window)
    window.show()
    yield window


class TestGUIWorkflow:

    def test_profile_dropdown_lists_all_builtins(self, main_window) -> None:
        """User clicks the profile dropdown and sees every bundled
        profile. Regression guard: a seeding bug or a dropdown-refresh
        race once left the dropdown empty on fresh installs.

        This is the widget-layer mirror of
        ``test_bundled_builtin_profile_loads_and_runs_through_pipeline``
        — that test proved the profiles LOAD cleanly; this one proves
        they actually make it into the QComboBox the user interacts
        with.
        """
        from src.application.profile_manager import BUILTIN_NAMES

        combo = main_window.profile_combo
        # Profile name is stored as itemData (label has a ⭐ prefix).
        names = [combo.itemData(i) for i in range(combo.count())]
        for expected in BUILTIN_NAMES:
            assert expected in names, (
                f"profile dropdown missing {expected!r}; got: {names}"
            )

    def test_changing_profile_in_dropdown_updates_current_profile(
        self, main_window, qtbot
    ) -> None:
        """Picking a profile in the dropdown updates the
        ``_current_profile`` state the Start button uses — regression
        for a wiring bug that left _current_profile pinned to the
        first-loaded profile regardless of the dropdown."""
        combo = main_window.profile_combo
        # Find the ``quick_reliable`` index and select it.
        for i in range(combo.count()):
            if combo.itemData(i) == "quick_reliable":
                combo.setCurrentIndex(i)
                break
        else:
            pytest.skip("quick_reliable not in dropdown (seeding failed)")

        # Let the dropdown's currentIndexChanged signal propagate.
        qtbot.wait(50)

        assert main_window._current_profile is not None
        assert main_window._current_profile.name == "quick_reliable"

    def test_job_bridge_completed_signal_is_emitted_by_on_complete(
        self, main_window, qtbot
    ) -> None:
        """Simulate a ParallelProcessor worker finishing: the
        ``_on_complete`` callback (which runs on a non-GUI thread
        in production) must surface as a ``_job_bridge.completed``
        Qt signal so the UI can update its widgets safely."""
        from src.core.models import JobResult
        from src.shared.types import JobStatus

        received: list[tuple[str, JobResult]] = []
        main_window._job_bridge.completed.connect(
            lambda job_id, result: received.append((job_id, result))
        )

        fake_result = JobResult(
            job_id="fake-job",
            status=JobStatus.COMPLETED,
            input_path="/tmp/in.pdf",
            output_path="/tmp/out.pdf",
        )
        # The callback MainWindow registers with ParallelProcessor. It
        # captures job_id via closure in ``_submit_item``; we call
        # the internal method directly with a job_id to bypass.
        main_window._on_job_complete("fake-job", fake_result)

        # Signal routed via QueuedConnection — need event loop spin.
        qtbot.wait(50)

        assert received == [("fake-job", fake_result)]

    def test_job_bridge_failed_signal_fires_on_worker_exception(
        self, main_window, qtbot
    ) -> None:
        """Worker raises → _on_error → _job_bridge.failed with the
        exception as a second arg."""
        errors: list[tuple[str, BaseException]] = []
        main_window._job_bridge.failed.connect(
            lambda job_id, exc: errors.append((job_id, exc))
        )

        fake_exc = RuntimeError("pretend worker died")
        main_window._on_job_failed("fake-job", fake_exc)
        qtbot.wait(50)

        assert len(errors) == 1
        job_id, exc = errors[0]
        assert job_id == "fake-job"
        assert str(exc) == "pretend worker died"
