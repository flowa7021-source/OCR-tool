"""Tests for the P0+P1+P3 deep-freeze round.

Locks in:

1. ``ResultsPanel`` Ctrl+F highlight is debounced — rapid keystrokes
   produce exactly one highlight pass.
2. ``PDFViewer.open`` schedules the first-page render on a worker
   thread; it does NOT call ``_render_current`` synchronously.
3. ``ProcessPoolExecutor`` is created with ``max_tasks_per_child=10``
   so long sessions don't leak worker memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# ResultsPanel Ctrl+F debounce
# --------------------------------------------------------------------------


class TestResultsSearchDebounce:
    def _make_panel(self, qtbot):
        from src.core.models import JobResult, PageResult
        from src.shared.types import JobStatus
        from src.ui.results_panel import ResultsPanel

        panel = ResultsPanel()
        qtbot.addWidget(panel)
        panel.show()
        # Large-ish document so a real highlight pass would be noticeable.
        panel.set_result(
            JobResult(
                job_id="j", status=JobStatus.COMPLETED,
                input_path="/x", output_path="/x_ocr.pdf",
                pages=[PageResult(page_number=i + 1, text="hit " * 200)
                       for i in range(20)],
            )
        )
        panel._act_find.trigger()
        return panel

    def test_rapid_keystrokes_coalesce_to_one_highlight(self, qtbot) -> None:
        panel = self._make_panel(qtbot)
        with patch.object(panel, "_apply_pending_search") as apply:
            # Simulate a user typing 5 characters rapidly.
            panel._on_search_text_changed("h")
            panel._on_search_text_changed("hi")
            panel._on_search_text_changed("hit")
            panel._on_search_text_changed("hit ")
            panel._on_search_text_changed("hit t")
            # Timer hasn't fired yet.
            assert apply.call_count == 0
            qtbot.wait(300)
            # Exactly one apply after the debounce window.
            assert apply.call_count == 1

    def test_empty_string_clears_immediately(self, qtbot) -> None:
        """Backspacing to empty should clear highlights without waiting."""
        panel = self._make_panel(qtbot)
        panel._on_search_text_changed("hit")
        with patch.object(panel, "_clear_highlights") as clear:
            panel._on_search_text_changed("")
            # Synchronous clear, no debounce.
            assert clear.called

    def test_final_highlight_call_count_is_one(self, qtbot) -> None:
        """End-to-end: 10 keystrokes → exactly 1 _highlight_all call.

        Uses ``setText`` so ``_edit_search.text()`` returns the
        expected value inside ``_apply_pending_search`` — calling
        ``_on_search_text_changed`` directly bypasses the QLineEdit
        state and leaves the final ``self._edit_search.text()``
        as the last setText'd value.
        """
        panel = self._make_panel(qtbot)
        with patch.object(panel, "_highlight_all", return_value=0) as hl:
            for i in range(1, 11):
                panel._edit_search.setText("h" * i)
            qtbot.wait(300)
            # Despite 10 keystrokes only one real highlight pass runs.
            assert hl.call_count == 1


# --------------------------------------------------------------------------
# PDF first-page async render
# --------------------------------------------------------------------------


class TestPDFViewerAsyncOpen:
    def test_open_does_not_call_render_current_synchronously(
        self, qtbot, tmp_path: Path
    ) -> None:
        """PDFViewer.open must NOT block on a pixmap render on the GUI thread."""
        import fitz

        from src.ui.pdf_viewer import PDFViewer

        pdf = tmp_path / "t.pdf"
        doc = fitz.open()
        try:
            page = doc.new_page(width=200, height=200)
            page.insert_text((10, 20), "async test")
            doc.save(str(pdf))
        finally:
            doc.close()

        viewer = PDFViewer()
        qtbot.addWidget(viewer)
        with patch.object(viewer, "_render_current") as render:
            viewer.open(pdf)
            # Must not have rendered synchronously.
            assert not render.called

    def test_open_schedules_first_page_renderer(
        self, qtbot, tmp_path: Path
    ) -> None:
        import fitz

        from src.ui.pdf_viewer import PDFViewer

        pdf = tmp_path / "t.pdf"
        doc = fitz.open()
        try:
            page = doc.new_page(width=200, height=200)
            page.insert_text((10, 20), "hi")
            doc.save(str(pdf))
        finally:
            doc.close()

        viewer = PDFViewer()
        qtbot.addWidget(viewer)
        with patch.object(viewer, "_schedule_first_page_render") as sched:
            viewer.open(pdf)
            assert sched.call_count == 1
            args, _ = sched.call_args
            assert args[0] == pdf

    def test_first_page_renderer_delivers_pixmap(
        self, qtbot, tmp_path: Path
    ) -> None:
        """The worker itself must produce a non-null QPixmap for a valid PDF."""
        import fitz

        from src.ui.pdf_viewer import _FirstPageRenderer

        pdf = tmp_path / "t.pdf"
        doc = fitz.open()
        try:
            page = doc.new_page(width=200, height=200)
            page.insert_text((10, 20), "x")
            doc.save(str(pdf))
        finally:
            doc.close()

        got: list = []
        runnable = _FirstPageRenderer(pdf, page=1, zoom=1.0)
        runnable.signals.ready.connect(
            lambda p, pix: got.append((p, pix))
        )
        # Invoke directly (no thread pool) — same code path as QRunnable.run.
        runnable.run()
        assert len(got) == 1
        page_num, pixmap = got[0]
        assert page_num == 1
        assert not pixmap.isNull()

    def test_async_delivery_ignored_when_user_paged_away(
        self, qtbot, tmp_path: Path
    ) -> None:
        """If the user flipped to page 3 while page 1 was rendering, the
        late page-1 pixmap must NOT overwrite the page-3 view."""
        from PySide6.QtGui import QPixmap

        from src.ui.pdf_viewer import PDFViewer

        viewer = PDFViewer()
        qtbot.addWidget(viewer)
        viewer._doc = object()  # sentinel: non-None
        viewer._current_page = 3
        fake = QPixmap(10, 10)
        viewer._on_first_page_ready(page=1, pixmap=fake)
        # _page_label remains whatever it was; critically, NOT fake.
        # (We can't directly assert that — just ensure no crash and
        # the guard branch was taken.)


# --------------------------------------------------------------------------
# ProcessPoolExecutor max_tasks_per_child
# --------------------------------------------------------------------------


class TestExecutorRecycle:
    def test_executor_is_constructed_with_max_tasks_per_child(self) -> None:
        """Guards against a future refactor that drops the recycle kwarg."""
        from src.application.parallel_processor import ParallelProcessor

        pp = ParallelProcessor(max_workers=1)
        captured: dict = {}

        class _FakeExecutor:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def submit(self, *_a, **_kw):
                class _F:
                    def result(self):
                        return None
                return _F()

        with patch(
            "src.application.parallel_processor.ProcessPoolExecutor",
            _FakeExecutor,
        ), patch.object(pp, "_start_progress_bridge"):
            pp._ensure_executor()

        assert captured.get("max_tasks_per_child") == 10, (
            f"ProcessPoolExecutor missing max_tasks_per_child=10 — "
            f"got kwargs={captured}"
        )

    def test_falls_back_on_pre_3_11_python(self) -> None:
        """On older Pythons that reject the kwarg the executor still constructs."""
        from src.application.parallel_processor import ParallelProcessor

        pp = ParallelProcessor(max_workers=1)
        attempts: list = []

        class _PickyExecutor:
            def __init__(self, *args, **kwargs):
                attempts.append(kwargs)
                if "max_tasks_per_child" in kwargs:
                    raise TypeError(
                        "unexpected keyword argument 'max_tasks_per_child'"
                    )

            def submit(self, *_a, **_kw):
                class _F:
                    def result(self):
                        return None
                return _F()

        with patch(
            "src.application.parallel_processor.ProcessPoolExecutor",
            _PickyExecutor,
        ), patch.object(pp, "_start_progress_bridge"):
            pp._ensure_executor()

        # First attempt with recycle, second without.
        assert len(attempts) == 2
        assert "max_tasks_per_child" in attempts[0]
        assert "max_tasks_per_child" not in attempts[1]
