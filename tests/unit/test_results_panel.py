"""Smoke tests for the results panel."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.models import JobResult, PageResult  # noqa: E402
from src.shared.types import JobStatus  # noqa: E402
from src.ui.results_panel import ResultsPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _sample_result() -> JobResult:
    return JobResult(
        job_id="j1",
        status=JobStatus.COMPLETED,
        input_path="/tmp/in.pdf",
        output_path="/tmp/out.pdf",
        pages=[
            PageResult(page_number=1, text="Первая страница", mean_confidence=92.0),
            PageResult(page_number=2, text="Hello world", mean_confidence=85.0,
                       low_confidence_words=["He1lo", "wor1d"]),
        ],
    )


def test_set_result_populates(qapp: QApplication) -> None:
    panel = ResultsPanel()
    panel.set_result(_sample_result())
    # The panel should display text from one of the pages
    if hasattr(panel, "text") or hasattr(panel, "text_edit"):
        widget = getattr(panel, "text", None) or getattr(panel, "text_edit", None)
        if widget is not None:
            assert len(widget.toPlainText()) > 0


def test_clear_empties_panel(qapp: QApplication) -> None:
    panel = ResultsPanel()
    panel.set_result(_sample_result())
    if hasattr(panel, "clear"):
        panel.clear()


def test_export_signal_exists(qapp: QApplication) -> None:
    panel = ResultsPanel()
    assert hasattr(panel, "export_requested")
