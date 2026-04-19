"""Real-OCR + live Qt UI: widgets react correctly to a real pipeline run.

Mocked UI tests in ``tests/unit/test_ui_interactions.py`` feed synthetic
JobResults into widgets. This module runs the ACTUAL pipeline with
real Tesseract, hands the JobResult to the Results panel, and
verifies the panel populates correctly. Catches wiring regressions
that mocks miss (signal/slot connections, thread affinity, etc.).

Skipped without PySide6 or without pytest-qt; both are installed on
PR CI via the ``pytest-qt`` line in ``ci.yml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from src.shared.types import JobStatus  # noqa: E402
from tests.integration._real_ocr_helpers import (  # noqa: E402
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


class TestResultsPanelWithRealJobResult:
    """Push a real JobResult into ResultsPanel; it must populate text."""

    def test_results_panel_shows_real_ocr_text(
        self,
        qtbot,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.ui.results_panel import ResultsPanel

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", "ui test content", pages=1,
        )
        profile = make_realistic_profile(binarization="otsu", dpi=200)
        result = run_pipeline(
            input_pdf, tmp_path / "out.pdf", profile,
            real_tesseract_wrapper,
        )
        assert result.status is JobStatus.COMPLETED, result.error

        panel = ResultsPanel()
        qtbot.addWidget(panel)
        panel.set_result(result)

        # The text viewer inside the panel must contain the recognised text.
        # We don't know the exact widget name across refactors, so walk
        # children and find QTextEdit-like ones. ``QObject.findChildren``
        # accepts exactly one type — call it separately for QTextEdit and
        # QPlainTextEdit and concatenate the results.
        from PySide6.QtWidgets import QPlainTextEdit, QTextEdit

        texts: list[str] = []
        for child in panel.findChildren(QTextEdit):
            texts.append(child.toPlainText())
        for child in panel.findChildren(QPlainTextEdit):
            texts.append(child.toPlainText())
        all_text = "\n".join(texts)

        recognised = (result.pages[0].text or "").strip()
        assert recognised, "real pipeline produced empty text"
        assert any(
            recognised[i:i + 4] in all_text
            for i in range(len(recognised) - 3)
        ), (
            f"ResultsPanel didn't render real OCR text. "
            f"Recognised: {recognised!r}. UI text: {all_text!r}"
        )


class TestQueuePanelAttachesToRealQueue:
    """QueuePanel attached to a real QueueManager must reflect additions."""

    def test_queue_panel_reflects_added_items(
        self,
        qtbot,
        tmp_path: Path,
    ) -> None:
        from src.application.queue_manager import QueueManager
        from src.core.models import OCRJobConfig, ProfileData, QueueItem
        from src.ui.queue_panel import QueuePanel

        # Minimal viable profile — don't need real OCR here, we're
        # testing the panel's reactivity to queue events.
        profile = ProfileData(name="ui-test")
        queue = QueueManager()
        panel = QueuePanel()
        qtbot.addWidget(panel)
        panel.attach_queue(queue)

        # Add two items — panel must show both
        for i in range(2):
            item = QueueItem(
                config=OCRJobConfig(
                    input_path=str(tmp_path / f"x{i}.pdf"),
                    output_path=str(tmp_path / f"y{i}.pdf"),
                    profile=profile,
                ),
            )
            queue.add(item)

        qtbot.waitUntil(
            lambda: panel.table.rowCount() == 2, timeout=2000,
        )
