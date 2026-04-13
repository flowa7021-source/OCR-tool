"""Progress indicator widget for the OCR queue.

Displays overall queue progress plus detail about the currently
processing file (its name, page counter and page-level progress bar).
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from src.shared.constants import COLOR_TEXT_SECONDARY

logger = logging.getLogger(__name__)


class ProgressWidget(QWidget):
    """Compact progress panel shown above/below the queue list."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the widget in its empty/idle state.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self._lbl_overall = QLabel("Обработано файлов: 0/0", self)
        layout.addWidget(self._lbl_overall)

        self._bar_overall = QProgressBar(self)
        self._bar_overall.setRange(0, 100)
        self._bar_overall.setValue(0)
        layout.addWidget(self._bar_overall)

        self._lbl_current_file = QLabel("Текущий файл: —", self)
        layout.addWidget(self._lbl_current_file)

        self._bar_current = QProgressBar(self)
        self._bar_current.setRange(0, 100)
        self._bar_current.setValue(0)
        layout.addWidget(self._bar_current)

        self._lbl_page = QLabel("Страница 0 из 0", self)
        self._lbl_page.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10pt;"
        )
        layout.addWidget(self._lbl_page)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_overall(self, current_jobs: int, total_jobs: int) -> None:
        """Update the overall queue progress bar and label.

        Args:
            current_jobs: Number of completed jobs.
            total_jobs: Total number of jobs in the queue.
        """
        current_jobs = max(0, int(current_jobs))
        total_jobs = max(0, int(total_jobs))
        self._lbl_overall.setText(f"Обработано файлов: {current_jobs}/{total_jobs}")
        if total_jobs <= 0:
            self._bar_overall.setValue(0)
            return
        pct = int(round(100.0 * current_jobs / total_jobs))
        self._bar_overall.setValue(max(0, min(100, pct)))

    def set_current_file(self, name: str, page: int, total: int) -> None:
        """Update the per-file indicators.

        Args:
            name: File name currently being processed.
            page: 1-based current page index.
            total: Total page count of the current file.
        """
        self._lbl_current_file.setText(f"Текущий файл: {name}")
        page = max(0, int(page))
        total = max(0, int(total))
        self._lbl_page.setText(f"Страница {page} из {total}")
        if total <= 0:
            self._bar_current.setValue(0)
            return
        pct = int(round(100.0 * page / total))
        self._bar_current.setValue(max(0, min(100, pct)))

    def clear(self) -> None:
        """Reset all indicators to their initial state."""
        self._lbl_overall.setText("Обработано файлов: 0/0")
        self._bar_overall.setValue(0)
        self._lbl_current_file.setText("Текущий файл: —")
        self._bar_current.setValue(0)
        self._lbl_page.setText("Страница 0 из 0")
