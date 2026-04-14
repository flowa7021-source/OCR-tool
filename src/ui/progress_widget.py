"""Progress indicator widget for the OCR queue.

Displays overall queue progress plus detail about the currently
processing file (its name, page counter, page-level progress bar
and an ETA computed from the rolling per-page rate).
"""

from __future__ import annotations

import logging
import time

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from src.shared.constants import COLOR_TEXT_SECONDARY

logger = logging.getLogger(__name__)


def _format_eta(seconds: float) -> str:
    """Format a wall-clock ETA in seconds → human-readable Russian.

    ``59 → "< 1 мин"``, ``150 → "~ 3 мин"``, ``3700 → "~ 1 ч 1 мин"``.
    """
    if seconds < 1:
        return ""
    if seconds < 60:
        return "< 1 мин"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"~ {minutes} мин"
    hours = minutes // 60
    minutes = minutes % 60
    if minutes == 0:
        return f"~ {hours} ч"
    return f"~ {hours} ч {minutes} мин"


class ProgressWidget(QWidget):
    """Compact progress panel shown above/below the queue list."""

    #: Number of recent per-page samples kept for ETA computation. A
    #: 5-sample moving average smooths out variance from page-to-page
    #: complexity without lagging behind abrupt changes.
    _ETA_WINDOW: int = 5

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

        # Rolling window of (timestamp, page_number) samples keyed by
        # the current file name so the ETA resets when we switch files.
        self._eta_file: str = ""
        self._eta_samples: list[tuple[float, int]] = []

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
        """Update the per-file indicators and recompute the ETA.

        Args:
            name: File name currently being processed.
            page: 1-based current page index.
            total: Total page count of the current file.
        """
        page = max(0, int(page))
        total = max(0, int(total))
        eta_text = self._update_eta(name, page, total)
        suffix = f" · осталось {eta_text}" if eta_text else ""
        self._lbl_current_file.setText(f"Текущий файл: {name}{suffix}")
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
        self._eta_file = ""
        self._eta_samples.clear()

    # ------------------------------------------------------------------
    # ETA
    # ------------------------------------------------------------------
    def _update_eta(self, name: str, page: int, total: int) -> str:
        """Maintain the moving window and return a formatted ETA string."""
        if name != self._eta_file:
            # New file / new session: reset the window.
            self._eta_file = name
            self._eta_samples.clear()
        now = time.monotonic()
        # Only record forward progress; repeated ticks on the same page
        # shouldn't inflate the rate estimate.
        if not self._eta_samples or page > self._eta_samples[-1][1]:
            self._eta_samples.append((now, page))
            if len(self._eta_samples) > self._ETA_WINDOW:
                self._eta_samples.pop(0)
        if len(self._eta_samples) < 2 or total <= 0 or page >= total:
            return ""
        t0, p0 = self._eta_samples[0]
        dt = now - t0
        dp = page - p0
        if dp <= 0 or dt <= 0:
            return ""
        sec_per_page = dt / dp
        remaining_pages = total - page
        return _format_eta(sec_per_page * remaining_pages)
