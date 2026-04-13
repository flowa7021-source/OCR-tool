"""Queue panel: table of jobs with drag-and-drop and context menu."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.models import QueueItem
from src.shared.types import JobStatus

if TYPE_CHECKING:
    from src.application.queue_manager import QueueManager

logger = logging.getLogger(__name__)


class _SignalBridge(QObject):
    """Translates QueueManager observer callbacks to Qt signals.

    The QueueManager may fire notifications from worker threads; using a
    QObject signal lets us marshal them safely into the UI thread via
    Qt.QueuedConnection.
    """

    queue_event = Signal(object, str)  # (QueueItem, event_type)


class QueuePanel(QWidget):
    """Displays the job queue as a table and emits user-action signals."""

    files_dropped = Signal(list)  # list[Path]
    remove_requested = Signal(str)
    move_up_requested = Signal(str)
    move_down_requested = Signal(str)
    pause_requested = Signal(str)
    resume_requested = Signal(str)
    cancel_requested = Signal(str)
    retry_requested = Signal(str)
    open_output_requested = Signal(str)

    COLS = ("", "Файл", "Страницы", "Прогресс", "Статус")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._queue: QueueManager | None = None
        self._bridge: _SignalBridge | None = None

        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

    # -------------------------------------------------------------- queue
    def attach_queue(self, qm: "QueueManager") -> None:
        """Subscribe to queue events and refresh immediately."""
        self._queue = qm
        self._bridge = _SignalBridge()
        self._bridge.queue_event.connect(self._on_queue_event, Qt.ConnectionType.QueuedConnection)
        try:
            qm.subscribe(
                lambda item, event: self._bridge.queue_event.emit(item, event)
                if self._bridge is not None
                else None
            )
        except AttributeError:
            logger.warning("QueueManager has no subscribe(); polling not implemented")
        self.refresh()

    @Slot(object, str)
    def _on_queue_event(self, _item: object, _event: str) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Rebuild table contents from the attached queue."""
        if self._queue is None:
            return
        items = self._queue.list_items()
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            self._set_row(row, item)

    def _set_row(self, row: int, item: QueueItem) -> None:
        status = item.status
        # status icon
        icon_item = QTableWidgetItem(status.icon)
        icon_item.setData(Qt.ItemDataRole.UserRole, item.job_id)
        icon_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, icon_item)
        # filename
        self.table.setItem(row, 1, QTableWidgetItem(item.file_name))
        # pages
        pages_text = f"{item.progress_current}/{item.progress_total}" if item.progress_total else "-"
        self.table.setItem(row, 2, QTableWidgetItem(pages_text))
        # progress bar
        bar = QProgressBar()
        bar.setMinimum(0)
        bar.setMaximum(100)
        bar.setValue(int(item.progress_pct))
        bar.setTextVisible(True)
        bar.setFormat(f"{item.progress_pct:.0f}%")
        self.table.setCellWidget(row, 3, bar)
        # status label
        label = status.label
        if status is JobStatus.FAILED and item.error_message:
            label = f"{label}: {item.error_message[:40]}"
        self.table.setItem(row, 4, QTableWidgetItem(label))

    # -------------------------------------------------------------- drop
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths: list[Path] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_dir():
                paths.extend(sorted(local.rglob("*.pdf")))
            elif local.suffix.lower() == ".pdf":
                paths.append(local)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # -------------------------------------------------------------- menu
    def _show_context_menu(self, pos: object) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        job_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(job_id, str):
            return

        menu = QMenu(self)
        actions: list[tuple[str, Signal]] = [
            ("Удалить", self.remove_requested),
            ("Вверх", self.move_up_requested),
            ("Вниз", self.move_down_requested),
            ("Пауза", self.pause_requested),
            ("Возобновить", self.resume_requested),
            ("Отменить", self.cancel_requested),
            ("Повторить", self.retry_requested),
            ("Открыть результат", self.open_output_requested),
        ]
        for label, signal in actions:
            act = QAction(label, self)
            act.triggered.connect(lambda _checked=False, s=signal, jid=job_id: s.emit(jid))
            menu.addAction(act)
        menu.exec(self.table.viewport().mapToGlobal(pos))
