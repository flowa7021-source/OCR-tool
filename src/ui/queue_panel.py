"""Queue panel: table of jobs with drag-and-drop and context menu."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
        self._filter_text: str = ""

        # Coalesce refresh bursts: a running OCR job emits a queue
        # event per page (1-10 Hz). Rebuilding the whole QTableWidget
        # at that rate visibly janks the GUI. This timer batches all
        # refresh requests into one real redraw per 100 ms window.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._do_refresh)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Filter row — becomes useful once the queue has more than a
        # handful of files. Substring match on file name, case-
        # insensitive. Empty filter shows everything.
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(QLabel("Фильтр:", self))
        self._edit_filter = QLineEdit(self)
        self._edit_filter.setPlaceholderText("Подстрока в имени файла…")
        self._edit_filter.setClearButtonEnabled(True)
        self._edit_filter.textChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._edit_filter, 1)
        layout.addLayout(filter_row)

        self.table = QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setAccessibleName("Очередь обработки OCR")
        self.table.setAccessibleDescription(
            "Список файлов в очереди с колонками: номер, файл, страницы, "
            "прогресс, статус. Используйте стрелки для навигации, Enter для "
            "выбора, правый клик для контекстного меню."
        )
        self._edit_filter.setAccessibleName("Фильтр очереди")
        self._edit_filter.setAccessibleDescription(
            "Оставляет в таблице только файлы, имя которых содержит эту "
            "подстроку. Регистр не важен."
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.setMinimumHeight(160)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(60)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 36)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 100)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 140)
        layout.addWidget(self.table)

    # -------------------------------------------------------------- queue
    def attach_queue(self, qm: QueueManager) -> None:
        """Subscribe to queue events and refresh immediately."""
        self._queue = qm
        self._bridge = _SignalBridge()
        self._bridge.queue_event.connect(self._on_queue_event, Qt.ConnectionType.QueuedConnection)
        qm.subscribe(
            lambda item, event: self._bridge.queue_event.emit(item, event)
            if self._bridge is not None
            else None
        )
        # Direct user action (attaching a queue) — refresh immediately so
        # the first render is synchronous and tests / the user see the
        # initial state straight away.
        self._do_refresh()

    @Slot(object, str)
    def _on_queue_event(self, _item: object, _event: str) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Schedule a batched table rebuild on the GUI thread.

        Starting the singleshot timer is a no-op if it's already
        pending — so 100 progress events in a 100 ms window cause one
        actual redraw. Previously each event was a full
        ``setRowCount`` + per-cell rebuild, which was a measurable
        GUI-thread freeze on multi-file batches.
        """
        self._refresh_timer.start()

    def _do_refresh(self) -> None:
        """Actual table rebuild. Called from the coalescing timer."""
        if self._queue is None:
            return
        items = self._queue.list_items()
        needle = self._filter_text.casefold()
        if needle:
            items = [i for i in items if needle in i.file_name.casefold()]
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            self._set_row(row, item)

    def _on_filter_changed(self, text: str) -> None:
        """Filter-line-edit hook: store the needle and re-render.

        Immediate (not debounced): human typing is at most ~10 Hz, and
        users expect the table to change the instant they type a
        character. Debounce is reserved for machine-driven progress
        bursts which can fire at 10× that rate.
        """
        self._filter_text = text.strip()
        self._do_refresh()

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
        from src.infrastructure.file_utils import is_valid_pdf

        paths: list[Path] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_dir():
                paths.extend(p for p in sorted(local.rglob("*.pdf")) if is_valid_pdf(p))
            elif local.suffix.lower() == ".pdf" and is_valid_pdf(local):
                paths.append(local)
            else:
                logger.info("Игнорируем drag-drop: %s — не валидный PDF", local)
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
        menu.addSeparator()
        clear_act = QAction("Очистить завершённые", self)
        clear_act.triggered.connect(self._on_clear_completed_requested)
        menu.addAction(clear_act)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _on_clear_completed_requested(self) -> None:
        """Menu entry-point: ask for confirmation before deleting rows.

        We show the prompt only when there's actually something to remove —
        an empty confirm dialog would be just noise. Tests bypass this
        wrapper and call ``_on_clear_completed`` directly.
        """
        from PySide6.QtWidgets import QMessageBox

        if self._queue is None:
            return
        terminal_statuses = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        count = sum(1 for it in self._queue.list_items() if it.status in terminal_statuses)
        if count == 0:
            return
        reply = QMessageBox.question(
            self,
            "Очистить очередь",
            f"Удалить из очереди {count} завершённых/отменённых заданий?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._on_clear_completed()

    def _on_clear_completed(self) -> None:
        """Remove every COMPLETED / FAILED / CANCELLED row from the queue."""
        if self._queue is None:
            return
        removed = self._queue.clear_completed()
        logger.info("Cleared %d terminal queue items", removed)
        # Direct user action — refresh immediately so the feedback is
        # instant, not 100 ms later.
        self._do_refresh()
