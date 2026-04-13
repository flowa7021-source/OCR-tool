"""PDF viewer widget using PyMuPDF for rendering.

Provides a thumbnail list + scrollable page view with zoom, keyboard
navigation, and optional before/after compare mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QIcon, QImage, QKeyEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.shared.constants import UI_THUMBNAIL_SIZE

logger = logging.getLogger(__name__)

_MIN_ZOOM: float = 0.1
_MAX_ZOOM: float = 8.0


def _import_fitz() -> Any:
    """Lazy import PyMuPDF to avoid hard import failure at module load.

    Returns:
        The ``fitz`` module.
    """
    import fitz  # type: ignore[import-not-found]

    return fitz


def _render_page_pixmap(doc: Any, page_index: int, zoom: float) -> QPixmap:
    """Render a PDF page to a :class:`QPixmap`.

    Args:
        doc: Opened PyMuPDF document.
        page_index: Zero-based page index.
        zoom: Display zoom factor (1.0 = 100 %).

    Returns:
        QPixmap of the rendered page at ``zoom * 2`` DPI scale.
    """
    fitz = _import_fitz()
    page = doc.load_page(page_index)
    matrix = fitz.Matrix(zoom * 2, zoom * 2)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = QImage(
        pix.samples,
        pix.width,
        pix.height,
        pix.stride,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(image)


class _ThumbSignals(QObject):
    """Signals emitted by :class:`_ThumbnailWorker`."""

    thumbnail_ready = Signal(int, QImage)
    finished = Signal()


class _ThumbnailWorker(QRunnable):
    """Background worker that renders PDF thumbnails one-by-one."""

    def __init__(self, doc_path: Path, page_count: int, thumb_size: int) -> None:
        super().__init__()
        self.doc_path = doc_path
        self.page_count = page_count
        self.thumb_size = thumb_size
        self.signals = _ThumbSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request the worker to stop at the next iteration."""
        self._cancelled = True

    def run(self) -> None:  # noqa: D401 - Qt entry point
        """Render thumbnails sequentially and emit them as they are ready."""
        try:
            fitz = _import_fitz()
            doc = fitz.open(str(self.doc_path))
        except Exception:  # pragma: no cover - runtime dependency
            logger.exception("Failed to open %s for thumbnails", self.doc_path)
            self.signals.finished.emit()
            return
        try:
            for i in range(min(self.page_count, doc.page_count)):
                if self._cancelled:
                    break
                try:
                    page = doc.load_page(i)
                    rect = page.rect
                    scale = self.thumb_size / max(rect.width, rect.height, 1.0)
                    matrix = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    image = QImage(
                        pix.samples,
                        pix.width,
                        pix.height,
                        pix.stride,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.signals.thumbnail_ready.emit(i, image)
                except Exception:
                    logger.exception("Failed to render thumbnail %d", i)
        finally:
            doc.close()
            self.signals.finished.emit()


class PDFViewer(QWidget):
    """Scrollable PDF viewer with thumbnails, zoom, and compare mode."""

    page_changed = Signal(int)
    document_opened = Signal(str)
    document_closed = Signal()
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the viewer widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._doc: Any = None
        self._document_path: Path | None = None
        self._current_page: int = 0
        self._zoom: float = 1.0
        self._thumb_worker: _ThumbnailWorker | None = None
        self._compare_splitter: QSplitter | None = None
        self._compare_before: PDFViewer | None = None
        self._compare_after: PDFViewer | None = None

        self._root_layout = QHBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._root_layout.addWidget(self._splitter)

        self._thumbs = QListWidget(self)
        self._thumbs.setIconSize(QSize(UI_THUMBNAIL_SIZE, UI_THUMBNAIL_SIZE))
        self._thumbs.setFixedWidth(UI_THUMBNAIL_SIZE + 40)
        self._thumbs.setSpacing(4)
        self._thumbs.itemClicked.connect(self._on_thumb_clicked)
        self._splitter.addWidget(self._thumbs)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._page_label = QLabel(self._scroll)
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._page_label)
        self._splitter.addWidget(self._scroll)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_page(self) -> int:
        """Return the 1-based current page number (0 when no document)."""
        return self._current_page

    @property
    def page_count(self) -> int:
        """Return the total number of pages in the open document."""
        if self._doc is None:
            return 0
        return int(self._doc.page_count)

    @property
    def zoom_factor(self) -> float:
        """Return the current zoom factor (1.0 = 100 %)."""
        return self._zoom

    @property
    def document_path(self) -> Path | None:
        """Return the :class:`~pathlib.Path` of the currently open document."""
        return self._document_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open(self, path: Path) -> None:
        """Open a PDF document and display its first page.

        Args:
            path: Path to the PDF file.
        """
        self.close_document()
        try:
            fitz = _import_fitz()
            self._doc = fitz.open(str(path))
        except Exception:
            logger.exception("Failed to open PDF %s", path)
            self._doc = None
            return
        self._document_path = path
        self._current_page = 1
        self._render_current()
        self._start_thumbnail_worker()
        self.document_opened.emit(str(path))
        self.page_changed.emit(self._current_page)

    def close_document(self) -> None:
        """Close the currently open document and reset state."""
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker = None
        self._thumbs.clear()
        self._page_label.clear()
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:  # pragma: no cover
                logger.exception("Error closing document")
        self._doc = None
        self._document_path = None
        self._current_page = 0
        self.document_closed.emit()

    def set_page(self, n: int) -> None:
        """Navigate to the given 1-based page.

        Args:
            n: Target page number (1-based).
        """
        if self._doc is None:
            return
        n = max(1, min(n, self.page_count))
        if n == self._current_page:
            return
        self._current_page = n
        self._render_current()
        self.page_changed.emit(self._current_page)

    def set_zoom(self, factor: float) -> None:
        """Set zoom to ``factor`` (clamped to [0.1, 8.0]).

        Args:
            factor: New zoom factor.
        """
        factor = max(_MIN_ZOOM, min(_MAX_ZOOM, float(factor)))
        if abs(factor - self._zoom) < 1e-4:
            return
        self._zoom = factor
        self._render_current()
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        """Increase zoom by 25 %."""
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """Decrease zoom by 20 %."""
        self.set_zoom(self._zoom / 1.25)

    def reset_zoom(self) -> None:
        """Restore zoom to 1.0 (100 %)."""
        self.set_zoom(1.0)

    def fit_to_width(self) -> None:
        """Scale the page so its width fills the scroll area viewport."""
        if self._doc is None:
            return
        try:
            page = self._doc.load_page(self._current_page - 1)
            page_width_pt = float(page.rect.width)
        except Exception:
            logger.exception("Could not query page width")
            return
        if page_width_pt <= 0:
            return
        # PDF points are 72 DPI; QPixmap width at zoom=1 is page_width_pt * 2.
        viewport_w = max(1, self._scroll.viewport().width() - 8)
        page_px_at_1 = page_width_pt * 2.0
        factor = viewport_w / page_px_at_1
        self.set_zoom(factor)

    def next_page(self) -> None:
        """Advance to the next page if possible."""
        self.set_page(self._current_page + 1)

    def prev_page(self) -> None:
        """Go to the previous page if possible."""
        self.set_page(self._current_page - 1)

    def set_compare_mode(
        self, before_path: Path | None, after_path: Path | None
    ) -> None:
        """Toggle before/after compare mode.

        When both paths are provided the main viewer area is replaced by a
        horizontal splitter of two mini-viewers. Passing ``None`` for either
        argument reverts to single-document mode.

        Args:
            before_path: Path to the "before" PDF, or ``None`` to disable.
            after_path: Path to the "after" PDF, or ``None`` to disable.
        """
        if before_path is not None and after_path is not None:
            if self._compare_splitter is None:
                self._compare_splitter = QSplitter(Qt.Orientation.Horizontal, self)
                self._compare_before = PDFViewer(self._compare_splitter)
                self._compare_after = PDFViewer(self._compare_splitter)
                self._compare_splitter.addWidget(self._compare_before)
                self._compare_splitter.addWidget(self._compare_after)
                # Hide standard splitter, show compare splitter
                self._splitter.hide()
                self._root_layout.addWidget(self._compare_splitter)
            if self._compare_before is not None:
                self._compare_before.open(before_path)
            if self._compare_after is not None:
                self._compare_after.open(after_path)
        else:
            if self._compare_splitter is not None:
                if self._compare_before is not None:
                    self._compare_before.close_document()
                if self._compare_after is not None:
                    self._compare_after.close_document()
                self._compare_splitter.setParent(None)
                self._compare_splitter.deleteLater()
                self._compare_splitter = None
                self._compare_before = None
                self._compare_after = None
                self._splitter.show()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        """Zoom on Ctrl+wheel, otherwise let the scroll area handle it."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        """Navigate pages with Up/Down/PageUp/PageDown/Home/End."""
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self.prev_page()
            event.accept()
            return
        if key in (Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self.next_page()
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self.set_page(1)
            event.accept()
            return
        if key == Qt.Key.Key_End:
            self.set_page(self.page_count)
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _render_current(self) -> None:
        """Render the current page into the main label."""
        if self._doc is None or self._current_page < 1:
            self._page_label.clear()
            return
        try:
            pix = _render_page_pixmap(
                self._doc, self._current_page - 1, self._zoom
            )
            self._page_label.setPixmap(pix)
            self._page_label.resize(pix.size())
        except Exception:
            logger.exception("Failed to render page %d", self._current_page)

    def _start_thumbnail_worker(self) -> None:
        """Launch the background thumbnail rendering worker."""
        if self._doc is None or self._document_path is None:
            return
        worker = _ThumbnailWorker(
            self._document_path, self.page_count, UI_THUMBNAIL_SIZE
        )
        worker.signals.thumbnail_ready.connect(self._on_thumb_ready)
        self._thumb_worker = worker
        # Pre-populate items so the ordering is stable.
        for i in range(self.page_count):
            item = QListWidgetItem(f"{i + 1}")
            item.setSizeHint(QSize(UI_THUMBNAIL_SIZE + 20, UI_THUMBNAIL_SIZE + 20))
            self._thumbs.addItem(item)
        QThreadPool.globalInstance().start(worker)

    def _on_thumb_ready(self, index: int, image: QImage) -> None:
        """Slot: assign a thumbnail pixmap to its list item."""
        if index < 0 or index >= self._thumbs.count():
            return
        item = self._thumbs.item(index)
        if item is None:
            return
        item.setIcon(QIcon(QPixmap.fromImage(image)))

    def _on_thumb_clicked(self, item: QListWidgetItem) -> None:
        """Slot: jump to the page corresponding to a clicked thumbnail."""
        row = self._thumbs.row(item)
        self.set_page(row + 1)
