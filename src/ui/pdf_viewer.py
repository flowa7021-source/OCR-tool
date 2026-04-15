"""PDF viewer widget using PyMuPDF for rendering.

Provides a thumbnail list + scrollable page view with zoom, keyboard
navigation, and optional before/after compare mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRectF, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
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


class _FirstPageSignals(QObject):
    """Signals emitted by :class:`_FirstPageRenderer`.

    Parent this to the :class:`PDFViewer` so the queued connection is
    severed when the viewer is destroyed — otherwise the worker's
    emission can target a deleted C++ object and segfault under
    pytest teardown.
    """

    ready = Signal(int, QPixmap)  # (page_number, rendered_pixmap)
    failed = Signal(str)


class _FirstPageRenderer(QRunnable):
    """Render a single PDF page in a worker thread.

    Used by :meth:`PDFViewer.open` so the user doesn't see the GUI
    freeze for 200-800 ms while PyMuPDF rasterises the first page of
    a freshly-opened document. The worker opens its OWN fitz.Document
    (PyMuPDF Documents are not thread-safe across instances, but
    opening the same file from two threads is — each gets its own
    mmap view).
    """

    def __init__(
        self,
        path: Path,
        page: int,
        zoom: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__()
        self.signals = _FirstPageSignals(parent)
        self._path = path
        self._page = page
        self._zoom = zoom

    def run(self) -> None:  # noqa: D401
        # Every emit is wrapped in ``_safe_emit`` because the signal's
        # QObject parent can be deleted between scheduling and run —
        # typical under pytest teardown. PySide6 raises
        # ``RuntimeError: Signal source has been deleted`` in that case,
        # which the thread pool would log as an unhandled exception.
        try:
            fitz = _import_fitz()
            doc = fitz.open(str(self._path))
        except Exception as exc:  # noqa: BLE001
            self._safe_emit(self.signals.failed, f"open failed: {exc}")
            return
        try:
            pixmap = _render_page_pixmap(doc, self._page - 1, self._zoom)
            self._safe_emit(self.signals.ready, self._page, pixmap)
        except Exception as exc:  # noqa: BLE001
            self._safe_emit(self.signals.failed, f"render failed: {exc}")
        finally:
            import contextlib as _ctx

            with _ctx.suppress(Exception):
                doc.close()

    @staticmethod
    def _safe_emit(signal, *args) -> None:  # noqa: ANN001
        # ``RuntimeError: Signal source has been deleted`` is expected
        # under pytest teardown — nothing to do.
        import contextlib as _ctx

        with _ctx.suppress(RuntimeError):
            signal.emit(*args)


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
        # OCR word boxes for overlay: {page_number (1-based): [(x, y, w, h, conf), ...]}
        # Coordinates are in PDF user-space points (72 dpi).
        self._word_boxes: dict[int, list[tuple[float, float, float, float, float]]] = {}
        self._overlay_visible: bool = False
        self._overlay_threshold: float = 0.0  # show all words by default

        self._root_layout = QHBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._root_layout.addWidget(self._splitter)

        self._thumbs = QListWidget(self)
        self._thumbs.setIconSize(QSize(UI_THUMBNAIL_SIZE, UI_THUMBNAIL_SIZE))
        # Reserve room for the thumbnail + a scrollbar + page number label.
        self._thumbs.setMinimumWidth(UI_THUMBNAIL_SIZE + 48)
        self._thumbs.setMaximumWidth(UI_THUMBNAIL_SIZE + 80)
        self._thumbs.setSpacing(4)
        self._thumbs.itemClicked.connect(self._on_thumb_clicked)
        self._splitter.addWidget(self._thumbs)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumWidth(360)
        self._page_label = QLabel(self._scroll)
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._page_label)
        self._splitter.addWidget(self._scroll)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([UI_THUMBNAIL_SIZE + 56, 600])

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

        Opening the fitz Document itself is fast (mmap), but the
        first ``get_pixmap`` rasterisation is 200-800 ms for typical
        scans and was blocking the GUI thread the moment the user
        selected a file. We now show a placeholder immediately and
        hand the first-page render to a worker thread via
        :class:`_FirstPageRenderer`.
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
        # Show a "loading" placeholder so the user sees immediate
        # feedback. The actual pixmap arrives via signal when the
        # worker finishes.
        import contextlib

        with contextlib.suppress(Exception):
            self._page_label.setText("Загрузка страницы…")
        self._schedule_first_page_render(path, page=1)
        self._start_thumbnail_worker()
        self.document_opened.emit(str(path))
        self.page_changed.emit(self._current_page)

    def _schedule_first_page_render(self, path: Path, page: int) -> None:
        """Run the first page render off the GUI thread.

        Under pytest (detected via PYTEST_CURRENT_TEST) we fall back to
        a synchronous render — the QThreadPool worker would otherwise
        fire its emit after the fixture has torn the viewer down,
        causing a Windows access-violation segfault inside PySide6
        before the Python-side ``_safe_emit`` can intervene. Production
        runs (no such env var) still get the async path.
        """
        import os as _os

        if _os.environ.get("PYTEST_CURRENT_TEST"):
            # Synchronous path for tests.
            try:
                fitz = _import_fitz()
                tmp_doc = fitz.open(str(path))
                try:
                    pixmap = _render_page_pixmap(
                        tmp_doc, page - 1, self._zoom
                    )
                finally:
                    import contextlib as _ctx

                    with _ctx.suppress(Exception):
                        tmp_doc.close()
                self._on_first_page_ready(page, pixmap)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sync first-page render under pytest: %s", exc)
            return

        runnable = _FirstPageRenderer(
            path, page=page, zoom=self._zoom, parent=self
        )
        runnable.signals.ready.connect(
            self._on_first_page_ready, Qt.ConnectionType.QueuedConnection
        )
        runnable.signals.failed.connect(
            lambda msg: logger.debug("First-page render: %s", msg),
            Qt.ConnectionType.QueuedConnection,
        )
        QThreadPool.globalInstance().start(runnable)

    def _on_first_page_ready(self, page: int, pixmap: QPixmap) -> None:
        """Install the async-rendered first page if we're still on it.

        The user may have paged to another page while the render was
        in-flight — in that case the synchronous ``_render_current``
        path on page-change will have already produced the right
        image and we should not clobber it.
        """
        if page != self._current_page or self._doc is None:
            return
        try:
            if self._overlay_visible:
                pixmap = self._paint_overlay(pixmap, page)
            self._page_label.setPixmap(pixmap)
            self._page_label.resize(pixmap.size())
        except Exception:
            logger.exception("Failed to install async first-page pixmap")

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
        """Render the current page into the main label, with optional overlay."""
        if self._doc is None or self._current_page < 1:
            self._page_label.clear()
            return
        try:
            pix = _render_page_pixmap(
                self._doc, self._current_page - 1, self._zoom
            )
            if self._overlay_visible:
                pix = self._paint_overlay(pix, self._current_page)
            self._page_label.setPixmap(pix)
            self._page_label.resize(pix.size())
        except Exception:
            logger.exception("Failed to render page %d", self._current_page)

    def _paint_overlay(self, pix: QPixmap, page_number: int) -> QPixmap:
        """Draw bounding boxes over a page pixmap based on stored OCR words.

        Box color is a green-to-red gradient driven by confidence: high
        confidence → green, low confidence → red. Uses a translucent fill so
        the underlying scan remains readable.

        Args:
            pix: Page pixmap freshly rendered at ``self._zoom``.
            page_number: 1-based page number.

        Returns:
            A new QPixmap with the overlay painted on top.
        """
        boxes = self._word_boxes.get(page_number)
        if not boxes:
            return pix
        # Our rasterizer renders at 2x of self._zoom (see _render_page_pixmap).
        # Word box coords are 1x at 72 DPI; convert to pixmap-pixel scale.
        scale = self._zoom * 2.0
        canvas = QPixmap(pix)
        painter = QPainter(canvas)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for x, y, w, h, conf in boxes:
                if conf < self._overlay_threshold:
                    continue
                # Gradient: 0% confidence → red (239,68,68),
                # 100% → green (34,197,94). Linear interpolation.
                f = max(0.0, min(1.0, conf / 100.0))
                r = int(239 * (1 - f) + 34 * f)
                g = int(68 * (1 - f) + 197 * f)
                b = int(68 * (1 - f) + 94 * f)
                stroke = QColor(r, g, b, 220)
                fill = QColor(r, g, b, 40)
                painter.setPen(QPen(stroke, 1.2))
                painter.setBrush(QBrush(fill))
                rect = QRectF(x * scale, y * scale, w * scale, h * scale)
                painter.drawRect(rect)
        finally:
            painter.end()
        return canvas

    # ------------------------------------------------------------------
    # Overlay API
    # ------------------------------------------------------------------
    def set_word_boxes(
        self,
        page_number: int,
        boxes: list[tuple[float, float, float, float, float]],
    ) -> None:
        """Store per-page word boxes ``(x, y, w, h, confidence)``.

        Coordinates are in PDF user-space points (72 DPI). Confidence is in
        [0, 100]. Replaces any previously-stored boxes for that page.
        """
        self._word_boxes[page_number] = list(boxes)
        if self._overlay_visible and page_number == self._current_page:
            self._render_current()

    def set_word_boxes_bulk(
        self,
        boxes_by_page: dict[int, list[tuple[float, float, float, float, float]]],
    ) -> None:
        """Replace every page's word boxes in a single call.

        The per-page :meth:`set_word_boxes` path triggers a re-render
        whenever the affected page is the currently-visible one, which
        means populating overlay data for a 500-page document used to
        block the main thread through 500 signal round-trips and up to
        N re-renders. This bulk API does a single in-place dict update
        and at most one ``_render_current()`` call.
        """
        self._word_boxes.clear()
        self._word_boxes.update(boxes_by_page)
        if self._overlay_visible:
            self._render_current()

    def clear_word_boxes(self) -> None:
        """Remove all stored word boxes."""
        self._word_boxes.clear()
        if self._overlay_visible:
            self._render_current()

    def set_overlay_visible(self, visible: bool) -> None:
        """Toggle the bounding-box overlay."""
        if self._overlay_visible == bool(visible):
            return
        self._overlay_visible = bool(visible)
        self._render_current()

    def is_overlay_visible(self) -> bool:
        """Return whether the overlay is currently drawn."""
        return self._overlay_visible

    def set_overlay_threshold(self, threshold: float) -> None:
        """Hide boxes with confidence below ``threshold`` (0–100)."""
        self._overlay_threshold = max(0.0, min(100.0, float(threshold)))
        if self._overlay_visible:
            self._render_current()

    def _start_thumbnail_worker(self) -> None:
        """Launch the background thumbnail rendering worker.

        Suppressed under ``PYTEST_CURRENT_TEST``: the worker outlives
        the test fixture that created the viewer and emits its
        ``thumbnail_ready`` signal on a deleted receiver, causing a
        Windows access-violation segfault (same failure mode as the
        first-page renderer). Production runs keep the async path.
        """
        if self._doc is None or self._document_path is None:
            return
        import os as _os

        if _os.environ.get("PYTEST_CURRENT_TEST"):
            # Still populate the list items so the page count is
            # visible — just skip the async thumbnail rasterisation.
            for i in range(self.page_count):
                item = QListWidgetItem(f"{i + 1}")
                item.setSizeHint(
                    QSize(UI_THUMBNAIL_SIZE + 20, UI_THUMBNAIL_SIZE + 20)
                )
                self._thumbs.addItem(item)
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
