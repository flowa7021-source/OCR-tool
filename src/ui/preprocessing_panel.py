"""Preprocessing panel with live before/after preview."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.models import (
    BackgroundConfig,
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    DewarpConfig,
    PreprocessConfig,
)
from src.shared.constants import (
    ADAPTIVE_BLOCK_SIZE_RANGE,
    ADAPTIVE_C_RANGE,
    CLAHE_CLIP_RANGE,
    CLAHE_TILE_RANGE,
    DESKEW_MANUAL_RANGE,
    NLM_H_RANGE,
    UI_PREVIEW_UPDATE_DEBOUNCE_MS,
)
from src.shared.types import BinarizationMethod, DenoiseMethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preview rasterisation LRU cache
# ---------------------------------------------------------------------------
#
# Every tick of a slider in this panel used to pay for a fresh
# ``fitz.open`` + ``get_pixmap`` for the same page. On a 300 DPI A4
# scan that's 80-150 ms — enough to feel laggy when dragging.
#
# The cache is an ordered dict keyed by ``(path, mtime, page_num, dpi)``.
# Keying on mtime means an external edit of the PDF invalidates the
# cached pixmap automatically without any explicit refresh.

_PREVIEW_CACHE_MAX: int = 3  # at ~10 MB / entry keeps total under 30 MB
_PREVIEW_CACHE: OrderedDict[tuple[str, int, int, int], Any] = OrderedDict()
_PREVIEW_CACHE_LOCK: threading.Lock = threading.Lock()


def _get_preview_cache() -> tuple[OrderedDict, threading.Lock]:
    """Return the module-level preview cache + its lock (test hook)."""
    return _PREVIEW_CACHE, _PREVIEW_CACHE_LOCK


def _cached_rasterize(pdf_path: Path, page_num: int, dpi: int) -> Any:
    """Return a numpy BGR/grayscale image for ``pdf_path``'s page ``page_num``.

    Hits the in-memory LRU when available; falls back to PyMuPDF
    rasterisation otherwise. Callers MUST treat the returned array as
    read-only (we return a shallow view to save another copy).
    """
    import fitz  # lazy
    import numpy as np

    cache, lock = _get_preview_cache()
    try:
        mtime = int(pdf_path.stat().st_mtime_ns)
    except OSError:
        mtime = 0
    key = (str(pdf_path), mtime, int(page_num), int(dpi))

    with lock:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    doc = fitz.open(str(pdf_path))
    try:
        if page_num < 1 or page_num > doc.page_count:
            raise ValueError(
                f"page {page_num} out of range 1..{doc.page_count}"
            )
        pix = doc[page_num - 1].get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            arr = arr[:, :, :3]
        arr = arr.copy()  # detach from PyMuPDF buffer so the cache survives doc.close
    finally:
        doc.close()

    with lock:
        cache[key] = arr
        cache.move_to_end(key)
        while len(cache) > _PREVIEW_CACHE_MAX:
            cache.popitem(last=False)
    return arr


def clear_preview_cache() -> None:
    """Drop every cached preview page (used by tests and UI reset hooks)."""
    cache, lock = _get_preview_cache()
    with lock:
        cache.clear()


class _PreviewWorker(QRunnable):
    """Background worker that renders a preview step."""

    class Signals(QObject):
        ready = Signal(object, object)  # (before_pixmap, after_pixmap)
        failed = Signal(str)

    def __init__(
        self,
        pdf_path: Path,
        page_num: int,
        stage: str,
        config: PreprocessConfig,
    ) -> None:
        super().__init__()
        self._pdf_path = pdf_path
        self._page_num = page_num
        self._stage = stage
        self._config = config
        self.signals = self.Signals()

    def run(self) -> None:  # noqa: D401 — Qt override
        try:
            from src.core.image_preprocessor import preview_step

            # Use the module-level LRU cache: the user typically drags
            # sliders while staying on the same page, so 99 % of preview
            # repaints skip rasterisation entirely.
            arr = _cached_rasterize(self._pdf_path, self._page_num, dpi=120)
            before = arr  # read-only view; preview_step is not mutating
            after = preview_step(arr, self._stage, self._config)

            before_pix = _ndarray_to_pixmap(before)
            after_pix = _ndarray_to_pixmap(after)
            self.signals.ready.emit(before_pix, after_pix)
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Preview rendering failed: %s", exc)
            self.signals.failed.emit(str(exc))


def _ndarray_to_pixmap(arr: Any) -> QPixmap:
    """Convert a numpy image (grayscale or BGR/RGB) to QPixmap."""
    import numpy as np

    if arr.ndim == 2:
        h, w = arr.shape
        arr = np.ascontiguousarray(arr)
        img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        # Assume BGR from OpenCV, convert to RGB
        h, w, n = arr.shape
        rgb = arr[:, :, ::-1].copy() if n == 3 else arr[:, :, :3][:, :, ::-1].copy()
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img)


class PreprocessingPanel(QWidget):
    """Widget to configure preprocessing pipeline with live preview.

    Emits :py:attr:`config_changed` (debounced) whenever any control changes.
    """

    config_changed = Signal(object)  # PreprocessConfig

    STAGES = ("original", "dewarp", "deskew", "contrast", "background", "denoise", "binarization")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = PreprocessConfig()
        self._preview_path: Path | None = None
        self._preview_page: int = 1
        self._suppress_signals = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(UI_PREVIEW_UPDATE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_and_preview)

        self._threadpool = QThreadPool.globalInstance()

        self._build_ui()
        self.set_config(self._config)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # Preview row
        self.before_label = QLabel("Оригинал")
        self.before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.before_label.setMinimumSize(200, 260)
        self.after_label = QLabel("Результат")
        self.after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.after_label.setMinimumSize(200, 260)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        preview_splitter.setChildrenCollapsible(False)
        preview_splitter.addWidget(self.before_label)
        preview_splitter.addWidget(self.after_label)
        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setSizes([260, 260])
        root.addWidget(preview_splitter, 1)

        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Этап предобработки:"))
        self.stage_combo = QComboBox(self)
        for s in self.STAGES:
            self.stage_combo.addItem(s)
        self.stage_combo.setCurrentText("binarization")
        self.stage_combo.currentIndexChanged.connect(self._on_any_change)
        stage_row.addWidget(self.stage_combo, 1)
        root.addLayout(stage_row)

        # Scrollable config area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form_layout = QVBoxLayout(inner)
        form_layout.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll, 2)

        form_layout.addWidget(self._build_deskew_group())
        form_layout.addWidget(self._build_dewarp_group())
        form_layout.addWidget(self._build_binarization_group())
        form_layout.addWidget(self._build_denoise_group())
        form_layout.addWidget(self._build_contrast_group())
        form_layout.addWidget(self._build_background_group())
        form_layout.addStretch(1)

    def _build_deskew_group(self) -> QGroupBox:
        g = QGroupBox("Коррекция наклона (deskew)")
        g.setCheckable(True)
        g.toggled.connect(self._on_any_change)
        self.deskew_group = g

        layout = QFormLayout(g)
        self.deskew_auto = QRadioButton("Автоопределение угла")
        self.deskew_manual_rb = QRadioButton("Задать угол вручную")
        rb_group = QButtonGroup(g)
        rb_group.addButton(self.deskew_auto)
        rb_group.addButton(self.deskew_manual_rb)
        self.deskew_auto.toggled.connect(self._on_any_change)
        self.deskew_manual_rb.toggled.connect(self._on_any_change)

        row = QHBoxLayout()
        row.addWidget(self.deskew_auto)
        row.addWidget(self.deskew_manual_rb)
        row.addStretch(1)
        layout.addRow(row)

        self.deskew_angle = QDoubleSpinBox(g)
        self.deskew_angle.setRange(*DESKEW_MANUAL_RANGE)
        self.deskew_angle.setDecimals(2)
        self.deskew_angle.setSingleStep(0.1)
        self.deskew_angle.valueChanged.connect(self._on_any_change)
        layout.addRow("Угол (°):", self.deskew_angle)
        return g

    def _build_dewarp_group(self) -> QGroupBox:
        g = QGroupBox("Выпрямление деформаций (dewarp)")
        g.setCheckable(True)
        g.toggled.connect(self._on_any_change)
        self.dewarp_group = g
        lay = QVBoxLayout(g)
        lay.addWidget(QLabel(
            "Для фотографий книг и деформированных страниц. Применяется до бинаризации."
        ))
        return g

    def _build_binarization_group(self) -> QGroupBox:
        g = QGroupBox("Бинаризация")
        lay = QFormLayout(g)

        self.bin_method = QComboBox(g)
        for m in BinarizationMethod:
            self.bin_method.addItem(m.value.upper(), userData=m.value)
        self.bin_method.currentIndexChanged.connect(self._on_any_change)
        lay.addRow("Метод:", self.bin_method)

        self.adaptive_block = QSpinBox(g)
        self.adaptive_block.setRange(*ADAPTIVE_BLOCK_SIZE_RANGE)
        self.adaptive_block.setSingleStep(2)
        self.adaptive_block.valueChanged.connect(self._snap_odd_and_emit)
        lay.addRow("Размер блока (adaptive):", self.adaptive_block)

        self.adaptive_c = QSpinBox(g)
        self.adaptive_c.setRange(*ADAPTIVE_C_RANGE)
        self.adaptive_c.valueChanged.connect(self._on_any_change)
        lay.addRow("Константа C (adaptive):", self.adaptive_c)

        self.sauvola_window = QSpinBox(g)
        self.sauvola_window.setRange(*ADAPTIVE_BLOCK_SIZE_RANGE)
        self.sauvola_window.valueChanged.connect(self._on_any_change)
        lay.addRow("Окно (Sauvola):", self.sauvola_window)

        self.sauvola_k = QDoubleSpinBox(g)
        self.sauvola_k.setRange(0.05, 0.5)
        self.sauvola_k.setSingleStep(0.05)
        self.sauvola_k.valueChanged.connect(self._on_any_change)
        lay.addRow("k (Sauvola):", self.sauvola_k)
        return g

    def _build_denoise_group(self) -> QGroupBox:
        g = QGroupBox("Шумоподавление")
        g.setCheckable(True)
        g.toggled.connect(self._on_any_change)
        self.denoise_group = g

        lay = QVBoxLayout(g)
        self.denoise_list = QListWidget(g)
        lay.addWidget(self.denoise_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Добавить", g)
        add_menu = QMenu(add_btn)
        for m in DenoiseMethod:
            action = add_menu.addAction(m.value)
            action.triggered.connect(lambda _checked=False, method=m: self._add_denoise_step(method))
        add_btn.setMenu(add_menu)
        btn_row.addWidget(add_btn)

        rm_btn = QPushButton("Удалить", g)
        rm_btn.clicked.connect(self._remove_denoise_step)
        btn_row.addWidget(rm_btn)

        up_btn = QPushButton("↑", g)
        up_btn.clicked.connect(lambda: self._move_denoise_step(-1))
        btn_row.addWidget(up_btn)

        down_btn = QPushButton("↓", g)
        down_btn.clicked.connect(lambda: self._move_denoise_step(1))
        btn_row.addWidget(down_btn)

        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        nlm_row = QFormLayout()
        self.nlm_h = QSpinBox(g)
        self.nlm_h.setRange(*NLM_H_RANGE)
        self.nlm_h.valueChanged.connect(self._on_denoise_h_changed)
        nlm_row.addRow("NLM h (для выбранного шага):", self.nlm_h)
        lay.addLayout(nlm_row)
        return g

    def _build_contrast_group(self) -> QGroupBox:
        g = QGroupBox("Контраст и яркость")
        lay = QFormLayout(g)

        self.clahe_check = QCheckBox("Включить CLAHE", g)
        self.clahe_check.toggled.connect(self._on_any_change)
        lay.addRow(self.clahe_check)

        self.clahe_clip = QDoubleSpinBox(g)
        self.clahe_clip.setRange(*CLAHE_CLIP_RANGE)
        self.clahe_clip.setSingleStep(0.5)
        self.clahe_clip.valueChanged.connect(self._on_any_change)
        lay.addRow("CLAHE clip limit:", self.clahe_clip)

        self.clahe_tile = QSpinBox(g)
        self.clahe_tile.setRange(*CLAHE_TILE_RANGE)
        self.clahe_tile.valueChanged.connect(self._on_any_change)
        lay.addRow("CLAHE tile size:", self.clahe_tile)

        self.manual_contrast = QCheckBox("Ручная коррекция", g)
        self.manual_contrast.toggled.connect(self._on_any_change)
        lay.addRow(self.manual_contrast)

        self.alpha = QDoubleSpinBox(g)
        self.alpha.setRange(0.5, 3.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.valueChanged.connect(self._on_any_change)
        lay.addRow("Контраст (α):", self.alpha)

        self.beta = QSpinBox(g)
        self.beta.setRange(-100, 100)
        self.beta.valueChanged.connect(self._on_any_change)
        lay.addRow("Яркость (β):", self.beta)
        return g

    def _build_background_group(self) -> QGroupBox:
        g = QGroupBox("Удаление теней и фона")
        g.setCheckable(True)
        g.toggled.connect(self._on_any_change)
        self.bg_group = g
        lay = QFormLayout(g)
        self.bg_kernel = QSpinBox(g)
        self.bg_kernel.setRange(11, 201)
        self.bg_kernel.setSingleStep(2)
        self.bg_kernel.valueChanged.connect(self._snap_odd_and_emit)
        lay.addRow("Размер ядра размытия:", self.bg_kernel)
        return g

    # --------------------------------------------------------- denoise mgmt
    def _add_denoise_step(self, method: DenoiseMethod) -> None:
        step = DenoiseStep(method=method)
        self._config.denoise.steps.append(step)
        self._refresh_denoise_list()
        self._on_any_change()

    def _remove_denoise_step(self) -> None:
        idx = self.denoise_list.currentRow()
        if 0 <= idx < len(self._config.denoise.steps):
            self._config.denoise.steps.pop(idx)
            self._refresh_denoise_list()
            self._on_any_change()

    def _move_denoise_step(self, delta: int) -> None:
        idx = self.denoise_list.currentRow()
        steps = self._config.denoise.steps
        new_idx = idx + delta
        if 0 <= idx < len(steps) and 0 <= new_idx < len(steps):
            steps[idx], steps[new_idx] = steps[new_idx], steps[idx]
            self._refresh_denoise_list()
            self.denoise_list.setCurrentRow(new_idx)
            self._on_any_change()

    def _refresh_denoise_list(self) -> None:
        self.denoise_list.clear()
        for step in self._config.denoise.steps:
            QListWidgetItem(f"{step.method.value} (ksize={step.ksize}, h={step.h})", self.denoise_list)

    def _on_denoise_h_changed(self, value: int) -> None:
        idx = self.denoise_list.currentRow()
        if 0 <= idx < len(self._config.denoise.steps):
            self._config.denoise.steps[idx].h = value
            self._refresh_denoise_list()
            self.denoise_list.setCurrentRow(idx)
        self._on_any_change()

    # --------------------------------------------------------- snapping
    @Slot(int)
    def _snap_odd_and_emit(self, value: int) -> None:
        if self._suppress_signals:
            return
        sender = self.sender()
        if isinstance(sender, QSpinBox) and value % 2 == 0:
            sender.blockSignals(True)
            sender.setValue(value + 1 if value + 1 <= sender.maximum() else value - 1)
            sender.blockSignals(False)
        self._on_any_change()

    # --------------------------------------------------------- state
    def set_config(self, cfg: PreprocessConfig) -> None:
        """Load the given config into all widgets."""
        self._suppress_signals = True
        try:
            self._config = cfg
            # Deskew
            self.deskew_group.setChecked(cfg.deskew.enabled)
            self.deskew_auto.setChecked(cfg.deskew.auto_detect)
            self.deskew_manual_rb.setChecked(not cfg.deskew.auto_detect)
            self.deskew_angle.setValue(cfg.deskew.manual_angle)
            # Dewarp
            self.dewarp_group.setChecked(cfg.dewarp.enabled)
            # Binarization
            idx = self.bin_method.findData(cfg.binarization.method.value)
            if idx >= 0:
                self.bin_method.setCurrentIndex(idx)
            self.adaptive_block.setValue(cfg.binarization.adaptive_block_size)
            self.adaptive_c.setValue(cfg.binarization.adaptive_c)
            self.sauvola_window.setValue(cfg.binarization.sauvola_window)
            self.sauvola_k.setValue(cfg.binarization.sauvola_k)
            # Denoise
            self.denoise_group.setChecked(cfg.denoise.enabled)
            self._refresh_denoise_list()
            # Contrast
            self.clahe_check.setChecked(cfg.contrast.clahe_enabled)
            self.clahe_clip.setValue(cfg.contrast.clahe_clip)
            self.clahe_tile.setValue(cfg.contrast.clahe_tile)
            self.manual_contrast.setChecked(cfg.contrast.manual_enabled)
            self.alpha.setValue(cfg.contrast.alpha)
            self.beta.setValue(cfg.contrast.beta)
            # Background
            self.bg_group.setChecked(cfg.background.enabled)
            self.bg_kernel.setValue(cfg.background.blur_kernel)
        finally:
            self._suppress_signals = False

    def get_config(self) -> PreprocessConfig:
        """Return a PreprocessConfig built from current widget values."""
        cfg = PreprocessConfig(
            deskew=DeskewConfig(
                enabled=self.deskew_group.isChecked(),
                auto_detect=self.deskew_auto.isChecked(),
                manual_angle=float(self.deskew_angle.value()),
            ),
            dewarp=DewarpConfig(enabled=self.dewarp_group.isChecked()),
            binarization=BinarizationConfig(
                method=BinarizationMethod(self.bin_method.currentData()),
                adaptive_block_size=self.adaptive_block.value(),
                adaptive_c=self.adaptive_c.value(),
                sauvola_window=self.sauvola_window.value(),
                sauvola_k=float(self.sauvola_k.value()),
            ),
            denoise=DenoiseConfig(
                enabled=self.denoise_group.isChecked(),
                steps=list(self._config.denoise.steps),
            ),
            contrast=ContrastConfig(
                clahe_enabled=self.clahe_check.isChecked(),
                clahe_clip=float(self.clahe_clip.value()),
                clahe_tile=self.clahe_tile.value(),
                manual_enabled=self.manual_contrast.isChecked(),
                alpha=float(self.alpha.value()),
                beta=self.beta.value(),
            ),
            background=BackgroundConfig(
                enabled=self.bg_group.isChecked(),
                blur_kernel=self.bg_kernel.value(),
            ),
        )
        self._config = cfg
        return cfg

    # --------------------------------------------------------- preview
    def set_preview_source(self, pdf_path: Path | None, page_num: int = 1) -> None:
        """Set the PDF page used as the preview source."""
        self._preview_path = pdf_path
        self._preview_page = page_num
        self._schedule_emit()

    # --------------------------------------------------------- debouncing
    def _on_any_change(self, *args: Any, **kwargs: Any) -> None:
        if self._suppress_signals:
            return
        self._schedule_emit()

    def _schedule_emit(self) -> None:
        self._debounce.start()

    def _emit_and_preview(self) -> None:
        cfg = self.get_config()
        self.config_changed.emit(cfg)
        self._trigger_preview(cfg)

    def _trigger_preview(self, cfg: PreprocessConfig) -> None:
        if self._preview_path is None or not self._preview_path.exists():
            return
        stage = self.stage_combo.currentText()
        worker = _PreviewWorker(self._preview_path, self._preview_page, stage, cfg)
        worker.signals.ready.connect(self._on_preview_ready)
        worker.signals.failed.connect(lambda msg: logger.warning("preview failed: %s", msg))
        self._threadpool.start(worker)

    @Slot(object, object)
    def _on_preview_ready(self, before: QPixmap, after: QPixmap) -> None:
        def _scale(p: QPixmap, label: QLabel) -> QPixmap:
            return p.scaled(
                max(label.width(), 100),
                max(label.height(), 100),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.before_label.setPixmap(_scale(before, self.before_label))
        self.after_label.setPixmap(_scale(after, self.after_label))
