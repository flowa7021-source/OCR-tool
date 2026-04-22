"""Editor widget for :class:`OCRConfig` (EasyOCR back-end)."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.models import OCRConfig
from src.shared.constants import COLOR_TEXT_SECONDARY, DPI_CHOICES
from src.shared.types import OCREngineKind
from src.ui.accessibility import describe

logger = logging.getLogger(__name__)

_DEBOUNCE_MS: int = 300

_LANG_LABELS: dict[str, str] = {"ru": "Русский", "en": "English"}


class SettingsPanel(QWidget):
    """Widget that edits a :class:`OCRConfig` in real time."""

    config_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating: bool = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_config)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(self._scroll.Shape.NoFrame)
        outer.addWidget(self._scroll)

        inner = QWidget(self._scroll)
        self._scroll.setWidget(inner)
        root = QVBoxLayout(inner)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # --- Engine group --------------------------------------------
        # Single item today (EasyOCR), but kept as a combo so a future
        # back-end can slot in without a layout change.
        self._group_engine = QGroupBox("OCR-движок", self)
        engine_layout = QVBoxLayout(self._group_engine)
        self._cmb_engine = QComboBox(self._group_engine)
        from src.application.engines.registry import list_engines

        for kind, label, available, msg in list_engines():
            display = label if available else f"{label} (недоступен)"
            self._cmb_engine.addItem(display, kind)
            idx = self._cmb_engine.count() - 1
            tooltip = kind.description if available else f"{kind.description}\n\n{msg}"
            self._cmb_engine.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
            if not available:
                self._cmb_engine.model().item(idx).setEnabled(False)
        engine_layout.addWidget(self._cmb_engine)
        self._lbl_engine_hint = QLabel(self._group_engine)
        self._lbl_engine_hint.setWordWrap(True)
        self._lbl_engine_hint.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        engine_layout.addWidget(self._lbl_engine_hint)
        root.addWidget(self._group_engine)

        # --- Languages group -----------------------------------------
        self._group_langs = QGroupBox("Языки", self)
        langs_layout = QVBoxLayout(self._group_langs)
        self._chk_rus = QCheckBox(_LANG_LABELS["ru"], self._group_langs)
        self._chk_eng = QCheckBox(_LANG_LABELS["en"], self._group_langs)
        self._chk_rus.setChecked(True)
        self._chk_eng.setChecked(True)
        langs_layout.addWidget(self._chk_rus)
        langs_layout.addWidget(self._chk_eng)

        primary_row = QHBoxLayout()
        primary_row.addWidget(QLabel("Основной язык:", self._group_langs))
        self._cmb_primary = QComboBox(self._group_langs)
        primary_row.addWidget(self._cmb_primary, 1)
        langs_layout.addLayout(primary_row)
        root.addWidget(self._group_langs)

        # --- Advanced group ------------------------------------------
        self._group_adv = QGroupBox("Дополнительно", self)
        adv_layout = QFormLayout(self._group_adv)

        self._cmb_dpi = QComboBox(self._group_adv)
        for dpi in DPI_CHOICES:
            self._cmb_dpi.addItem(str(dpi), int(dpi))
        adv_layout.addRow("DPI:", self._cmb_dpi)

        self._edit_allowlist = QLineEdit(self._group_adv)
        describe(
            self._edit_allowlist,
            name="Allowlist символов",
            description=(
                "Если задан — EasyOCR распознаёт только перечисленные "
                "символы. Полезно для узких доменов (штрих-коды, номера "
                "счетов). Пусто = без ограничения."
            ),
        )
        adv_layout.addRow("Allowlist символов:", self._edit_allowlist)

        conf_row = QHBoxLayout()
        self._slider_conf = QSlider(Qt.Orientation.Horizontal, self._group_adv)
        self._slider_conf.setRange(0, 100)
        self._lbl_conf = QLabel("0%", self._group_adv)
        self._lbl_conf.setMinimumWidth(40)
        conf_row.addWidget(self._slider_conf, 1)
        conf_row.addWidget(self._lbl_conf)
        adv_layout.addRow("Порог confidence:", conf_row)

        self._spin_min_keep = QDoubleSpinBox(self._group_adv)
        self._spin_min_keep.setRange(0.0, 1.0)
        self._spin_min_keep.setDecimals(2)
        self._spin_min_keep.setSingleStep(0.05)
        describe(
            self._spin_min_keep,
            name="min_keep_confidence",
            description=(
                "Engine-side минимум confidence (0..1). Боксы ниже этого "
                "порога отбрасываются ДО фильтра confidence_threshold. "
                "Защищает downstream от hallucination-блоков EasyOCR на "
                "штампах / границах."
            ),
        )
        adv_layout.addRow("min_keep_confidence (engine):", self._spin_min_keep)

        self._chk_gpu = QCheckBox("Использовать GPU (CUDA)", self._group_adv)
        self._chk_gpu.setToolTip(
            "Если CUDA недоступна, EasyOCR автоматически откатывается на CPU."
        )
        adv_layout.addRow("GPU:", self._chk_gpu)

        self._chk_skip_text = QCheckBox("Пропускать страницы с текстом", self._group_adv)
        adv_layout.addRow("Skip text:", self._chk_skip_text)

        self._spin_max_pages = QSpinBox(self._group_adv)
        self._spin_max_pages.setRange(0, 10000)
        self._spin_max_pages.setSpecialValueText("без ограничения")
        self._spin_max_pages.setSuffix(" стр.")
        self._spin_max_pages.setToolTip(
            "Режим предпросмотра: обработать только первые N страниц "
            "документа. 0 — без ограничения."
        )
        adv_layout.addRow("Ограничение страниц:", self._spin_max_pages)

        self._chk_drop_low_conf = QCheckBox(
            "Удалять слова ниже порога confidence", self._group_adv,
        )
        adv_layout.addRow("drop_low_conf_words:", self._chk_drop_low_conf)

        self._chk_soft_rescue = QCheckBox(
            "Soft-rescue для shape-credible слов", self._group_adv,
        )
        adv_layout.addRow("soft_rescue_dropped_words:", self._chk_soft_rescue)

        self._chk_adaptive_conf = QCheckBox(
            "Адаптивный порог confidence по странице", self._group_adv,
        )
        adv_layout.addRow("adaptive_confidence_threshold:", self._chk_adaptive_conf)

        self._chk_user_words = QCheckBox(
            "Fuzzy-rescue по словарю ru_lexicon", self._group_adv,
        )
        adv_layout.addRow("user_words_fuzzy_rescue:", self._chk_user_words)

        root.addWidget(self._group_adv)
        root.addStretch(1)

        # Initial defaults.
        self.set_config(OCRConfig())

        # Wiring.
        self._chk_rus.toggled.connect(self._on_language_toggled)
        self._chk_eng.toggled.connect(self._on_language_toggled)
        self._cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        self._cmb_primary.currentIndexChanged.connect(self._schedule_emit)
        self._cmb_dpi.currentIndexChanged.connect(self._schedule_emit)
        self._edit_allowlist.textChanged.connect(self._schedule_emit)
        self._slider_conf.valueChanged.connect(self._on_conf_changed)
        self._spin_min_keep.valueChanged.connect(self._schedule_emit)
        self._chk_gpu.toggled.connect(self._schedule_emit)
        self._chk_skip_text.toggled.connect(self._schedule_emit)
        self._spin_max_pages.valueChanged.connect(self._schedule_emit)
        self._chk_drop_low_conf.toggled.connect(self._schedule_emit)
        self._chk_soft_rescue.toggled.connect(self._schedule_emit)
        self._chk_adaptive_conf.toggled.connect(self._schedule_emit)
        self._chk_user_words.toggled.connect(self._schedule_emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, cfg: OCRConfig) -> None:
        """Populate the widgets from a config object without emitting signals."""
        self._updating = True
        try:
            self._chk_rus.blockSignals(True)
            self._chk_eng.blockSignals(True)
            self._chk_rus.setChecked("ru" in cfg.languages)
            self._chk_eng.setChecked("en" in cfg.languages)
            self._chk_rus.blockSignals(False)
            self._chk_eng.blockSignals(False)
            self._rebuild_primary_combo(preferred=cfg.primary_language)

            self._set_combo_data(self._cmb_engine, cfg.engine)
            self._update_engine_hint()
            self._set_combo_data(self._cmb_dpi, int(cfg.dpi))

            self._edit_allowlist.blockSignals(True)
            self._edit_allowlist.setText(cfg.allowlist)
            self._edit_allowlist.blockSignals(False)

            self._slider_conf.blockSignals(True)
            self._slider_conf.setValue(int(cfg.confidence_threshold))
            self._lbl_conf.setText(f"{int(cfg.confidence_threshold)}%")
            self._slider_conf.blockSignals(False)

            self._spin_min_keep.blockSignals(True)
            self._spin_min_keep.setValue(float(cfg.min_keep_confidence))
            self._spin_min_keep.blockSignals(False)

            self._chk_gpu.blockSignals(True)
            self._chk_gpu.setChecked(bool(cfg.gpu))
            self._chk_gpu.blockSignals(False)

            self._chk_skip_text.blockSignals(True)
            self._chk_skip_text.setChecked(cfg.skip_text)
            self._chk_skip_text.blockSignals(False)

            self._spin_max_pages.blockSignals(True)
            self._spin_max_pages.setValue(int(getattr(cfg, "max_pages", 0) or 0))
            self._spin_max_pages.blockSignals(False)

            for chk, value in (
                (self._chk_drop_low_conf, cfg.drop_low_conf_words),
                (self._chk_soft_rescue, cfg.soft_rescue_dropped_words),
                (self._chk_adaptive_conf, cfg.adaptive_confidence_threshold),
                (self._chk_user_words, cfg.user_words_fuzzy_rescue),
            ):
                chk.blockSignals(True)
                chk.setChecked(bool(value))
                chk.blockSignals(False)
        finally:
            self._updating = False

    def get_config(self) -> OCRConfig:
        """Build an :class:`OCRConfig` from the current widget state."""
        languages: list[str] = []
        if self._chk_rus.isChecked():
            languages.append("ru")
        if self._chk_eng.isChecked():
            languages.append("en")
        if not languages:
            languages = ["ru"]

        primary = self._cmb_primary.currentData()
        if not isinstance(primary, str) or primary not in languages:
            primary = languages[0]

        dpi = self._cmb_dpi.currentData() or 300
        engine = self._cmb_engine.currentData() or OCREngineKind.EASYOCR

        from src.shared.validators import (
            ValidationError,
            validate_confidence,
            validate_dpi,
            validate_languages,
        )

        try:
            languages = validate_languages(languages)
        except ValidationError:
            languages = ["ru"]
        try:
            dpi_value = validate_dpi(int(dpi))
        except ValidationError:
            dpi_value = 300
        try:
            confidence = validate_confidence(float(self._slider_conf.value()))
        except ValidationError:
            confidence = 60.0

        return OCRConfig(
            engine=OCREngineKind(engine) if not isinstance(engine, OCREngineKind) else engine,
            languages=languages,
            primary_language=primary,
            dpi=dpi_value,
            allowlist=self._edit_allowlist.text(),
            confidence_threshold=confidence,
            min_keep_confidence=float(self._spin_min_keep.value()),
            gpu=self._chk_gpu.isChecked(),
            skip_text=self._chk_skip_text.isChecked(),
            max_pages=int(self._spin_max_pages.value()),
            drop_low_conf_words=self._chk_drop_low_conf.isChecked(),
            soft_rescue_dropped_words=self._chk_soft_rescue.isChecked(),
            adaptive_confidence_threshold=self._chk_adaptive_conf.isChecked(),
            user_words_fuzzy_rescue=self._chk_user_words.isChecked(),
        )

    def _on_engine_changed(self, _idx: int) -> None:
        self._update_engine_hint()
        self._schedule_emit()

    def _update_engine_hint(self) -> None:
        kind = self._cmb_engine.currentData()
        if not isinstance(kind, OCREngineKind):
            self._lbl_engine_hint.clear()
            return
        try:
            from src.application.engines.registry import get_engine

            engine = get_engine(kind)
            ok, msg = engine.is_available()
        except KeyError as exc:
            ok, msg = False, str(exc)
        if ok:
            self._lbl_engine_hint.setText("Готов к использованию.")
        else:
            self._lbl_engine_hint.setText(msg)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _schedule_emit(self, *_args: Any) -> None:
        if self._updating:
            return
        self._debounce.start()

    def _emit_config(self) -> None:
        try:
            cfg = self.get_config()
        except Exception:
            logger.exception("Failed to build OCRConfig from panel")
            return
        self.config_changed.emit(cfg)

    def _on_language_toggled(self, _checked: bool) -> None:
        self._rebuild_primary_combo()
        self._schedule_emit()

    def _on_conf_changed(self, value: int) -> None:
        self._lbl_conf.setText(f"{int(value)}%")
        self._schedule_emit()

    def _rebuild_primary_combo(self, preferred: str | None = None) -> None:
        self._cmb_primary.blockSignals(True)
        try:
            prev = preferred if preferred is not None else self._cmb_primary.currentData()
            self._cmb_primary.clear()
            if self._chk_rus.isChecked():
                self._cmb_primary.addItem(_LANG_LABELS["ru"], "ru")
            if self._chk_eng.isChecked():
                self._cmb_primary.addItem(_LANG_LABELS["en"], "en")
            if self._cmb_primary.count() == 0:
                self._cmb_primary.addItem(_LANG_LABELS["ru"], "ru")
            if prev is not None:
                idx = self._cmb_primary.findData(prev)
                if idx >= 0:
                    self._cmb_primary.setCurrentIndex(idx)
        finally:
            self._cmb_primary.blockSignals(False)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        combo.blockSignals(True)
        try:
            idx = combo.findData(value)
            if idx < 0 and hasattr(value, "value"):
                idx = combo.findData(value.value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
