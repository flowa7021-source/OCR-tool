"""Editor widget for :class:`OCRConfig`.

Exposes a :class:`SettingsPanel` that renders all Tesseract/OCRmyPDF
parameters as Qt widgets and emits a debounced ``config_changed`` signal
whenever the user changes any field.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from src.shared.types import OEM, PSM, Language, OCREngineKind, OptimizeLevel

logger = logging.getLogger(__name__)

_DEBOUNCE_MS: int = 300


class SettingsPanel(QWidget):
    """Widget that edits a :class:`OCRConfig` in real time."""

    config_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the panel with default :class:`OCRConfig` values.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._updating: bool = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_config)

        # Wrap the whole panel in a QScrollArea so every group is reachable
        # even when the parent splitter is squeezed.
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
                # Keep the item visible (the user might want to download it)
                # but show it greyed out by removing the enabled flag.

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
        self._chk_rus = QCheckBox("Русский", self._group_langs)
        self._chk_eng = QCheckBox("English", self._group_langs)
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

        # --- PSM group -----------------------------------------------
        self._group_psm = QGroupBox("Сегментация страницы (PSM)", self)
        psm_layout = QVBoxLayout(self._group_psm)
        self._cmb_psm = QComboBox(self._group_psm)
        for psm in PSM:
            self._cmb_psm.addItem(psm.label, psm)
            idx = self._cmb_psm.count() - 1
            self._cmb_psm.setItemData(idx, psm.description, Qt.ItemDataRole.ToolTipRole)
        psm_layout.addWidget(self._cmb_psm)
        root.addWidget(self._group_psm)

        # --- OEM group -----------------------------------------------
        self._group_oem = QGroupBox("Движок OCR (OEM)", self)
        oem_layout = QVBoxLayout(self._group_oem)
        self._cmb_oem = QComboBox(self._group_oem)
        for oem in OEM:
            self._cmb_oem.addItem(oem.label, oem)
            idx = self._cmb_oem.count() - 1
            self._cmb_oem.setItemData(idx, oem.description, Qt.ItemDataRole.ToolTipRole)
        oem_layout.addWidget(self._cmb_oem)
        root.addWidget(self._group_oem)

        # --- Advanced group ------------------------------------------
        self._group_adv = QGroupBox("Дополнительно", self)
        adv_layout = QFormLayout(self._group_adv)

        self._cmb_dpi = QComboBox(self._group_adv)
        for dpi in DPI_CHOICES:
            self._cmb_dpi.addItem(str(dpi), int(dpi))
        adv_layout.addRow("DPI:", self._cmb_dpi)

        self._edit_whitelist = QLineEdit(self._group_adv)
        adv_layout.addRow("Whitelist символов:", self._edit_whitelist)

        self._edit_blacklist = QLineEdit(self._group_adv)
        adv_layout.addRow("Blacklist символов:", self._edit_blacklist)

        self._spin_timeout = QSpinBox(self._group_adv)
        self._spin_timeout.setRange(10, 600)
        adv_layout.addRow("Таймаут страницы (сек):", self._spin_timeout)

        conf_row = QHBoxLayout()
        self._slider_conf = QSlider(Qt.Orientation.Horizontal, self._group_adv)
        self._slider_conf.setRange(0, 100)
        self._lbl_conf = QLabel("0%", self._group_adv)
        self._lbl_conf.setMinimumWidth(40)
        conf_row.addWidget(self._slider_conf, 1)
        conf_row.addWidget(self._lbl_conf)
        adv_layout.addRow("Порог confidence:", conf_row)

        self._cmb_optimize = QComboBox(self._group_adv)
        for lvl in OptimizeLevel:
            self._cmb_optimize.addItem(f"{int(lvl)} — {lvl.name}", lvl)
        adv_layout.addRow("Optimize:", self._cmb_optimize)

        self._chk_skip_text = QCheckBox("Пропускать страницы с текстом", self._group_adv)
        adv_layout.addRow("Skip text:", self._chk_skip_text)

        # Preview mode: process only the first N pages. 0 = no limit.
        self._spin_max_pages = QSpinBox(self._group_adv)
        self._spin_max_pages.setRange(0, 10000)
        self._spin_max_pages.setSpecialValueText("без ограничения")
        self._spin_max_pages.setSuffix(" стр.")
        self._spin_max_pages.setToolTip(
            "Режим предпросмотра: обработать только первые N страниц "
            "документа. Удобно для проверки настроек профиля на большом "
            "PDF перед полным запуском. 0 — без ограничения."
        )
        adv_layout.addRow("Ограничение страниц:", self._spin_max_pages)

        root.addWidget(self._group_adv)
        root.addStretch(1)

        # Initial defaults.
        self.set_config(OCRConfig())

        # Wiring.
        self._chk_rus.toggled.connect(self._on_language_toggled)
        self._chk_eng.toggled.connect(self._on_language_toggled)
        self._cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        self._cmb_primary.currentIndexChanged.connect(self._schedule_emit)
        self._cmb_psm.currentIndexChanged.connect(self._schedule_emit)
        self._cmb_oem.currentIndexChanged.connect(self._schedule_emit)
        self._cmb_dpi.currentIndexChanged.connect(self._schedule_emit)
        self._edit_whitelist.textChanged.connect(self._schedule_emit)
        self._edit_blacklist.textChanged.connect(self._schedule_emit)
        self._spin_timeout.valueChanged.connect(self._schedule_emit)
        self._slider_conf.valueChanged.connect(self._on_conf_changed)
        self._cmb_optimize.currentIndexChanged.connect(self._schedule_emit)
        self._chk_skip_text.toggled.connect(self._schedule_emit)
        self._spin_max_pages.valueChanged.connect(self._schedule_emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, cfg: OCRConfig) -> None:
        """Populate the widgets from a config object without emitting signals.

        Args:
            cfg: Source :class:`OCRConfig`.
        """
        self._updating = True
        try:
            self._chk_rus.blockSignals(True)
            self._chk_eng.blockSignals(True)
            self._chk_rus.setChecked(Language.RUSSIAN.value in cfg.languages)
            self._chk_eng.setChecked(Language.ENGLISH.value in cfg.languages)
            self._chk_rus.blockSignals(False)
            self._chk_eng.blockSignals(False)
            self._rebuild_primary_combo(preferred=cfg.primary_language)

            self._set_combo_data(self._cmb_engine, cfg.engine)
            self._update_engine_hint()
            self._set_combo_data(self._cmb_psm, cfg.psm)
            self._set_combo_data(self._cmb_oem, cfg.oem)
            self._set_combo_data(self._cmb_dpi, int(cfg.dpi))
            self._set_combo_data(self._cmb_optimize, cfg.optimize_level)

            self._edit_whitelist.blockSignals(True)
            self._edit_whitelist.setText(cfg.char_whitelist)
            self._edit_whitelist.blockSignals(False)

            self._edit_blacklist.blockSignals(True)
            self._edit_blacklist.setText(cfg.char_blacklist)
            self._edit_blacklist.blockSignals(False)

            self._spin_timeout.blockSignals(True)
            self._spin_timeout.setValue(int(cfg.tesseract_timeout))
            self._spin_timeout.blockSignals(False)

            self._slider_conf.blockSignals(True)
            self._slider_conf.setValue(int(cfg.confidence_threshold))
            self._lbl_conf.setText(f"{int(cfg.confidence_threshold)}%")
            self._slider_conf.blockSignals(False)

            self._chk_skip_text.blockSignals(True)
            self._chk_skip_text.setChecked(cfg.skip_text)
            self._chk_skip_text.blockSignals(False)

            self._spin_max_pages.blockSignals(True)
            self._spin_max_pages.setValue(int(getattr(cfg, "max_pages", 0) or 0))
            self._spin_max_pages.blockSignals(False)
        finally:
            self._updating = False

    def get_config(self) -> OCRConfig:
        """Build an :class:`OCRConfig` from the current widget state.

        Returns:
            A new :class:`OCRConfig` snapshot.
        """
        languages: list[str] = []
        if self._chk_rus.isChecked():
            languages.append(Language.RUSSIAN.value)
        if self._chk_eng.isChecked():
            languages.append(Language.ENGLISH.value)
        if not languages:
            languages = [Language.RUSSIAN.value]

        primary = self._cmb_primary.currentData()
        if not isinstance(primary, str) or primary not in languages:
            primary = languages[0]

        psm = self._cmb_psm.currentData() or PSM.AUTO
        oem = self._cmb_oem.currentData() or OEM.LSTM_ONLY
        dpi = self._cmb_dpi.currentData() or 300
        optimize = self._cmb_optimize.currentData() or OptimizeLevel.LOSSLESS
        engine = self._cmb_engine.currentData() or OCREngineKind.TESSERACT

        # Defensive validation: every UI widget is bounded, but profile
        # JSON loads bypass the widgets, so coerce + clamp on the way out
        # of get_config too.
        from src.shared.validators import (
            ValidationError,
            validate_confidence,
            validate_dpi,
            validate_languages,
        )

        try:
            languages = validate_languages(languages)
        except ValidationError:
            languages = ["rus"]
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
            psm=PSM(psm),
            oem=OEM(oem),
            dpi=dpi_value,
            char_whitelist=self._edit_whitelist.text(),
            char_blacklist=self._edit_blacklist.text(),
            confidence_threshold=confidence,
            tesseract_timeout=int(self._spin_timeout.value()),
            optimize_level=OptimizeLevel(int(optimize)),
            skip_text=self._chk_skip_text.isChecked(),
            max_pages=int(self._spin_max_pages.value()),
        )

    def _on_engine_changed(self, _idx: int) -> None:
        """Refresh the engine availability hint and re-emit config."""
        self._update_engine_hint()
        self._schedule_emit()

    def _update_engine_hint(self) -> None:
        """Show availability message under the engine dropdown."""
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
        """Restart the debounce timer so ``config_changed`` is coalesced."""
        if self._updating:
            return
        self._debounce.start()

    def _emit_config(self) -> None:
        """Emit the current config. Called by the debounce timer."""
        try:
            cfg = self.get_config()
        except Exception:
            logger.exception("Failed to build OCRConfig from panel")
            return
        self.config_changed.emit(cfg)

    def _on_language_toggled(self, _checked: bool) -> None:
        """Rebuild the primary-language combo when checkboxes change."""
        self._rebuild_primary_combo()
        self._schedule_emit()

    def _on_conf_changed(self, value: int) -> None:
        """Update the confidence label and schedule an emit."""
        self._lbl_conf.setText(f"{int(value)}%")
        self._schedule_emit()

    def _rebuild_primary_combo(self, preferred: str | None = None) -> None:
        """Rebuild the primary-language combo from the checked languages.

        Args:
            preferred: Optional language code to select if present.
        """
        self._cmb_primary.blockSignals(True)
        try:
            prev = preferred if preferred is not None else self._cmb_primary.currentData()
            self._cmb_primary.clear()
            if self._chk_rus.isChecked():
                self._cmb_primary.addItem(Language.RUSSIAN.label, Language.RUSSIAN.value)
            if self._chk_eng.isChecked():
                self._cmb_primary.addItem(Language.ENGLISH.label, Language.ENGLISH.value)
            if self._cmb_primary.count() == 0:
                # Always keep at least one option.
                self._cmb_primary.addItem(Language.RUSSIAN.label, Language.RUSSIAN.value)
            # Restore prior selection if still available.
            if prev is not None:
                idx = self._cmb_primary.findData(prev)
                if idx >= 0:
                    self._cmb_primary.setCurrentIndex(idx)
        finally:
            self._cmb_primary.blockSignals(False)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        """Select the combo entry whose ``data`` equals ``value``.

        Args:
            combo: Target combo box.
            value: Value to match against :meth:`QComboBox.itemData`.
        """
        combo.blockSignals(True)
        try:
            idx = combo.findData(value)
            if idx < 0 and hasattr(value, "value"):
                idx = combo.findData(value.value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
