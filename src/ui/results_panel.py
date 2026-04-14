"""Results panel: view recognised text and trigger exports.

Displays a :class:`JobResult` produced by the OCR pipeline. The user can
navigate pages, inspect low-confidence words and request exports.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.models import JobResult, PageResult
from src.shared.constants import COLOR_TEXT_SECONDARY
from src.shared.types import ExportFormat

logger = logging.getLogger(__name__)


class ResultsPanel(QWidget):
    """Displays the text output of a completed OCR job."""

    export_requested = Signal(object, object)
    # Fired when the user wants to open the produced searchable PDF with the
    # system default handler (no path argument — the panel's bound JobResult
    # knows where the file lives).
    open_pdf_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the panel with an empty state.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._result: JobResult | None = None

        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self._cmb_page = QComboBox(self)
        self._cmb_page.setMinimumWidth(140)
        top_row.addWidget(self._cmb_page)

        self._lbl_conf = QLabel("Средний confidence: —", self)
        top_row.addWidget(self._lbl_conf)

        self._lbl_path = QLabel("", self)
        self._lbl_path.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        self._lbl_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        top_row.addWidget(self._lbl_path, 1)
        root.addLayout(top_row)

        self._text_edit = QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setMinimumHeight(160)
        self._text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        root.addWidget(self._text_edit, 1)

        # Collapsible low-confidence list.
        self._btn_toggle_lowconf = QToolButton(self)
        self._btn_toggle_lowconf.setText("Слова с низкой уверенностью")
        self._btn_toggle_lowconf.setCheckable(True)
        self._btn_toggle_lowconf.setChecked(False)
        self._btn_toggle_lowconf.setArrowType(Qt.ArrowType.RightArrow)
        self._btn_toggle_lowconf.toggled.connect(self._on_toggle_lowconf)
        root.addWidget(self._btn_toggle_lowconf)

        self._list_lowconf = QListWidget(self)
        self._list_lowconf.setMaximumHeight(120)
        self._list_lowconf.hide()
        root.addWidget(self._list_lowconf)

        # Export buttons.
        btn_row = QHBoxLayout()
        self._btn_save_pdf = QPushButton("Сохранить PDF как…", self)
        self._btn_save_pdf.setToolTip(
            "Сохранить итоговый searchable PDF (скан + невидимый текстовый слой) "
            "в выбранное место."
        )
        self._btn_txt = QPushButton("Сохранить TXT", self)
        self._cmb_txt_encoding = QComboBox(self)
        self._cmb_txt_encoding.addItem("UTF-8", userData="utf-8")
        self._cmb_txt_encoding.addItem("UTF-8 с BOM", userData="utf-8-sig")
        self._cmb_txt_encoding.addItem("Windows-1251", userData="cp1251")
        self._cmb_txt_encoding.setToolTip(
            "Кодировка TXT-файла. Windows-1251 нужна для совместимости со "
            "старыми редакторами на Windows."
        )
        self._btn_docx = QPushButton("Сохранить DOCX", self)
        self._btn_copy = QPushButton("Копировать текст", self)
        self._btn_open_pdf = QPushButton("Открыть PDF", self)
        self._btn_open_pdf.setToolTip(
            "Открыть итоговый PDF в программе по умолчанию."
        )
        btn_row.addWidget(self._btn_save_pdf)
        btn_row.addWidget(self._btn_txt)
        btn_row.addWidget(self._cmb_txt_encoding)
        btn_row.addWidget(self._btn_docx)
        btn_row.addWidget(self._btn_copy)
        btn_row.addWidget(self._btn_open_pdf)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Wiring.
        self._cmb_page.currentIndexChanged.connect(self._on_page_changed)
        self._btn_save_pdf.clicked.connect(
            lambda: self.export_requested.emit(ExportFormat.PDF, None)
        )
        self._btn_txt.clicked.connect(
            lambda: self.export_requested.emit(ExportFormat.TXT, None)
        )
        self._btn_docx.clicked.connect(
            lambda: self.export_requested.emit(ExportFormat.DOCX, None)
        )
        self._btn_copy.clicked.connect(
            lambda: self.export_requested.emit(ExportFormat.CLIPBOARD, None)
        )
        self._btn_open_pdf.clicked.connect(self.open_pdf_requested.emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_result(self, result: JobResult) -> None:
        """Bind a :class:`JobResult` to the panel.

        Args:
            result: Completed job result to display.
        """
        self._result = result
        self._cmb_page.blockSignals(True)
        self._cmb_page.clear()
        for page in result.pages:
            self._cmb_page.addItem(f"Страница {page.page_number}", page.page_number)
        self._cmb_page.blockSignals(False)

        avg = result.average_confidence
        self._lbl_conf.setText(f"Средний confidence: {avg:.1f}%")
        self._lbl_path.setText(str(result.output_path or result.input_path))

        if self._cmb_page.count() > 0:
            self._cmb_page.setCurrentIndex(0)
            self._on_page_changed(0)
        else:
            self._text_edit.clear()
            self._list_lowconf.clear()

    def txt_encoding(self) -> str:
        """Return the TXT-export encoding currently selected by the user."""
        data = self._cmb_txt_encoding.currentData()
        return str(data) if data else "utf-8"

    def clear(self) -> None:
        """Reset the panel to an empty state."""
        self._result = None
        self._cmb_page.blockSignals(True)
        self._cmb_page.clear()
        self._cmb_page.blockSignals(False)
        self._text_edit.clear()
        self._list_lowconf.clear()
        self._lbl_conf.setText("Средний confidence: —")
        self._lbl_path.setText("")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _on_page_changed(self, idx: int) -> None:
        """Update the text edit and low-confidence list for page ``idx``.

        Args:
            idx: Zero-based combo index selected by the user.
        """
        if self._result is None or idx < 0 or idx >= len(self._result.pages):
            self._text_edit.clear()
            self._list_lowconf.clear()
            return
        page: PageResult = self._result.pages[idx]
        self._text_edit.setPlainText(page.text)
        self._list_lowconf.clear()
        for word in page.low_confidence_words:
            self._list_lowconf.addItem(word)

    def _on_toggle_lowconf(self, checked: bool) -> None:
        """Expand or collapse the low-confidence words list.

        Args:
            checked: Whether the toggle button is pressed.
        """
        self._list_lowconf.setVisible(checked)
        self._btn_toggle_lowconf.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

