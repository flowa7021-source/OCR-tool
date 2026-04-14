"""Results panel: view recognised text and trigger exports.

Displays a :class:`JobResult` produced by the OCR pipeline. The user can
navigate pages, inspect low-confidence words and request exports.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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


def QTextDocumentFindFlag_Backward():  # noqa: N802 — matches Qt enum name
    """Return the backward-search flag as a Qt FindFlag value.

    Wrapped so the module loads even when the specific Qt binding name
    differs across PySide6 versions. QTextDocument.FindBackward is the
    canonical location.
    """
    from PySide6.QtGui import QTextDocument

    return QTextDocument.FindFlag.FindBackward


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

        # Search bar — hidden by default, toggled by Ctrl+F / the
        # "Поиск" button. Escape hides it and clears highlights.
        self._search_row = QHBoxLayout()
        self._edit_search = QLineEdit(self)
        self._edit_search.setPlaceholderText("Найти в тексте страницы…")
        self._edit_search.returnPressed.connect(self._find_next)
        self._edit_search.textChanged.connect(self._on_search_text_changed)
        self._btn_find_next = QPushButton("▼", self)
        self._btn_find_next.setMaximumWidth(40)
        self._btn_find_next.setToolTip("Следующее совпадение (Enter)")
        self._btn_find_next.clicked.connect(self._find_next)
        self._btn_find_prev = QPushButton("▲", self)
        self._btn_find_prev.setMaximumWidth(40)
        self._btn_find_prev.setToolTip("Предыдущее (Shift+Enter)")
        self._btn_find_prev.clicked.connect(self._find_prev)
        self._lbl_search_count = QLabel("", self)
        self._lbl_search_count.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        self._search_row.addWidget(self._edit_search, 1)
        self._search_row.addWidget(self._btn_find_prev)
        self._search_row.addWidget(self._btn_find_next)
        self._search_row.addWidget(self._lbl_search_count)
        self._search_container = QWidget(self)
        self._search_container.setLayout(self._search_row)
        self._search_container.hide()
        root.addWidget(self._search_container)

        # Ctrl+F toggles the search bar (scoped to this panel).
        self._act_find = QAction("Поиск", self)
        self._act_find.setShortcut(QKeySequence.StandardKey.Find)
        self._act_find.triggered.connect(self._toggle_search)
        self.addAction(self._act_find)
        self._act_find_esc = QAction("Закрыть поиск", self)
        self._act_find_esc.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self._act_find_esc.triggered.connect(self._close_search)
        self._search_container.addAction(self._act_find_esc)

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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _toggle_search(self) -> None:
        """Show the search bar and focus the input."""
        self._search_container.setVisible(True)
        self._edit_search.setFocus()
        self._edit_search.selectAll()

    def _close_search(self) -> None:
        """Hide the search bar and clear any highlighted matches."""
        self._search_container.hide()
        self._clear_highlights()
        self._lbl_search_count.setText("")

    def _on_search_text_changed(self, text: str) -> None:
        """Re-highlight every occurrence and update the match counter."""
        self._clear_highlights()
        if not text:
            self._lbl_search_count.setText("")
            return
        count = self._highlight_all(text)
        self._lbl_search_count.setText(f"{count}")
        # Auto-jump to the first match on every keystroke.
        if count > 0:
            cursor = self._text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self._text_edit.setTextCursor(cursor)
            self._find_next()

    def _highlight_all(self, needle: str) -> int:
        """Yellow-highlight every occurrence of ``needle``; return count."""
        doc = self._text_edit.document()
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.GlobalColor.yellow)
        fmt.setForeground(Qt.GlobalColor.black)
        cursor = QTextCursor(doc)
        count = 0
        while True:
            cursor = doc.find(needle, cursor)
            if cursor.isNull():
                break
            cursor.mergeCharFormat(fmt)
            count += 1
        return count

    def _clear_highlights(self) -> None:
        doc = self._text_edit.document()
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.SelectionType.Document)
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.GlobalColor.transparent)
        cursor.mergeCharFormat(fmt)

    def _find_next(self) -> None:
        needle = self._edit_search.text()
        if not needle:
            return
        found = self._text_edit.find(needle)
        if not found:
            # Wrap around
            cursor = self._text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self._text_edit.setTextCursor(cursor)
            self._text_edit.find(needle)

    def _find_prev(self) -> None:
        needle = self._edit_search.text()
        if not needle:
            return
        found = self._text_edit.find(needle, QTextDocumentFindFlag_Backward())
        if not found:
            cursor = self._text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._text_edit.setTextCursor(cursor)
            self._text_edit.find(needle, QTextDocumentFindFlag_Backward())

    def _on_toggle_lowconf(self, checked: bool) -> None:
        """Expand or collapse the low-confidence words list.

        Args:
            checked: Whether the toggle button is pressed.
        """
        self._list_lowconf.setVisible(checked)
        self._btn_toggle_lowconf.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

