"""UI panel: structured fields extracted by the post-OCR parser.

Displays :attr:`JobResult.parsed` (a :class:`ParsedDocument`) as a
read-only table whose column layout mirrors the Excel workbook the
user will get from "Export in Excel…" — same order, same confidence-
colour bands, so what the user sees on-screen round-trips into the
``.xlsx`` without surprises.

Kept deliberately thin: all persistence and parser invocation happen
upstream (pipeline → ``JobResult.parsed``). This widget just
renders, plus a single signal (:attr:`export_requested`) that asks
the main window's :class:`ExportManager` wiring to save the result.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from src.core.models import JobResult
from src.shared.types import ExportFormat

logger = logging.getLogger(__name__)


# (header_label, json_dict_key) — order and labels MUST mirror
# :data:`src.tn_parser.excel.COLUMNS` so the on-screen table, the
# exported xlsx and the snapshot sidecar all speak the same language.
# The last tuple carries the sentinel key ``"__confidence__"``
# because confidence is computed, not a field in ``to_json_dict``.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Транспортная накладная", "waybill"),
    ("Дата", "date"),
    ("№", "number"),
    ("Грузоотправитель", "shipper"),
    ("Грузополучатель", "consignee"),
    ("Груз", "cargo"),
    ("Объём", "volume"),
    ("Водитель", "driver"),
    ("Транспортное средство", "vehicle"),
    ("Прием груза", "reception"),
    ("Источник файл", "source"),
    ("Примечание", "note"),
    ("Уверенность, %", "__confidence__"),
)

_CONF_COL_INDEX = len(_COLUMNS) - 1
_CONF_SENTINEL_KEY = "__confidence__"

# Same byte-exact hex values as src/tn_parser/excel.py so an
# operator eyeballing the on-screen colour and then the exported
# workbook doesn't see two subtly different greens / reds / yellows
# and wonder if the pipeline introduced drift somewhere.
_COLOR_LOW = QColor("#F4CCCC")  # < 50 %
_COLOR_MID = QColor("#FFF2CC")  # 50 – 69 %
_COLOR_OK = QColor("#D9EAD3")  # ≥ 70 %


def _coerce_confidence_pct(row: dict[str, Any]) -> int:
    """Normalise a parser row's confidence into a 0-100 int.

    ``ParsedRow.to_json_dict`` stores confidence as a dict of per-
    field floats under ``"confidence"`` PLUS an ``"overall_confidence"``
    scalar. Prefer the scalar (cheap, already rounded on save);
    fall back to the mean of the per-field dict only when the
    scalar is missing (older snapshots, hand-edited JSON).
    """
    scalar = row.get("overall_confidence")
    if isinstance(scalar, (int, float)):
        return max(0, min(100, round(float(scalar) * 100)))
    conf_map = row.get("confidence") or {}
    floats = [v for v in conf_map.values() if isinstance(v, (int, float))]
    if not floats:
        return 0
    return max(0, min(100, round(sum(floats) / len(floats) * 100)))


class _ParsedRowsModel(QAbstractTableModel):
    """Qt table model backed by ``ParsedDocument.rows`` (list of dicts).

    Dicts, not dataclasses — :class:`~src.core.models.ParsedDocument`
    intentionally stores rows in the JSON-dict form produced by
    ``ParsedRow.to_json_dict`` so the core / domain layer doesn't
    take a compile-time dependency on ``src.tn_parser``. The model
    treats them as opaque key-value records.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []

    # ---- mutation ----------------------------------------------------
    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def clear(self) -> None:
        self.set_rows([])

    # ---- QAbstractTableModel overrides -------------------------------
    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return len(_COLUMNS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation is Qt.Orientation.Horizontal:
            return _COLUMNS[section][0]
        return section + 1

    def data(
        self,
        index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None
        row_idx, col_idx = index.row(), index.column()
        row = self._rows[row_idx]
        key = _COLUMNS[col_idx][1]

        if role == Qt.ItemDataRole.DisplayRole:
            if key == _CONF_SENTINEL_KEY:
                return _coerce_confidence_pct(row)
            return str(row.get(key, ""))

        if role == Qt.ItemDataRole.BackgroundRole and col_idx == _CONF_COL_INDEX:
            pct = _coerce_confidence_pct(row)
            if pct < 50:
                return QBrush(_COLOR_LOW)
            if pct < 70:
                return QBrush(_COLOR_MID)
            return QBrush(_COLOR_OK)

        if role == Qt.ItemDataRole.TextAlignmentRole and col_idx == _CONF_COL_INDEX:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.ToolTipRole and key != _CONF_SENTINEL_KEY:
            # Full value for fields that overflow the column width —
            # the "Груз" and "Грузоотправитель" cells can be several
            # hundred characters long on real waybills, so a tooltip
            # is the only way to read them without resizing.
            return str(row.get(key, ""))

        return None


class InvoiceParserPanel(QWidget):
    """Read-only table of fields extracted from a finished job.

    Empty-state placeholder tells the user *why* there's nothing to
    show ("enable extract.enabled in the profile") rather than
    leaving them guessing. The ``Export to Excel…`` button is
    disabled until rows are loaded.

    Signals:
        export_requested(ExportFormat, Path|None): Raised when the
            user clicks ``Экспорт в Excel…``. ``Path`` is always
            ``None`` — the main window's existing export handler
            pops the Save dialog, consistent with how
            :class:`~src.ui.results_panel.ResultsPanel` routes
            TXT / DOCX / PDF exports through the same plumbing.
        row_double_clicked(int): Row index. Reserved for a future
            "jump to PDF page" wiring; emitted unconditionally so
            the integration can land without another panel release.
    """

    export_requested = Signal(object, object)
    row_double_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Top toolbar: summary on the left, single export button right.
        top_row = QHBoxLayout()
        self._summary_label = QLabel("Нет распознанных полей", self)
        top_row.addWidget(self._summary_label)
        top_row.addStretch(1)
        self._export_button = QPushButton("Экспорт в Excel…", self)
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._on_export_clicked)
        top_row.addWidget(self._export_button)
        root.addLayout(top_row)

        # Table.
        self._model = _ParsedRowsModel(self)
        self._table = QTableView(self)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows,
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive,
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.verticalHeader().setVisible(False)
        self._table.doubleClicked.connect(self._on_double_click)
        root.addWidget(self._table, 1)

        # Placeholder shown only when the table is empty — guides the
        # user to flip ``extract.enabled`` if they expected rows.
        self._placeholder = QLabel(
            "Извлечённые поля появятся здесь после обработки файла "
            "профилем, у которого включено `extract.enabled` "
            "(например, встроенный `tn_upd`).",
            self,
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "color: #888; font-style: italic; padding: 16px;",
        )
        root.addWidget(self._placeholder)

        self._update_empty_state()

    # ---- public API --------------------------------------------------

    def set_result(self, result: JobResult | None) -> None:
        """Populate the table from ``result.parsed``.

        Accepts ``None`` (cleared state) and job results whose
        ``parsed`` is ``None`` / has an empty row list — both paths
        reset the table and disable the export button.
        """
        rows: list[dict[str, Any]] = []
        overall_pct: int = 0
        if result is not None and result.parsed is not None:
            rows = list(result.parsed.rows)
            overall_pct = round(result.parsed.overall_confidence * 100)

        if not rows:
            self._model.clear()
            self._summary_label.setText("Нет распознанных полей")
            self._export_button.setEnabled(False)
            self._update_empty_state()
            return

        self._model.set_rows(rows)
        self._summary_label.setText(
            f"Извлечено строк: {len(rows)} · "
            f"средняя уверенность: {overall_pct}%",
        )
        self._export_button.setEnabled(True)
        # Size columns to content on fresh data; the user can still
        # drag dividers afterwards — we use Interactive mode above.
        self._table.resizeColumnsToContents()
        self._update_empty_state()

    def clear(self) -> None:
        """Reset to the empty-state placeholder."""
        self.set_result(None)

    def row_count(self) -> int:
        """Number of rows in the model (convenience for tests)."""
        return self._model.rowCount()

    # ---- internals ---------------------------------------------------

    def _update_empty_state(self) -> None:
        has_rows = self._model.rowCount() > 0
        self._placeholder.setVisible(not has_rows)
        self._table.setVisible(has_rows)

    def _on_export_clicked(self) -> None:
        # Path=None asks MainWindow to pop a Save dialog. Matches the
        # contract documented on export_requested.
        self.export_requested.emit(ExportFormat.EXCEL, None)

    def _on_double_click(self, index: QModelIndex) -> None:
        if index.isValid():
            self.row_double_clicked.emit(index.row())


__all__ = ["InvoiceParserPanel"]
