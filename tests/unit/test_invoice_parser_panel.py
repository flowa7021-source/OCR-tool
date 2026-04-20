"""Tests for :class:`src.ui.invoice_parser_panel.InvoiceParserPanel`.

The panel is mostly display code; the tests therefore focus on the
behavioural contract that the main window relies on:

* Empty state: no rows → placeholder visible, export button disabled.
* ``set_result`` with a populated :class:`ParsedDocument` → table
  hydrates, export button enables, summary reflects row count +
  average confidence.
* Confidence column shows an int percentage (0–100) and receives a
  distinct background colour for each of the three bands —
  byte-exact hex match with :mod:`src.tn_parser.excel` so on-screen
  and workbook colours don't drift.
* ``export_requested(EXCEL, None)`` fires when the user clicks the
  export button — the signal shape the main window already wires.
* ``row_double_clicked`` emits the row index — reserved for the
  future "jump to PDF page" integration.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.models import (  # noqa: E402
    JobResult,
    ParsedDocument,
)
from src.shared.types import ExportFormat, JobStatus  # noqa: E402
from src.ui.invoice_parser_panel import (  # noqa: E402
    _COLOR_LOW,
    _COLOR_MID,
    _COLOR_OK,
    _CONF_COL_INDEX,
    InvoiceParserPanel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _row_dict(
    *,
    number: str = "ТН-001",
    confidence: float = 0.85,
    shipper: str = 'ООО "Ромашка", ИНН 7701234567',
) -> dict:
    """Build a parser-row dict mirroring ``ParsedRow.to_json_dict``."""
    return {
        "waybill": f"Транспортная накладная № {number}",
        "date": "15.03.2024",
        "number": number,
        "shipper": shipper,
        "consignee": 'ООО "Василёк"',
        "cargo": "Мука",
        "volume": "500 шт",
        "driver": "Иванов И.И.",
        "vehicle": "KAMAZ\nА123ВС777",
        "reception": "г. Москва",
        "source": "test.pdf",
        "note": "",
        "overall_confidence": confidence,
    }


def _job_with_rows(rows: list[dict], avg: float) -> JobResult:
    return JobResult(
        job_id="j1",
        status=JobStatus.COMPLETED,
        input_path="/tmp/test.pdf",
        output_path="/tmp/out.pdf",
        parsed=ParsedDocument(rows=rows, overall_confidence=avg),
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def test_panel_starts_empty(qapp: QApplication) -> None:
    """Fresh panel: zero rows, export disabled, placeholder visible."""
    panel = InvoiceParserPanel()
    assert panel.row_count() == 0
    assert panel._export_button.isEnabled() is False
    assert panel._placeholder.isVisible() is False  # not shown() yet
    # After forcing a layout via show() the placeholder should be
    # visible and the table hidden.
    panel.show()
    qapp.processEvents()
    assert panel._placeholder.isVisible() is True
    assert panel._table.isVisible() is False
    panel.hide()


def test_set_result_with_none_clears(qapp: QApplication) -> None:
    """Passing ``None`` resets the table to empty-state."""
    panel = InvoiceParserPanel()
    panel.set_result(_job_with_rows([_row_dict()], avg=0.85))
    assert panel.row_count() == 1
    panel.set_result(None)
    assert panel.row_count() == 0
    assert panel._export_button.isEnabled() is False


def test_set_result_with_parsed_none_clears(qapp: QApplication) -> None:
    """JobResult with ``parsed=None`` is a common path (extract off)."""
    panel = InvoiceParserPanel()
    result = JobResult(
        job_id="j2",
        status=JobStatus.COMPLETED,
        input_path="/tmp/x.pdf",
        output_path="/tmp/x.ocr.pdf",
        parsed=None,
    )
    panel.set_result(result)
    assert panel.row_count() == 0
    assert panel._export_button.isEnabled() is False


def test_set_result_with_empty_rows_clears(qapp: QApplication) -> None:
    """Parsed but zero rows → same as no-data (don't fake a table)."""
    panel = InvoiceParserPanel()
    panel.set_result(_job_with_rows([], avg=0.0))
    assert panel.row_count() == 0
    assert panel._export_button.isEnabled() is False


# ---------------------------------------------------------------------------
# Populated state
# ---------------------------------------------------------------------------


def test_set_result_populates_table_and_summary(qapp: QApplication) -> None:
    """Two rows → table has 2 rows, summary shows count + avg %."""
    panel = InvoiceParserPanel()
    rows = [
        _row_dict(number="ТН-001", confidence=0.90),
        _row_dict(number="ТН-002", confidence=0.60),
    ]
    panel.set_result(_job_with_rows(rows, avg=0.75))
    assert panel.row_count() == 2
    assert panel._export_button.isEnabled() is True
    summary = panel._summary_label.text()
    assert "Извлечено строк: 2" in summary
    assert "75%" in summary


def test_table_displays_waybill_and_confidence(qapp: QApplication) -> None:
    """Column 0 shows waybill text; last column shows int conf %."""
    panel = InvoiceParserPanel()
    panel.set_result(_job_with_rows([_row_dict(number="ТН-999", confidence=0.82)], 0.82))
    model = panel._model
    # First column — waybill.
    idx0 = model.index(0, 0)
    assert "ТН-999" in model.data(idx0, Qt.ItemDataRole.DisplayRole)
    # Last column — integer 0-100.
    conf_idx = model.index(0, _CONF_COL_INDEX)
    val = model.data(conf_idx, Qt.ItemDataRole.DisplayRole)
    assert isinstance(val, int)
    assert val == 82


# ---------------------------------------------------------------------------
# Confidence colour bands — byte-exact match with src/tn_parser/excel.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence,expected_color",
    [
        (0.20, _COLOR_LOW),   # <50 → red
        (0.49, _COLOR_LOW),
        (0.50, _COLOR_MID),   # 50-69 → yellow
        (0.69, _COLOR_MID),
        (0.70, _COLOR_OK),    # ≥70 → green
        (0.95, _COLOR_OK),
    ],
)
def test_confidence_column_background_bands(
    qapp: QApplication,
    confidence: float,
    expected_color: QColor,
) -> None:
    """Each band paints a distinct background; matches excel.py palette."""
    panel = InvoiceParserPanel()
    panel.set_result(_job_with_rows([_row_dict(confidence=confidence)], confidence))
    model = panel._model
    idx = model.index(0, _CONF_COL_INDEX)
    brush = model.data(idx, Qt.ItemDataRole.BackgroundRole)
    assert brush is not None
    assert brush.color().name() == expected_color.name()


def test_non_confidence_column_has_no_background(qapp: QApplication) -> None:
    """Only the confidence column gets a band colour — avoids visual noise."""
    panel = InvoiceParserPanel()
    panel.set_result(_job_with_rows([_row_dict(confidence=0.20)], 0.20))
    model = panel._model
    # Column 0 (waybill) should NOT have a background brush.
    idx = model.index(0, 0)
    assert model.data(idx, Qt.ItemDataRole.BackgroundRole) is None


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def test_export_button_emits_request_signal(qapp: QApplication, qtbot) -> None:
    """Clicking ``Экспорт в Excel…`` fires ``export_requested(EXCEL, None)``."""
    panel = InvoiceParserPanel()
    panel.set_result(_job_with_rows([_row_dict()], 0.80))
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.export_requested, timeout=1000) as blocker:
        panel._export_button.click()
    # Signal args: (ExportFormat, Path|None).
    fmt, path = blocker.args
    assert fmt is ExportFormat.EXCEL
    assert path is None


def test_double_click_emits_row_index(qapp: QApplication, qtbot) -> None:
    """Double-click row N → ``row_double_clicked`` fires with N."""
    panel = InvoiceParserPanel()
    rows = [_row_dict(number="row-0"), _row_dict(number="row-1")]
    panel.set_result(_job_with_rows(rows, 0.80))
    qtbot.addWidget(panel)
    idx = panel._model.index(1, 0)
    with qtbot.waitSignal(panel.row_double_clicked, timeout=1000) as blocker:
        panel._on_double_click(idx)
    assert blocker.args == [1]


# ---------------------------------------------------------------------------
# Edge case: confidence derived from per-field dict (legacy shape)
# ---------------------------------------------------------------------------


def test_confidence_falls_back_to_per_field_mean(qapp: QApplication) -> None:
    """Older snapshot shape: no ``overall_confidence`` — mean of ``confidence``."""
    panel = InvoiceParserPanel()
    legacy = _row_dict(confidence=0.0)
    legacy.pop("overall_confidence")
    legacy["confidence"] = {"date": 1.0, "number": 0.5}
    panel.set_result(_job_with_rows([legacy], avg=0.0))
    conf_idx = panel._model.index(0, _CONF_COL_INDEX)
    val = panel._model.data(conf_idx, Qt.ItemDataRole.DisplayRole)
    # (1.0 + 0.5) / 2 = 0.75 → 75 %
    assert val == 75
