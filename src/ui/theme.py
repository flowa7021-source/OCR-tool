"""Dark-theme stylesheet and theme application helpers."""

from __future__ import annotations

import logging

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from src.shared.constants import (
    COLOR_ACCENT,
    COLOR_ACCENT_ALT,
    COLOR_BG_CONTROL,
    COLOR_BG_MAIN,
    COLOR_BG_PANEL,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
)

logger = logging.getLogger(__name__)


DARK_QSS: str = f"""
/* ===== Base ===== */
QWidget {{
    background-color: {COLOR_BG_MAIN};
    color: {COLOR_TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    font-size: 13px;
    selection-background-color: {COLOR_ACCENT};
    selection-color: #ffffff;
}}

QMainWindow {{
    background-color: {COLOR_BG_MAIN};
}}

QMainWindow::separator {{
    background: {COLOR_BORDER};
    width: 2px;
    height: 2px;
}}

QToolTip {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    padding: 6px 8px;
    border-radius: 4px;
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: {COLOR_BG_CONTROL};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
    color: #ffffff;
}}
QPushButton:pressed {{
    background-color: {COLOR_ACCENT_ALT};
    border-color: {COLOR_ACCENT_ALT};
}}
QPushButton:disabled {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_SECONDARY};
    border-color: {COLOR_BORDER};
}}
QPushButton:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
    color: #ffffff;
}}

/* ===== Inputs ===== */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLOR_BG_CONTROL};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {COLOR_ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {COLOR_ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {COLOR_TEXT_SECONDARY};
}}

QComboBox::drop-down {{
    width: 18px;
    border-left: 1px solid {COLOR_BORDER};
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    selection-background-color: {COLOR_ACCENT};
    selection-color: #ffffff;
    outline: 0;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {COLOR_BG_CONTROL};
    border: none;
    width: 14px;
}}

/* ===== Sliders ===== */
QSlider::groove:horizontal {{
    height: 6px;
    background: {COLOR_BG_CONTROL};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_ACCENT};
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_ACCENT_ALT};
    border-radius: 3px;
}}

/* ===== Check / radio ===== */
QCheckBox, QRadioButton {{
    color: {COLOR_TEXT_PRIMARY};
    spacing: 6px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLOR_BORDER};
    background: {COLOR_BG_CONTROL};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {COLOR_BG_PANEL};
}}

/* ===== Group boxes ===== */
QGroupBox {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {COLOR_TEXT_PRIMARY};
}}

/* ===== Lists / tables ===== */
QListWidget, QTableWidget, QTreeWidget {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    gridline-color: {COLOR_BORDER};
    outline: 0;
}}
QListWidget::item, QTableWidget::item, QTreeWidget::item {{
    padding: 4px;
}}
QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected {{
    background: {COLOR_ACCENT};
    color: #ffffff;
}}
QHeaderView::section {{
    background-color: {COLOR_BG_CONTROL};
    color: {COLOR_TEXT_PRIMARY};
    border: none;
    border-right: 1px solid {COLOR_BORDER};
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 4px 8px;
    font-weight: bold;
}}

/* ===== Progress bar ===== */
QProgressBar {{
    background-color: {COLOR_BG_CONTROL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    color: {COLOR_TEXT_PRIMARY};
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 3px;
}}

/* ===== Tabs ===== */
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    background: {COLOR_BG_PANEL};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {COLOR_BG_CONTROL};
    color: {COLOR_TEXT_SECONDARY};
    border: 1px solid {COLOR_BORDER};
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {COLOR_ACCENT};
    color: #ffffff;
    border-color: {COLOR_ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {COLOR_TEXT_PRIMARY};
}}

/* ===== Scrollbars ===== */
QScrollBar:vertical {{
    background: {COLOR_BG_MAIN};
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BG_CONTROL};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLOR_ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {COLOR_BG_MAIN};
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {COLOR_BG_CONTROL};
    border-radius: 6px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLOR_ACCENT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ===== Menu / toolbar / status ===== */
QMenuBar {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_PRIMARY};
    border-bottom: 1px solid {COLOR_BORDER};
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 10px;
}}
QMenuBar::item:selected {{
    background: {COLOR_ACCENT};
    color: #ffffff;
}}
QMenu {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 22px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background: {COLOR_ACCENT};
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background: {COLOR_BORDER};
    margin: 4px 6px;
}}

QToolBar {{
    background: {COLOR_BG_PANEL};
    border-bottom: 1px solid {COLOR_BORDER};
    spacing: 4px;
    padding: 4px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: {COLOR_TEXT_PRIMARY};
    padding: 4px 8px;
    border-radius: 4px;
}}
QToolBar QToolButton:hover {{
    background: {COLOR_ACCENT};
    color: #ffffff;
}}
QToolBar QToolButton:checked {{
    background: {COLOR_ACCENT_ALT};
}}

QStatusBar {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_TEXT_SECONDARY};
    border-top: 1px solid {COLOR_BORDER};
}}

/* ===== Splitter ===== */
QSplitter::handle {{
    background: {COLOR_BORDER};
}}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QSplitter::handle:hover {{ background: {COLOR_ACCENT}; }}

/* ===== DockWidget ===== */
QDockWidget {{
    color: {COLOR_TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: {COLOR_BG_PANEL};
    padding: 6px 10px;
    border-bottom: 1px solid {COLOR_BORDER};
    font-weight: bold;
}}

/* Status coloring helpers */
QLabel[role="success"] {{ color: {COLOR_SUCCESS}; }}
QLabel[role="warning"] {{ color: {COLOR_WARNING}; }}
QLabel[role="error"] {{ color: {COLOR_ERROR}; }}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the dark stylesheet and a matching palette to the QApplication.

    Args:
        app: Running :class:`QApplication` instance.
    """
    if app is None:  # pragma: no cover - defensive
        raise ValueError("QApplication instance is required")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_BG_MAIN))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_BG_CONTROL))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_BG_PANEL))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLOR_BG_PANEL))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLOR_BG_CONTROL))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor(COLOR_ACCENT_ALT))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(COLOR_TEXT_SECONDARY),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(COLOR_TEXT_SECONDARY),
    )

    app.setPalette(palette)
    app.setStyleSheet(DARK_QSS)
    logger.debug("Dark theme applied")
