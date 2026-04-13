"""Simple read-only log viewer dialog."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Tail a maximum of this many bytes from the end of the file to keep the
# viewer snappy even with megabyte-sized logs.
_TAIL_BYTES = 512 * 1024
_REFRESH_MS = 1500


class LogViewer(QDialog):
    """Show the tail of the application log file with live updates."""

    def __init__(self, log_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Журнал — {log_path.name}")
        self.setMinimumSize(900, 500)
        self._log_path = log_path

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(str(log_path)))
        header.addStretch(1)
        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self._refresh)
        header.addWidget(self.btn_refresh)
        self.btn_clear = QPushButton("Очистить файл")
        self.btn_clear.clicked.connect(self._clear_file)
        header.addWidget(self.btn_clear)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.accept)
        header.addWidget(self.btn_close)
        root.addLayout(header)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.text.setFont(mono)
        root.addWidget(self.text, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        self._refresh()

    # ----------------------------------------------------------- refresh
    def _refresh(self) -> None:
        try:
            if not self._log_path.exists():
                self.text.setPlainText(f"(файл не существует: {self._log_path})")
                return
            size = self._log_path.stat().st_size
            with self._log_path.open("rb") as f:
                if size > _TAIL_BYTES:
                    f.seek(size - _TAIL_BYTES)
                    # Skip a partial line
                    f.readline()
                data = f.read()
            text = data.decode("utf-8", errors="replace")
        except OSError as exc:
            text = f"(ошибка чтения: {exc})"
        # Preserve scroll position if user scrolled up
        sb = self.text.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.text.setPlainText(text)
        if at_bottom:
            cursor = self.text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text.setTextCursor(cursor)
            sb.setValue(sb.maximum())

    def _clear_file(self) -> None:
        resp = QMessageBox.question(
            self,
            "Очистить лог",
            "Удалить текущее содержимое лог-файла? Это необратимо.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            self._log_path.write_text("", encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self._refresh()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().closeEvent(event)
