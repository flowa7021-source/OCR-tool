"""Modal dialog that downloads an HTR model with live progress."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.infrastructure.model_manager import (
    ModelDownloadError,
    ModelManager,
    ModelSpec,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _DownloadWorker(QObject):
    """Runs in a QThread; bridges :class:`ModelManager` to Qt signals."""

    progress = Signal(int, int, str)  # bytes_done, bytes_total, current_file
    finished = Signal(bool, str)  # success, message

    def __init__(
        self,
        manager: ModelManager,
        model_id: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._model_id = model_id
        self._cancel = False

    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._cancel = True

    def run(self) -> None:
        try:
            self._manager.download(
                self._model_id,
                progress_callback=lambda d, t, n: self.progress.emit(d, t, n),
                cancel=lambda: self._cancel,
            )
        except ModelDownloadError as exc:
            logger.warning("Model download failed: %s", exc)
            self.finished.emit(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected model-download error")
            self.finished.emit(False, f"Неожиданная ошибка: {exc}")
        else:
            self.finished.emit(True, "")


class ModelDownloadDialog(QDialog):
    """User-facing download dialog for an HTR model."""

    def __init__(
        self,
        manager: ModelManager,
        spec: ModelSpec,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Скачать {spec.label}")
        self.setMinimumWidth(520)
        self._manager = manager
        self._spec = spec

        root = QVBoxLayout(self)

        size_mb = max(1, spec.total_size_bytes // (1024 * 1024))
        info = QLabel(
            f"<b>{spec.label}</b><br>"
            f"{spec.description}<br><br>"
            f"Объём загрузки: ~{size_mb} МБ. После завершения движок "
            f"станет доступен в выпадающем списке OCR-движков.",
            self,
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("Ожидание…")
        root.addWidget(self._progress)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._buttons = QDialogButtonBox(self)
        self._btn_start = QPushButton("Начать", self)
        self._btn_cancel = QPushButton("Отмена", self)
        self._buttons.addButton(self._btn_start, QDialogButtonBox.ButtonRole.AcceptRole)
        self._buttons.addButton(self._btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        root.addWidget(self._buttons)

        self._btn_start.clicked.connect(self._start_download)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)

        self._thread: QThread | None = None
        self._worker: _DownloadWorker | None = None

    # ------------------------------------------------------------------
    def _start_download(self) -> None:
        if self._thread is not None:
            return  # already running
        self._btn_start.setEnabled(False)
        self._progress.setFormat("Скачивание…")
        self._status.setText("Подключение к Hugging Face Hub…")

        self._thread = QThread(self)
        self._worker = _DownloadWorker(self._manager, self._spec.model_id)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, done: int, total: int, name: str) -> None:
        if total > 0:
            pct = int(min(100, max(0, 100 * done / total)))
            self._progress.setValue(pct)
            done_mb = done / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self._progress.setFormat(f"{pct}% — {done_mb:.1f} / {total_mb:.1f} МБ")
        self._status.setText(f"Файл: {name}")

    def _on_finished(self, ok: bool, message: str) -> None:
        if ok:
            self._progress.setValue(100)
            self._progress.setFormat("Готово")
            self._status.setText("Модель успешно скачана.")
            self._btn_cancel.setText("Закрыть")
            self._btn_start.setEnabled(False)
        else:
            self._status.setText(f"Ошибка: {message}")
            self._btn_start.setEnabled(True)
            self._btn_start.setText("Повторить")

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None and self._thread is not None and self._thread.isRunning():
            self._worker.cancel()
            self._status.setText("Отмена…")
            return
        self.reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
