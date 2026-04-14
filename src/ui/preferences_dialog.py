"""Application preferences dialog."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.infrastructure.config_storage import AppSettings, SettingsStorage
from src.shared.constants import (
    MAX_PARALLEL_WORKERS,
    MIN_PARALLEL_WORKERS,
    UI_AUTOSAVE_INTERVAL_PAGES,
)

logger = logging.getLogger(__name__)


class PreferencesDialog(QDialog):
    """Modal dialog that edits :class:`AppSettings`."""

    def __init__(
        self,
        settings_storage: SettingsStorage,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._storage = settings_storage
        self._settings: AppSettings = settings_storage.load()

        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        perf_group = QGroupBox("Производительность")
        perf_form = QFormLayout(perf_group)
        self.workers = QSpinBox(self)
        self.workers.setRange(MIN_PARALLEL_WORKERS, MAX_PARALLEL_WORKERS)
        self.workers.setToolTip(
            "Количество параллельных процессов обработки файлов.\n"
            "Рекомендуется 2–4 в зависимости от числа ядер CPU."
        )
        perf_form.addRow("Параллельных воркеров:", self.workers)
        root.addWidget(perf_group)

        reliability_group = QGroupBox("Надёжность")
        rel_form = QFormLayout(reliability_group)
        self.autosave = QSpinBox(self)
        self.autosave.setRange(0, 500)
        self.autosave.setSpecialValueText("Отключено")
        self.autosave.setSuffix(" стр.")
        self.autosave.setToolTip(
            "Частота автосохранения промежуточных результатов при длинных заданиях. "
            "0 — отключено."
        )
        rel_form.addRow("Интервал автосохранения:", self.autosave)
        root.addWidget(reliability_group)

        ui_group = QGroupBox("Интерфейс")
        ui_form = QFormLayout(ui_group)
        self.dark_theme = QCheckBox("Тёмная тема", self)
        self.dark_theme.setToolTip(
            "Переключение темы вступает в силу после перезапуска приложения."
        )
        ui_form.addRow(self.dark_theme)
        root.addWidget(ui_group)

        note = QLabel(
            "<i>Некоторые настройки применяются после перезапуска.</i>",
            self,
        )
        note.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _populate(self) -> None:
        self.workers.setValue(self._settings.parallel_workers)
        self.autosave.setValue(
            self._settings.autosave_interval_pages
            if self._settings.autosave_interval_pages
            else UI_AUTOSAVE_INTERVAL_PAGES
        )
        self.dark_theme.setChecked(self._settings.theme == "dark")

    # --------------------------------------------------------------- save
    def _on_accept(self) -> None:
        from src.shared.validators import ValidationError, validate_workers

        try:
            workers = validate_workers(self.workers.value())
        except ValidationError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Настройки", str(exc))
            return
        self._settings.parallel_workers = workers
        self._settings.autosave_interval_pages = self.autosave.value()
        self._settings.theme = "dark" if self.dark_theme.isChecked() else "light"
        try:
            self._storage.save(self._settings)
        except Exception as exc:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox

            logger.exception("Failed to save preferences: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить настройки: {exc}")
            return
        self.accept()

    def settings(self) -> AppSettings:
        """Return the (possibly modified) settings after the dialog closes."""
        return self._settings
