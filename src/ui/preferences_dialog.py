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
    QLineEdit,
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
        self._initial_theme: str = (self._settings.theme or "dark").lower()

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

        # Disk-cache budget. Permanent cache under %LOCALAPPDATA%/
        # ocr-cache/ — second run of the same PDF with the same profile
        # is served from it for free. Tighter than 500 MB is rarely
        # useful; 0 disables the cache entirely.
        from PySide6.QtWidgets import QComboBox

        self.cache_budget = QComboBox(self)
        for label, value in (
            ("Отключён", 0),
            ("512 МБ", 512),
            ("1 ГБ", 1024),
            ("2 ГБ (по-умолчанию)", 2048),
            ("5 ГБ", 5120),
        ):
            self.cache_budget.addItem(label, value)
        self.cache_budget.setToolTip(
            "Лимит дискового кэша результатов OCR. Повторный прогон "
            "того же PDF с тем же профилем в рамках лимита "
            "возвращает результат мгновенно."
        )
        rel_form.addRow("Кэш OCR:", self.cache_budget)
        root.addWidget(reliability_group)

        ui_group = QGroupBox("Интерфейс")
        ui_form = QFormLayout(ui_group)
        self.dark_theme = QCheckBox("Тёмная тема", self)
        self.dark_theme.setToolTip(
            "Переключение темы применяется мгновенно (без перезапуска)."
        )
        self.dark_theme.toggled.connect(self._on_theme_toggled)
        ui_form.addRow(self.dark_theme)

        self.notify_on_complete = QCheckBox(
            "Уведомление при завершении задания", self
        )
        self.notify_on_complete.setToolTip(
            "Всплывающее уведомление системного трея, когда очередь OCR "
            "обработана и главное окно не в фокусе. Ошибки сигнализируются "
            "независимо от этой настройки."
        )
        ui_form.addRow(self.notify_on_complete)
        root.addWidget(ui_group)

        # --- LLM fallback (opt-in) -----------------------------------------
        # CLAUDE.md: приложение должно работать полностью оффлайн.
        # Пустой ключ здесь = LLM-fallback выключен; парсер ТН/УПД
        # тихо деградирует до regex-only пути. Пользователь заполняет
        # поле, только если явно хочет рескью на низко-confidence
        # строках (≤ 40 %) с тарифицируемым Claude API.
        llm_group = QGroupBox("LLM-фоллбэк (опционально)", self)
        llm_form = QFormLayout(llm_group)
        self.llm_api_key = QLineEdit(self)
        # EchoMode.PasswordEchoOnEdit — показывает значение пока
        # пользователь печатает, потом скрывает точками. Даёт
        # возможность проверить вставку ключа без постоянно
        # открытой секретной строки.
        self.llm_api_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.llm_api_key.setPlaceholderText(
            "sk-ant-api03-… (оставьте пустым, чтобы отключить)"
        )
        self.llm_api_key.setToolTip(
            "Anthropic API key для Claude-fallback парсера ТН/УПД. "
            "Хранится в settings.json пользователя (НЕ в профиле — "
            "экспорт профиля не утекает ключ). Используется, когда "
            "профиль включает ``extract.llm_fallback.enabled=True`` "
            "И строка парсера имеет общую уверенность ниже 40 %. "
            "При пустом ключе парсер работает без LLM."
        )
        llm_form.addRow("Anthropic API key:", self.llm_api_key)

        llm_hint = QLabel(
            "Ключ передаётся в ANTHROPIC_API_KEY окружения на время "
            "сессии. Установите пакет Anthropic SDK отдельно: "
            "<code>pip install \"ocr-studio[llm]\"</code>.",
            self,
        )
        llm_hint.setWordWrap(True)
        llm_hint.setTextFormat(Qt.TextFormat.RichText)
        llm_form.addRow(llm_hint)
        root.addWidget(llm_group)

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
        buttons.rejected.connect(self._on_reject)
        root.addWidget(buttons)

    def _populate(self) -> None:
        self.workers.setValue(self._settings.parallel_workers)
        self.autosave.setValue(
            self._settings.autosave_interval_pages
            if self._settings.autosave_interval_pages
            else UI_AUTOSAVE_INTERVAL_PAGES
        )
        self.dark_theme.setChecked(self._settings.theme == "dark")
        self.notify_on_complete.setChecked(bool(self._settings.notify_on_complete))
        # Select the cache-budget combo item matching the stored value.
        saved_cache = int(self._settings.ocr_cache_max_mb)
        idx = self.cache_budget.findData(saved_cache)
        if idx < 0:
            # Non-standard value — show the closest preset. The user's
            # custom number is still honoured until they change it.
            idx = self.cache_budget.findData(2048)
        self.cache_budget.setCurrentIndex(max(0, idx))
        self.llm_api_key.setText(self._settings.anthropic_api_key or "")

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
        self._settings.notify_on_complete = bool(self.notify_on_complete.isChecked())
        cache_mb = self.cache_budget.currentData()
        if isinstance(cache_mb, int):
            self._settings.ocr_cache_max_mb = int(cache_mb)
        # Strip the API key — users routinely copy-paste a trailing
        # newline or space from password managers, which makes the
        # stored credential silently invalid.
        self._settings.anthropic_api_key = self.llm_api_key.text().strip()
        try:
            self._storage.save(self._settings)
        except Exception as exc:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox

            logger.exception("Failed to save preferences: %s", exc)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить настройки: {exc}")
            return

        # Reflect the (possibly changed) key into the process
        # environment immediately — no restart required. Mirrors the
        # same call made at app startup in :func:`src.app.create_application`.
        try:
            from src.infrastructure.llm_credentials import apply_to_environment

            apply_to_environment(self._settings)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM credential re-apply failed: %s", exc)
        # Sync the main window's View→Theme checkbox with the saved choice.
        parent = self.parent()
        action = getattr(parent, "action_light_theme", None)
        if action is not None:
            try:
                action.blockSignals(True)
                action.setChecked(self._settings.theme == "light")
            finally:
                action.blockSignals(False)
        self.accept()

    # -------------------------------------------------------------- cancel
    def _on_reject(self) -> None:
        """Restore the pre-dialog theme if the user touched the toggle."""
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import apply_theme

        app = QApplication.instance()
        if app is not None:
            try:
                apply_theme(app, self._initial_theme)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Restoring theme on cancel failed: %s", exc)
        # Restore the main window's menu checkbox too.
        parent = self.parent()
        action = getattr(parent, "action_light_theme", None)
        if action is not None:
            try:
                action.blockSignals(True)
                action.setChecked(self._initial_theme == "light")
            finally:
                action.blockSignals(False)
        self.reject()

    def settings(self) -> AppSettings:
        """Return the (possibly modified) settings after the dialog closes."""
        return self._settings

    # ----------------------------------------------------------- live theme
    def _on_theme_toggled(self, checked: bool) -> None:
        """Re-apply the stylesheet immediately so the user sees the change.

        The preference itself is only persisted on OK. If the user cancels,
        the main window restores the previous theme from settings on close.
        """
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import apply_theme

        app = QApplication.instance()
        if app is None:
            return
        try:
            apply_theme(app, "dark" if checked else "light")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Live theme re-apply failed: %s", exc)

        # Let the main window sync its menu checkbox if one is present.
        parent = self.parent()
        action = getattr(parent, "action_light_theme", None)
        if action is not None:
            try:
                action.blockSignals(True)
                action.setChecked(not checked)
            finally:
                action.blockSignals(False)
