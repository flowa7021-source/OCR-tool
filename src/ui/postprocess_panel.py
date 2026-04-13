"""Postprocessing panel: edit text corrections and custom regex rules."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.models import PostprocessConfig, RegexRule
from src.shared.constants import UI_PREVIEW_UPDATE_DEBOUNCE_MS

logger = logging.getLogger(__name__)


class PostprocessPanel(QWidget):
    """Widget for editing :class:`PostprocessConfig`.

    Exposes toggles for builtin corrections (Russian / English autocorrection,
    hyphen merge, whitespace + Unicode normalization, artifact removal) and a
    table of user-defined regex rules.
    """

    config_changed = Signal(object)  # PostprocessConfig

    _RULE_COLUMNS = ("", "Шаблон", "Замена", "Regex", "Case", "Описание")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = PostprocessConfig()
        self._suppress = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(UI_PREVIEW_UPDATE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_config)

        self._build_ui()
        self.set_config(self._config)

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Built-in toggles
        builtin_group = QGroupBox("Встроенные правила")
        b_layout = QVBoxLayout(builtin_group)
        self.cb_rus = self._make_checkbox("Автокоррекция русского (О↔0, З↔3, Ч↔4, б↔6 …)")
        self.cb_eng = self._make_checkbox("Автокоррекция английского (l↔1, O↔0, rn→m …)")
        self.cb_hyphen = self._make_checkbox("Объединять слова с переносом по строке")
        self.cb_ws = self._make_checkbox("Нормализовать пробелы и табуляции")
        self.cb_unicode = self._make_checkbox("Нормализовать Unicode (NFC)")
        self.cb_artifacts = self._make_checkbox("Удалять OCR-артефакты (одиночные не-буквенные символы)")
        for cb in (self.cb_rus, self.cb_eng, self.cb_hyphen, self.cb_ws, self.cb_unicode, self.cb_artifacts):
            b_layout.addWidget(cb)
        root.addWidget(builtin_group)

        # Custom rules
        rules_group = QGroupBox("Пользовательские правила (найти → заменить)")
        r_layout = QVBoxLayout(rules_group)

        self.rules_table = QTableWidget(0, len(self._RULE_COLUMNS), rules_group)
        self.rules_table.setHorizontalHeaderLabels(self._RULE_COLUMNS)
        self.rules_table.verticalHeader().setVisible(False)
        self.rules_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.rules_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.rules_table.itemChanged.connect(self._on_rule_cell_changed)
        r_layout.addWidget(self.rules_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Добавить правило", rules_group)
        add_btn.clicked.connect(self._add_rule)
        btn_row.addWidget(add_btn)

        del_btn = QPushButton("Удалить", rules_group)
        del_btn.clicked.connect(self._delete_selected_rule)
        btn_row.addWidget(del_btn)

        import_btn = QPushButton("Импорт JSON…", rules_group)
        import_btn.clicked.connect(self._import_rules)
        btn_row.addWidget(import_btn)

        export_btn = QPushButton("Экспорт JSON…", rules_group)
        export_btn.clicked.connect(self._export_rules)
        btn_row.addWidget(export_btn)
        btn_row.addStretch(1)
        r_layout.addLayout(btn_row)

        root.addWidget(rules_group, 1)

        # Quick preview row
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Тест:"))
        self.preview_input = QLineEdit(self)
        self.preview_input.setPlaceholderText("Вставьте строку для проверки правил…")
        self.preview_input.textChanged.connect(self._refresh_preview)
        preview_row.addWidget(self.preview_input, 1)
        self.preview_output = QLineEdit(self)
        self.preview_output.setReadOnly(True)
        preview_row.addWidget(self.preview_output, 1)
        root.addLayout(preview_row)

    def _make_checkbox(self, text: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.toggled.connect(self._on_toggle)
        return cb

    # --------------------------------------------------------------- state
    def set_config(self, cfg: PostprocessConfig) -> None:
        """Populate all widgets from ``cfg``."""
        self._suppress = True
        try:
            self._config = cfg
            self.cb_rus.setChecked(cfg.autocorrect_russian)
            self.cb_eng.setChecked(cfg.autocorrect_english)
            self.cb_hyphen.setChecked(cfg.merge_hyphenated)
            self.cb_ws.setChecked(cfg.normalize_whitespace)
            self.cb_unicode.setChecked(cfg.normalize_unicode)
            self.cb_artifacts.setChecked(cfg.remove_artifacts)
            self._rebuild_rules_table(cfg.custom_rules)
        finally:
            self._suppress = False
        self._refresh_preview()

    def get_config(self) -> PostprocessConfig:
        """Return a :class:`PostprocessConfig` reflecting current widgets."""
        rules = self._rules_from_table()
        cfg = PostprocessConfig(
            autocorrect_russian=self.cb_rus.isChecked(),
            autocorrect_english=self.cb_eng.isChecked(),
            merge_hyphenated=self.cb_hyphen.isChecked(),
            normalize_whitespace=self.cb_ws.isChecked(),
            normalize_unicode=self.cb_unicode.isChecked(),
            remove_artifacts=self.cb_artifacts.isChecked(),
            custom_rules=rules,
        )
        self._config = cfg
        return cfg

    # --------------------------------------------------------------- rules
    def _rebuild_rules_table(self, rules: list[RegexRule]) -> None:
        self.rules_table.blockSignals(True)
        self.rules_table.setRowCount(len(rules))
        for row, rule in enumerate(rules):
            self._set_rule_row(row, rule)
        self.rules_table.blockSignals(False)

    def _set_rule_row(self, row: int, rule: RegexRule) -> None:
        # Enabled checkbox
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        chk.setCheckState(Qt.CheckState.Checked if rule.enabled else Qt.CheckState.Unchecked)
        self.rules_table.setItem(row, 0, chk)
        self.rules_table.setItem(row, 1, QTableWidgetItem(rule.pattern))
        self.rules_table.setItem(row, 2, QTableWidgetItem(rule.replacement))
        regex = QTableWidgetItem()
        regex.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        regex.setCheckState(Qt.CheckState.Checked if rule.is_regex else Qt.CheckState.Unchecked)
        self.rules_table.setItem(row, 3, regex)
        cs = QTableWidgetItem()
        cs.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        cs.setCheckState(Qt.CheckState.Checked if rule.case_sensitive else Qt.CheckState.Unchecked)
        self.rules_table.setItem(row, 4, cs)
        self.rules_table.setItem(row, 5, QTableWidgetItem(rule.description))

    def _rules_from_table(self) -> list[RegexRule]:
        rules: list[RegexRule] = []
        for row in range(self.rules_table.rowCount()):
            pattern_item = self.rules_table.item(row, 1)
            repl_item = self.rules_table.item(row, 2)
            if pattern_item is None or not pattern_item.text():
                continue
            rules.append(
                RegexRule(
                    pattern=pattern_item.text(),
                    replacement=repl_item.text() if repl_item is not None else "",
                    enabled=self._check_state(row, 0),
                    is_regex=self._check_state(row, 3),
                    case_sensitive=self._check_state(row, 4),
                    description=(self.rules_table.item(row, 5).text()
                                 if self.rules_table.item(row, 5) else ""),
                )
            )
        return rules

    def _check_state(self, row: int, col: int) -> bool:
        item = self.rules_table.item(row, col)
        return item is not None and item.checkState() == Qt.CheckState.Checked

    def _add_rule(self) -> None:
        rule = RegexRule(pattern="", replacement="", description="")
        row = self.rules_table.rowCount()
        self.rules_table.insertRow(row)
        self._set_rule_row(row, rule)
        self.rules_table.editItem(self.rules_table.item(row, 1))

    def _delete_selected_rule(self) -> None:
        rows = sorted({idx.row() for idx in self.rules_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.rules_table.removeRow(row)
        self._schedule_emit()

    def _import_rules(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт правил", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            import json
            from pathlib import Path

            data = json.loads(Path(path).read_text(encoding="utf-8"))
            rules = [
                RegexRule(
                    pattern=r.get("pattern", ""),
                    replacement=r.get("replacement", ""),
                    enabled=r.get("enabled", True),
                    is_regex=r.get("is_regex", True),
                    case_sensitive=r.get("case_sensitive", True),
                    description=r.get("description", ""),
                )
                for r in data
            ]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл: {exc}")
            return
        self._rebuild_rules_table(rules)
        self._schedule_emit()

    def _export_rules(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт правил", "rules.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            import json
            from pathlib import Path

            rules = self._rules_from_table()
            payload = [
                {
                    "pattern": r.pattern,
                    "replacement": r.replacement,
                    "enabled": r.enabled,
                    "is_regex": r.is_regex,
                    "case_sensitive": r.case_sensitive,
                    "description": r.description,
                }
                for r in rules
            ]
            Path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {exc}")

    # --------------------------------------------------------------- change
    def _on_toggle(self, _checked: bool = False) -> None:
        if not self._suppress:
            self._schedule_emit()

    def _on_rule_cell_changed(self, _item: QTableWidgetItem) -> None:
        if not self._suppress:
            self._schedule_emit()

    def _schedule_emit(self) -> None:
        self._debounce.start()

    def _emit_config(self) -> None:
        cfg = self.get_config()
        self.config_changed.emit(cfg)
        self._refresh_preview()

    # --------------------------------------------------------------- preview
    def _refresh_preview(self, *args: Any) -> None:
        text = self.preview_input.text()
        if not text:
            self.preview_output.setText("")
            return
        try:
            from src.core.text_postprocessor import TextPostprocessor

            processed = TextPostprocessor().process(text, self.get_config())
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview postprocess failed: %s", exc)
            processed = text
        self.preview_output.setText(processed)
