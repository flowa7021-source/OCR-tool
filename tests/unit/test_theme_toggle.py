"""Runtime theme toggle: apply_theme() + View menu + Preferences live apply.

These verify that switching between ``"dark"`` and ``"light"`` actually
swaps the stylesheet at runtime and that persistence round-trips via
``SettingsStorage``. A broken toggle would strand users on whichever
theme the app started with — hard to notice in CI without an explicit
test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestApplyTheme:
    def test_dark_installs_dark_qss(self, qtbot) -> None:
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import DARK_QSS, apply_theme

        app = QApplication.instance()
        assert app is not None
        apply_theme(app, "dark")
        assert app.styleSheet() == DARK_QSS

    def test_light_installs_light_qss(self, qtbot) -> None:
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import LIGHT_QSS, apply_theme

        app = QApplication.instance()
        assert app is not None
        apply_theme(app, "light")
        assert app.styleSheet() == LIGHT_QSS

    def test_unknown_kind_falls_back_to_dark(self, qtbot) -> None:
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import DARK_QSS, apply_theme

        app = QApplication.instance()
        apply_theme(app, "neon-green")  # type: ignore[arg-type]
        assert app.styleSheet() == DARK_QSS

    def test_round_trip_dark_to_light_to_dark(self, qtbot) -> None:
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import DARK_QSS, LIGHT_QSS, apply_theme

        app = QApplication.instance()
        apply_theme(app, "dark")
        apply_theme(app, "light")
        assert app.styleSheet() == LIGHT_QSS
        apply_theme(app, "dark")
        assert app.styleSheet() == DARK_QSS


class TestPreferencesDialogLive:
    def test_toggle_applies_live_stylesheet(self, qtbot, tmp_path: Path) -> None:
        """Flipping the checkbox updates QApplication.styleSheet immediately."""
        from PySide6.QtWidgets import QApplication

        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.preferences_dialog import PreferencesDialog
        from src.ui.theme import DARK_QSS, LIGHT_QSS, apply_theme

        storage = SettingsStorage(config_dir=tmp_path)
        settings = storage.load()
        settings.theme = "dark"
        storage.save(settings)

        app = QApplication.instance()
        apply_theme(app, "dark")

        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        # Dark → flip checkbox off → should become light.
        dlg.dark_theme.setChecked(False)
        assert app.styleSheet() == LIGHT_QSS
        # Flip it back → dark again.
        dlg.dark_theme.setChecked(True)
        assert app.styleSheet() == DARK_QSS

    def test_cancel_restores_initial_theme(self, qtbot, tmp_path: Path) -> None:
        from PySide6.QtWidgets import QApplication

        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.preferences_dialog import PreferencesDialog
        from src.ui.theme import DARK_QSS, apply_theme

        storage = SettingsStorage(config_dir=tmp_path)
        settings = storage.load()
        settings.theme = "dark"
        storage.save(settings)

        app = QApplication.instance()
        apply_theme(app, "dark")

        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        dlg.dark_theme.setChecked(False)
        # User cancels: dialog should roll back to dark.
        dlg._on_reject()
        assert app.styleSheet() == DARK_QSS
        # Persisted setting unchanged.
        assert storage.load().theme == "dark"

    def test_ok_persists_new_theme(self, qtbot, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.preferences_dialog import PreferencesDialog

        storage = SettingsStorage(config_dir=tmp_path)
        settings = storage.load()
        settings.theme = "dark"
        storage.save(settings)

        dlg = PreferencesDialog(storage)
        qtbot.addWidget(dlg)
        dlg.dark_theme.setChecked(False)
        dlg._on_accept()
        assert storage.load().theme == "light"


class TestAppSettingsThemeField:
    def test_theme_default_is_dark(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        assert storage.load().theme == "dark"

    def test_theme_round_trip_to_disk(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        settings = storage.load()
        settings.theme = "light"
        storage.save(settings)

        # Re-open through a fresh storage to confirm file round-trip
        storage2 = SettingsStorage(config_dir=tmp_path)
        assert storage2.load().theme == "light"
