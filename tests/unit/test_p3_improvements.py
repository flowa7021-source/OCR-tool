"""P3 improvements: config backup, tray notifier, accessibility labels.

Skips anything that requires a real desktop tray (QSystemTrayIcon is
mocked so these run under ``QT_QPA_PLATFORM=offscreen`` on CI).
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# Config backup round-trip
# --------------------------------------------------------------------------


class TestConfigBackupRoundTrip:
    def _storages(self, tmp_path: Path):
        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage

        settings = SettingsStorage(config_dir=tmp_path / "cfg")
        profiles = ProfileStorage(profiles_dir=tmp_path / "profiles")
        return settings, profiles

    def test_export_contains_manifest_settings_and_profiles(
        self, tmp_path: Path
    ) -> None:
        from src.application.config_backup import (
            BACKUP_SCHEMA_VERSION,
            export_config,
        )
        from src.core.models import ProfileData

        settings, profiles = self._storages(tmp_path)
        # Seed a custom profile that should end up in the ZIP.
        custom = ProfileData(name="mine")
        profiles.save(custom)

        target = tmp_path / "out.zip"
        export_config(target, settings, profiles)

        with zipfile.ZipFile(target) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "settings.json" in names
            assert "profiles/mine.json" in names
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["schema_version"] == BACKUP_SCHEMA_VERSION

    def test_import_restores_profiles_and_settings(self, tmp_path: Path) -> None:
        from src.application.config_backup import export_config, import_config
        from src.core.models import ProfileData

        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src_root.mkdir()
        dst_root.mkdir()

        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage

        src_settings = SettingsStorage(config_dir=src_root / "cfg")
        src_profiles = ProfileStorage(profiles_dir=src_root / "profiles")
        saved = src_settings.load()
        saved.theme = "light"
        saved.parallel_workers = 3
        src_settings.save(saved)
        src_profiles.save(ProfileData(name="travel"))

        backup = tmp_path / "bak.zip"
        export_config(backup, src_settings, src_profiles)

        dst_settings = SettingsStorage(config_dir=dst_root / "cfg")
        dst_profiles = ProfileStorage(profiles_dir=dst_root / "profiles")
        # Baseline for the destination.
        dst_settings.save(dst_settings.load())  # ensure file exists

        result = import_config(backup, dst_settings, dst_profiles)
        assert result.settings_restored is True
        assert "travel.json" in result.profiles_added
        restored = SettingsStorage(config_dir=dst_root / "cfg").load()
        assert restored.theme == "light"
        assert restored.parallel_workers == 3

    def test_import_rejects_non_zip(self, tmp_path: Path) -> None:
        from src.application.config_backup import BackupFormatError, import_config

        bogus = tmp_path / "fake.zip"
        bogus.write_bytes(b"this is not a zip file at all")
        s, p = self._storages(tmp_path)
        with pytest.raises(BackupFormatError):
            import_config(bogus, s, p)

    def test_import_rejects_zip_without_manifest(self, tmp_path: Path) -> None:
        from src.application.config_backup import BackupFormatError, import_config

        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("random.txt", "hello")
        s, p = self._storages(tmp_path)
        with pytest.raises(BackupFormatError):
            import_config(path, s, p)

    def test_import_rejects_newer_schema_version(self, tmp_path: Path) -> None:
        from src.application.config_backup import BackupFormatError, import_config

        path = tmp_path / "future.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "manifest.json",
                json.dumps({"schema_version": 99, "app_version": "x"}),
            )
        s, p = self._storages(tmp_path)
        with pytest.raises(BackupFormatError):
            import_config(path, s, p)

    def test_import_skips_path_traversal(self, tmp_path: Path) -> None:
        """A malicious archive trying to write outside profiles_dir is ignored."""
        from src.application.config_backup import import_config

        path = tmp_path / "evil.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
            zf.writestr("profiles/../../etc_passwd.json", '{"name":"evil"}')
        s, p = self._storages(tmp_path)
        result = import_config(path, s, p)
        assert result.profiles_added == []

    def test_import_overwrite_flag_honoured(self, tmp_path: Path) -> None:
        from src.application.config_backup import export_config, import_config
        from src.core.models import OCRConfig, ProfileData
        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage

        src_profiles = ProfileStorage(profiles_dir=tmp_path / "src")
        src_profiles.save(ProfileData(name="shared", ocr=OCRConfig(dpi=600)))

        dst_profiles = ProfileStorage(profiles_dir=tmp_path / "dst")
        dst_profiles.save(ProfileData(name="shared", ocr=OCRConfig(dpi=200)))

        backup = tmp_path / "b.zip"
        export_config(
            backup,
            SettingsStorage(config_dir=tmp_path / "src_cfg"),
            src_profiles,
        )

        # Preserve existing: dst should still have dpi=200 afterwards.
        res = import_config(
            backup,
            SettingsStorage(config_dir=tmp_path / "dst_cfg"),
            dst_profiles,
            overwrite_existing_profiles=False,
        )
        assert "shared.json" not in res.profiles_added
        assert "shared.json" not in res.profiles_overwritten
        assert dst_profiles.load("shared").ocr.dpi == 200

        # Same backup, overwrite=True: dpi now 600.
        res2 = import_config(
            backup,
            SettingsStorage(config_dir=tmp_path / "dst_cfg"),
            dst_profiles,
            overwrite_existing_profiles=True,
        )
        assert "shared.json" in res2.profiles_overwritten
        assert dst_profiles.load("shared").ocr.dpi == 600


# --------------------------------------------------------------------------
# Tray notifier
# --------------------------------------------------------------------------


class TestTrayNotifier:
    def test_falls_back_silently_when_tray_unavailable(self, qtbot) -> None:
        from PySide6.QtGui import QIcon

        from src.ui.tray_notifier import TrayNotifier

        with patch(
            "src.ui.tray_notifier.QSystemTrayIcon.isSystemTrayAvailable",
            return_value=False,
        ):
            n = TrayNotifier(QIcon())
        # No-op; just must not raise.
        n.notify_complete("OCR", "done")
        n.notify_failure("OCR", "boom")
        n.shutdown()
        assert n._tray is None

    def test_notify_delegates_to_show_message(self, qtbot) -> None:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QSystemTrayIcon

        from src.ui.tray_notifier import TrayNotifier

        with (
            patch.object(
                QSystemTrayIcon, "isSystemTrayAvailable", return_value=True
            ),
            patch.object(QSystemTrayIcon, "showMessage") as mock_show,
            patch.object(QSystemTrayIcon, "show"),
        ):
            n = TrayNotifier(QIcon())
            n.notify_complete("Title", "Msg")
            assert mock_show.called
            args = mock_show.call_args[0]
            assert args[0] == "Title"
            assert args[1] == "Msg"


# --------------------------------------------------------------------------
# AppSettings new field
# --------------------------------------------------------------------------


class TestNotifyOnCompleteSetting:
    def test_default_is_true(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        s = SettingsStorage(config_dir=tmp_path).load()
        assert s.notify_on_complete is True

    def test_round_trip(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        settings = storage.load()
        settings.notify_on_complete = False
        storage.save(settings)
        assert (
            SettingsStorage(config_dir=tmp_path).load().notify_on_complete is False
        )


# --------------------------------------------------------------------------
# Accessibility helper + widget wiring
# --------------------------------------------------------------------------


class TestAccessibility:
    def test_describe_sets_name_description_and_tooltip(self, qtbot) -> None:
        from PySide6.QtWidgets import QLineEdit

        from src.ui.accessibility import describe

        w = QLineEdit()
        qtbot.addWidget(w)
        describe(w, name="N", description="D")
        assert w.accessibleName() == "N"
        assert w.accessibleDescription() == "D"
        assert w.toolTip() == "D"

    def test_describe_tooltip_override(self, qtbot) -> None:
        from PySide6.QtWidgets import QLineEdit

        from src.ui.accessibility import describe

        w = QLineEdit()
        qtbot.addWidget(w)
        describe(w, name="n", description="long description", tooltip="short")
        assert w.toolTip() == "short"
        assert w.accessibleDescription() == "long description"

    def test_preprocessing_panel_has_accessible_labels(self, qtbot) -> None:
        from src.ui.preprocessing_panel import PreprocessingPanel

        p = PreprocessingPanel()
        qtbot.addWidget(p)
        assert p.clahe_clip.accessibleName()
        assert p.clahe_clip.accessibleDescription()
        assert p.sauvola_k.accessibleDescription()
        assert p.nlm_h.accessibleDescription()

    def test_queue_panel_table_has_accessible_metadata(self, qtbot) -> None:
        from src.ui.queue_panel import QueuePanel

        p = QueuePanel()
        qtbot.addWidget(p)
        assert p.table.accessibleName()
        assert p.table.accessibleDescription()
        assert p._edit_filter.accessibleName()


# --------------------------------------------------------------------------
# Release workflow YAML
# --------------------------------------------------------------------------


class TestReleaseWorkflow:
    def test_release_yml_parses_and_has_expected_jobs(self) -> None:
        yaml = pytest.importorskip("yaml")

        path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "jobs" in data
        jobs = set(data["jobs"].keys())
        assert {"gate-tests", "sign", "publish"}.issubset(jobs)

    def test_release_yml_triggers_on_version_tags(self) -> None:
        yaml = pytest.importorskip("yaml")

        path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # YAML 1.1 parses the unquoted "on" as True (boolean), so the
        # dict key may be either the string "on" or the bool True.
        on = data.get("on", data.get(True))
        assert on is not None, f"No 'on' trigger found in {list(data.keys())}"
        tags = on["push"]["tags"]
        assert "v*" in tags


# --------------------------------------------------------------------------
# MainWindow toast suppression logic
# --------------------------------------------------------------------------


class TestCompletionToastGating:
    def _make_window(self, qtbot, tmp_path: Path):
        from src.application.export_manager import ExportManager
        from src.application.parallel_processor import ParallelProcessor
        from src.application.profile_manager import ProfileManager
        from src.application.queue_manager import QueueManager
        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage
        from src.ui.main_window import MainWindow

        pm = ProfileManager(ProfileStorage(profiles_dir=tmp_path / "profiles"))
        pm.initialize_builtins()
        w = MainWindow(
            profile_manager=pm,
            queue_manager=QueueManager(),
            parallel_processor=ParallelProcessor(max_workers=1),
            export_manager=ExportManager(),
            settings_storage=SettingsStorage(config_dir=tmp_path / "cfg"),
        )
        qtbot.addWidget(w)
        w._tray = MagicMock()
        return w

    def test_no_toast_when_disabled(self, qtbot, tmp_path: Path) -> None:
        from src.core.models import JobResult
        from src.shared.types import JobStatus

        w = self._make_window(qtbot, tmp_path)
        settings = w._settings_storage.load()
        settings.notify_on_complete = False
        w._settings_storage.save(settings)
        # Force inactive window so the isActiveWindow guard doesn't fire.
        with patch.object(w, "isActiveWindow", return_value=False):
            w._maybe_emit_completion_toast(
                JobResult(
                    job_id="j", status=JobStatus.COMPLETED,
                    input_path="x", output_path="x_ocr.pdf",
                )
            )
        assert not w._tray.notify_complete.called

    def test_failure_always_toasts_when_window_inactive(
        self, qtbot, tmp_path: Path
    ) -> None:
        from src.core.models import JobResult
        from src.shared.types import JobStatus

        w = self._make_window(qtbot, tmp_path)
        with patch.object(w, "isActiveWindow", return_value=False):
            w._maybe_emit_completion_toast(
                JobResult(
                    job_id="j", status=JobStatus.FAILED,
                    input_path="x", output_path="x_ocr.pdf",
                    error="boom",
                )
            )
        assert w._tray.notify_failure.called

    def test_success_skipped_when_window_active(
        self, qtbot, tmp_path: Path
    ) -> None:
        from src.core.models import JobResult
        from src.shared.types import JobStatus

        w = self._make_window(qtbot, tmp_path)
        with patch.object(w, "isActiveWindow", return_value=True):
            w._maybe_emit_completion_toast(
                JobResult(
                    job_id="j", status=JobStatus.COMPLETED,
                    input_path="x", output_path="x_ocr.pdf",
                )
            )
        assert not w._tray.notify_complete.called
