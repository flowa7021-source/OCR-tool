"""Tests for the latest fix-and-harden pass.

Covers:
    * Regex ReDoS watchdog (text_postprocessor._substitute_with_timeout)
    * Profile schema migration (_migrate_profile_dict)
    * AppSettings schema_version round-trip
    * Diagnostics zip build
    * Update-checker version parsing + comparison
    * Single-instance guard socket-name isolation
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# ReDoS watchdog
# ---------------------------------------------------------------------------


class TestRegexWatchdog:
    def test_fast_regex_completes(self) -> None:
        from src.core.text_postprocessor import _substitute_with_timeout

        pat = re.compile(r"\bfoo\b")
        out = _substitute_with_timeout(pat, "BAR", "hello foo world", timeout=1.0)
        assert out == "hello BAR world"

    def test_slow_regex_times_out(self) -> None:
        """The watchdog kicks in regardless of WHY the regex is slow.

        Python's re module optimises many textbook ReDoS patterns, so
        an actual catastrophic-backtracking payload isn't reliable on
        all hardware. Instead we install a fake pattern whose ``sub``
        just sleeps: the watchdog semantics (kill after N seconds) are
        the contract we care about.
        """
        import time

        from src.core.text_postprocessor import (
            RegexTimeoutError,
            _substitute_with_timeout,
        )

        class _SleepingPattern:
            pattern = "<sleep>"

            @staticmethod
            def sub(_replacement: str, _text: str) -> str:
                time.sleep(2.0)  # longer than any reasonable timeout
                return "unreachable"

        with pytest.raises(RegexTimeoutError):
            _substitute_with_timeout(_SleepingPattern(), "", "any text", timeout=0.2)

    def test_custom_rule_redos_is_logged_not_raised(self, caplog, monkeypatch) -> None:
        """A bad custom rule must not kill the pipeline.

        We patch ``_substitute_with_timeout`` to always raise
        :class:`RegexTimeoutError`, modelling the "every custom rule
        hangs" worst case, and assert that the overall process still
        returns the untouched input text.
        """
        import logging

        from src.core import text_postprocessor
        from src.core.models import PostprocessConfig, RegexRule
        from src.core.text_postprocessor import (
            RegexTimeoutError,
            TextPostprocessor,
        )

        def _fake_sub(_pat, _replacement, _text, timeout=1.0):  # noqa: ARG001
            raise RegexTimeoutError("simulated ReDoS")

        monkeypatch.setattr(text_postprocessor, "_substitute_with_timeout", _fake_sub)

        caplog.set_level(logging.WARNING, logger="src.core.text_postprocessor")
        cfg = PostprocessConfig(
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
            remove_artifacts=False,
            custom_rules=[
                RegexRule(
                    pattern=r"^(a+)+b$",
                    replacement="",
                    is_regex=True,
                    enabled=True,
                )
            ],
        )
        text = "hello world"
        result = TextPostprocessor().process(text, cfg)
        assert result == text
        assert any("Правило #0" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Profile schema migration
# ---------------------------------------------------------------------------


class TestProfileSchemaMigration:
    def test_v0_profile_gets_engine_field(self) -> None:
        from src.core.models import ProfileData

        old = {
            # No schema_version → treated as v0
            "name": "legacy",
            "description": "old profile",
            "ocr": {
                # No engine field either
                "languages": ["rus"],
                "primary_language": "rus",
            },
        }
        profile = ProfileData.from_dict(old)
        from src.shared.types import OCREngineKind

        assert profile.ocr.engine is OCREngineKind.TESSERACT
        assert profile.schema_version == 1

    def test_current_version_passes_through(self) -> None:
        from src.core.models import ProfileData

        original = ProfileData(name="x")
        d = original.to_dict()
        assert d["schema_version"] == 1
        restored = ProfileData.from_dict(d)
        assert restored.schema_version == 1

    def test_future_version_logs_and_loads_best_effort(self, caplog) -> None:
        import logging

        from src.core.models import ProfileData

        caplog.set_level(logging.WARNING)
        data = {"name": "future", "schema_version": 999, "ocr": {}}
        profile = ProfileData.from_dict(data)
        # Best-effort: the loader returns a usable profile
        assert profile.name == "future"
        assert any(
            "schema_version" in record.message.lower() for record in caplog.records
        )


class TestSettingsSchemaVersion:
    def test_missing_version_is_filled(self, tmp_path: Path) -> None:
        import json

        from src.infrastructure.config_storage import (
            SETTINGS_SCHEMA_VERSION,
            SettingsStorage,
        )

        (tmp_path / "settings.json").write_text(
            json.dumps({"parallel_workers": 2}),  # no schema_version
            encoding="utf-8",
        )
        storage = SettingsStorage(config_dir=tmp_path)
        settings = storage.load()
        assert settings.schema_version == SETTINGS_SCHEMA_VERSION
        assert settings.parallel_workers == 2

    def test_future_version_logs_warning(self, tmp_path: Path, caplog) -> None:
        import json
        import logging

        from src.infrastructure.config_storage import SettingsStorage

        caplog.set_level(logging.WARNING)
        (tmp_path / "settings.json").write_text(
            json.dumps({"schema_version": 999, "parallel_workers": 2}),
            encoding="utf-8",
        )
        storage = SettingsStorage(config_dir=tmp_path)
        storage.invalidate_cache()  # avoid the initial empty-defaults cache
        storage.load()
        assert any(
            "schema_version" in r.message.lower() for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Diagnostics zip
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_zip_contents(self, tmp_path: Path) -> None:
        from src.application.diagnostics import build_diagnostics_zip

        # Fake settings + log file
        settings = tmp_path / "settings.json"
        settings.write_text('{"last_profile": "default"}', encoding="utf-8")
        log = tmp_path / "ocr-studio.log"
        log.write_text("INFO: hello\n", encoding="utf-8")
        # Rotated backup should also be picked up
        (tmp_path / "ocr-studio.log.1").write_text("rotated\n", encoding="utf-8")

        target = tmp_path / "diag.zip"
        build_diagnostics_zip(target, settings_path=settings, log_file=log)

        assert target.exists()
        with zipfile.ZipFile(target) as zf:
            names = zf.namelist()
            assert "environment.json" in names
            assert "settings.json" in names
            assert "README.txt" in names
            assert any(n.startswith("logs/") and n.endswith(".log") for n in names)
            assert "logs/ocr-studio.log.1" in names
            # README documents what's NOT included
            readme = zf.read("README.txt").decode("utf-8")
            assert "User profiles" in readme

    def test_missing_settings_doesnt_crash(self, tmp_path: Path) -> None:
        from src.application.diagnostics import build_diagnostics_zip

        target = tmp_path / "diag.zip"
        build_diagnostics_zip(
            target,
            settings_path=tmp_path / "does-not-exist.json",
            log_file=tmp_path / "no-log-either.log",
        )
        with zipfile.ZipFile(target) as zf:
            # Only environment.json + README — no settings entry
            names = zf.namelist()
            assert "environment.json" in names
            assert "settings.json" not in names


# ---------------------------------------------------------------------------
# Update checker
# ---------------------------------------------------------------------------


class TestUpdateChecker:
    def test_parse_version_strips_v(self) -> None:
        from src.application.update_checker import _parse_version

        assert _parse_version("v1.2.3") == (1, 2, 3)
        assert _parse_version("1.2.3") == (1, 2, 3)
        assert _parse_version("V2.0") == (2, 0)

    def test_parse_version_strips_prerelease(self) -> None:
        from src.application.update_checker import _parse_version

        assert _parse_version("1.2.3-rc1") == (1, 2, 3)
        assert _parse_version("1.0.0-dev.abc123") == (1, 0, 0)

    def test_is_newer(self) -> None:
        from src.application.update_checker import is_newer

        assert is_newer("1.0.0", "1.0.1") is True
        assert is_newer("1.0.0", "1.1.0") is True
        assert is_newer("1.0.0", "2.0.0") is True
        assert is_newer("1.0.1", "1.0.0") is False
        assert is_newer("1.0.0", "1.0.0") is False
        assert is_newer("1.0.0", "v1.0.1") is True

    def test_check_once_handles_http_failure(self, monkeypatch) -> None:
        from src.application import update_checker

        def _fail(*_args, **_kwargs):
            raise OSError("network down")

        monkeypatch.setattr(update_checker.urllib.request, "urlopen", _fail)
        result = update_checker.check_once(repo="doesnt/matter", timeout=0.1)
        assert result is None


# ---------------------------------------------------------------------------
# Single-instance guard (socket-name only — no real server in tests)
# ---------------------------------------------------------------------------


class TestSingleInstance:
    def test_socket_name_is_user_scoped(self) -> None:
        from src.ui.single_instance import _socket_name

        name = _socket_name()
        assert name.startswith("ocr-studio-")
        assert len(name) > len("ocr-studio-")  # user suffix appended
