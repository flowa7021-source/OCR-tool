"""Tests for :mod:`src.infrastructure.config_storage`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.models import ProfileData
from src.infrastructure.config_storage import (
    AppSettings,
    ProfileStorage,
    SettingsStorage,
)


class TestProfileStorage:
    def test_empty_dir_lists_only_bundled(self, tmp_profiles_dir: Path) -> None:
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        # Bundled profiles may exist; user list must not crash and must be a list.
        assert isinstance(storage.list_profiles(), list)

    def test_save_load_roundtrip(
        self, tmp_profiles_dir: Path, sample_profile: ProfileData
    ) -> None:
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        storage.save(sample_profile)
        loaded = storage.load(sample_profile.name)
        assert loaded.to_dict() == sample_profile.to_dict()

    def test_save_atomic_no_tmp_leftover(
        self, tmp_profiles_dir: Path, sample_profile: ProfileData
    ) -> None:
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        storage.save(sample_profile)
        leftovers = list(tmp_profiles_dir.glob("*.tmp"))
        assert leftovers == []
        assert (tmp_profiles_dir / f"{sample_profile.name}.json").exists()

    def test_load_missing_raises(self, tmp_profiles_dir: Path) -> None:
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        with pytest.raises(FileNotFoundError):
            storage.load("does-not-exist-xyz")

    def test_list_profiles_includes_saved(
        self, tmp_profiles_dir: Path, sample_profile: ProfileData
    ) -> None:
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        storage.save(sample_profile)
        names = [p.name for p in storage.list_profiles()]
        assert sample_profile.name in names

    def test_malformed_json_surfaces_via_rejected_profiles(
        self, tmp_profiles_dir: Path, sample_profile: ProfileData
    ) -> None:
        """Malformed JSON in profiles dir doesn't break listing — it
        surfaces via :attr:`ProfileStorage.rejected_profiles` so a
        future UI can show 'N profiles skipped'.

        Previously the bad file was silently logged at error-level,
        which in a ``--windowed`` PyInstaller build goes to a log the
        user never sees. They notice only that "my profile disappeared".
        """
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)

        # Drop a valid profile.
        storage.save(sample_profile)

        # And a deliberately malformed one — not valid JSON.
        broken = tmp_profiles_dir / "broken.json"
        broken.write_bytes(b"{ this is not valid json")

        # And a JSON-but-not-a-profile file: missing required structure.
        stray = tmp_profiles_dir / "random_data.json"
        stray.write_text('{"completely": "unrelated"}', encoding="utf-8")

        profiles = storage.list_profiles()
        names = [p.name for p in profiles]
        # Good profile still listed.
        assert sample_profile.name in names

        # Rejected surface populated.
        rejected_paths = [p.name for p, _ in storage.rejected_profiles]
        assert "broken.json" in rejected_paths, (
            f"broken JSON should be in rejected_profiles, got {rejected_paths}"
        )
        # Each rejection has a non-empty reason string.
        for _, reason in storage.rejected_profiles:
            assert reason, "rejected entry has empty reason"

    def test_rejected_profiles_empty_by_default(
        self, tmp_profiles_dir: Path
    ) -> None:
        """Fresh storage with no files has empty rejected list — callers
        can read it before the first list_profiles() without AttributeError."""
        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        assert storage.rejected_profiles == ()

    def test_atomic_write_retries_on_transient_permission_error(
        self, tmp_profiles_dir: Path, sample_profile: ProfileData, monkeypatch
    ) -> None:
        """A transient PermissionError from os.replace — the AV / indexer
        / OneDrive-sync scenario — must be retried, not surfaced as
        'failed to save profile' on the first hiccup.
        """
        import os

        from src.infrastructure import config_storage

        # First two calls raise PermissionError, third succeeds.
        real_replace = os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError(
                    "simulated AV briefly holding the target file"
                )
            return real_replace(src, dst)

        monkeypatch.setattr(config_storage.os, "replace", flaky_replace)

        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        # Must not raise.
        storage.save(sample_profile)

        assert calls["n"] == 3, (
            f"expected 2 retries then success; got {calls['n']} replace calls"
        )
        # Saved file landed correctly.
        loaded = storage.load(sample_profile.name)
        assert loaded.to_dict() == sample_profile.to_dict()

    def test_seed_rolls_back_partial_copy_on_failure(
        self, tmp_profiles_dir: Path, monkeypatch
    ) -> None:
        """Disk fills after the first bundled profile is copied — the
        dir must end up empty again, not half-seeded.

        Without rollback, next launch sees ``*.json`` already present
        and skips seeding, leaving the user permanently missing 5-ish
        builtin profiles with no recovery path short of deleting the
        whole profiles dir by hand.
        """
        import shutil as _shutil

        from src.infrastructure import config_storage

        # Need at least 2 bundled profiles so we can simulate "succeed
        # on first, fail on second".
        bundled = list(config_storage.BUNDLED_PROFILES_DIR.glob("*.json"))
        assert len(bundled) >= 2, (
            "test precondition: at least 2 bundled profiles required; "
            f"found {bundled}"
        )

        real_copy2 = _shutil.copy2
        copies: list[str] = []

        def flaky_copy2(src, dst, *a, **kw):
            copies.append(str(dst))
            if len(copies) == 2:
                raise OSError(28, "No space left on device")  # ENOSPC
            return real_copy2(src, dst, *a, **kw)

        monkeypatch.setattr(config_storage.shutil, "copy2", flaky_copy2)

        # Constructing the storage seeds the dir.
        config_storage.ProfileStorage(profiles_dir=tmp_profiles_dir)

        # Must have attempted at least 2 copies, and rolled back to
        # empty state.
        assert len(copies) >= 2
        remaining = list(tmp_profiles_dir.glob("*.json"))
        assert remaining == [], (
            f"partial seed not rolled back; leftover files: {remaining}"
        )

    def test_atomic_write_gives_up_after_persistent_failure(
        self, tmp_profiles_dir: Path, sample_profile: ProfileData, monkeypatch
    ) -> None:
        """If os.replace permanently fails, the PermissionError eventually
        propagates — we don't want a silent data-loss where the user
        thinks their save succeeded but it didn't."""
        import pytest

        from src.infrastructure import config_storage

        def always_fail(src, dst):
            raise PermissionError("permanently locked")

        monkeypatch.setattr(config_storage.os, "replace", always_fail)

        storage = ProfileStorage(profiles_dir=tmp_profiles_dir)
        with pytest.raises(PermissionError):
            storage.save(sample_profile)


class TestSettingsStorage:
    def test_load_returns_defaults_on_empty(self, tmp_path: Path) -> None:
        storage = SettingsStorage(config_dir=tmp_path / "cfg")
        settings = storage.load()
        defaults = AppSettings()
        assert settings.theme == defaults.theme
        assert settings.parallel_workers == defaults.parallel_workers
        assert settings.recent_files == []

    def test_save_load_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        storage = SettingsStorage(config_dir=tmp_path / "cfg")
        original = AppSettings(
            last_profile="custom",
            last_input_dir="/tmp/in",
            last_output_dir="/tmp/out",
            parallel_workers=3,
            theme="light",
            autosave_interval_pages=25,
            recent_files=["/a.pdf", "/b.pdf"],
        )
        storage.save(original)
        loaded = storage.load()
        assert loaded.last_profile == "custom"
        assert loaded.last_input_dir == "/tmp/in"
        assert loaded.last_output_dir == "/tmp/out"
        assert loaded.parallel_workers == 3
        assert loaded.theme == "light"
        assert loaded.autosave_interval_pages == 25
        assert loaded.recent_files == ["/a.pdf", "/b.pdf"]

    def test_save_atomic_no_tmp_leftover(self, tmp_path: Path) -> None:
        cfg_dir = tmp_path / "cfg"
        storage = SettingsStorage(config_dir=cfg_dir)
        storage.save(AppSettings())
        assert list(cfg_dir.glob("*.tmp")) == []
        assert (cfg_dir / "settings.json").exists()
