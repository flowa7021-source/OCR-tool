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
