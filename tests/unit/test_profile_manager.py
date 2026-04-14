"""Tests for :mod:`src.application.profile_manager`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.profile_manager import BUILTIN_NAMES, ProfileManager
from src.infrastructure.config_storage import ProfileStorage


@pytest.fixture
def manager(tmp_profiles_dir: Path) -> ProfileManager:
    return ProfileManager(ProfileStorage(profiles_dir=tmp_profiles_dir))


def test_initialize_builtins_creates_all(manager: ProfileManager) -> None:
    manager.initialize_builtins()
    names = {p.name for p in manager.list_profiles()}
    for expected in BUILTIN_NAMES:
        assert expected in names
    assert len(BUILTIN_NAMES) == 5


def test_builtins_marked_builtin(manager: ProfileManager) -> None:
    manager.initialize_builtins()
    for name in BUILTIN_NAMES:
        profile = manager.load(name)
        assert profile.builtin is True


def test_cannot_delete_builtin(manager: ProfileManager) -> None:
    manager.initialize_builtins()
    with pytest.raises(PermissionError) as exc_info:
        manager.delete("default")
    msg = str(exc_info.value).lower()
    assert "builtin" in msg or "встроен" in msg


def test_duplicate_creates_non_builtin(manager: ProfileManager) -> None:
    manager.initialize_builtins()
    copy = manager.duplicate("default", "my_copy")
    assert copy.name == "my_copy"
    assert copy.builtin is False

    reloaded = manager.load("my_copy")
    assert reloaded.name == "my_copy"
    assert reloaded.builtin is False


def test_initialize_builtins_is_idempotent(manager: ProfileManager) -> None:
    """Second call must not overwrite user edits to builtin profiles."""
    manager.initialize_builtins()

    # User edits a builtin.
    profile = manager.load("default")
    profile.description = "USER EDITED"
    manager.save(profile)

    # Re-run seeding.
    manager.initialize_builtins()

    after = manager.load("default")
    assert after.description == "USER EDITED"
