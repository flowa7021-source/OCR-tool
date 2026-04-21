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
    """``initialize_builtins`` сидирует все builtin'ы из BUILTIN_NAMES.

    С декабря 2026 builtin один — ``universal_accurate``. Раньше было
    7 (default / quick_reliable / low_quality_scan / contracts_ru /
    english_text / tn_upd / universal_accurate); пользователь попросил
    собрать best-of-all в единый ``universal_accurate``, остальные
    удалили (см. ProfileManager docstring).
    """
    manager.initialize_builtins()
    names = {p.name for p in manager.list_profiles()}
    for expected in BUILTIN_NAMES:
        assert expected in names
    # Апрель 2026: 2 builtin'а — universal_accurate (aggressive
    # preprocessing) + universal_clean (minimal preprocessing для
    # чистых сканов / digital-экспортов).
    assert len(BUILTIN_NAMES) == 2
    assert BUILTIN_NAMES == ("universal_accurate", "universal_clean")


def test_builtins_marked_builtin(manager: ProfileManager) -> None:
    manager.initialize_builtins()
    for name in BUILTIN_NAMES:
        profile = manager.load(name)
        assert profile.builtin is True


def test_cannot_delete_builtin(manager: ProfileManager) -> None:
    """Builtin профиль нельзя удалить — storage бросает PermissionError."""
    manager.initialize_builtins()
    with pytest.raises(PermissionError) as exc_info:
        manager.delete("universal_accurate")
    msg = str(exc_info.value).lower()
    assert "builtin" in msg or "встроен" in msg


def test_duplicate_creates_non_builtin(manager: ProfileManager) -> None:
    manager.initialize_builtins()
    copy = manager.duplicate("universal_accurate", "my_copy")
    assert copy.name == "my_copy"
    assert copy.builtin is False

    reloaded = manager.load("my_copy")
    assert reloaded.name == "my_copy"
    assert reloaded.builtin is False


def test_initialize_builtins_re_seeds_builtin(
    manager: ProfileManager,
) -> None:
    """Builtin перерезаписывается при повторном вызове — по контракту
    с декабря 2026 (см. ProfileManager.initialize_builtins docstring).

    Это важно: после апгрейда приложения builtin-профиль обновляется
    сам, без необходимости пользователю чистить ~/.ocrstudio/profiles/
    вручную. Кастомизация ведётся через ``duplicate`` (профиль с
    builtin=False — не трогается).
    """
    manager.initialize_builtins()
    profile = manager.load("universal_accurate")
    profile.description = "EDITED IN BUILTIN — будет затёрт"
    profile.builtin = True
    manager.save(profile)
    manager.initialize_builtins()
    after = manager.load("universal_accurate")
    # builtin ре-сидирован — кастомное описание затёрто каноническим.
    assert after.description != "EDITED IN BUILTIN — будет затёрт"


def test_user_copy_survives_re_seed(manager: ProfileManager) -> None:
    """Пользовательская копия (builtin=False) НЕ перезаписывается
    при повторном initialize_builtins. Это паритет с правилом «builtin
    обновляется, custom — нет»."""
    manager.initialize_builtins()
    copy = manager.duplicate("universal_accurate", "my_copy")
    copy.description = "MY CUSTOMISATION"
    manager.save(copy)
    manager.initialize_builtins()
    reloaded = manager.load("my_copy")
    assert reloaded.builtin is False
    assert reloaded.description == "MY CUSTOMISATION"
