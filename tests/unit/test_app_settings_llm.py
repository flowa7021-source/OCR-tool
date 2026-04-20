"""Tests for the ``anthropic_api_key`` field on :class:`AppSettings`.

Schema v2 adds the key to ``settings.json``. The tests pin:

* Default empty key → offline-first stance.
* Round-trip preserves the key byte-exactly (including long
  ``sk-ant-…`` strings and any Unicode that leaks in through
  clipboard paste).
* v1 settings.json files load cleanly with default empty key
  (the whole migration premise).
* The key is stored at the TOP level of settings.json, NOT in a
  nested profile — guarantees ``ProfileManager.export_profile``
  can't accidentally serialise it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.config_storage import (
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    SettingsStorage,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_key_is_empty() -> None:
    """No key out of the box — enforces offline-first."""
    assert AppSettings().anthropic_api_key == ""


def test_schema_version_is_2() -> None:
    """PR #8 targets exactly v2 — guard against accidental bumps."""
    assert SETTINGS_SCHEMA_VERSION == 2


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_to_dict_includes_key() -> None:
    """Serialised form exposes the key under the expected JSON path."""
    settings = AppSettings(anthropic_api_key="sk-ant-api03-round-trip")
    data = settings.to_dict()
    assert data["anthropic_api_key"] == "sk-ant-api03-round-trip"
    assert data["schema_version"] == SETTINGS_SCHEMA_VERSION


def test_round_trip_preserves_key() -> None:
    original = AppSettings(anthropic_api_key="sk-ant-api03-keepEXACT-01234567")
    reloaded = AppSettings.from_dict(original.to_dict())
    assert reloaded.anthropic_api_key == original.anthropic_api_key


@pytest.mark.parametrize(
    "key",
    [
        "",
        "sk-ant-api03-ascii-only",
        "sk-ant-api03-with-unicode-тест",   # clipboard pastes can smuggle Cyrillic
        "sk-ant-api03-" + "x" * 256,         # long keys shouldn't truncate
    ],
)
def test_round_trip_parametrised(key: str) -> None:
    data = AppSettings(anthropic_api_key=key).to_dict()
    assert AppSettings.from_dict(data).anthropic_api_key == key


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------


def test_v1_settings_load_with_empty_key() -> None:
    """A pre-v2 settings.json has no ``anthropic_api_key`` field."""
    v1_data = {
        "schema_version": 1,
        "last_profile": "default",
        "parallel_workers": 2,
        "theme": "dark",
        # No anthropic_api_key — this is what v1 looks like.
    }
    loaded = AppSettings.from_dict(v1_data)
    assert loaded.schema_version == SETTINGS_SCHEMA_VERSION
    assert loaded.anthropic_api_key == ""
    # Non-migrated fields still come through.
    assert loaded.last_profile == "default"
    assert loaded.parallel_workers == 2


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


def test_storage_persists_key_to_disk(tmp_path: Path) -> None:
    storage = SettingsStorage(config_dir=tmp_path)
    settings = storage.load()
    settings.anthropic_api_key = "sk-ant-api03-disk-persist"
    storage.save(settings)

    raw = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert raw["anthropic_api_key"] == "sk-ant-api03-disk-persist"

    fresh = SettingsStorage(config_dir=tmp_path)
    assert fresh.load().anthropic_api_key == "sk-ant-api03-disk-persist"


def test_key_is_not_stored_in_profile_dict() -> None:
    """Sanity: ``ProfileData.to_dict`` must not contain the LLM key."""
    from src.core.models import ProfileData

    profile = ProfileData(name="test")
    data = profile.to_dict()
    # Walk the whole nested structure and fail on any "anthropic" key.
    def _walk(obj):  # noqa: ANN001
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "anthropic" not in str(k).lower(), (
                    f"profile dict leaks anthropic key at {k!r}: {v!r}"
                )
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
