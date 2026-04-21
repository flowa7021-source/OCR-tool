"""Tests for the ``extract`` section added in profile schema v11.

The section wires a post-OCR structured-field parser (ТН / УПД in the
first iteration) onto profiles. Two contracts must hold:

1. **Backwards compatibility** — any profile JSON authored at schema
   v1..v10 loads cleanly and produces ``extract.enabled=False``, so the
   OCR pipeline behaves exactly as it did before v11.
2. **Round-trip stability** — a v11 profile with a non-default
   ``extract`` section (the ``tn_upd`` builtin) serialises and
   deserialises without drift, so exported profiles reimport byte-
   equal.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.application.profile_manager import ProfileManager
from src.core.models import (
    PROFILE_SCHEMA_VERSION,
    ExtractConfig,
    LlmFallbackConfig,
    ProfileData,
)
from src.infrastructure.config_storage import ProfileStorage

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


def test_current_schema_version_is_11() -> None:
    """Gate against accidental bumps: PR #2 targets exactly v11."""
    assert PROFILE_SCHEMA_VERSION == 12


def test_extract_defaults_are_offline_safe() -> None:
    """Defaults MUST keep the offline-first guarantee.

    A profile created without arguments must have ``extract.enabled=False``
    and ``extract.llm_fallback.enabled=False``. Otherwise a fresh install
    on an air-gapped Windows box would silently try to reach the
    Anthropic API.
    """
    cfg = ExtractConfig()
    assert cfg.enabled is False
    assert cfg.llm_fallback.enabled is False

    llm = LlmFallbackConfig()
    assert llm.enabled is False
    assert llm.model == "claude-opus-4-7"
    assert llm.max_text_chars == 20_000


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _v9_profile_dict() -> dict:
    """Minimal v9 profile — the shape shipped in profiles/default.json."""
    return {
        "name": "legacy_v9",
        "description": "",
        "schema_version": 9,
        "preprocess": {},
        "ocr": {},
        "postprocess": {},
    }


def test_migration_v9_adds_extract_disabled() -> None:
    """A v9 profile migrates to current schema with extract.enabled=False."""
    loaded = ProfileData.from_dict(_v9_profile_dict())
    assert loaded.schema_version == PROFILE_SCHEMA_VERSION
    assert loaded.extract.enabled is False
    # Defaults are picked up even if JSON had no `extract` key.
    assert loaded.extract.kind == "tn_upd"
    assert loaded.extract.llm_fallback.enabled is False


def test_migration_v10_adds_extract_disabled() -> None:
    """v10 → v11 sets the extract section without touching OCR fields."""
    d = _v9_profile_dict()
    d["schema_version"] = 10
    d["ocr"] = {"soft_rescue_dropped_words": True}  # v10-authored
    loaded = ProfileData.from_dict(d)
    assert loaded.schema_version == 12
    assert loaded.ocr.soft_rescue_dropped_words is True
    assert loaded.extract.enabled is False


def test_migration_preserves_existing_extract_section() -> None:
    """If a JSON already has ``extract``, migration MUST NOT overwrite it.

    This protects future schema bumps: once a user's profile has an
    ``extract.enabled=True`` setting, re-loading must not reset it to
    False.
    """
    d = _v9_profile_dict()
    d["schema_version"] = 12
    d["extract"] = {
        "enabled": True,
        "kind": "tn_upd",
        "llm_fallback": {"enabled": True, "min_confidence": 0.55},
    }
    loaded = ProfileData.from_dict(d)
    assert loaded.extract.enabled is True
    assert loaded.extract.llm_fallback.enabled is True
    assert loaded.extract.llm_fallback.min_confidence == 0.55


# ---------------------------------------------------------------------------
# Round-trip for the merged universal_accurate builtin
# (с декабря 2026 — единственный builtin, парсер ТН/УПД встроен через
# extract.kind="tn_upd"; см. ProfileManager._build_universal_accurate)
# ---------------------------------------------------------------------------


def test_universal_accurate_builder_enables_extract(tmp_path: Path) -> None:
    """``_build_universal_accurate`` produces extract.enabled=True
    с kind="tn_upd" — парсер ТН/УПД встроен в единственный builtin."""
    mgr = ProfileManager(ProfileStorage(profiles_dir=tmp_path))
    profile = mgr._build_universal_accurate()
    assert profile.name == "universal_accurate"
    assert profile.extract.enabled is True
    assert profile.extract.kind == "tn_upd"
    assert profile.extract.multi_document is True
    # LLM fallback остаётся opt-in: пользователь явно включает в
    # настройках (ANTHROPIC_API_KEY в settings.json).
    assert profile.extract.llm_fallback.enabled is False


def test_universal_accurate_profile_round_trips(tmp_path: Path) -> None:
    """Serialising and reloading the builder output is byte-stable."""
    mgr = ProfileManager(ProfileStorage(profiles_dir=tmp_path))
    original = mgr._build_universal_accurate()
    rebuilt = ProfileData.from_dict(original.to_dict())
    assert rebuilt.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# Bundled profile on disk
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_PROFILES_DIR = _REPO_ROOT / "profiles"


def test_bundled_universal_accurate_profile_exists_and_loads() -> None:
    """``profiles/universal_accurate.json`` ships и загружается под текущей
    схемой. С декабря 2026 — единственный bundled профиль (default /
    quick_reliable / low_quality_scan / contracts_ru / english_text /
    tn_upd удалены и слиты в universal_accurate)."""
    path = _BUNDLED_PROFILES_DIR / "universal_accurate.json"
    assert path.exists(), f"Missing bundled profile: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == PROFILE_SCHEMA_VERSION
    profile = ProfileData.from_dict(data)
    assert profile.name == "universal_accurate"
    assert profile.extract.enabled is True
    assert profile.extract.kind == "tn_upd"


def test_all_bundled_v9_profiles_migrate_cleanly() -> None:
    """Every pre-v11 profile in the bundle loads without errors.

    Catches the class of migration bugs where a field added later in
    the chain (e.g. v10's ``soft_rescue_dropped_words``) is assumed
    present by a v11 migration step — the migration must layer, not
    skip forward.
    """
    for path in sorted(_BUNDLED_PROFILES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = ProfileData.from_dict(data)
        assert profile.schema_version == PROFILE_SCHEMA_VERSION, path.name
        # extract section must materialise even for the v9 bundled profiles.
        assert hasattr(profile, "extract"), path.name
