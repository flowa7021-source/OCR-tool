"""Persistence for user profiles and application settings.

All files are written atomically: content is first written to a
``*.tmp`` sibling and then :func:`os.replace`-d onto the final path.
JSON files are UTF-8 with ``ensure_ascii=False`` and 2-space indent so
they remain human-editable for Russian profile names.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.models import ProfileData
from src.shared.constants import (
    APP_VERSION,
    BUNDLED_PROFILES_DIR,
    CONFIG_DIR,
    DEFAULT_PARALLEL_WORKERS,
    PROFILES_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist ``data`` as JSON to ``path``.

    Args:
        path: Destination file path.
        data: JSON-serializable dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:  # pragma: no cover - not all FS support fsync
            pass
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and decode a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class ProfileStorage:
    """Manages builtin and user profiles on disk."""

    def __init__(self, profiles_dir: Path | None = None) -> None:
        """Create a new storage instance.

        Args:
            profiles_dir: Directory holding user profile JSON files.
                Defaults to :data:`PROFILES_DIR`.
        """
        self.profiles_dir: Path = profiles_dir if profiles_dir is not None else PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._seed_from_bundled_if_empty()

    # -- seeding -----------------------------------------------------------
    def _seed_from_bundled_if_empty(self) -> None:
        """Copy bundled profiles into the user dir on first launch."""
        try:
            if any(self.profiles_dir.glob("*.json")):
                return
            if not BUNDLED_PROFILES_DIR.exists():
                return
            for src in BUNDLED_PROFILES_DIR.glob("*.json"):
                dst = self.profiles_dir / src.name
                shutil.copy2(str(src), str(dst))
                logger.info("Seeded profile from bundle: %s", dst.name)
        except OSError as exc:
            logger.warning("Unable to seed profiles from bundle: %s", exc)

    # -- path helpers ------------------------------------------------------
    def _path_for(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.json"

    def _bundled_path_for(self, name: str) -> Path:
        return BUNDLED_PROFILES_DIR / f"{name}.json"

    # -- CRUD --------------------------------------------------------------
    def list_profiles(self) -> list[ProfileData]:
        """Return all profiles, builtins first then user-defined.

        Both groups are individually sorted by name. Builtin profiles come
        from :data:`BUNDLED_PROFILES_DIR`; user profiles from
        :attr:`profiles_dir`. If the same name exists in both, the user
        copy wins.
        """
        profiles: dict[str, ProfileData] = {}
        builtin_names: set[str] = set()

        if BUNDLED_PROFILES_DIR.exists():
            for file in sorted(BUNDLED_PROFILES_DIR.glob("*.json")):
                try:
                    data = _read_json(file)
                    data["builtin"] = True
                    profile = ProfileData.from_dict(data)
                    profile.builtin = True
                    profiles[profile.name] = profile
                    builtin_names.add(profile.name)
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    logger.error("Failed to load builtin profile %s: %s", file, exc)

        for file in sorted(self.profiles_dir.glob("*.json")):
            try:
                data = _read_json(file)
                profile = ProfileData.from_dict(data)
                # User files override builtins by name; preserve the
                # builtin flag only if the user hasn't explicitly changed it.
                if profile.name not in builtin_names:
                    profile.builtin = bool(data.get("builtin", False))
                profiles[profile.name] = profile
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                logger.error("Failed to load profile %s: %s", file, exc)

        builtins = sorted(
            (p for p in profiles.values() if p.builtin), key=lambda p: p.name
        )
        user = sorted(
            (p for p in profiles.values() if not p.builtin), key=lambda p: p.name
        )
        return builtins + user

    def load(self, name: str) -> ProfileData:
        """Load a profile by name.

        User directory is preferred; falls back to bundled profiles.

        Args:
            name: Profile name (without extension).

        Returns:
            The loaded :class:`ProfileData` instance.

        Raises:
            FileNotFoundError: No profile with this name exists.
        """
        user_path = self._path_for(name)
        if user_path.exists():
            data = _read_json(user_path)
            return ProfileData.from_dict(data)

        bundled = self._bundled_path_for(name)
        if bundled.exists():
            data = _read_json(bundled)
            data["builtin"] = True
            profile = ProfileData.from_dict(data)
            profile.builtin = True
            return profile

        raise FileNotFoundError(f"Профиль не найден: {name}")

    def save(self, profile: ProfileData) -> Path:
        """Persist ``profile`` to the user profiles directory atomically.

        Args:
            profile: Profile to save.

        Returns:
            Final path of the saved file.
        """
        path = self._path_for(profile.name)
        _atomic_write_json(path, profile.to_dict())
        logger.info("Saved profile: %s", path)
        return path

    def delete(self, name: str) -> None:
        """Delete a user-saved profile.

        Builtin profiles (``profile.builtin == True``) are never removed.

        Args:
            name: Profile name.

        Raises:
            FileNotFoundError: No user profile with this name.
            PermissionError: Profile is marked builtin.
        """
        path = self._path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Профиль не найден: {name}")

        try:
            data = _read_json(path)
            if data.get("builtin", False):
                raise PermissionError(f"Нельзя удалить встроенный профиль: {name}")
        except json.JSONDecodeError:
            logger.warning("Profile %s is malformed; proceeding with deletion", name)

        path.unlink()
        logger.info("Deleted profile: %s", name)

    def exists(self, name: str) -> bool:
        """Return True if a profile exists in the user or bundled directory."""
        return self._path_for(name).exists() or self._bundled_path_for(name).exists()

    def export_to(self, profile: ProfileData, path: Path) -> None:
        """Write ``profile`` as JSON to ``path`` (atomic)."""
        _atomic_write_json(path, profile.to_dict())
        logger.info("Exported profile %s to %s", profile.name, path)

    def import_from(self, path: Path) -> ProfileData:
        """Import a profile JSON from ``path``.

        The resulting profile is always marked ``builtin=False``.

        Args:
            path: File to read.

        Returns:
            Parsed :class:`ProfileData`.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ValueError: File is not valid profile JSON.
        """
        if not path.exists():
            raise FileNotFoundError(f"Файл профиля не найден: {path}")
        try:
            data = _read_json(path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Некорректный JSON в {path}: {exc}") from exc
        if "name" not in data or not isinstance(data.get("name"), str):
            raise ValueError(f"Файл не является профилем (нет 'name'): {path}")

        profile = ProfileData.from_dict(data)
        profile.builtin = False
        logger.info("Imported profile from %s", path)
        return profile


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------


@dataclass
class AppSettings:
    """Per-user application settings (non-profile state)."""

    last_profile: str = "default"
    last_input_dir: str = ""
    last_output_dir: str = ""
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    window_geometry: str = ""  # base64 Qt geometry blob
    window_state: str = ""  # base64 Qt window state blob
    theme: str = "dark"
    autosave_interval_pages: int = 10
    recent_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "last_profile": self.last_profile,
            "last_input_dir": self.last_input_dir,
            "last_output_dir": self.last_output_dir,
            "parallel_workers": self.parallel_workers,
            "window_geometry": self.window_geometry,
            "window_state": self.window_state,
            "theme": self.theme,
            "autosave_interval_pages": self.autosave_interval_pages,
            "recent_files": list(self.recent_files),
            "app_version": APP_VERSION,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppSettings:
        """Build an instance from a dictionary, tolerating missing keys."""
        defaults = cls()
        recent = data.get("recent_files", defaults.recent_files)
        if not isinstance(recent, list):
            recent = []
        recent = [str(item) for item in recent][:10]
        return cls(
            last_profile=str(data.get("last_profile", defaults.last_profile)),
            last_input_dir=str(data.get("last_input_dir", defaults.last_input_dir)),
            last_output_dir=str(data.get("last_output_dir", defaults.last_output_dir)),
            parallel_workers=int(
                data.get("parallel_workers", defaults.parallel_workers)
            ),
            window_geometry=str(data.get("window_geometry", defaults.window_geometry)),
            window_state=str(data.get("window_state", defaults.window_state)),
            theme=str(data.get("theme", defaults.theme)),
            autosave_interval_pages=int(
                data.get("autosave_interval_pages", defaults.autosave_interval_pages)
            ),
            recent_files=recent,
        )

    def add_recent_file(self, path: str) -> None:
        """Insert ``path`` at the front of the recent-files list (capped at 10)."""
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        del self.recent_files[10:]


class SettingsStorage:
    """Load and save :class:`AppSettings` to ``CONFIG_DIR/settings.json``."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """Create a new settings storage.

        Args:
            config_dir: Directory for the settings file. Defaults to
                :data:`CONFIG_DIR`.
        """
        self.config_dir: Path = config_dir if config_dir is not None else CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.path: Path = self.config_dir / "settings.json"

    def load(self) -> AppSettings:
        """Load settings from disk, returning defaults on any error."""
        if not self.path.exists():
            return AppSettings()
        try:
            data = _read_json(self.path)
            return AppSettings.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.error("Failed to load settings (%s); using defaults", exc)
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        """Atomically persist ``settings`` to disk."""
        _atomic_write_json(self.path, settings.to_dict())
        logger.debug("Settings saved to %s", self.path)
