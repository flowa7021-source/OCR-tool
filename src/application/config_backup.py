"""Export and import user configuration (settings + profiles) as a ZIP.

Used by "Настройки → Экспорт конфигурации" / "Импорт конфигурации" to
give users a portable backup that survives OS reinstall, machine swap,
or moving between Windows / Linux builds.

Bundle layout:

    OCRStudio-config-YYYYMMDD-HHMMSS.zip
    ├── manifest.json                  schema_version, app_version, timestamp
    ├── settings.json                  AppSettings
    └── profiles/
        ├── my_profile_a.json
        └── my_profile_b.json

Built-in profiles are **not** exported: they come from the installation
bundle, so re-importing them would shadow the read-only originals. Only
user-written / user-modified profiles are included.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from src.infrastructure.config_storage import ProfileStorage, SettingsStorage
from src.shared.constants import APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)


BACKUP_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_SETTINGS_NAME = "settings.json"
_PROFILES_DIR = "profiles"


def default_backup_name() -> str:
    """Suggest a timestamped filename for the backup archive."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"OCRStudio-config-{stamp}.zip"


def export_config(
    target: Path,
    settings_storage: SettingsStorage,
    profile_storage: ProfileStorage,
) -> Path:
    """Write a ZIP backup of settings + user profiles to ``target``.

    Overwrites any existing file at ``target``. Returns the final path.

    Args:
        target: Destination ``.zip`` path.
        settings_storage: Source of :class:`AppSettings`.
        profile_storage: Source of user profiles; only files in its
            ``profiles_dir`` are included (no bundled profiles).
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Serialise settings through AppSettings.to_dict so we ride the
    # versioned schema (and never leak disk-cache fields).
    settings = settings_storage.load()
    settings_payload = json.dumps(settings.to_dict(), ensure_ascii=False, indent=2)

    profiles_dir = profile_storage.profiles_dir
    included: list[str] = []

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            _MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        zf.writestr(_SETTINGS_NAME, settings_payload)
        if profiles_dir.exists():
            for pf in sorted(profiles_dir.glob("*.json")):
                # Preserve the original file verbatim so user edits
                # (timestamps, custom keys) round-trip exactly.
                zf.write(pf, arcname=f"{_PROFILES_DIR}/{pf.name}")
                included.append(pf.name)

    logger.info(
        "Exported config to %s (settings.json + %d profile(s): %s)",
        target,
        len(included),
        ", ".join(included) if included else "—",
    )
    return target


class ImportResult:
    """Small record returned from :func:`import_config` for UI display."""

    __slots__ = ("settings_restored", "profiles_added", "profiles_overwritten", "schema_version")

    def __init__(
        self,
        *,
        settings_restored: bool,
        profiles_added: list[str],
        profiles_overwritten: list[str],
        schema_version: int,
    ) -> None:
        self.settings_restored = settings_restored
        self.profiles_added = profiles_added
        self.profiles_overwritten = profiles_overwritten
        self.schema_version = schema_version


class BackupFormatError(RuntimeError):
    """Raised when the provided file doesn't look like our backup."""


def import_config(
    source: Path,
    settings_storage: SettingsStorage,
    profile_storage: ProfileStorage,
    *,
    overwrite_existing_profiles: bool = True,
) -> ImportResult:
    """Restore settings + profiles from a ZIP produced by :func:`export_config`.

    Raises:
        BackupFormatError: The file isn't a valid backup or schema is too new.

    Args:
        source: Path to the backup ``.zip``.
        settings_storage: Destination for ``settings.json``.
        profile_storage: Destination for profile files.
        overwrite_existing_profiles: If False, pre-existing user profiles
            with the same name are kept and the imported copy is skipped.
    """
    source = Path(source)
    if not source.exists():
        raise BackupFormatError(f"Файл не найден: {source}")
    if not zipfile.is_zipfile(source):
        raise BackupFormatError(f"Не ZIP-архив: {source}")

    with zipfile.ZipFile(source, "r") as zf:
        names = set(zf.namelist())
        if _MANIFEST_NAME not in names:
            raise BackupFormatError(
                "В архиве отсутствует manifest.json — это не бэкап OCR Studio."
            )

        try:
            manifest = json.loads(zf.read(_MANIFEST_NAME).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackupFormatError(f"Повреждённый manifest.json: {exc}") from exc

        schema = int(manifest.get("schema_version", 0) or 0)
        if schema > BACKUP_SCHEMA_VERSION:
            raise BackupFormatError(
                f"Версия бэкапа {schema} новее поддерживаемой "
                f"({BACKUP_SCHEMA_VERSION}). Обновите приложение."
            )

        # Settings are optional (users may hand-craft a profile-only bundle).
        settings_restored = False
        if _SETTINGS_NAME in names:
            try:
                raw = zf.read(_SETTINGS_NAME).decode("utf-8")
                data = json.loads(raw)
                # Round-trip through AppSettings to run schema migrations.
                from src.infrastructure.config_storage import AppSettings

                settings = AppSettings.from_dict(data)
                settings_storage.save(settings)
                settings_restored = True
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                raise BackupFormatError(
                    f"Повреждённый settings.json в архиве: {exc}"
                ) from exc

        added: list[str] = []
        overwritten: list[str] = []
        profile_dir = profile_storage.profiles_dir
        profile_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(names):
            if not name.startswith(f"{_PROFILES_DIR}/") or not name.endswith(".json"):
                continue
            basename = Path(name).name
            # Reject path traversal attempts (e.g. "profiles/../../etc/pwd.json").
            if ".." in Path(name).parts or Path(basename).name != basename:
                logger.warning("Пропускаю подозрительный путь в архиве: %s", name)
                continue
            dest = profile_dir / basename
            exists = dest.exists()
            if exists and not overwrite_existing_profiles:
                continue
            # Validate JSON BEFORE writing — refuse to install corrupt profiles.
            try:
                payload = zf.read(name).decode("utf-8")
                json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.error("Пропускаю повреждённый профиль %s: %s", basename, exc)
                continue
            dest.write_text(payload, encoding="utf-8")
            (overwritten if exists else added).append(basename)

    logger.info(
        "Imported config from %s: settings=%s, added=%d, overwritten=%d",
        source,
        settings_restored,
        len(added),
        len(overwritten),
    )
    return ImportResult(
        settings_restored=settings_restored,
        profiles_added=added,
        profiles_overwritten=overwritten,
        schema_version=schema,
    )


def _rm_tree_best_effort(path: Path) -> None:
    """Utility used in rollback paths. Swallows errors."""
    import contextlib

    with contextlib.suppress(Exception):
        shutil.rmtree(path, ignore_errors=True)
