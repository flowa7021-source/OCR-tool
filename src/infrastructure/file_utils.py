"""Filesystem helpers.

All functions are safe for Windows paths containing Cyrillic characters and
spaces: paths are manipulated exclusively through :class:`pathlib.Path`
and never via string concatenation.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from pathlib import Path

from src.shared.constants import TEMP_DIR

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if it does not exist.

    Args:
        path: Directory to create.

    Returns:
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_unique_path(path: Path) -> Path:
    """Return a non-existing path derived from ``path``.

    If ``path`` already exists, suffixes ``" (1)"``, ``" (2)"``, ... are
    appended to the stem until a free name is found. Capped at 9999
    attempts.

    Args:
        path: Desired output path.

    Returns:
        A path that currently does not exist.

    Raises:
        FileExistsError: If 9999 candidate paths already exist.
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for counter in range(1, 10_000):
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"Не удалось подобрать уникальное имя для {path} за 9999 попыток"
    )


def suggest_output_path(
    input_path: Path,
    suffix: str = "_ocr",
    ext: str | None = None,
) -> Path:
    """Suggest an output path based on an input file.

    Args:
        input_path: Source file path.
        suffix: String appended to the stem (e.g. ``"_ocr"``).
        ext: Optional new extension including the dot (e.g. ``".txt"``). If
            ``None`` the original extension is preserved.

    Returns:
        New path in the same directory.
    """
    new_ext = ext if ext is not None else input_path.suffix
    if new_ext and not new_ext.startswith("."):
        new_ext = f".{new_ext}"
    return input_path.with_name(f"{input_path.stem}{suffix}{new_ext}")


def cleanup_temp_dir(temp_dir: Path, older_than_hours: int = 24) -> int:
    """Remove files in ``temp_dir`` older than ``older_than_hours``.

    Iterates recursively; directories that become empty are also removed.
    Individual errors are logged at ``warning`` level but do not abort the
    operation.

    Args:
        temp_dir: Directory to clean up.
        older_than_hours: Age threshold in hours.

    Returns:
        Number of files successfully removed.
    """
    if not temp_dir.exists():
        return 0

    threshold = time.time() - (older_than_hours * 3600)
    removed = 0

    for item in sorted(temp_dir.rglob("*"), reverse=True):
        try:
            if item.is_file():
                if item.stat().st_mtime < threshold:
                    item.unlink()
                    removed += 1
            elif item.is_dir():
                # Remove empty directories opportunistically.
                with contextlib.suppress(OSError):
                    item.rmdir()
        except OSError as exc:
            logger.warning("Failed to remove %s: %s", item, exc)

    return removed


def create_temp_workdir(prefix: str = "ocr_") -> Path:
    """Create a unique subdirectory under :data:`TEMP_DIR`.

    Args:
        prefix: Prefix for the directory name.

    Returns:
        Path to the newly created directory.
    """
    ensure_dir(TEMP_DIR)
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(TEMP_DIR)))
    logger.debug("Created temporary workdir: %s", path)
    return path


def bytes_human(size: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size: Size in bytes.

    Returns:
        String such as ``"1.23 MB"``.
    """
    value = float(size)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024.0 or unit == "PB":
            if unit == "B":
                return f"{sign}{int(value)} {unit}"
            return f"{sign}{value:.2f} {unit}"
        value /= 1024.0
    return f"{sign}{value:.2f} PB"


def is_valid_pdf(path: Path) -> bool:
    """Quick structural check: does the file start with ``%PDF-``?

    Args:
        path: File to probe.

    Returns:
        True if the header is present, False otherwise (including missing
        files or I/O errors).
    """
    try:
        if not path.is_file():
            return False
        with open(path, "rb") as fh:
            header = fh.read(5)
        return header == b"%PDF-"
    except OSError as exc:
        logger.warning("is_valid_pdf: cannot read %s: %s", path, exc)
        return False
