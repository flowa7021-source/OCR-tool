"""Auto-detect resources from a user-installed OCR Studio.

Normally the app finds its bundled Tesseract, Ghostscript, and
``tessdata`` under ``APP_ROOT/resources/`` — meaningful inside a
PyInstaller ``--windowed`` install where ``APP_ROOT`` resolves to the
``_internal`` directory. In a **developer checkout** (``git clone`` +
``pip install -e .[dev]``) that path doesn't exist: the dev is running
Python against source files, no PyInstaller bundle is next to them.

Historically the fallback was ``shutil.which("tesseract")`` — fine on
Linux where ``apt install tesseract-ocr`` puts it on PATH, but on
Windows the dev would either have to install Tesseract separately
(Chocolatey / MSI) or manually copy the bundled binaries out of an
existing OCR Studio install. That second option is exactly the dev
workflow we want to unblock: the user already HAS OCR Studio
installed and its ``_internal/resources/`` is sitting right there on
disk; we just need to find it.

This module walks the standard Windows install locations and returns
the first ``_internal/resources`` directory it finds. Callers that
resolve external binaries (``TesseractWrapper`` + ``external_tools``)
use the returned path as an extra search root, **after** the
PyInstaller-bundled path (which takes precedence inside a real frozen
build) and **before** ``shutil.which`` (so the installed app's
Tesseract wins over any random system-wide one the dev might have).

No-op on non-Windows platforms and when no OCR Studio is installed.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Standard Windows install locations produced by the Inno Setup
# installer we ship. Ordered most-specific first so per-user installs
# (the default for a double-click install without admin) win over
# system-wide ones — a dev with both shouldn't have the wrong pair
# picked for them.
_WINDOWS_CANDIDATE_ROOTS: tuple[str, ...] = (
    # LOCALAPPDATA\Programs\<name> — Inno Setup default for per-user.
    "{LOCALAPPDATA}/Programs/OCR Studio",
    # PROGRAMFILES\<name> — Inno Setup default for per-machine install.
    "{PROGRAMFILES}/OCR Studio",
    # 32-bit install on 64-bit Windows.
    "{PROGRAMFILES(X86)}/OCR Studio",
)


@functools.lru_cache(maxsize=1)
def find_installed_ocr_studio_resources() -> Path | None:
    """Return the ``_internal/resources`` directory of an installed app.

    Cached for the lifetime of the process: filesystem probes during
    engine-availability checks are cheap individually but add up when
    the UI refreshes the engine dropdown every second.

    Returns:
        The resolved path, or ``None`` if no OCR Studio install was
        found. Non-Windows platforms always return ``None``.
    """
    if sys.platform != "win32":
        return None

    environ = os.environ
    for template in _WINDOWS_CANDIDATE_ROOTS:
        try:
            # Resolve ``{LOCALAPPDATA}`` / etc. against ``os.environ``
            # without leaking a ``KeyError`` when a variable is absent
            # on exotic installs (headless Windows Server without a
            # user session, stripped environments under SSH).
            expanded = template.format_map(
                _SafeEnvDict(environ)  # type: ignore[arg-type]
            )
        except (KeyError, ValueError):
            continue
        if "{" in expanded:
            # At least one substitution was missing — skip.
            continue
        candidate = Path(expanded) / "_internal" / "resources"
        if candidate.is_dir():
            logger.info(
                "Dev-mode: found installed OCR Studio resources at %s — "
                "CLI will reuse its bundled Tesseract / Ghostscript "
                "without needing a separate system-wide install.",
                candidate,
            )
            return candidate
    return None


class _SafeEnvDict(dict[str, str]):
    """dict wrapper that returns ``"{KEY}"`` for missing keys.

    Lets ``str.format_map`` leave unexpanded placeholders intact
    instead of raising ``KeyError``. The caller then sees ``"{" in
    expanded`` and knows the substitution was incomplete.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
