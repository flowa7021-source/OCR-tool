"""Discover and register bundled external binaries on ``PATH``.

OCRmyPDF spawns several binaries as subprocesses — Tesseract, Ghostscript,
and optionally pngquant/unpaper/jbig2enc. It locates them through
``shutil.which(name)``, which walks the ``PATH`` environment variable.
Our installer ships these binaries under ``resources/<tool>/``, but
unless that directory is on ``PATH`` OCRmyPDF will error out with the
unhelpful::

    MissingDependencyError: Could not find program 'tesseract' on the PATH

even though the executable is sitting right next to the app.

This module centralises the two problems:

1. **Discovery** — given our bundle layout (``resources/tesseract/``,
   ``resources/ghostscript/``, …) find the actual executable name,
   accounting for platform differences (e.g. ``gswin64c.exe`` on
   Windows vs ``gs`` on Linux).
2. **Registration** — prepend every directory containing a discovered
   binary to ``os.environ["PATH"]``, idempotently, so any subprocess
   spawned by any library picks up our copy.

And one diagnostic helper:

3. **Pre-flight check** — returns the set of OCRmyPDF-required tools
   that are missing, so the pipeline can fail fast with a clear
   error message instead of letting OCRmyPDF raise cryptic tracebacks
   mid-job.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from src.shared.constants import (
    GHOSTSCRIPT_BIN_DIR,
    TESSERACT_BIN_DIR,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalTool:
    """Metadata for a bundled external binary.

    Attributes:
        name: Logical tool name (``"tesseract"``, ``"ghostscript"``).
        bundle_dir: Directory in the installation where we place the
            binary via ``--add-data``.
        candidate_names: Executable basenames to probe, in preference
            order. First match wins. Having multiple helps with
            platform / distribution differences (Ghostscript on
            Windows ships as ``gswin64c.exe``, on Linux as ``gs``).
        required_for_ocrmypdf: When True, a missing binary breaks
            OCRmyPDF entirely and we should fail fast.
    """

    name: str
    bundle_dir: Path
    candidate_names: tuple[str, ...]
    required_for_ocrmypdf: bool


# Windows vs POSIX binary names. Using ``sys.platform`` keeps the table
# readable while still producing the correct executable list at import
# time — no conditional lookups at call sites.
_IS_WINDOWS = sys.platform == "win32"

_TESSERACT_CANDIDATES = (
    ("tesseract.exe",) if _IS_WINDOWS else ("tesseract",)
)

# Ghostscript on Windows is ``gswin64c.exe`` (console) / ``gswin64.exe``
# (GUI). OCRmyPDF explicitly prefers the console build because it can
# attach stdin/stdout. On Linux the binary is called ``gs``.
_GHOSTSCRIPT_CANDIDATES = (
    ("gswin64c.exe", "gswin32c.exe", "gs.exe")
    if _IS_WINDOWS
    else ("gs",)
)

# Full registry of tools we might bundle. Missing entries in the bundle
# aren't fatal — they just mean the feature (jbig2, unpaper, …) is
# unavailable. ``required_for_ocrmypdf=True`` is reserved for the two
# hard dependencies OCRmyPDF cannot run without.
REGISTRY: tuple[ExternalTool, ...] = (
    ExternalTool(
        name="tesseract",
        bundle_dir=TESSERACT_BIN_DIR,
        candidate_names=_TESSERACT_CANDIDATES,
        required_for_ocrmypdf=True,
    ),
    ExternalTool(
        name="ghostscript",
        bundle_dir=GHOSTSCRIPT_BIN_DIR,
        candidate_names=_GHOSTSCRIPT_CANDIDATES,
        required_for_ocrmypdf=True,
    ),
)


def locate(tool: ExternalTool) -> Path | None:
    """Return the path to ``tool``'s bundled binary, or ``None`` if absent.

    Looks under :attr:`ExternalTool.bundle_dir` (and one level of
    subdirectories, to be tolerant of installers that nest under
    ``bin/``) for any of :attr:`ExternalTool.candidate_names`.
    Falls back to ``shutil.which(name)`` so the function also works
    in development where the tool is only on the system PATH.
    """
    bundle_dir = tool.bundle_dir
    if bundle_dir.is_dir():
        for candidate in tool.candidate_names:
            direct = bundle_dir / candidate
            if direct.is_file():
                return direct
            for child in bundle_dir.iterdir():
                if child.is_dir():
                    nested = child / candidate
                    if nested.is_file():
                        return nested
    # Dev mode fallback — maybe the user has it on their PATH.
    for candidate in tool.candidate_names:
        system = shutil.which(candidate)
        if system:
            return Path(system)
    return None


def ensure_on_path() -> dict[str, Path | None]:
    """Register every bundled tool directory on ``os.environ['PATH']``.

    This is the single entry point called at worker startup (and
    optionally on the host) so that every subprocess-spawning library
    — ``pytesseract``, ``ocrmypdf``, ``subprocess.run(["gs", ...])`` —
    sees our bundled binaries.

    Returns:
        ``{tool_name: resolved_path | None}`` for diagnostics. A
        ``None`` value means the binary couldn't be located in the
        bundle or on the system PATH, so OCRmyPDF will fail if the
        tool is marked required.
    """
    resolved: dict[str, Path | None] = {}
    existing = os.environ.get("PATH", "").split(os.pathsep)
    # Build a set for O(1) membership; preserve order in a list so we
    # know what to prepend.
    existing_set = {p for p in existing if p}

    prepend: list[str] = []
    for tool in REGISTRY:
        path = locate(tool)
        resolved[tool.name] = path
        if path is None:
            level = logging.ERROR if tool.required_for_ocrmypdf else logging.INFO
            logger.log(
                level,
                "External tool %r not found (bundle_dir=%s, required=%s)",
                tool.name, tool.bundle_dir, tool.required_for_ocrmypdf,
            )
            continue
        bin_dir = str(path.parent)
        if bin_dir not in existing_set:
            prepend.append(bin_dir)
            existing_set.add(bin_dir)
            logger.info(
                "Registered %s on PATH: %s (binary=%s)",
                tool.name, bin_dir, path.name,
            )

    if prepend:
        os.environ["PATH"] = (
            os.pathsep.join(prepend) + os.pathsep + os.environ.get("PATH", "")
        )

    return resolved


def verify_required_for_ocrmypdf() -> list[str]:
    """Return the names of required external tools that are missing.

    Empty list means OCRmyPDF can run. A non-empty list is the
    best-effort answer to "what would break if I called
    ``ocrmypdf.ocr(...)`` right now?". Use this for a pre-flight check
    in the pipeline so end users see a meaningful error message
    instead of OCRmyPDF's internal ``MissingDependencyError``
    stacktrace.
    """
    missing: list[str] = []
    for tool in REGISTRY:
        if not tool.required_for_ocrmypdf:
            continue
        if locate(tool) is None:
            missing.append(tool.name)
    return missing
