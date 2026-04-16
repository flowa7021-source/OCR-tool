"""Locate and configure the Tesseract OCR binary.

Provides a small wrapper class that discovers the Tesseract executable and
its ``tessdata`` directory (checking bundled resources, environment
variables and finally the system ``PATH``), verifies that the required
languages are available and configures ``pytesseract`` accordingly.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.shared.constants import (
    TESSDATA_DIR,
    TESSERACT_BIN_DIR,
    TESSERACT_EXE_NAME,
    TESSERACT_VERSION,
)

logger = logging.getLogger(__name__)


class TesseractNotFoundError(RuntimeError):
    """Raised when the Tesseract binary cannot be located."""


class TessdataNotFoundError(RuntimeError):
    """Raised when no usable tessdata directory is found."""


# Windows-only flag to hide the console window of spawned processes.
if sys.platform == "win32":  # pragma: no cover - platform-specific
    _CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
else:
    _CREATE_NO_WINDOW = 0


_VERSION_RE = re.compile(r"tesseract\s+([\d.]+)", re.IGNORECASE)


class TesseractWrapper:
    """Wrapper around the Tesseract binary with cached class-level state."""

    # Class-level cache so repeated instantiation does not re-probe the FS.
    _binary_path: Path | None = None
    _tessdata_path: Path | None = None
    _version: str | None = None
    _configured: bool = False

    # ------------------------------------------------------------------
    # Binary / tessdata discovery
    # ------------------------------------------------------------------
    def find_tesseract_binary(self) -> Path:
        """Locate the Tesseract executable.

        Search order:
            1. Bundled binary at ``TESSERACT_BIN_DIR / TESSERACT_EXE_NAME``.
            2. ``TESSERACT_CMD`` environment variable.
            3. ``shutil.which("tesseract")``.

        Returns:
            Path to the Tesseract executable.

        Raises:
            TesseractNotFoundError: No usable binary exists.
        """
        if TesseractWrapper._binary_path is not None:
            return TesseractWrapper._binary_path

        candidates: list[Path] = []

        bundled = TESSERACT_BIN_DIR / TESSERACT_EXE_NAME
        candidates.append(bundled)

        env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
        if env_cmd:
            candidates.append(Path(env_cmd))

        system = shutil.which("tesseract")
        if system:
            candidates.append(Path(system))

        for candidate in candidates:
            if candidate and candidate.exists() and candidate.is_file():
                logger.info("Tesseract binary found: %s", candidate)
                TesseractWrapper._binary_path = candidate
                return candidate

        searched = ", ".join(str(c) for c in candidates) or "<none>"
        raise TesseractNotFoundError(
            f"Tesseract executable not found. Searched: {searched}"
        )

    def find_tessdata_dir(self) -> Path:
        """Locate a ``tessdata`` directory containing at least one traineddata file.

        Search order:
            1. ``TESSDATA_DIR`` (bundled resources).
            2. ``TESSDATA_PREFIX`` environment variable.
            3. ``tessdata/`` sibling next to the Tesseract binary.

        Returns:
            Path to a directory containing at least one ``*.traineddata`` file.

        Raises:
            TessdataNotFoundError: No candidate contains traineddata.
        """
        if TesseractWrapper._tessdata_path is not None:
            return TesseractWrapper._tessdata_path

        candidates: list[Path] = [TESSDATA_DIR]

        env_prefix = os.environ.get("TESSDATA_PREFIX", "").strip()
        if env_prefix:
            env_path = Path(env_prefix)
            candidates.append(env_path)
            # TESSDATA_PREFIX sometimes points to parent of tessdata/.
            candidates.append(env_path / "tessdata")

        try:
            binary = self.find_tesseract_binary()
            candidates.append(binary.parent / "tessdata")
        except TesseractNotFoundError:
            logger.debug("Binary missing while searching tessdata; continuing")

        for candidate in candidates:
            if (
                candidate.exists()
                and candidate.is_dir()
                and any(candidate.glob("*.traineddata"))
            ):
                logger.info("Tessdata directory found: %s", candidate)
                TesseractWrapper._tessdata_path = candidate
                return candidate

        searched = ", ".join(str(c) for c in candidates) or "<none>"
        raise TessdataNotFoundError(
            f"No tessdata directory with *.traineddata found. Searched: {searched}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def available_languages(self) -> list[str]:
        """Return sorted list of language codes present in the tessdata dir."""
        tessdata = self.find_tessdata_dir()
        langs = sorted({p.stem for p in tessdata.glob("*.traineddata")})
        return langs

    def get_version(self) -> str:
        """Invoke ``tesseract --version`` and parse the first line.

        Returns:
            Version string such as ``"5.5.0"``. Empty string if parsing failed.
        """
        if TesseractWrapper._version is not None:
            return TesseractWrapper._version

        binary = self.find_tesseract_binary()
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=_CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("Timeout while querying tesseract version: %s", exc)
            return ""
        except OSError as exc:
            logger.error("Failed to run tesseract --version: %s", exc)
            return ""

        # Tesseract prints version to stderr historically; try both.
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        first_line = next((line for line in output.splitlines() if line.strip()), "")
        match = _VERSION_RE.search(first_line)
        version = match.group(1) if match else ""
        TesseractWrapper._version = version
        logger.info("Tesseract version detected: %s", version or "<unknown>")
        return version

    # ------------------------------------------------------------------
    # Verification + configuration
    # ------------------------------------------------------------------
    def verify(self) -> tuple[bool, str]:
        """Perform a full health-check of the Tesseract install.

        Checks binary presence, parses its version (warning-only on
        mismatch with :data:`TESSERACT_VERSION`) and ensures the ``rus`` and
        ``eng`` languages are available. On success ``pytesseract`` is
        configured.

        Returns:
            Tuple ``(ok, message)``.
        """
        try:
            binary = self.find_tesseract_binary()
        except TesseractNotFoundError as exc:
            return False, f"Tesseract не найден: {exc}"

        try:
            tessdata = self.find_tessdata_dir()
        except TessdataNotFoundError as exc:
            return False, f"Tessdata не найдена: {exc}"

        version = self.get_version()
        messages: list[str] = [
            f"Бинарный файл: {binary}",
            f"Tessdata: {tessdata}",
            f"Версия: {version or '<не определено>'}",
        ]

        # Major.minor comparison only; patch differences are tolerated.
        if version:
            expected_parts = TESSERACT_VERSION.split(".")[:2]
            actual_parts = version.split(".")[:2]
            if expected_parts != actual_parts:
                warn = (
                    f"Версия Tesseract {version} отличается от ожидаемой "
                    f"{TESSERACT_VERSION} (major.minor)"
                )
                logger.warning(warn)
                messages.append(f"Предупреждение: {warn}")

        langs = self.available_languages()
        missing = [lang for lang in ("rus", "eng") if lang not in langs]
        if missing:
            msg = f"Отсутствуют языки: {', '.join(missing)}"
            messages.append(msg)
            return False, " | ".join(messages)

        messages.append(f"Языки: {', '.join(langs)}")

        try:
            self.configure_pytesseract()
        except Exception as exc:  # pragma: no cover - pytesseract optional at import
            logger.error("Failed to configure pytesseract: %s", exc)
            messages.append(f"Не удалось настроить pytesseract: {exc}")
            return False, " | ".join(messages)

        return True, " | ".join(messages)

    def configure_pytesseract(self) -> None:
        """Set ``pytesseract.tesseract_cmd``, ``TESSDATA_PREFIX`` and ``PATH``.

        ``pytesseract.tesseract_cmd`` only helps code that goes through
        the ``pytesseract`` library. OCRmyPDF — which is the actual OCR
        engine we run — uses its own ``shutil.which("tesseract")`` to
        locate the binary. If the bundled ``tesseract.exe`` isn't on
        ``PATH``, OCRmyPDF raises::

            MissingDependencyError: Could not find program 'tesseract'

        even though we just "configured" it for pytesseract. Prepending
        the bundled binary's directory to ``PATH`` makes it visible to
        every subprocess — pytesseract, ocrmypdf's subprocess module,
        ghostscript spawning tesseract, all uniformly.

        Idempotent: re-running it doesn't stack duplicate entries on PATH.
        """
        binary = self.find_tesseract_binary()
        tessdata = self.find_tessdata_dir()

        # pytesseract is imported lazily so that infrastructure tests that
        # don't need OCR can still run without the dependency installed.
        import pytesseract  # noqa: WPS433 - local import intentional

        pytesseract.pytesseract.tesseract_cmd = str(binary)
        os.environ["TESSDATA_PREFIX"] = str(tessdata)

        bin_dir = str(binary.parent)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if bin_dir not in path_entries:
            os.environ["PATH"] = (
                bin_dir + os.pathsep + os.environ.get("PATH", "")
            )
            logger.info("Prepended Tesseract bin dir to PATH: %s", bin_dir)

        TesseractWrapper._configured = True
        logger.info(
            "pytesseract configured (cmd=%s, TESSDATA_PREFIX=%s)", binary, tessdata
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Clear cached discovery results (primarily for tests)."""
        cls._binary_path = None
        cls._tessdata_path = None
        cls._version = None
        cls._configured = False

    def refresh(self) -> tuple[bool, str]:
        """Re-probe the filesystem and reconfigure pytesseract.

        Call this after any runtime change to the Tesseract install —
        for example when the user just dropped a new ``*.traineddata``
        into the tessdata directory, or after a bundled Tesseract
        update. Cheaper than restarting the app, and the only supported
        way to make the process pick up the new files without a
        restart.

        Returns the result of :meth:`verify` on the refreshed cache.
        """
        type(self).reset()
        return self.verify()
