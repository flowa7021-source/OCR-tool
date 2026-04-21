"""Build a portable diagnostics bundle for bug reports.

The zip contains everything a maintainer needs to triage a problem
*without* any user documents or OCR output — just the app's own
state:

    * rotating log file and any existing backups
    * ``settings.json`` (personal preferences, no credentials)
    * Python / OS / Qt version strings
    * engine availability (EasyOCR import + torch / CUDA status)
    * installed package versions for our declared dependencies

User profiles live one directory over; since they may contain user
text (custom regex rules, descriptions) we deliberately skip them.
"""

from __future__ import annotations

import io
import json
import logging
import platform
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_read(path: Path, max_bytes: int = 5 * 1024 * 1024) -> bytes | None:
    """Read a file, capped at ``max_bytes``. Returns None on any error."""
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("rb") as fh:
            return fh.read(max_bytes)
    except OSError as exc:
        logger.warning("Could not read %s for diagnostics: %s", path, exc)
        return None


def _collect_environment() -> dict[str, str]:
    """Gather non-identifying environment info."""
    info: dict[str, str] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split("\n")[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "frozen": str(getattr(sys, "frozen", False)),
    }
    try:
        import PySide6

        info["pyside6_version"] = PySide6.__version__
    except Exception:  # noqa: BLE001
        info["pyside6_version"] = "(not installed)"

    # Declared project dependencies: snapshot their versions too.
    for package in (
        "opencv-python",
        "PyMuPDF",
        "easyocr",
        "numpy",
        "scikit-image",
        "Pillow",
        "psutil",
        "python-docx",
        "torch",
        "transformers",
        "tiktoken",
    ):
        try:
            from importlib.metadata import version as _pkg_version

            info[f"pkg.{package}"] = _pkg_version(package)
        except Exception:  # noqa: BLE001
            info[f"pkg.{package}"] = "(not installed)"
    return info


def _collect_engine_status() -> dict[str, dict[str, object]]:
    """Probe each registered OCR engine and report availability."""
    status: dict[str, dict[str, object]] = {}
    try:
        from src.application.engines.registry import list_engines

        for kind, name, available, msg in list_engines():
            status[kind.value] = {
                "name": name,
                "available": bool(available),
                "message": msg,
            }
    except Exception as exc:  # noqa: BLE001
        status["_error"] = {"message": str(exc)}
    return status


def _collect_engine_details() -> dict[str, object]:
    """Probe the EasyOCR engine: torch + CUDA + bundled-models dir."""
    from src.application.engines.registry import get_engine
    from src.shared.constants import EASYOCR_MODELS_DIR
    from src.shared.types import OCREngineKind

    details: dict[str, object] = {}
    try:
        engine = get_engine(OCREngineKind.EASYOCR)
        ok, msg = engine.is_available()
        details["available"] = bool(ok)
        details["message"] = msg
    except Exception as exc:  # noqa: BLE001
        details["available"] = False
        details["message"] = f"(probe failed: {exc})"

    try:
        import torch

        details["torch_version"] = torch.__version__
        details["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            details["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        details["torch_version"] = f"(not installed: {exc})"
        details["cuda_available"] = False

    details["models_dir"] = str(EASYOCR_MODELS_DIR)
    try:
        if EASYOCR_MODELS_DIR.is_dir():
            details["models_files"] = sorted(
                p.name for p in EASYOCR_MODELS_DIR.iterdir()
            )
        else:
            details["models_files"] = "(directory missing)"
    except Exception as exc:  # noqa: BLE001
        details["models_files"] = f"(scan failed: {exc})"
    return details


def build_diagnostics_zip(
    target: Path,
    *,
    settings_path: Path | None = None,
    log_file: Path | None = None,
) -> Path:
    """Write a diagnostics zip to ``target`` and return the path.

    Args:
        target: Destination ``.zip`` path. Parent directory is created
            if missing; any existing file is overwritten.
        settings_path: Location of ``settings.json``. Defaults to the
            canonical ``CONFIG_DIR / settings.json``.
        log_file: Active log file path. Defaults to
            ``LOGS_DIR / LOG_FILE_NAME``. All rotated backups are
            included automatically.

    Returns:
        The final ``target`` path.
    """
    from src.shared.constants import CONFIG_DIR, LOG_FILE_NAME, LOGS_DIR

    settings_path = settings_path or (CONFIG_DIR / "settings.json")
    log_file = log_file or (LOGS_DIR / LOG_FILE_NAME)

    target.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        env_blob = json.dumps(
            {
                "environment": _collect_environment(),
                "engines": _collect_engine_status(),
                "engine_details": _collect_engine_details(),
            },
            ensure_ascii=False,
            indent=2,
        )
        zf.writestr("environment.json", env_blob)

        settings_bytes = _safe_read(settings_path)
        if settings_bytes is not None:
            zf.writestr("settings.json", settings_bytes)

        # Active log + its rotated siblings
        for candidate in sorted(log_file.parent.glob(f"{log_file.name}*")):
            blob = _safe_read(candidate)
            if blob is not None:
                zf.writestr(f"logs/{candidate.name}", blob)

        # Per-worker diagnostic logs (detailed pipeline trace written by
        # each ProcessPoolExecutor worker) + native-crash dumps produced
        # by faulthandler. These are the single most useful artefact when
        # diagnosing "OCR didn't produce any output" because the host's
        # ocr-studio.log can't see inside the worker processes.
        for pattern in ("worker-*.log", "worker-crash-*.log"):
            for candidate in sorted(log_file.parent.glob(pattern)):
                blob = _safe_read(candidate)
                if blob is not None:
                    zf.writestr(f"logs/{candidate.name}", blob)

        # A short README so recipients know what's in here and what
        # deliberately isn't (user PDFs + profiles).
        readme = io.BytesIO()
        readme.write(
            (
                "OCR Studio diagnostics bundle\n"
                "==============================\n"
                f"Generated: {datetime.now(UTC).isoformat()}\n\n"
                "Contents:\n"
                "  environment.json   — Python / OS / package versions,\n"
                "                        EasyOCR + torch / CUDA availability\n"
                "  settings.json      — application preferences\n"
                "  logs/*.log*        — rotating application log + backups\n"
                "  logs/worker-*.log  — per-worker pipeline trace (PID-keyed)\n"
                "  logs/worker-crash* — native-crash dumps (faulthandler)\n\n"
                "NOT included (for privacy):\n"
                "  * Input PDFs and OCR results\n"
                "  * User profiles (custom regex rules may be sensitive)\n"
                "  * EasyOCR model weights\n"
            ).encode()
        )
        zf.writestr("README.txt", readme.getvalue())

    logger.info("Wrote diagnostics to %s (%d bytes)", target, target.stat().st_size)
    return target
