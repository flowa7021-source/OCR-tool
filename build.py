"""PyInstaller build script for OCR Studio.

Builds a standalone Windows distribution in `dist/OCRStudio/`. Bundles Tesseract
5.5.0 binaries, tessdata language files, QSS styles, icons, and the default
profile JSONs.

Usage (on Windows):
    python build.py                # full clean build
    python build.py --skip-clean   # keep previous build artifacts

Prerequisites:
    * Python 3.11+ with dependencies from requirements.txt installed
    * PyInstaller >= 6.6
    * Tesseract 5.5.0 binaries placed in `resources/tesseract/` (tesseract.exe,
      all DLLs). See docs/vendoring.md (TODO) for exact file list.
    * `resources/tessdata/rus.traineddata` and `eng.traineddata` present.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
APP_NAME = "OCRStudio"


def _sep() -> str:
    """PyInstaller path separator (';' on Windows, ':' elsewhere)."""
    return ";" if os.name == "nt" else ":"


def clean() -> None:
    """Remove previous build artifacts."""
    for path in (DIST_DIR, BUILD_DIR):
        if path.exists():
            print(f"[build] removing {path}")
            shutil.rmtree(path, ignore_errors=True)
    for spec in PROJECT_ROOT.glob("*.spec"):
        try:
            spec.unlink()
        except OSError:
            pass


def ensure_resources() -> None:
    """Validate that bundleable resources exist before building."""
    tessdata = PROJECT_ROOT / "resources" / "tessdata"
    required = ("rus.traineddata", "eng.traineddata")
    missing = [f for f in required if not (tessdata / f).exists()]
    if missing:
        print(
            f"[build] WARNING: missing tessdata files: {missing}. "
            "Place them in resources/tessdata/ before shipping."
        )

    tess_bin = PROJECT_ROOT / "resources" / "tesseract" / "tesseract.exe"
    if os.name == "nt" and not tess_bin.exists():
        print(
            f"[build] WARNING: Tesseract binary not found at {tess_bin}. "
            "Application will fall back to system PATH."
        )


def build_pyinstaller(onefile: bool = False) -> int:
    """Invoke PyInstaller and return its exit code."""
    sep = _sep()
    entry = PROJECT_ROOT / "src" / "main.py"

    args: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        APP_NAME,
        "--noconfirm",
        "--clean",
        "--windowed",  # no console window
        "--onedir" if not onefile else "--onefile",
        f"--paths={PROJECT_ROOT}",
        # Bundled assets
        f"--add-data=resources/tessdata{sep}resources/tessdata",
        f"--add-data=resources/tesseract{sep}resources/tesseract",
        f"--add-data=resources/icons{sep}resources/icons",
        f"--add-data=resources/styles{sep}resources/styles",
        f"--add-data=profiles{sep}profiles",
        # Hidden imports that PyInstaller sometimes misses
        "--collect-submodules=ocrmypdf",
        "--collect-submodules=pikepdf",
        "--collect-data=ocrmypdf",
        "--collect-data=pikepdf",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=skimage.filters",
    ]

    # Attach Windows .ico if it was generated/placed before the build.
    ico = PROJECT_ROOT / "resources" / "icons" / "app.ico"
    if ico.exists():
        args.append(f"--icon={ico}")

    args.append(str(entry))
    print("[build] running:", " ".join(args))
    return subprocess.call(args, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OCR Studio distribution")
    parser.add_argument("--skip-clean", action="store_true", help="keep previous build")
    parser.add_argument("--onefile", action="store_true", help="single .exe (slower startup)")
    opts = parser.parse_args()

    if not opts.skip_clean:
        clean()

    ensure_resources()

    rc = build_pyinstaller(onefile=opts.onefile)
    if rc != 0:
        print(f"[build] PyInstaller failed with exit code {rc}")
        return rc

    print(f"[build] Success. Output: {DIST_DIR / APP_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
