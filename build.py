"""PyInstaller build script for OCR Studio.

Builds a standalone Windows distribution in `dist/OCRStudio/`. Bundles
Tesseract 5.5.0 binaries, tessdata language files, QSS styles, icons,
the default profile JSONs, and (when ``--with-htr`` is passed) the
torch + transformers Python deps for the GOT-OCR 2.0 engine. Model
weights themselves are NOT bundled — they download to AppData on
first use, regardless of HTR.

Usage (on Windows):
    python build.py                # default install: ~250 MB
    python build.py --with-htr     # bundle torch/transformers: ~2.5 GB
    python build.py --skip-clean   # keep previous build artifacts

Prerequisites:
    * Python 3.11+ with dependencies from requirements.txt installed
    * PyInstaller >= 6.6
    * Tesseract 5.5.0 binaries placed in `resources/tesseract/`
      (tesseract.exe + all DLLs).
    * `resources/tessdata/rus.traineddata` and `eng.traineddata` present.
    * For ``--with-htr``: the ``htr`` extras must be installed
      (``pip install -e ".[htr]"`` or ``pip install torch transformers
      Pillow tiktoken``).
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

    # Ghostscript is a HARD dependency of OCRmyPDF; a missing bundle
    # means every OCR job fails with "Could not find program 'gs'".
    gs_bin = PROJECT_ROOT / "resources" / "ghostscript" / "gswin64c.exe"
    if os.name == "nt" and not gs_bin.exists():
        print(
            f"[build] WARNING: Ghostscript binary not found at {gs_bin}. "
            "OCRmyPDF will fail at runtime unless Ghostscript is on the "
            "system PATH. Build the CI workflow or run the Ghostscript "
            "download step manually before packaging for end users."
        )


def build_pyinstaller(onefile: bool = False, with_htr: bool = False) -> int:
    """Invoke PyInstaller and return its exit code.

    Args:
        onefile: Use ``--onefile`` instead of ``--onedir``.
        with_htr: Bundle torch / transformers / Pillow / tiktoken so the
            GOT-OCR 2.0 engine is selectable out of the box. Adds
            roughly 2 GB to the resulting bundle.
    """
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
        # Ghostscript is optional on developer machines (the CI workflow
        # downloads + drops it into resources/ghostscript/; source checkouts
        # typically don't have it). Adding --add-data for a missing source
        # makes PyInstaller fail the entire build, so gate on existence.
        *(
            [f"--add-data=resources/ghostscript{sep}resources/ghostscript"]
            if (PROJECT_ROOT / "resources" / "ghostscript").is_dir()
            else []
        ),
        # Hidden imports that PyInstaller sometimes misses
        "--collect-submodules=ocrmypdf",
        "--collect-submodules=pikepdf",
        "--collect-data=ocrmypdf",
        "--collect-data=pikepdf",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=skimage.filters",
    ]

    if with_htr:
        # GOT-OCR 2.0 needs the entire torch + transformers + tokenizer
        # stack. PyInstaller's static analyser can't follow `from_pretrained`
        # dynamic loading, so we collect everything explicitly.
        args.extend(
            [
                "--collect-all=torch",
                "--collect-all=transformers",
                "--collect-all=tokenizers",
                "--collect-all=tiktoken",
                "--collect-all=PIL",
                "--collect-data=safetensors",
                "--collect-submodules=safetensors",
                # GOT-OCR 2.0 weights ship with `trust_remote_code=True`
                # Python files, so transformers will exec() them at runtime.
                # The hidden-imports below cover the symbols those files
                # reference (verified by inspecting the HF repo).
                "--hidden-import=torch._dynamo",
                "--hidden-import=torch._dynamo.config",
                "--hidden-import=torch._inductor",
                "--hidden-import=torchvision",
                "--hidden-import=transformers.models.auto",
                "--hidden-import=transformers.modeling_utils",
                "--hidden-import=transformers.generation",
            ]
        )

        # Bundle pre-downloaded GOT-OCR 2.0 weights if the CI step
        # (".github/scripts/download_got_ocr2.py") has populated the
        # tree. Users get handwriting OCR out-of-the-box instead of
        # having to fetch ~580 MB from HuggingFace after install.
        # Absent in typical source checkouts → skipped silently.
        bundled_models = PROJECT_ROOT / "resources" / "models"
        if (bundled_models / "got_ocr2").is_dir():
            args.append(
                f"--add-data=resources/models{sep}resources/models"
            )
            print(
                f"[build] bundling GOT-OCR 2.0 weights from {bundled_models}"
            )

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
    parser.add_argument(
        "--with-htr",
        action="store_true",
        help="bundle torch + transformers for the GOT-OCR 2.0 engine (~+2 GB)",
    )
    opts = parser.parse_args()

    if not opts.skip_clean:
        clean()

    ensure_resources()

    if opts.with_htr:
        # Fail-fast guard: if [htr] extras aren't installed the build
        # will quietly miss the modules and produce a broken bundle.
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            print(
                f"[build] ERROR: --with-htr requires torch + transformers; "
                f"missing: {exc.name}.\n"
                f"        Install via: pip install -e \".[htr]\""
            )
            return 2

    rc = build_pyinstaller(onefile=opts.onefile, with_htr=opts.with_htr)
    if rc != 0:
        print(f"[build] PyInstaller failed with exit code {rc}")
        return rc

    print(f"[build] Success. Output: {DIST_DIR / APP_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
