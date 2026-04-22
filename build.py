"""PyInstaller build script for OCR Studio (EasyOCR edition).

Builds a standalone Windows distribution in ``dist/OCRStudio/``. Bundles
pre-downloaded EasyOCR weights (CRAFT detector + ru/en recognizers),
QSS styles, icons, Russian lexicon and the default profile JSONs.

Usage:
    python build.py                # default install
    python build.py --skip-clean   # keep previous build artifacts
    python build.py --onefile      # single-file .exe (slower startup)

Prerequisites:
    * Python 3.11+ with requirements.txt installed (incl. torch + easyocr).
    * EasyOCR model weights downloaded into ``resources/easyocr_models/``.
      Run ``python scripts/prefetch_easyocr_models.py`` first — the
      check below is a hard gate (missing models → fresh installs hang
      on first-run download and can fail offline).
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

#: Model files EasyOCR expects under ``~/.EasyOCR/model/`` at runtime.
#: Bundled verbatim so offline first-run works on any fresh machine.
EASYOCR_REQUIRED_MODELS = (
    "craft_mlt_25k.pth",   # CRAFT text detector (shared, ~83 MB)
    # ru+en recognizer (~15 MB). If ``scripts/finetune/finetune_recognizer.py``
    # has been run, this file is the fine-tuned checkpoint and the stock
    # version is kept next to it as ``cyrillic_g2.pth.stock`` for rollback.
    # PyInstaller bundles the current file as-is — no special casing needed.
    "cyrillic_g2.pth",
)


def _sep() -> str:
    return ";" if os.name == "nt" else ":"


def clean() -> None:
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
    """Hard-gate build on presence of EasyOCR model weights."""
    models_dir = PROJECT_ROOT / "resources" / "easyocr_models"
    missing = [m for m in EASYOCR_REQUIRED_MODELS
               if not (models_dir / m).exists()]
    if missing:
        raise SystemExit(
            f"[build] ERROR: resources/easyocr_models/ missing: {missing}.\n"
            "Run: python scripts/prefetch_easyocr_models.py\n"
            "This bundles the CRAFT detector + ru/en recognizer so offline "
            "first-run works without a 400 MB download from HuggingFace."
        )

    lex = PROJECT_ROOT / "resources" / "ru_lexicon.txt"
    if not lex.exists():
        raise SystemExit(
            f"[build] ERROR: {lex} missing. Restore from git "
            "(git checkout -- resources/ru_lexicon.txt)."
        )


def build_pyinstaller(onefile: bool = False) -> int:
    sep = _sep()
    entry = PROJECT_ROOT / "src" / "main.py"

    args: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm", "--clean", "--windowed",
        "--onedir" if not onefile else "--onefile",
        f"--paths={PROJECT_ROOT}",
        # Bundled assets
        f"--add-data=resources/easyocr_models{sep}resources/easyocr_models",
        f"--add-data=resources/ru_lexicon.txt{sep}resources",
        f"--add-data=resources/icons{sep}resources/icons",
        f"--add-data=resources/styles{sep}resources/styles",
        f"--add-data=profiles{sep}profiles",
        # expected/ ground-truth catalog (tn_parser ИНН lookup) —
        # optional on dev checkouts.
        *(
            [f"--add-data=expected{sep}expected"]
            if (PROJECT_ROOT / "expected").is_dir() else []
        ),
        # PyTorch / EasyOCR hidden imports + data files.
        "--collect-all=easyocr",
        "--collect-all=torch",
        "--collect-all=torchvision",
        "--collect-submodules=skimage",
        "--hidden-import=PIL._tkinter_finder",
        # ТН / УПД parser deps.
        "--collect-all=rapidfuzz",
        "--collect-all=scripts",
        "--collect-submodules=src.tn_parser",
        "--collect-all=openpyxl",
        "--collect-all=pymorphy3",
        "--collect-all=pymorphy3_dicts_ru",
    ]

    ico = PROJECT_ROOT / "resources" / "icons" / "app.ico"
    if ico.exists():
        args.append(f"--icon={ico}")

    args.append(str(entry))
    print("[build] running:", " ".join(args))
    return subprocess.call(args, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OCR Studio")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--onefile", action="store_true")
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
