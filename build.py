"""PyInstaller build script for OCR Studio.

Builds a standalone Windows distribution in `dist/OCRStudio/`. Bundles
Tesseract 5.5.0 binaries, tessdata language files, QSS styles, icons,
and the default profile JSONs.

Usage (on Windows):
    python build.py                # default install: ~250 MB
    python build.py --skip-clean   # keep previous build artifacts

Prerequisites:
    * Python 3.11+ with dependencies from requirements.txt installed
    * PyInstaller >= 6.6
    * Tesseract 5.5.0 binaries placed in `resources/tesseract/`
      (tesseract.exe + all DLLs).
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

    # Stage D of Initiative 1: user-words + user-patterns. These are
    # checked into the repo under ``resources/tessdata/`` and picked
    # up automatically by the ``--add-data=resources/tessdata`` hook
    # below, but someone deleting them would silently regress Russian
    # accuracy (ИНН / КПП / dates / entity abbreviations) with no
    # visible error. Refuse to build so the regression is caught at
    # packaging time rather than after release.
    required_user_dicts = ("user-words.rus", "user-patterns.rus")
    missing_user_dicts = [
        f for f in required_user_dicts if not (tessdata / f).exists()
    ]
    if missing_user_dicts:
        raise SystemExit(
            f"[build] ERROR: tessdata/ missing required user-dict files: "
            f"{missing_user_dicts}.\n"
            "These files hold Russian business vocabulary and regex "
            "patterns (ИНН / КПП / dates) that Tesseract loads at OCR "
            "time to improve accuracy. They live in the repo under "
            "resources/tessdata/ and are bundled automatically by the "
            "--add-data hook; only a manual delete or a broken checkout "
            "would remove them. Restore them from git (``git checkout "
            "-- resources/tessdata/user-words.rus "
            "resources/tessdata/user-patterns.rus``) and re-run."
        )

    # The ``configs/`` subdirectory of tessdata holds Tesseract's
    # output-format params (``hocr``, ``txt``, ``pdf``, etc.). Without
    # these the bundled Tesseract runs but cannot emit hOCR — every
    # OCRmyPDF call ends in the dreaded
    # ``FileNotFoundError: ..._ocr_hocr.hocr`` graft crash. Refuse to
    # build (rather than warn) because shipping without them produces
    # a binary that fails on every page of every document.
    configs = tessdata / "configs"
    required_configs = ("hocr", "txt", "pdf")
    missing_configs = [
        c for c in required_configs if not (configs / c).exists()
    ]
    if missing_configs:
        raise SystemExit(
            f"[build] ERROR: tessdata/configs/ missing required files: "
            f"{missing_configs}.\n"
            "These are tiny text files from "
            "github.com/tesseract-ocr/tesseract/tree/main/tessdata/configs "
            "that tell Tesseract which output formats to produce. Without "
            "them OCRmyPDF crashes on every page with "
            "'_ocr_hocr.hocr not found'.\n"
            "The CI workflow downloads them automatically; if you're "
            "building locally, run:\n"
            f"  mkdir -p {configs}\n"
            "  for f in hocr txt pdf; do\n"
            "    curl -fsSL "
            "https://github.com/tesseract-ocr/tesseract/raw/main/tessdata/configs/$f "
            f"-o {configs}/$f\n"
            "  done"
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


def build_pyinstaller(onefile: bool = False) -> int:
    """Invoke PyInstaller and return its exit code.

    Args:
        onefile: Use ``--onefile`` instead of ``--onedir``.
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
        # ``expected/`` ground-truth JSON corpus — used by
        # :func:`src.core.doc_catalog.load_default_catalog` at worker
        # startup to seed the ИНН / ОГРН auto-correct lookup table.
        # Optional on dev checkouts (the folder may not exist yet)
        # for the same reason we gate Ghostscript above.
        *(
            [f"--add-data=expected{sep}expected"]
            if (PROJECT_ROOT / "expected").is_dir()
            else []
        ),
        # Hidden imports that PyInstaller sometimes misses
        "--collect-submodules=ocrmypdf",
        "--collect-submodules=pikepdf",
        "--collect-data=ocrmypdf",
        "--collect-data=pikepdf",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=skimage.filters",
        # ТН / УПД parser dependencies.
        #
        # * ``rapidfuzz`` is a Cython C-extension (Levenshtein / token-
        #   sort matchers). PyInstaller's static analyser collects the
        #   top-level ``rapidfuzz`` module but sometimes misses the
        #   compiled ``.pyd`` siblings (``rapidfuzz.distance``,
        #   ``rapidfuzz.process_cpp``) that `sections.py` loads through
        #   re-export. Missing .pyd → ``ImportError: DLL load failed``
        #   the first time a user opens the ``tn_upd`` profile. The
        #   ``--collect-all`` directive fetches submodules + data +
        #   binaries in one go — the safest mode for C-ext libraries.
        # * ``src.tn_parser`` submodules (``fields``, ``validators``,
        #   ``sections``, …) are all relative-imported inside the
        #   package, which PyInstaller usually follows. Declaring the
        #   whole subtree explicitly costs nothing and protects
        #   against the "lazy import inside a function missed by the
        #   static analyser" failure mode.
        # * ``scripts`` ships the four ``ocr-cli parser <cmd>``
        #   backing modules. ``cli.py`` imports them via
        #   ``importlib.import_module`` — PyInstaller cannot see that
        #   call at build time, so without ``--collect-all=scripts``
        #   every parser subcommand in the installed build would
        #   raise ``ModuleNotFoundError: scripts.run_golden`` (or
        #   similar) even though the CLI entry itself works. Using
        #   ``collect-all`` instead of ``collect-submodules`` picks up
        #   the (empty) ``__init__.py`` alongside the leaf modules.
        # * ``openpyxl`` is pure-Python and usually auto-detected
        #   through ``src.tn_parser.excel``. The explicit hidden-
        #   import is a belt-and-braces for stripped builds where
        #   an ``__all__`` reshuffle in a future openpyxl release
        #   could hide the import from PyInstaller's scanner.
        "--collect-all=rapidfuzz",
        "--collect-all=scripts",
        "--collect-submodules=src.tn_parser",
        # ``openpyxl`` has ~80 submodules (cell/, styles/, writer/, …) —
        # ``--hidden-import=openpyxl`` only collects the top-level
        # package and leaves ``import openpyxl.workbook`` failing at
        # runtime inside the frozen bundle. Use ``--collect-all`` so
        # every submodule + data file (``_constants.py``, schemas)
        # ships together. The parser-smoke step asserts an ``openpyxl``
        # directory lives under the install tree; without this flag
        # PyInstaller collapses the whole thing into a single .pyc
        # inside ``base_library.zip`` and the check can't find it.
        "--collect-all=openpyxl",
        # ``pymorphy3`` also lazy-loads language dictionaries via
        # ``importlib.import_module(f'pymorphy3_dicts_{lang}')``. Same
        # "static analyser misses dynamic import" failure mode.
        "--collect-all=pymorphy3",
        "--collect-all=pymorphy3_dicts_ru",
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
