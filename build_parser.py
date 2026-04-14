"""PyInstaller build script for transport_parser.

Produces dist/TransportParser/ (onedir) containing TransportParser.exe.

Usage (on Windows runner):
    python -m pip install pyinstaller
    python build_parser.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "TransportParser"
ENTRY = ROOT / "transport_parser.py"


def clean() -> None:
    for d in (DIST, BUILD):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    for spec in ROOT.glob("*.spec"):
        try:
            spec.unlink()
        except OSError:
            pass


def build(onefile: bool = False) -> int:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", APP_NAME,
        "--windowed",
    ]
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    cmd.append(str(ENTRY))
    print("[build] running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onefile", action="store_true",
                        help="Build single-file exe instead of directory.")
    parser.add_argument("--skip-clean", action="store_true")
    args = parser.parse_args()

    if not args.skip_clean:
        clean()

    rc = build(onefile=args.onefile)
    if rc != 0:
        return rc

    out = DIST / (f"{APP_NAME}.exe" if args.onefile else APP_NAME)
    print(f"[build] done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
