"""Pre-download EasyOCR model weights into resources/easyocr_models/.

Run this ONCE before ``python build.py`` so the installer can bundle
the weights and ship offline-ready. EasyOCR otherwise downloads
~400 MB from HuggingFace on first use of each language.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "resources" / "easyocr_models"
REQUIRED = ("craft_mlt_25k.pth", "cyrillic_g2.pth")


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    import easyocr
    # ``Reader(["ru", "en"])`` loads cyrillic_g2.pth, which recognises
    # both Cyrillic and Latin glyphs (EasyOCR's mixed-script combo).
    # That's the only recognizer we need at runtime; english_g2.pth
    # is a pure-Latin model we don't ship.
    print("[prefetch] Loading ru+en EasyOCR reader…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    cache = Path(reader.model_storage_directory)
    print(f"[prefetch] Cache dir: {cache}")

    missing = []
    for name in REQUIRED:
        src = cache / name
        dst = TARGET / name
        if not src.exists():
            missing.append(name)
            continue
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"[prefetch] Already bundled: {name}")
            continue
        shutil.copy2(src, dst)
        print(f"[prefetch] Copied: {name} ({dst.stat().st_size / 1e6:.1f} MB)")

    if missing:
        print(f"[prefetch] ERROR: missing after reader init: {missing}",
              file=sys.stderr)
        return 1
    print(f"[prefetch] Done. Models in: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
