"""Rasterise ``resources/icons/app.svg`` into a multi-size Windows ``.ico``.

Run from the repo root::

    python .github/scripts/generate_ico.py

Used by ``.github/workflows/build-installer.yml`` — moved out of the
workflow YAML because the previous in-line ``python - <<'EOF'`` heredoc
was a bash construct and crashed PowerShell with
``Missing file specification after redirection operator``.

Failures are soft: if ``cairosvg`` or ``Pillow`` aren't installed, or
the SVG is missing, we print a diagnostic and exit 0 — the workflow's
``continue-on-error: true`` step previously had the same semantics
and PyInstaller falls back to a generic icon.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

ICON_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    svg_path = repo_root / "resources" / "icons" / "app.svg"
    ico_path = repo_root / "resources" / "icons" / "app.ico"

    if not svg_path.exists():
        print(f"ICO generation skipped: source SVG not found at {svg_path}")
        return 0

    try:
        import cairosvg  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"ICO generation skipped: {exc}")
        return 0

    svg_bytes = svg_path.read_bytes()
    images = []
    for sz in ICON_SIZES:
        try:
            png = cairosvg.svg2png(bytestring=svg_bytes, output_width=sz, output_height=sz)
        except Exception as exc:  # noqa: BLE001 - cairosvg can raise anything
            print(f"ICO generation skipped: cairosvg failed at {sz}px: {exc}")
            return 0
        images.append(Image.open(BytesIO(png)).convert("RGBA"))

    try:
        images[0].save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in ICON_SIZES],
            append_images=images[1:],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ICO generation skipped: Pillow save failed: {exc}")
        return 0

    print(f"Wrote {ico_path} ({ico_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
