"""Build a searchable PDF from page images + OCR word boxes.

Writes each page as a raster image and stamps an invisible text layer
(PDF render_mode=3) with per-word bounding boxes so downstream code
(UI preview, tn_parser layout_anchor, exporters) can ``get_text()``
the result exactly as it would from an ocrmypdf-produced PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from src.shared.constants import RESOURCES_DIR

# DejaVu Sans covers Cyrillic, Greek, Latin Extended, Arabic Basic and more.
# Base-14 PDF fonts (helv/cour/…) are Latin-only — Cyrillic chars become "·".
_FONT_FILE = RESOURCES_DIR / "DejaVuSans.ttf"
_FONT_NAME = "dejavu"


@dataclass
class PageWords:
    """One page: raster image bytes + OCR word boxes in pixel coords."""

    image_png: bytes
    width_px: int
    height_px: int
    dpi: int
    # (x, y, w, h, conf[0..100], text) in pixel coordinates
    words: list[tuple[float, float, float, float, float, str]]


def build_searchable_pdf(pages: list[PageWords], output_pdf: Path) -> None:
    """Assemble a searchable PDF from the given pages."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    font_file_str = str(_FONT_FILE) if _FONT_FILE.exists() else None
    doc = fitz.open()
    try:
        for p in pages:
            pt_w = p.width_px * 72.0 / p.dpi
            pt_h = p.height_px * 72.0 / p.dpi
            page = doc.new_page(width=pt_w, height=pt_h)
            page.insert_image(page.rect, stream=p.image_png)
            if font_file_str:
                page.insert_font(fontname=_FONT_NAME, fontfile=font_file_str)
            scale = 72.0 / p.dpi  # px → pt
            for x, y, w, h, _conf, text in p.words:
                if not text.strip() or w <= 0 or h <= 0:
                    continue
                # Invisible text: render_mode=3 hides glyphs but keeps
                # them selectable/searchable.
                rect = fitz.Rect(x * scale, y * scale,
                                 (x + w) * scale, (y + h) * scale)
                fontsize = max(1.0, rect.height * 0.8)
                extra = (
                    {"fontname": _FONT_NAME, "fontfile": font_file_str}
                    if font_file_str else {}
                )
                page.insert_text(
                    (rect.x0, rect.y1 - (rect.height - fontsize) / 2),
                    text,
                    fontsize=fontsize,
                    render_mode=3,
                    **extra,
                )
        doc.save(str(output_pdf), garbage=3, deflate=True)
    finally:
        doc.close()
