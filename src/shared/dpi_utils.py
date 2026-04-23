"""Source-DPI utilities — detect the native resolution of images
embedded in a PDF page.

A page's render DPI (e.g., 300) is set by us when rasterising for OCR.
The *source* DPI is whatever the scanner captured. When source < render,
PyMuPDF's upsampling is pure interpolation — no added detail — and
OCR quality is capped by the source. Detecting low source DPI enables
targeted super-resolution before the OCR pass.

Extracted from ``src/application/pipeline.py`` so the OCR engine layer
can consume it without importing the whole pipeline module.
"""

from __future__ import annotations


def estimate_page_source_dpi(page) -> int | None:  # noqa: ANN001
    """Best-effort estimate of the native DPI of the largest raster on
    ``page``. Returns ``None`` for all-vector / text-only pages.

    Uses ``page.get_image_info()`` to find the biggest embedded image,
    then back-computes DPI from pixel dimensions vs the bbox in PDF
    points (72 pt = 1 inch).
    """
    try:
        infos = page.get_image_info()
    except (AttributeError, RuntimeError):
        return None
    if not infos:
        return None

    # Use the largest image on the page — main scan, not the logo.
    try:
        biggest = max(
            infos,
            key=lambda d: int(d.get("width", 0)) * int(d.get("height", 0)),
        )
    except (ValueError, KeyError, TypeError):
        return None
    img_w = int(biggest.get("width") or 0)
    img_h = int(biggest.get("height") or 0)
    bbox = biggest.get("bbox") or (0, 0, 0, 0)
    if len(bbox) != 4:
        return None
    page_w_pt = max(1.0, float(bbox[2]) - float(bbox[0]))
    page_h_pt = max(1.0, float(bbox[3]) - float(bbox[1]))
    if img_w <= 0 or img_h <= 0:
        return None
    dpi_x = img_w * 72.0 / page_w_pt
    dpi_y = img_h * 72.0 / page_h_pt
    # Use the SMALLER of the two axes — more conservative, matches the
    # pipeline-level advisory logic and avoids over-estimating DPI on
    # non-square embedded images.
    return int(round(min(dpi_x, dpi_y)))


__all__ = ["estimate_page_source_dpi"]
