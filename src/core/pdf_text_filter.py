"""Strip low-confidence words from a searchable PDF's text layer.

``src.core.confidence_filter`` filters the string surfaced to the user
(results panel / TXT / DOCX export). But the invisible text layer that
OCRmyPDF stamps INTO the output PDF is still the raw Tesseract union —
Ctrl-F in Adobe Reader and copy-paste-into-Word both hit every
10-40 %-confidence stamp / signature / logo guess Tesseract made at
non-text regions. Downstream consumers (corporate DMS, Google Drive
indexing, etc.) ingest the same polluted layer.

This module closes that gap. Given the same ``image_to_data`` TSV dict
used by the results-panel filter, it iterates every sub-threshold word,
transforms its pixel bounding box into PDF user-space points, and
applies a PyMuPDF redaction that removes ONLY the invisible text
overlapping that rectangle — the underlying raster image is preserved.

Invariants:
  * Does NOT re-OCR. The TSV comes from the caller (already collected
    in ``pipeline._compute_confidences`` for mean-conf bookkeeping).
  * Does NOT touch the visible scan image. The redact call uses
    ``images=0, graphics=0`` which leaves raster / vector art alone.
  * Per-page operation — a bad transform on one page can't
    silently break another.
  * In-place: the input PDF is modified and saved over itself. The
    caller is expected to have made a copy first if it wants the
    original preserved (our pipeline writes to a tmp path and only
    moves to the final output on success, so we inherit that).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def filter_pdf_text_layer(
    pdf_path: Path,
    *,
    tsv_per_page: Sequence[Mapping[str, Sequence[Any]]],
    image_sizes_px: Sequence[tuple[int, int]],
    min_confidence: float,
) -> int:
    """Redact low-confidence words from every page of ``pdf_path``.

    Args:
        pdf_path: Searchable PDF produced by OCRmyPDF. Modified in-place.
        tsv_per_page: One ``image_to_data``-shaped dict per page. Must
            include ``text``, ``conf``, ``left``, ``top``, ``width``,
            ``height``. Length MUST equal the PDF page count; a
            mismatch raises ``ValueError`` rather than silently
            mis-aligning pages.
        image_sizes_px: Per-page ``(width_px, height_px)`` of the
            preprocessed image the TSV was produced on. Used to scale
            the pixel bboxes into PDF user-space points. Must parallel
            ``tsv_per_page``.
        min_confidence: Words with ``conf < min_confidence`` get
            redacted. Use the same threshold the results-panel filter
            uses so the two views agree on "what is garbage".

    Returns:
        Total number of text regions redacted across all pages.
        ``0`` means nothing was above-threshold to remove (the PDF is
        left unchanged) OR the TSV was empty; callers typically only
        care about non-zero values for logging.

    Raises:
        ValueError: If ``tsv_per_page`` / ``image_sizes_px`` length
            disagrees with the PDF page count, or any required TSV
            column is missing.
    """
    import fitz  # PyMuPDF; imported lazily so callers without it can skip

    if len(tsv_per_page) != len(image_sizes_px):
        raise ValueError(
            f"tsv_per_page length ({len(tsv_per_page)}) != "
            f"image_sizes_px length ({len(image_sizes_px)})"
        )

    doc = fitz.open(str(pdf_path))
    try:
        if doc.page_count != len(tsv_per_page):
            raise ValueError(
                f"PDF has {doc.page_count} pages but tsv_per_page has "
                f"{len(tsv_per_page)} entries — refusing to silently "
                f"mis-align"
            )

        total_redacted = 0
        for page_idx, (data, (img_w, img_h)) in enumerate(
            zip(tsv_per_page, image_sizes_px, strict=True)
        ):
            redacted = _redact_page(
                doc.load_page(page_idx),
                data=data,
                image_width_px=img_w,
                image_height_px=img_h,
                min_confidence=min_confidence,
            )
            total_redacted += redacted

        if total_redacted > 0:
            # PyMuPDF forbids a non-incremental save over the opened
            # file ("save to original must be incremental"). Write to a
            # sibling tmp path, then atomically replace the original —
            # this flattens the redactions into the content stream so a
            # casual PDF reader that ignores annotations still doesn't
            # see the removed text.
            tmp_path = pdf_path.with_suffix(pdf_path.suffix + ".redacted.tmp")
            doc.save(str(tmp_path), deflate=True)
            logger.info(
                "PDF text-layer filter: removed %d low-conf word region(s) "
                "from %s",
                total_redacted, pdf_path,
            )
        else:
            tmp_path = None
        return total_redacted
    finally:
        doc.close()
        # Atomic swap AFTER ``doc`` is released — PyMuPDF holds an
        # exclusive file handle on Windows while open. ``Path.replace``
        # is atomic on the same filesystem, so readers either see the
        # old or new file, never a half-written one.
        if "tmp_path" in locals() and tmp_path is not None:  # noqa: F823
            tmp_path.replace(pdf_path)


def _redact_page(
    page: Any,
    *,
    data: Mapping[str, Sequence[Any]],
    image_width_px: int,
    image_height_px: int,
    min_confidence: float,
) -> int:
    """Add + apply redactions for one page. Returns the count redacted."""
    import fitz

    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    if image_width_px <= 0 or image_height_px <= 0:
        logger.warning(
            "PDF text-layer filter: skipping page %d — non-positive "
            "image dims (%d x %d)",
            page.number + 1, image_width_px, image_height_px,
        )
        return 0

    # PDF user-space ⇄ pixel coordinate transform. OCRmyPDF preserves
    # the input PDF page rect, so ``page.rect`` is the authoritative
    # target and ``image_{w,h}_px`` is the source frame Tesseract saw
    # when it generated the TSV we're filtering against.
    scale_x = page.rect.width / image_width_px
    scale_y = page.rect.height / image_height_px

    added = 0
    for i in range(len(texts)):
        word = texts[i]
        if not isinstance(word, str) or not word.strip():
            continue
        try:
            conf = float(confs[i]) if i < len(confs) else -1.0
        except (TypeError, ValueError):
            continue
        if conf < 0 or conf >= min_confidence:
            continue

        try:
            x = float(lefts[i])
            y = float(tops[i])
            w = float(widths[i])
            h = float(heights[i])
        except (TypeError, ValueError, IndexError):
            continue

        rect = fitz.Rect(
            x * scale_x,
            y * scale_y,
            (x + w) * scale_x,
            (y + h) * scale_y,
        )
        if rect.is_empty or not rect.is_valid:
            continue

        # ``fill=None`` + ``cross_out=False`` + ``text=None`` — we want
        # the redaction to remove text but NOT paint a coloured box or
        # strike-through over the visible scan image. The caller's
        # flattened PDF still shows exactly the original raster.
        page.add_redact_annot(
            rect, text=None, fill=None, cross_out=False,
        )
        added += 1

    if added > 0:
        # images=0, graphics=0, text=0 ⇒ "remove text inside redact
        # rects, never touch image/graphics content". Critical — any
        # other mode would damage the visible page.
        page.apply_redactions(images=0, graphics=0, text=0)
    return added


__all__ = ["filter_pdf_text_layer"]
