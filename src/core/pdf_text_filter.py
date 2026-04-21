"""Strip low-confidence words from a searchable PDF's text layer.

Given a per-page TSV-shaped dict (same keys as
``src.core.confidence_filter`` consumes — ``text``, ``conf``, ``left``,
``top``, ``width``, ``height``, ``block_num``), iterate every word
below the confidence threshold, transform its pixel bounding box into
PDF user-space points, and apply a PyMuPDF redaction that removes the
invisible text overlapping that rectangle without touching the
underlying raster.
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
    redact_noisy_blocks: bool = False,  # noqa: ARG001 — kept for back-compat
) -> int:
    """Redact low-confidence words from every page of ``pdf_path``.

    Args:
        pdf_path: Searchable PDF. Modified in-place.
        tsv_per_page: One TSV-shaped dict per page. Required keys:
            ``text``, ``conf``, ``left``, ``top``, ``width``,
            ``height``. Length MUST equal the PDF page count.
        image_sizes_px: Per-page ``(width_px, height_px)`` used to
            scale pixel bboxes into PDF user-space points.
        min_confidence: Words with ``conf < min_confidence`` get
            redacted. Use the same threshold the results-panel filter
            uses so the two views agree on "what is garbage".
        redact_noisy_blocks: Accepted for backwards compatibility;
            ignored. The engine-specific block heuristic has been
            removed — per-word redaction is the only pass.

    Returns:
        Total number of text regions redacted across all pages. ``0``
        means nothing was redacted (PDF unchanged).

    Raises:
        ValueError: If the ``tsv_per_page`` / ``image_sizes_px`` /
            PDF page counts disagree.
    """
    import fitz  # PyMuPDF; imported lazily so callers without it can skip

    if len(tsv_per_page) != len(image_sizes_px):
        raise ValueError(
            f"tsv_per_page length ({len(tsv_per_page)}) != "
            f"image_sizes_px length ({len(image_sizes_px)})"
        )

    doc = fitz.open(str(pdf_path))
    tmp_path: Path | None = None
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
            # file. Write to a sibling tmp path; swap in after close.
            tmp_path = pdf_path.with_suffix(pdf_path.suffix + ".redacted.tmp")
            doc.save(str(tmp_path), deflate=True)
            logger.info(
                "PDF text-layer filter: removed %d low-conf word region(s) "
                "from %s",
                total_redacted, pdf_path,
            )
        return total_redacted
    finally:
        doc.close()
        if tmp_path is not None:
            tmp_path.replace(pdf_path)


def find_noisy_blocks(
    data: Mapping[str, Sequence[Any]],
    *,
    min_confidence: float,
    min_block_words: int = 3,
    noise_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes of layout blocks that are majority-noise.

    A block qualifies when at least ``noise_ratio`` of its words are
    below ``min_confidence`` AND it has at least ``min_block_words``
    words. Returned bboxes are ``(left, top, width, height)`` in
    pixel coordinates.

    Retained as a pure helper for callers that still want the
    block-level diagnostic; the PDF redactor no longer uses it.
    """
    texts = list(data.get("text", []))
    confs = list(data.get("conf", []))
    blocks = list(data.get("block_num", []))
    lefts = list(data.get("left", []))
    tops = list(data.get("top", []))
    widths = list(data.get("width", []))
    heights = list(data.get("height", []))

    per_block: dict[int, dict[str, Any]] = {}

    n = len(texts)
    for i in range(n):
        word = texts[i]
        if not isinstance(word, str) or not word.strip():
            continue
        try:
            conf = float(confs[i]) if i < len(confs) else -1.0
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        try:
            blk = int(blocks[i])
            x = float(lefts[i])
            y = float(tops[i])
            w = float(widths[i])
            h = float(heights[i])
        except (TypeError, ValueError, IndexError):
            continue

        rec = per_block.setdefault(
            blk,
            {"total": 0, "low": 0, "l": x, "t": y, "r": x + w, "b": y + h},
        )
        rec["total"] += 1
        if conf < min_confidence:
            rec["low"] += 1
        rec["l"] = min(rec["l"], x)
        rec["t"] = min(rec["t"], y)
        rec["r"] = max(rec["r"], x + w)
        rec["b"] = max(rec["b"], y + h)

    noisy: list[tuple[int, int, int, int]] = []
    for rec in per_block.values():
        if rec["total"] < min_block_words:
            continue
        if rec["low"] / rec["total"] < noise_ratio:
            continue
        noisy.append((
            int(rec["l"]),
            int(rec["t"]),
            int(rec["r"] - rec["l"]),
            int(rec["b"] - rec["t"]),
        ))
    return noisy


def _redact_page(
    page: Any,
    *,
    data: Mapping[str, Sequence[Any]],
    image_width_px: int,
    image_height_px: int,
    min_confidence: float,
) -> int:
    """Add + apply per-word redactions for one page. Returns the count."""
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

        page.add_redact_annot(
            rect, text=None, fill=None, cross_out=False,
        )
        added += 1

    if added > 0:
        # images=0, graphics=0, text=0 ⇒ "remove text inside redact
        # rects, never touch image/graphics content".
        page.apply_redactions(images=0, graphics=0, text=0)
    return added


__all__ = ["filter_pdf_text_layer", "find_noisy_blocks"]
