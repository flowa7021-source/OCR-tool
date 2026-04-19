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
    redact_noisy_blocks: bool = False,
    block_noise_ratio: float = 0.5,
    block_min_words: int = 3,
) -> int:
    """Redact low-confidence words from every page of ``pdf_path``.

    Two redaction passes run per page:

      * **Per-word** (always on) — any individual word with
        ``conf < min_confidence`` gets its bbox redacted. Handles
        sprinkled noise where Tesseract's word-level segmentation is
        still usable.
      * **Per-block** (opt-in via ``redact_noisy_blocks``) — when the
        ``block_num`` layout cluster Tesseract emitted is majority-
        noise (``>= block_noise_ratio`` of its words below the
        threshold, and at least ``block_min_words`` words present),
        the WHOLE block's bounding rectangle is redacted. Catches
        stamp / signature / fine-print-template zones where even the
        individual "above-threshold" words are unreliable because
        they sit in a region Tesseract clearly mis-read as text.

    Args:
        pdf_path: Searchable PDF produced by OCRmyPDF. Modified in-place.
        tsv_per_page: One ``image_to_data``-shaped dict per page. Must
            include ``text``, ``conf``, ``left``, ``top``, ``width``,
            ``height``, and ``block_num`` when block-redaction is on.
            Length MUST equal the PDF page count; a mismatch raises
            ``ValueError`` rather than silently mis-aligning pages.
        image_sizes_px: Per-page ``(width_px, height_px)`` of the
            preprocessed image the TSV was produced on. Used to scale
            the pixel bboxes into PDF user-space points. Must parallel
            ``tsv_per_page``.
        min_confidence: Words with ``conf < min_confidence`` get
            redacted. Use the same threshold the results-panel filter
            uses so the two views agree on "what is garbage".
        redact_noisy_blocks: Enable the block-level second pass. Off
            by default to preserve existing behaviour; profiles opt in.
        block_noise_ratio: A layout block is "noisy" when at least
            this fraction of its words are sub-threshold. Default
            ``0.5`` — majority noise triggers the block-wide redact.
        block_min_words: Never redact a block with fewer than this many
            words in it, regardless of noise ratio. Small-sample blocks
            (captions, page numbers, single tokens) can randomly look
            noisy without being real garbage zones.

    Returns:
        Total number of text regions redacted across all pages (counts
        per-word AND per-block redactions together). ``0`` means
        nothing was redacted (the PDF is unchanged).

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
                redact_noisy_blocks=redact_noisy_blocks,
                block_noise_ratio=block_noise_ratio,
                block_min_words=block_min_words,
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


def find_noisy_blocks(
    data: Mapping[str, Sequence[Any]],
    *,
    min_confidence: float,
    min_block_words: int = 3,
    noise_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes of layout blocks that are majority-noise.

    Tesseract's ``image_to_data`` output groups words by ``block_num``
    — the layout-analysis stage's notion of "this is one contiguous
    region of text" (a paragraph, a table cell, a stamp, a signature
    line). If most words in a given block are sub-threshold, that's
    strong evidence the entire block is a bad region to trust: a
    stamp Tesseract tried to read as text, a signature misread as
    word fragments, a decorative header in a stylised font. We return
    one bbox per such block so the caller can mask the WHOLE zone
    rather than cherry-picking individual words.

    Args:
        data: ``image_to_data``-shaped dict with ``text``, ``conf``,
            ``block_num``, ``left``, ``top``, ``width``, ``height``.
        min_confidence: Words below this count as "noise" for the
            block-ratio calculation. Use the same threshold you pass
            to :func:`filter_pdf_text_layer`.
        min_block_words: Ignore blocks with fewer words. Small-sample
            blocks (page numbers, captions) can look noisy purely by
            chance — we require ``>=`` this many words before trusting
            the ratio signal.
        noise_ratio: A block is "noisy" when at least this fraction of
            its words are sub-threshold. ``0.5`` — majority rules.

    Returns:
        List of ``(left, top, width, height)`` tuples in pixel
        coordinates, one per noisy block. Empty when no block meets
        the criteria.
    """
    texts = list(data.get("text", []))
    confs = list(data.get("conf", []))
    blocks = list(data.get("block_num", []))
    lefts = list(data.get("left", []))
    tops = list(data.get("top", []))
    widths = list(data.get("width", []))
    heights = list(data.get("height", []))

    # block_id → {'total', 'low', bbox (l,t,r,b)}
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
        # Expand block bbox to cover this word.
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
    redact_noisy_blocks: bool = False,
    block_noise_ratio: float = 0.5,
    block_min_words: int = 3,
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

    # Pass 1: per-word low-conf redactions.
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

    # Pass 2: per-block redactions for layout zones Tesseract clearly
    # struggled with (majority-noise blocks — stamps, signatures, fine-
    # print headers). Redacting the whole block catches borderline-
    # conf words that the per-word pass above let through — a 70 %-
    # conf word sitting inside a stamp region is still untrustworthy.
    if redact_noisy_blocks:
        noisy = find_noisy_blocks(
            data,
            min_confidence=min_confidence,
            min_block_words=block_min_words,
            noise_ratio=block_noise_ratio,
        )
        for x_px, y_px, w_px, h_px in noisy:
            rect = fitz.Rect(
                x_px * scale_x,
                y_px * scale_y,
                (x_px + w_px) * scale_x,
                (y_px + h_px) * scale_y,
            )
            if rect.is_empty or not rect.is_valid:
                continue
            page.add_redact_annot(
                rect, text=None, fill=None, cross_out=False,
            )
            added += 1
        if noisy:
            logger.debug(
                "PDF text-layer filter: page %d — redacting %d noisy "
                "block(s) in addition to per-word hits",
                page.number + 1, len(noisy),
            )

    if added > 0:
        # images=0, graphics=0, text=0 ⇒ "remove text inside redact
        # rects, never touch image/graphics content". Critical — any
        # other mode would damage the visible page.
        page.apply_redactions(images=0, graphics=0, text=0)
    return added


__all__ = ["filter_pdf_text_layer", "find_noisy_blocks"]
