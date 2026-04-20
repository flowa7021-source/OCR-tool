"""Per-block PSM rescue: re-OCR low-confidence layout blocks with a
forgiving page-segmentation mode.

Tesseract's default PSM (AUTO = 3) runs full layout analysis and
sometimes misreads dense tables / invoice grids — the analyser
fragments a single table row into multiple blocks, or tries to
detect columns that aren't there, and the per-word confidence
collapses. PSM=SINGLE_BLOCK (6) skips layout analysis and treats
the crop as one continuous block of text; on table-heavy content
that recovers the individual cell text the default pass mangled.

This module runs AFTER the primary image_to_data pass:

  1. Group the TSV rows by ``block_num``.
  2. For each block with mean per-word confidence below 60 % AND
     ≥ 3 words, crop the block's bounding box out of the page
     image and re-OCR it with ``--psm 6``.
  3. If the re-OCR result has higher mean confidence (clear lift),
     swap the block's TSV entries with the new ones in-place.

Zero risk of a regression — a failed rescue leaves the original
TSV untouched, same fail-open contract as the word-level rescues.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - cv2 is a real dep
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - numpy is a real dep
    np = None  # type: ignore[assignment]

try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI lint jobs without tesseract deps
    pytesseract = None  # type: ignore[assignment]


#: A block qualifies for re-OCR rescue when its mean per-word
#: confidence sits below this value. 60 % is the empirical
#: boundary: above that the block is either correct or borderline-
#: but-trusted (word-level rescues handle that), below that the
#: block is fragmented / misread and a PSM-level re-OCR has a
#: fighting chance to recover the cell text.
_BLOCK_RESCUE_MAX_MEAN_CONFIDENCE: float = 60.0

#: Minimum word count for a block to be rescueable. Short blocks
#: (1-2 words) are usually page markers or stamp fragments; the
#: word-level rescues cover them. Rescue at block level only when
#: there's enough text to benefit from layout-level retuning.
_BLOCK_RESCUE_MIN_WORDS: int = 3

#: Minimum confidence lift for a block rescue to swap. Same
#: threshold as the word-level image rescues — a small conf win
#: is measurement noise in the noisy-block regime.
_BLOCK_MIN_CONFIDENCE_LIFT: float = 5.0

#: Padding around the block bbox before cropping. Wider than the
#: per-word rescue padding because blocks already include their
#: own inter-word spacing; the extra 12 px covers the thin ruled
#: lines that table cells sit inside.
_BLOCK_BBOX_PADDING_PX: int = 12


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_block_aggregates(
    data: Mapping[str, list[Any]],
) -> dict[int, dict[str, Any]]:
    """Group TSV rows by ``block_num``; compute per-block aggregates.

    Returns ``{block_num: {"indices": [...], "mean_conf": ..., "bbox":
    (x1, y1, x2, y2), "word_count": ...}}``. ``bbox`` is the union of
    every word's bounding box in the block, so the rescue crop
    captures the whole layout region the primary pass put together.
    """
    texts = data.get("text", [])
    confs = data.get("conf", [])
    blocks = data.get("block_num", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    per_block: dict[int, dict[str, Any]] = {}
    n = min(len(texts), len(confs), len(blocks))
    for i in range(n):
        text = texts[i] if isinstance(texts[i], str) else ""
        if not text.strip():
            continue
        conf = _safe_float(confs[i])
        if conf < 0:
            continue
        block_num = _safe_int(blocks[i])
        left = _safe_int(lefts[i])
        top = _safe_int(tops[i])
        width = _safe_int(widths[i])
        height = _safe_int(heights[i])
        if width <= 0 or height <= 0:
            continue
        right = left + width
        bottom = top + height
        entry = per_block.setdefault(
            block_num,
            {
                "indices": [],
                "conf_sum": 0.0,
                "word_count": 0,
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom,
            },
        )
        entry["indices"].append(i)
        entry["conf_sum"] += conf
        entry["word_count"] += 1
        entry["x1"] = min(entry["x1"], left)
        entry["y1"] = min(entry["y1"], top)
        entry["x2"] = max(entry["x2"], right)
        entry["y2"] = max(entry["y2"], bottom)

    # Finalize: compute mean, emit compact bbox tuple.
    finalized: dict[int, dict[str, Any]] = {}
    for block_num, entry in per_block.items():
        if entry["word_count"] == 0:
            continue
        mean_conf = entry["conf_sum"] / entry["word_count"]
        finalized[block_num] = {
            "indices": entry["indices"],
            "mean_conf": mean_conf,
            "word_count": entry["word_count"],
            "bbox": (entry["x1"], entry["y1"], entry["x2"], entry["y2"]),
        }
    return finalized


def _select_rescue_candidates(
    aggregates: dict[int, dict[str, Any]],
) -> list[int]:
    """Return block_num keys that qualify for rescue.

    Gate: mean_conf below :data:`_BLOCK_RESCUE_MAX_MEAN_CONFIDENCE`
    AND word_count at least :data:`_BLOCK_RESCUE_MIN_WORDS`. Stamp
    blocks (handled by ``redact_noisy_blocks`` and the handwritten-
    marker system) are NOT additionally rescued here — a rescue on
    a truly-handwritten block produces more noise, not less.
    """
    return [
        block_num
        for block_num, info in aggregates.items()
        if info["mean_conf"] < _BLOCK_RESCUE_MAX_MEAN_CONFIDENCE
        and info["word_count"] >= _BLOCK_RESCUE_MIN_WORDS
    ]


def _crop_block(image: Any, bbox: tuple[int, int, int, int]) -> Any | None:
    """Pad + clamp the bbox and slice the crop out of ``image``.

    Returns ``None`` when the image shape / bbox is unusable, so
    callers fail open.
    """
    if image is None or np is None:
        return None
    try:
        h, w = image.shape[:2]
    except (AttributeError, IndexError, ValueError):
        return None
    x1, y1, x2, y2 = bbox
    pad = _BLOCK_BBOX_PADDING_PX
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _reocr_block_crop(
    crop: Any, lang: str, base_config: str,
) -> dict[str, list[Any]] | None:
    """Re-OCR the block crop with PSM=SINGLE_BLOCK.

    Replaces the user's PSM (usually AUTO) in ``base_config`` with
    PSM=6. Returns the full ``image_to_data`` TSV dict on success,
    ``None`` on any Tesseract / pytesseract error.
    """
    if pytesseract is None:  # pragma: no cover
        return None
    # Strip any existing --psm flag (AND its value token, since it
    # follows --psm as a separate argument) before appending our own.
    # Keeps other user-specified -c params like
    # ``preserve_interword_spaces`` intact.
    cleaned_parts: list[str] = []
    skip_next = False
    for tok in base_config.split():
        if skip_next:
            skip_next = False
            continue
        if tok == "--psm":
            skip_next = True
            continue
        if tok.startswith("--psm"):
            continue
        cleaned_parts.append(tok)
    config = " ".join(cleaned_parts + ["--psm", "6"])

    try:
        return pytesseract.image_to_data(
            crop,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("block PSM rescue: image_to_data failed: %s", exc)
        return None


def _tsv_mean_conf(tsv: Mapping[str, list[Any]]) -> tuple[float, int]:
    """Return ``(mean_conf, word_count)`` over real words in ``tsv``."""
    total = 0.0
    count = 0
    for word, conf in zip(
        tsv.get("text", []), tsv.get("conf", []), strict=False,
    ):
        if not isinstance(word, str) or not word.strip():
            continue
        c = _safe_float(conf)
        if c < 0:
            continue
        total += c
        count += 1
    if count == 0:
        return 0.0, 0
    return total / count, count


def _tsv_reconstructed_text(tsv: Mapping[str, list[Any]]) -> str:
    """Serialise the rescue TSV back to text for a side-by-side
    confidence comparison — same ``(line_num)`` grouping as the
    final reconstruct pass, so the comparison reflects the shape
    of text we'd actually deliver."""
    rows: list[tuple[int, int, str]] = []
    texts = tsv.get("text", [])
    line_nums = tsv.get("line_num", [])
    word_nums = tsv.get("word_num", [])
    for i, word in enumerate(texts):
        if not isinstance(word, str) or not word.strip():
            continue
        rows.append((
            _safe_int(line_nums[i] if i < len(line_nums) else 0),
            _safe_int(word_nums[i] if i < len(word_nums) else 0),
            word.strip(),
        ))
    rows.sort()
    # Join words in the same line with space, lines with newline.
    out_lines: list[str] = []
    current_line: int | None = None
    buf: list[str] = []
    for line, _wn, word in rows:
        if line != current_line:
            if buf:
                out_lines.append(" ".join(buf))
            current_line = line
            buf = [word]
        else:
            buf.append(word)
    if buf:
        out_lines.append(" ".join(buf))
    return "\n".join(out_lines)


def rescue_low_conf_blocks(
    data: dict[str, list[Any]],
    image: Any,
    lang: str,
    base_config: str,
) -> int:
    """Re-OCR low-confidence blocks in-place; return count of rescued
    blocks.

    Mutates ``data`` — specifically ``data["text"]`` and
    ``data["conf"]`` — replacing entries for rescued blocks with
    the new PSM=SINGLE_BLOCK readings. The other TSV columns
    (``left`` / ``top`` / etc) are left unchanged because the
    rescue operates on the CONTENT of a block while keeping the
    block's position in the page invariant; the bboxes Tesseract
    emits for the crop are relative to the crop, not the page,
    and re-mapping them adds complexity without a clear gain for
    the downstream confidence / word-conf filter.

    Returns:
        Number of blocks whose content was swapped. Zero when no
        block qualified OR no rescue cleared the lift threshold.
    """
    if image is None or cv2 is None or np is None:
        return 0
    aggregates = _compute_block_aggregates(data)
    candidates = _select_rescue_candidates(aggregates)
    if not candidates:
        return 0

    rescued = 0
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for block_num in candidates:
        info = aggregates[block_num]
        crop = _crop_block(image, info["bbox"])
        if crop is None:
            continue
        new_tsv = _reocr_block_crop(crop, lang, base_config)
        if new_tsv is None:
            continue
        new_mean, new_count = _tsv_mean_conf(new_tsv)
        if new_count == 0:
            continue
        if new_mean < info["mean_conf"] + _BLOCK_MIN_CONFIDENCE_LIFT:
            continue

        # Replace block's TSV entries with new PSM=6 readings. We
        # write the new words into the existing index slots in the
        # SAME order, and blank out any remaining indices from the
        # old block so no old stale text survives.
        new_words = [
            (w.strip(), str(_safe_float(c)))
            for w, c in zip(
                new_tsv.get("text", []), new_tsv.get("conf", []),
                strict=False,
            )
            if isinstance(w, str) and w.strip() and _safe_float(c) >= 0
        ]
        old_indices = info["indices"]
        # Overwrite old entries with new words; pad new words with
        # placeholder rows if we have fewer new words than old.
        for slot, (word, conf_str) in zip(
            old_indices, new_words, strict=False,
        ):
            texts[slot] = word
            confs[slot] = conf_str
        # Any leftover old slots get cleared — setting text to empty
        # and conf to -1 makes ``reconstruct_text_from_tsv`` skip them.
        if len(old_indices) > len(new_words):
            for slot in old_indices[len(new_words):]:
                texts[slot] = ""
                confs[slot] = "-1"

        logger.info(
            "block PSM rescue: block %d mean_conf %.1f → %.1f (%d words)",
            block_num, info["mean_conf"], new_mean, new_count,
        )
        rescued += 1
    return rescued


__all__ = ["rescue_low_conf_blocks"]
