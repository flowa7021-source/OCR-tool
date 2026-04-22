"""Confidence-based retry: re-run OCR on low-confidence word crops with
alternative preprocessing.

The stock EasyOCR pipeline sees each page once with a single set of
preprocessing params. For words where the recogniser was uncertain
(low softmax confidence), trying a different crop-level treatment
often rescues 20-40% of them:

- 2× bicubic upscale — sharper edges, bigger strokes for the CRNN
- Contrast stretch (histogram endpoints) — fixes faded text
- Morphological thickening — fills in broken strokes on low-DPI scans
- CLAHE on the crop alone — local contrast without over-sharpening
  the whole page

This module is engine-agnostic: pass in any callable
``recognise(arr: np.ndarray) -> list[(bbox, text, conf)]`` and the
module wires up the retry loop. For EasyOCR, wrap
``lambda arr: reader.readtext(arr, detail=1, paragraph=False,
allowlist=..., **craft_kwargs)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Callable contract: takes a grayscale crop, returns a list of
# (bbox, text, conf) triples compatible with EasyOCR's readtext output.
_BBox = list[list[float]]
_RawResult = tuple[_BBox, str, float]
Recogniser = Callable[[np.ndarray], list[_RawResult]]


@dataclass
class RetryStats:
    """Counters for observability — logged per-page."""
    attempted: int = 0
    improved: int = 0
    unchanged: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "improved": self.improved,
            "unchanged": self.unchanged,
            "failed": self.failed,
        }


def _upscale_bicubic(crop: np.ndarray, factor: float = 2.0) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(
        crop, (int(w * factor), int(h * factor)),
        interpolation=cv2.INTER_CUBIC,
    )


def _stretch_contrast(crop: np.ndarray,
                      pct_low: float = 2.0,
                      pct_high: float = 98.0) -> np.ndarray:
    lo = float(np.percentile(crop, pct_low))
    hi = float(np.percentile(crop, pct_high))
    if hi - lo < 10:  # near-flat crop — don't touch
        return crop
    out = (crop.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _clahe_local(crop: np.ndarray,
                 clip_limit: float = 3.0,
                 tile_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    return clahe.apply(crop)


def _morph_thicken(crop: np.ndarray, kernel: int = 2) -> np.ndarray:
    # Dilate on a polarity-inverted image so that text (dark) grows wider.
    inv = cv2.bitwise_not(crop)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    dilated = cv2.dilate(inv, k)
    return cv2.bitwise_not(dilated)


# Ordered list of (name, transform). Tried until a higher-conf result
# appears, or all variants exhausted.
_VARIANTS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("upscale2x",  lambda c: _upscale_bicubic(c, 2.0)),
    ("stretch",    _stretch_contrast),
    ("clahe",      _clahe_local),
    ("thicken",    _morph_thicken),
    ("upscale3x",  lambda c: _upscale_bicubic(c, 3.0)),
)


def _clip_crop(
    page: np.ndarray, bbox: _BBox, padding: int = 4,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Extract the crop region from the page, with a small margin.

    Returns the crop + the (y0, x0) page offset so bbox can be mapped
    back into page coordinates after re-recognition.
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(page.shape[1], int(max(xs)) + padding)
    y1 = min(page.shape[0], int(max(ys)) + padding)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return page[y0:y1, x0:x1].copy(), (y0, x0)


def retry_low_confidence(
    page: np.ndarray,
    raw_results: list[_RawResult],
    recognise: Recogniser,
    threshold: float = 0.5,
    min_improvement: float = 0.05,
) -> tuple[list[_RawResult], RetryStats]:
    """Re-run ``recognise`` on each crop whose confidence is below
    ``threshold`` using alternative preprocessing. Returns the merged
    result list (same length + order as input) and a RetryStats counter.

    Args:
        page: full-page grayscale array.
        raw_results: outputs from the first OCR pass.
        recognise: callable that runs recognition on a crop.
        threshold: retry if original conf < this.
        min_improvement: only replace if new conf ≥ old + this (avoids
            flapping on noise-level confidence differences).
    """
    stats = RetryStats()
    out: list[_RawResult] = []
    for bbox, text, conf in raw_results:
        if conf >= threshold:
            out.append((bbox, text, conf))
            continue
        stats.attempted += 1
        clipped = _clip_crop(page, bbox)
        if clipped is None:
            stats.failed += 1
            out.append((bbox, text, conf))
            continue
        crop, _offset = clipped

        best_text, best_conf = text, conf
        for _variant_name, transform in _VARIANTS:
            try:
                variant = transform(crop)
            except cv2.error as e:  # pragma: no cover — opencv edge cases
                logger.debug("Variant %s failed on crop: %s", _variant_name, e)
                continue
            try:
                cand_raw = recognise(variant)
            except Exception as e:  # pragma: no cover — recogniser failure
                logger.debug("Recogniser failed on variant %s: %s",
                             _variant_name, e)
                continue
            # Merge all candidate texts on the variant crop — take the
            # highest-confidence one. If CRAFT split the crop into more
            # than one box, concatenate their texts in left-to-right order.
            if not cand_raw:
                continue
            cand_raw_sorted = sorted(
                cand_raw, key=lambda r: min(p[0] for p in r[0]),
            )
            cand_text = " ".join(t for _, t, _ in cand_raw_sorted).strip()
            cand_conf = max(c for _, _, c in cand_raw_sorted)
            if cand_conf >= best_conf + min_improvement and cand_text:
                best_text, best_conf = cand_text, cand_conf
        if best_text != text or best_conf > conf:
            stats.improved += 1
            out.append((bbox, best_text, best_conf))
        else:
            stats.unchanged += 1
            out.append((bbox, text, conf))
    return out, stats


__all__ = ["retry_low_confidence", "RetryStats", "Recogniser"]
