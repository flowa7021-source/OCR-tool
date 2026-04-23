"""Lightweight handwriting vs printed-text region classifier.

Our ТН/УПД scans mix pre-printed form fields (large, regular, high-
frequency glyphs) with handwritten fills (irregular stroke thickness,
wider stroke spacing, curved baselines). EasyOCR's CRNN recogniser is
trained on printed text and struggles on handwriting — its output is
often a high-confidence gibberish string.

This module classifies a word-crop as ``printed`` or ``handwritten``
using three cheap statistical features:

1. Stroke thickness variance — printed text has uniform ink, handwriting
   varies.
2. Baseline stability — printed text sits on a predictable line;
   handwriting drifts.
3. Edge density distribution — printed glyphs have sharp edges,
   handwriting has softer, thicker ones.

A hand-tuned threshold on the combined feature vector classifies each
crop. No trained model required — the features themselves are
discriminative enough for a coarse binary gate.

Downstream:
- Printed crops go to the regular CRNN.
- Handwritten crops get skipped (or routed to a separate model —
  TrOCR-handwritten, if the user wired one in). For now we just flag
  them so the pipeline can mark those words as
  ``confidence=0, text=""`` rather than emit confident garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HandwritingFeatures:
    stroke_variance: float    # stroke-thickness standard deviation
    baseline_slope: float     # absolute slope of lower edge (rad)
    edge_density: float       # fraction of pixels that are edges
    score: float              # 0 = printed, 1 = handwritten

    def as_dict(self) -> dict[str, float]:
        return {
            "stroke_variance": round(self.stroke_variance, 4),
            "baseline_slope": round(self.baseline_slope, 4),
            "edge_density": round(self.edge_density, 4),
            "score": round(self.score, 3),
        }


def _stroke_variance(binary: np.ndarray) -> float:
    """Measure stroke-width variability via distance transform.

    For each ink pixel, the distance transform gives the distance to
    the nearest background pixel — twice this is the local stroke
    width. Variance of those widths across the image is low for
    printed text (all strokes ≈ 1-2 px at OCR resolution), high for
    handwriting (can range 1-5 px in one word).
    """
    inv = cv2.bitwise_not(binary)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    ink_mask = inv > 0
    if not ink_mask.any():
        return 0.0
    widths = 2.0 * dt[ink_mask]
    return float(np.std(widths))


def _baseline_slope(binary: np.ndarray) -> float:
    """Estimate the slope of the lower edge (baseline) using its
    bottom-most ink pixel per column, fitted via least-squares.
    """
    inv = cv2.bitwise_not(binary)
    h, w = inv.shape
    bottoms = np.zeros(w, dtype=np.float32)
    any_ink = np.zeros(w, dtype=bool)
    for x in range(w):
        col = np.where(inv[:, x] > 0)[0]
        if len(col):
            bottoms[x] = float(col.max())
            any_ink[x] = True
    if any_ink.sum() < 8:
        return 0.0
    xs = np.arange(w, dtype=np.float32)[any_ink]
    ys = bottoms[any_ink]
    # Least-squares line y = m × x + b
    n = len(xs)
    m = (n * (xs * ys).sum() - xs.sum() * ys.sum()) / (
        n * (xs * xs).sum() - xs.sum() ** 2 + 1e-9
    )
    return float(abs(m))


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels above Canny edge detection threshold."""
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / (255.0 * edges.size))


def classify_crop(
    crop: np.ndarray,
    threshold: float = 0.55,
) -> tuple[bool, HandwritingFeatures]:
    """Classify ``crop`` (grayscale HxW uint8) as handwritten or printed.

    Returns ``(is_handwritten, features)``. ``is_handwritten`` is True
    when the combined score is ≥ ``threshold``. Defaults tuned on our
    corpus; lower threshold classifies more as handwritten (fewer
    false-printed), higher = more conservative.
    """
    if crop.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D array, got shape {crop.shape}",
        )
    if crop.size < 64:
        # Too small to measure robustly.
        feats = HandwritingFeatures(
            stroke_variance=0.0, baseline_slope=0.0,
            edge_density=0.0, score=0.0,
        )
        return (False, feats)

    _, binary = cv2.threshold(
        crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    sv = _stroke_variance(binary)
    bs = _baseline_slope(binary)
    ed = _edge_density(crop)

    # Heuristic blend — coefficients tuned by eye-balling a handful of
    # real crops. Pre-production; refine with a labeled training set.
    # Normalise each feature to ~0..1 empirically:
    sv_n = min(1.0, sv / 3.0)        # printed: 0.3-0.8, handwritten: 1.5+
    bs_n = min(1.0, bs / 0.15)       # printed: 0-0.05, handwritten: 0.1+
    ed_n = max(0.0, 1.0 - ed * 10.0)  # printed has higher edge density
    score = 0.5 * sv_n + 0.3 * bs_n + 0.2 * ed_n
    feats = HandwritingFeatures(
        stroke_variance=sv,
        baseline_slope=bs,
        edge_density=ed,
        score=score,
    )
    return (score >= threshold, feats)


__all__ = ["classify_crop", "HandwritingFeatures"]
