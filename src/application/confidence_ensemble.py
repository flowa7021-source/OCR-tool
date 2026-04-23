"""Confidence-weighted per-word ensemble of two OCR runs.

Runs the OCR engine twice on the same page with DIFFERENT preprocessing
profiles (typically ``universal_accurate`` and ``universal_hardscan``),
then merges the two word lists by picking the higher-confidence detection
for each spatial bbox cluster.

Why this works: accurate + hardscan expose the recognizer to different
regimes (no binarisation vs Sauvola, different CLAHE strengths,
different CRAFT thresholds). Words that are easy show up at high conf
in both runs; words that sit at a boundary (say, a smudged digit)
typically score well in ONE run and poorly in the other. Taking
max-conf per bbox gets the best of both without retraining.

Cost: 2× inference time. Worth it for high-value documents or
once-per-job validation passes; too slow for interactive preview.

This module operates on EasyOCR-shape raw results
(``list[(bbox, text, conf)]``) so it's independent of engine internals.
Pipeline-level integration happens in a future iteration (needs a
flag to invoke two engine runs); for now, use directly from a
script — see :mod:`scripts.ensemble_run`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BBox = list[list[float]]
_RawResult = tuple[_BBox, str, float]


@dataclass
class MergeStats:
    total_a: int = 0
    total_b: int = 0
    from_a: int = 0
    from_b: int = 0
    consensus: int = 0
    conflicts: int = 0  # both present, disagree on text

    def to_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in (
            "total_a", "total_b", "from_a", "from_b",
            "consensus", "conflicts",
        )}


def _bbox_center(bbox: _BBox) -> tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _bbox_iou(a: _BBox, b: _BBox) -> float:
    xs_a = [p[0] for p in a]
    ys_a = [p[1] for p in a]
    xs_b = [p[0] for p in b]
    ys_b = [p[1] for p in b]
    ax0, ay0 = min(xs_a), min(ys_a)
    ax1, ay1 = max(xs_a), max(ys_a)
    bx0, by0 = min(xs_b), min(ys_b)
    bx1, by1 = max(xs_b), max(ys_b)
    inter_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_x * inter_y
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_results(
    results_a: list[_RawResult],
    results_b: list[_RawResult],
    iou_threshold: float = 0.3,
) -> tuple[list[_RawResult], MergeStats]:
    """Merge two OCR result lists spatially.

    For each bbox in ``results_a``, find the best-IoU match in
    ``results_b``. If found (IoU ≥ threshold):
      - Take the text with higher confidence.
      - Use the bbox of the winner.
    If no match in B, keep the A result.

    Then add any B-only results (no A overlap) to the output.
    """
    stats = MergeStats(total_a=len(results_a), total_b=len(results_b))
    matched_b: set[int] = set()
    merged: list[_RawResult] = []

    for bbox_a, text_a, conf_a in results_a:
        best_iou = 0.0
        best_idx = -1
        for j, (bbox_b, _, _) in enumerate(results_b):
            if j in matched_b:
                continue
            iou = _bbox_iou(bbox_a, bbox_b)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_b.add(best_idx)
            bbox_b, text_b, conf_b = results_b[best_idx]
            if text_a == text_b:
                stats.consensus += 1
                merged.append((bbox_a, text_a, max(conf_a, conf_b)))
                stats.from_a += 1  # counted once per consensus
            elif conf_a >= conf_b:
                stats.from_a += 1
                if text_a != text_b:
                    stats.conflicts += 1
                merged.append((bbox_a, text_a, conf_a))
            else:
                stats.from_b += 1
                stats.conflicts += 1
                merged.append((bbox_b, text_b, conf_b))
        else:
            merged.append((bbox_a, text_a, conf_a))
            stats.from_a += 1

    # Add B-only results (no A match)
    for j, (bbox_b, text_b, conf_b) in enumerate(results_b):
        if j not in matched_b:
            merged.append((bbox_b, text_b, conf_b))
            stats.from_b += 1

    return merged, stats


__all__ = ["merge_results", "MergeStats"]
