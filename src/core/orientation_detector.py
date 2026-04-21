"""Pre-OCR page-orientation detector (no-op under the EasyOCR engine).

Historically this module ran an OSD pass to detect 90°/180°/270°
rotated pages. The EasyOCR engine handles mildly rotated text
internally and no lightweight pure-OpenCV replacement has proven
reliable, so the module now returns ``0`` (the "already upright"
signal) and logs a DEBUG line — the preprocessor keeps calling it
unchanged. If a robust detector lands later, restore the contract by
returning one of ``{0, 90, 180, 270}``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


#: Retained for backwards compatibility with existing profile JSON
#: (``AutoRotateConfig.min_confidence`` defaults to this).
MIN_ORIENTATION_CONFIDENCE: float = 1.0
_MIN_IMAGE_DIMENSION: int = 64


def _parse_osd_output(osd_text: str) -> tuple[int, float] | None:
    """Parse ``Rotate: <deg>`` + ``Orientation confidence: <f>`` lines.

    Retained as a pure-text helper for callers (and tests) that still
    feed it legacy OSD strings. The live detection path no longer
    calls it.
    """
    rotate: int | None = None
    conf: float | None = None
    for line in osd_text.splitlines():
        if line.startswith("Rotate:"):
            try:
                rotate = int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
        elif line.startswith("Orientation confidence:"):
            try:
                conf = float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    if rotate is None or conf is None:
        return None
    normalised = rotate % 360
    snapped = round(normalised / 90) * 90 % 360
    return snapped, conf


def detect_orientation(
    image: Any,
    *,
    min_confidence: float = MIN_ORIENTATION_CONFIDENCE,  # noqa: ARG001
) -> int | None:
    """Return the degrees to rotate ``image`` so text reads LTR-TTB.

    Currently a no-op: returns ``0`` for any usable image, ``None``
    for tiny / non-array inputs. The caller's rotation branch is a
    no-op on ``0`` so the image flows through unchanged.
    """
    try:
        h, w = image.shape[:2]
    except (AttributeError, IndexError, ValueError):
        return None
    if h < _MIN_IMAGE_DIMENSION or w < _MIN_IMAGE_DIMENSION:
        return None
    logger.debug(
        "orientation_detector: detection disabled under EasyOCR engine; "
        "returning 0 (no rotation)"
    )
    return 0


__all__ = [
    "MIN_ORIENTATION_CONFIDENCE",
    "_parse_osd_output",
    "detect_orientation",
]
