"""Pre-OCR page-orientation detector.

Companion to :mod:`src.core.script_detector`. Where that module picks
the right language for a page, this one picks the right rotation:
detects 90° / 180° / 270° off-axis pages (landscape scanned into a
portrait PDF, upside-down phone snaps, legal-size pages fed
sideways) and returns the amount to rotate so the text reads left-
to-right, top-to-bottom.

Runs BEFORE deskew. The two passes compose cleanly:

  * orientation detector — fixes the "scanner rotated the paper"
    class of errors in 90° increments (the angles deskew cannot
    find because row-variance projection is symmetric under 90°
    turns and Tesseract's LSTM at `-l rus+eng` reads gibberish
    from sideways text).
  * deskew — applies the residual ±5° tilt correction.

Design choices mirror :mod:`script_detector`:

* One Tesseract OSD call per page (~50 ms at 150 DPI).
* ``pytesseract.image_to_osd`` is the source of truth — reuses the
  same model we already ship.
* Fails open (returns 0°) so OSD errors never block the pipeline.
* Confidence floor (:data:`MIN_ORIENTATION_CONFIDENCE`) below which
  we prefer a potentially-wrong 0° to a confidently-wrong 180° —
  Tesseract can read rightside-up text on a sideways page more
  gracefully than upside-down text on a sideways flip.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI lint jobs without tesseract deps
    pytesseract = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


#: Below this ``Orientation confidence`` value reported by Tesseract
#: OSD we refuse to rotate. OSD reports orientation confidence on a
#: ~0–10 scale; clean text-heavy pages typically report 3+, while
#: logo-only / stamp-only / heavily-blank pages report 0–1 with
#: random orientation guesses. 1.0 is tight enough to filter the
#: random-guess regime while still catching legitimate sideways
#: scans (which consistently report 2+).
MIN_ORIENTATION_CONFIDENCE: float = 1.0
_MIN_IMAGE_DIMENSION: int = 64


def _parse_osd_output(osd_text: str) -> tuple[int, float] | None:
    """Extract ``(rotation_degrees, orientation_confidence)`` from OSD.

    Returns ``None`` when any required line is missing or malformed
    — callers treat that as "don't rotate".
    """
    rotate: int | None = None
    conf: float | None = None
    for line in osd_text.splitlines():
        # Tesseract reports the rotation needed to straighten via the
        # ``Rotate:`` line (the ``Orientation in degrees:`` line is
        # the current orientation, which is less useful — we'd have
        # to negate it). The two fields always agree modulo 360 but
        # we take ``Rotate:`` for clarity.
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
    # Normalise to the canonical set {0, 90, 180, 270}. Tesseract has
    # been observed to emit negative values or values ≥ 360 on
    # pathological inputs.
    normalised = rotate % 360
    # Snap to the nearest 90° — Tesseract OSD only ever means to
    # report 90° increments; any off-grid number is measurement noise.
    snapped = round(normalised / 90) * 90 % 360
    return snapped, conf


def detect_orientation(
    image: Any,
    *,
    min_confidence: float = MIN_ORIENTATION_CONFIDENCE,
) -> int | None:
    """Detect how many degrees to rotate ``image`` so text reads LTR-TTB.

    Returns:
        One of ``0``, ``90``, ``180``, ``270`` on a confident
        detection. ``None`` when OSD failed, confidence was below
        ``min_confidence``, or the image is too small to analyse.
        ``0`` is a legitimate "already upright" answer and callers
        should treat it as such (no rotation needed).
    """
    try:
        h, w = image.shape[:2]
    except (AttributeError, IndexError, ValueError):
        return None
    if h < _MIN_IMAGE_DIMENSION or w < _MIN_IMAGE_DIMENSION:
        return None
    if pytesseract is None:  # pragma: no cover
        return None
    try:
        osd_text = pytesseract.image_to_osd(image)
    except Exception as exc:  # noqa: BLE001 — pytesseract raises TesseractError
        # Common reasons: too few glyphs on the page, a photo of a
        # stamp, a QR-code-only page. Not an error we need to surface.
        logger.debug("orientation_detector: OSD failed (%s)", exc)
        return None
    parsed = _parse_osd_output(osd_text)
    if parsed is None:
        logger.debug("orientation_detector: OSD output unparseable")
        return None
    rotate, confidence = parsed
    if confidence < min_confidence:
        logger.debug(
            "orientation_detector: rotate=%d but confidence %.2f < %.2f; "
            "leaving image alone",
            rotate, confidence, min_confidence,
        )
        return None
    if rotate == 0:
        logger.debug(
            "orientation_detector: page already upright (confidence %.2f)",
            confidence,
        )
        return 0
    logger.info(
        "orientation_detector: rotate by %d° (confidence %.2f)",
        rotate, confidence,
    )
    return rotate


__all__ = [
    "MIN_ORIENTATION_CONFIDENCE",
    "detect_orientation",
]
