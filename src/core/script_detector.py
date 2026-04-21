"""Dominant-script detector (no-op under the EasyOCR engine).

Historically this module ran an OSD pass to decide whether a page
was predominantly Cyrillic or Latin so the OCR language could be
narrowed. The EasyOCR engine takes a fixed multi-language Reader, so
per-page script hints are not useful; the function is kept as a
stable API (returns ``None`` — no preference).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


#: Retained for backwards compatibility; callers may still reference it.
MIN_SCRIPT_CONFIDENCE: float = 2.0
_MIN_IMAGE_DIMENSION: int = 64


def detect_dominant_script(
    image: Any,
    *,
    min_confidence: float = MIN_SCRIPT_CONFIDENCE,  # noqa: ARG001
) -> str | None:
    """Return ``"ru"`` / ``"en"`` / ``None`` for the dominant script.

    Always returns ``None`` under the current engine (no preference).
    Kept as a stable entry point for callers that still plumb the
    optional hint through.
    """
    try:
        h, w = image.shape[:2]
    except (AttributeError, IndexError, ValueError):
        return None
    if h < _MIN_IMAGE_DIMENSION or w < _MIN_IMAGE_DIMENSION:
        return None
    logger.debug(
        "script_detector: detection disabled under EasyOCR engine; "
        "returning None (no language preference)"
    )
    return None


class ScriptDetector:
    """Stateful wrapper around :func:`detect_dominant_script`."""

    def __init__(
        self, *, min_confidence: float = MIN_SCRIPT_CONFIDENCE,
    ) -> None:
        self.min_confidence = min_confidence

    def detect(self, image: Any) -> str | None:
        return detect_dominant_script(
            image, min_confidence=self.min_confidence,
        )


__all__ = [
    "MIN_SCRIPT_CONFIDENCE",
    "ScriptDetector",
    "detect_dominant_script",
]
