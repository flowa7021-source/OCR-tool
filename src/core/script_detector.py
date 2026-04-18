"""Pre-OCR script detector for pipeline pages.

Tesseract's ``-l rus+eng`` picks the wrong script for visually
identical characters (``О/O``, ``А/A``, ``Е/E``, ``К/K``, ``Н/Н``,
``Р/Р``) when the page is pure Russian — the LSTM ranks the Latin
interpretation on some glyphs because its confidence vectors don't
always favour the dominant script. Post-processing catches most of
these via :mod:`src.core.text_postprocessor.normalize_cyrillic_latin_confusion`,
but prevention at the OCR stage is better than cure at the post-OCR
stage — the glyph is recognised correctly first time and no
information is lost.

This module exposes :func:`detect_dominant_script`, a single-call
helper that runs Tesseract's OSD (Orientation / Script Detection)
mode on a page image. If OSD reports Cyrillic or Latin with
confidence above :data:`MIN_SCRIPT_CONFIDENCE`, the helper returns
the corresponding language code (``"rus"`` or ``"eng"``). Anything
else — low confidence, an unsupported script, an OSD error — yields
``None``, and callers are expected to fall back to the user's
configured multi-language setting.

Design choices:

* **One call per page, not per region.** OSD is fast (~50 ms at
  150 DPI) but still a Tesseract subprocess.

* **Pytesseract, not custom LSTM.** We already ship Tesseract; its
  OSD model is trained on hundreds of scripts.

* **Fail open, not closed.** Any error from ``image_to_osd``
  returns ``None`` — script detection is an optimisation, not a
  requirement.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI lint jobs without tesseract deps
    pytesseract = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


MIN_SCRIPT_CONFIDENCE: float = 1.0
_MIN_IMAGE_DIMENSION: int = 64
_SCRIPT_TO_LANG: dict[str, str] = {
    "Cyrillic": "rus",
    "Latin": "eng",
}


def _parse_osd_output(osd_text: str) -> tuple[str, float] | None:
    """Extract ``(script_name, script_confidence)`` from OSD output."""
    script_name: str | None = None
    script_conf: float | None = None
    for line in osd_text.splitlines():
        if line.startswith("Script:"):
            script_name = line.split(":", 1)[1].strip()
        elif line.startswith("Script confidence:"):
            try:
                script_conf = float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    if not script_name or script_conf is None:
        return None
    return script_name, script_conf


def detect_dominant_script(
    image: Any,
    *,
    min_confidence: float = MIN_SCRIPT_CONFIDENCE,
) -> str | None:
    """Detect the dominant script of a page image.

    Returns ``"rus"``, ``"eng"`` or ``None``.
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
    except Exception as exc:  # noqa: BLE001
        logger.debug("script_detector: OSD failed (%s)", exc)
        return None
    parsed = _parse_osd_output(osd_text)
    if parsed is None:
        logger.debug("script_detector: OSD output unparseable")
        return None
    script_name, confidence = parsed
    if confidence < min_confidence:
        return None
    lang = _SCRIPT_TO_LANG.get(script_name)
    if lang is None:
        return None
    logger.info(
        "script_detector: detected %s → lang=%s (confidence %.2f)",
        script_name, lang, confidence,
    )
    return lang


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
