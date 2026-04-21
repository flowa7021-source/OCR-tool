"""Per-field OCR retry via EasyOCR.

Pipeline retry works at the PAGE level. When a specific field is
empty or garbage while other fields are fine, this module re-runs
OCR on a targeted bbox.

Entry point — :func:`rescue_field`. The EasyOCR ``Reader`` is
obtained from the engine registry on first use (lazy) unless the
caller injects one explicitly; tests override ``_easyocr_readtext``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _crop_bbox(
    raster: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Safely crop a region from ``raster``. ``None`` when ``bbox`` is
    entirely out of bounds or has a negative start."""
    x, y, w, h = bbox
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    height, width = raster.shape[:2]
    if x >= width or y >= height:
        return None
    x_end = min(x + w, width)
    y_end = min(y + h, height)
    return raster[y:y_end, x:x_end]


def _get_reader(languages: list[str] | None = None):
    """Lazy EasyOCR Reader via the engine registry."""
    from src.application.engines import get_engine
    from src.shared.types import OCREngineKind

    engine = get_engine(OCREngineKind.EASYOCR)
    return engine._get_reader(languages or ["ru", "en"], gpu=False)


def _easyocr_readtext(
    img: np.ndarray,
    *,
    reader: Any | None = None,
    allowlist: str | None = None,
) -> list[tuple[Any, str, float]]:
    """Thin wrapper around ``reader.readtext`` — monkeypatchable in tests.

    Returns the raw EasyOCR output: ``[(bbox, text, conf0..1), ...]``.
    """
    if reader is None:
        reader = _get_reader()
    return reader.readtext(img, detail=1, paragraph=False, allowlist=allowlist)


def rescue_field(
    raster: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    reader: Any | None = None,
    allowlist: str | None = None,
    min_token_conf: float = 0.3,
    # Kept for backwards-compatibility with older callers; ignored.
    psm: int | None = None,  # noqa: ARG001
    lang: str | None = None,  # noqa: ARG001
    whitelist: str | None = None,
    user_words: str | None = None,  # noqa: ARG001
) -> tuple[str, float]:
    """Re-OCR region ``bbox`` via EasyOCR.

    Args:
        raster: 2D or 3D ndarray of the preprocessed page.
        bbox: ``(x, y, w, h)`` in raster pixel coords.
        reader: Optional pre-loaded ``easyocr.Reader``. When ``None``
            the module loads the default one from the engine registry.
        allowlist: Optional character allowlist passed to EasyOCR.
            Falls back to ``whitelist`` when that legacy kwarg is set.
        min_token_conf: Per-token threshold in ``[0, 1]``. Tokens
            below this are skipped.

    Returns:
        ``(text, conf)`` with ``conf`` in ``[0, 1]``. ``("", 0.0)`` on
        any failure (bbox out-of-bounds, EasyOCR crash, no tokens left).
    """
    cropped = _crop_bbox(raster, bbox)
    if cropped is None or cropped.size == 0:
        return "", 0.0

    effective_allowlist = allowlist if allowlist is not None else whitelist

    try:
        raw = _easyocr_readtext(
            cropped, reader=reader, allowlist=effective_allowlist,
        )
    except Exception as exc:  # noqa: BLE001 — rescue must not propagate
        logger.debug("rescue_field: EasyOCR call failed: %s", exc)
        return "", 0.0

    words: list[str] = []
    confs: list[float] = []
    for _bbox, text, conf in raw:
        text_s = str(text or "").strip()
        if not text_s:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_f < min_token_conf:
            continue
        words.append(text_s)
        confs.append(conf_f)

    if not words:
        return "", 0.0

    return " ".join(words), sum(confs) / len(confs)


__all__ = ["rescue_field"]
