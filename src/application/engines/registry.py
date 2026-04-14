"""Engine registry. Maps :class:`OCREngineKind` values to instances.

Engines are imported lazily so that heavy dependencies (e.g. torch for
GOT-OCR2) aren't loaded until the user actually selects them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.shared.types import OCREngineKind

if TYPE_CHECKING:
    from src.application.engines.base import OCREngine

logger = logging.getLogger(__name__)

_CACHE: dict[OCREngineKind, OCREngine] = {}


def get_engine(kind: OCREngineKind) -> OCREngine:
    """Return the engine instance for ``kind``, constructing it lazily.

    Raises:
        KeyError: If the requested engine is not registered.
    """
    cached = _CACHE.get(kind)
    if cached is not None:
        return cached

    if kind is OCREngineKind.TESSERACT:
        from src.application.engines.tesseract_engine import TesseractEngine

        engine: OCREngine = TesseractEngine()
    elif kind is OCREngineKind.GOT_OCR2:
        # Lazy import: only attempts to load torch/transformers when
        # the user actually switches to this engine.
        try:
            from src.application.engines.got_ocr_engine import GOTOCREngine

            engine = GOTOCREngine()
        except ImportError as exc:
            raise KeyError(
                f"Движок GOT-OCR2 не установлен: {exc}. "
                "Установите 'ocr-studio[htr]' или скачайте модель."
            ) from exc
    else:  # pragma: no cover — exhaustive guard
        raise KeyError(f"Unknown engine kind: {kind}")

    _CACHE[kind] = engine
    return engine


def list_engines() -> list[tuple[OCREngineKind, str, bool, str]]:
    """Return ``[(kind, name, available, availability_message), ...]``.

    Used to populate the UI engine dropdown without forcing heavy
    imports just to know which engines are registered.
    """
    result: list[tuple[OCREngineKind, str, bool, str]] = []
    for kind in OCREngineKind:
        try:
            engine = get_engine(kind)
        except KeyError as exc:
            result.append((kind, kind.label, False, str(exc)))
            continue
        ok, msg = engine.is_available()
        result.append((kind, engine.name, ok, msg))
    return result


def reset_cache() -> None:
    """Clear the engine cache (test hook)."""
    _CACHE.clear()
