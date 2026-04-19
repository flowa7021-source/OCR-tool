"""Engine registry. Maps :class:`OCREngineKind` values to instances.

Only Tesseract is registered. The lazy-import pattern is preserved so
a future second engine can be added without restructuring this file.
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
    """Clear the engine cache, releasing any expensive resources first.

    Every cached engine gets :meth:`OCREngine.unload` called before it
    is dropped. For Tesseract this is a no-op; the hook stays in place
    for any future engine that holds onto heavy resources.
    """
    for engine in list(_CACHE.values()):
        try:
            engine.unload()
        except Exception:  # noqa: BLE001
            logger.debug("engine.unload raised", exc_info=True)
    _CACHE.clear()
