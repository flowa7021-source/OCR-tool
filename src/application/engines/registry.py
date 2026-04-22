"""Engine registry. Maps :class:`OCREngineKind` to instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.shared.types import OCREngineKind

if TYPE_CHECKING:
    from src.application.engines.base import OCREngine

logger = logging.getLogger(__name__)

_CACHE: dict[OCREngineKind, OCREngine] = {}


def get_engine(kind: OCREngineKind) -> OCREngine:
    """Return the engine for ``kind``, constructing it lazily."""
    cached = _CACHE.get(kind)
    if cached is not None:
        return cached

    if kind is OCREngineKind.EASYOCR:
        from src.application.engines.easyocr_engine import EasyOCREngine
        engine: OCREngine = EasyOCREngine()
    else:  # pragma: no cover
        raise KeyError(f"Unknown engine kind: {kind}")

    _CACHE[kind] = engine
    return engine


def list_engines() -> list[tuple[OCREngineKind, str, bool, str]]:
    """Return ``[(kind, name, available, availability_message), ...]``."""
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
    """Clear the engine cache, releasing any expensive resources first."""
    for engine in list(_CACHE.values()):
        try:
            engine.unload()
        except Exception:  # noqa: BLE001
            logger.debug("engine.unload raised", exc_info=True)
    _CACHE.clear()
