"""Pluggable OCR back-ends.

Each engine implements :class:`OCREngine` and is selected per job via
:attr:`~src.core.models.OCRConfig.engine`. The default engine is
Tesseract (bundled), alternative engines can be added without touching
the pipeline.
"""

from src.application.engines.base import OCREngine, PageOCRResult
from src.application.engines.registry import get_engine, list_engines
from src.application.engines.tesseract_engine import TesseractEngine

__all__ = [
    "OCREngine",
    "PageOCRResult",
    "TesseractEngine",
    "get_engine",
    "list_engines",
]
