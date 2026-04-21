"""Pluggable OCR back-ends.

Each engine implements :class:`OCREngine` and is selected per job via
:attr:`~src.core.models.OCRConfig.engine`.
"""

from src.application.engines.base import OCREngine, PageOCRResult
from src.application.engines.easyocr_engine import EasyOCREngine
from src.application.engines.registry import get_engine, list_engines

__all__ = [
    "OCREngine",
    "PageOCRResult",
    "EasyOCREngine",
    "get_engine",
    "list_engines",
]
