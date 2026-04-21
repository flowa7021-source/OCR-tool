"""Abstract OCR engine interface.

Engines receive a preprocessed PDF (rasterised, deskewed, denoised,
binarised by :class:`~src.core.image_preprocessor.ImagePreprocessor`)
and must produce a searchable PDF + one :class:`PageOCRResult` per
page.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.core.models import OCRConfig
from src.shared.types import OCREngineKind

logger = logging.getLogger(__name__)


# progress_callback(page_current, page_total, stage_name)
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PageOCRResult:
    """One page worth of OCR output."""

    page_number: int  # 1-based
    text: str = ""
    mean_confidence: float = 0.0  # 0..100
    # (x, y, w, h, confidence[0..100], text) in pixel coords at the
    # engine's rasterisation DPI. Callers that need PDF user-space
    # convert via ``scale = 72 / dpi``.
    word_boxes: list[tuple[float, float, float, float, float, str]] = field(
        default_factory=list
    )
    error: str | None = None


class EngineNotAvailableError(RuntimeError):
    """Raised when an engine is selected but its requirements are missing."""


class OCREngine(ABC):
    """Abstract base class for all OCR engines."""

    kind: OCREngineKind

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str]: ...

    @abstractmethod
    def run(
        self,
        preprocessed_pdf: Path,
        output_pdf: Path,
        config: OCRConfig,
        progress_callback: ProgressCallback | None = None,
        *,
        original_input_pdf: Path | None = None,
    ) -> list[PageOCRResult]: ...

    def unload(self) -> None:
        """Release expensive resources (model weights, GPU buffers)."""
        return
