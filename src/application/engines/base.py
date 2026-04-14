"""Abstract OCR engine interface.

Engines are responsible for turning a *preprocessed* PDF into a
searchable PDF and per-page text results. They receive a PDF that has
already been rasterised, deskewed, denoised, binarised etc. by the
pipeline — OpenCV preprocessing is intentionally kept outside the
engine so all engines share it.

Contract:
    * ``name`` / ``description`` — human-readable labels for UI
    * ``kind`` — matches the :class:`OCREngineKind` enum value
    * ``is_available()`` — cheap availability probe (no model download,
      no heavyweight imports); used by the UI to grey out unavailable
      entries and explain *why*
    * ``run()`` — actually do OCR. MUST produce a valid searchable PDF
      at ``output_pdf`` and return one :class:`PageOCRResult` per page.
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
    """One page worth of OCR output returned by an engine."""

    page_number: int  # 1-based
    text: str = ""
    mean_confidence: float = 0.0
    # Word-level bounding boxes in PDF user-space points (72 DPI).
    # Each tuple is (x, y, w, h, confidence[0-100], text).
    word_boxes: list[tuple[float, float, float, float, float, str]] = field(
        default_factory=list
    )
    error: str | None = None


class EngineNotAvailableError(RuntimeError):
    """Raised when an engine is selected but its requirements are missing.

    The string message is shown directly to the user, so it should be in
    Russian and explain what's missing (model, optional package, etc.).
    """


class OCREngine(ABC):
    """Abstract base class for all OCR engines."""

    #: The :class:`OCREngineKind` this implementation handles.
    kind: OCREngineKind

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name (e.g. "Tesseract 5")."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown in the engine picker tooltip."""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return ``(ok, message)`` describing current readiness.

        ``ok`` is True only when the engine can run right now (binaries
        found, model downloaded, optional deps importable). When False,
        ``message`` MUST tell the user what to do in Russian.
        """

    @abstractmethod
    def run(
        self,
        preprocessed_pdf: Path,
        output_pdf: Path,
        config: OCRConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[PageOCRResult]:
        """Execute OCR on ``preprocessed_pdf`` and write searchable PDF.

        Args:
            preprocessed_pdf: Path to a PDF already rasterised and
                preprocessed by the pipeline.
            output_pdf: Destination path for the searchable PDF.
                Parent directory will be created if missing.
            config: User-facing OCR configuration.
            progress_callback: Optional ``(current, total, stage)``
                callback; may be called from worker threads.

        Returns:
            One :class:`PageOCRResult` per page of the input PDF.

        Raises:
            EngineNotAvailableError: When the engine cannot run and the
                caller should surface a user-friendly message.
            RuntimeError: For unrecoverable internal failures.
        """
