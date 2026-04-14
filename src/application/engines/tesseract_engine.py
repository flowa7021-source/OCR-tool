"""Tesseract 5 back-end (via OCRmyPDF)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.application.engines.base import (
    EngineNotAvailableError,
    OCREngine,
    PageOCRResult,
    ProgressCallback,
)
from src.application.ocrmypdf_integration import (
    OCRmyPDFError,
    map_ocr_config,
    run_ocrmypdf,
)
from src.core.models import OCRConfig
from src.shared.types import OCREngineKind

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
    """Default back-end: Tesseract 5 driven by OCRmyPDF.

    The heavy lifting still happens inside OCRmyPDF (which handles the
    hOCR → invisible-text-layer conversion). This class is a thin
    adapter so the pipeline can talk to Tesseract through the same
    :class:`OCREngine` interface used by experimental engines.
    """

    kind = OCREngineKind.TESSERACT

    @property
    def name(self) -> str:
        return "Tesseract 5"

    @property
    def description(self) -> str:
        return (
            "Быстрый LSTM-движок, работает со всеми печатными "
            "документами. Встроен в дистрибутив."
        )

    def is_available(self) -> tuple[bool, str]:
        """Verify Tesseract binary and tessdata are reachable."""
        try:
            from src.infrastructure.tesseract_wrapper import TesseractWrapper

            wrapper = TesseractWrapper()
            return wrapper.verify()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tesseract availability probe failed: %s", exc)
            return False, f"Ошибка проверки Tesseract: {exc}"

    def run(
        self,
        preprocessed_pdf: Path,
        output_pdf: Path,
        config: OCRConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> list[PageOCRResult]:
        """Invoke OCRmyPDF and return per-page results.

        Unlike alternative engines, text extraction happens AFTER the
        searchable PDF is produced — the pipeline post-processes the
        output PDF to gather text and confidence. This adapter only
        runs the OCR step itself and returns empty :class:`PageOCRResult`
        stubs; the pipeline fills them in via ``_extract_and_postprocess``.
        Keeping this shape preserves the existing OCRmyPDF optimisations
        (threading, PDF/A, optimize=1 lossless reflow).
        """
        ok, msg = self.is_available()
        if not ok:
            raise EngineNotAvailableError(msg)

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        options = map_ocr_config(config, preprocessed_pdf, output_pdf)

        if progress_callback is not None:
            try:
                progress_callback(0, 1, "ocr")
            except Exception:  # noqa: BLE001
                logger.debug("progress_callback raised", exc_info=True)

        try:
            run_ocrmypdf(options)
        except OCRmyPDFError:
            raise
        except Exception as exc:
            logger.exception("Unexpected OCRmyPDF failure")
            raise RuntimeError(f"Tesseract/OCRmyPDF failed: {exc}") from exc

        if progress_callback is not None:
            try:
                progress_callback(1, 1, "ocr")
            except Exception:  # noqa: BLE001
                logger.debug("progress_callback raised", exc_info=True)

        # Per-page text and confidence are filled in later by the
        # pipeline (it already reads both from the resulting PDF and
        # via pytesseract.image_to_data). We return stubs preserving
        # page numbering so the engine contract is satisfied.
        import fitz  # lazy

        doc = fitz.open(str(output_pdf))
        try:
            return [PageOCRResult(page_number=i + 1) for i in range(doc.page_count)]
        finally:
            doc.close()
