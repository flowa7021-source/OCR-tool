"""Tesseract 5 back-end (via OCRmyPDF).

Processes each page INDIVIDUALLY through ``ocrmypdf.ocr`` so that a
Tesseract crash on one page does not kill the rest. This is the
primary design, not a fallback — production logs showed that batch
OCR on multi-page PDFs fails whenever a single page has a complex
element (stamp, rotated table, handwritten signature) that crashes
Tesseract's layout analysis.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
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
    """Tesseract 5 via OCRmyPDF — page-by-page processing.

    Each page is split into a single-page PDF, OCR'd independently
    via ``ocrmypdf.ocr``, then merged back. A page that crashes
    Tesseract is kept as a raster-only page (no text layer) — the
    user gets 3/4 pages searchable instead of a total failure.
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
        """OCR each page independently and assemble the output PDF.

        Unlike the old batch approach (one ``ocrmypdf.ocr`` call on
        the whole PDF), this processes pages one by one:

          1. Split the preprocessed PDF into N single-page PDFs.
          2. For each page: ``ocrmypdf.ocr`` in its own try/except.
             Success → searchable page. Failure → original raster.
          3. Merge all pages into the final output PDF.

        Progress fires per page, so the UI shows "page 2/4" instead
        of sitting at 0% for 2 minutes.
        """
        ok, msg = self.is_available()
        if not ok:
            raise EngineNotAvailableError(msg)

        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        import fitz

        src = fitz.open(str(preprocessed_pdf))
        page_count = src.page_count

        if page_count == 0:
            src.close()
            raise OCRmyPDFError("Preprocessed PDF has zero pages")

        work_dir = Path(tempfile.mkdtemp(prefix="ocr-pages-"))
        page_results: list[Path] = []
        ok_count = 0

        try:
            # 1. Split into single-page PDFs.
            page_pdfs: list[Path] = []
            for i in range(page_count):
                single = fitz.open()
                try:
                    single.insert_pdf(src, from_page=i, to_page=i)
                    p = work_dir / f"page_{i + 1:04d}.pdf"
                    single.save(str(p))
                    page_pdfs.append(p)
                finally:
                    single.close()
            src.close()

            # 2. OCR each page independently.
            for i, page_pdf in enumerate(page_pdfs):
                if progress_callback is not None:
                    with contextlib.suppress(Exception):
                        progress_callback(i, page_count, "ocr")

                page_out = work_dir / f"page_{i + 1:04d}_ocr.pdf"
                page_opts = map_ocr_config(config, page_pdf, page_out)

                try:
                    run_ocrmypdf(page_opts)
                    page_results.append(page_out)
                    ok_count += 1
                    logger.info(
                        "Page %d/%d OCR'd successfully", i + 1, page_count
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Page %d/%d FAILED (%s: %s); keeping original "
                        "raster (no text layer on this page)",
                        i + 1, page_count,
                        type(exc).__name__, exc,
                    )
                    page_results.append(page_pdf)

            # 3. Merge into the final output PDF.
            merged = fitz.open()
            try:
                for result_pdf in page_results:
                    if result_pdf.exists():
                        with fitz.open(str(result_pdf)) as p:
                            merged.insert_pdf(p)
                merged.save(str(output_pdf))
            finally:
                merged.close()

            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(page_count, page_count, "ocr")

            if ok_count == 0:
                raise OCRmyPDFError(
                    f"Ни одна из {page_count} страниц не была распознана. "
                    "Tesseract не смог обработать этот документ. "
                    "Попробуйте другой профиль (quick_reliable) или "
                    "уменьшите DPI."
                )

            if ok_count < page_count:
                logger.warning(
                    "%d/%d pages OCR'd, %d pages left as raster only",
                    ok_count, page_count, page_count - ok_count,
                )

            logger.info(
                "OCR complete: %d/%d pages with text layer, output=%s",
                ok_count, page_count, output_pdf,
            )

        finally:
            with contextlib.suppress(Exception):
                src.close()
            shutil.rmtree(work_dir, ignore_errors=True)

        return [
            PageOCRResult(page_number=i + 1) for i in range(page_count)
        ]
