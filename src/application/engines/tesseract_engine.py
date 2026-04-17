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
        except OCRmyPDFError as batch_err:
            # Batch OCR on all pages failed (timeout, Tesseract crash
            # on a specific page, graft-hocr-miss). Instead of failing
            # the entire job, try per-page fallback: OCR each page
            # individually so that pages 1-3 succeed even if page 4
            # crashes Tesseract.
            logger.warning(
                "Batch OCR failed (%s); trying per-page fallback so "
                "partial results are preserved.",
                batch_err,
            )
            try:
                self._per_page_fallback(
                    preprocessed_pdf, output_pdf, config, progress_callback
                )
            except Exception as fallback_err:  # noqa: BLE001
                # Even per-page fallback failed entirely. Surface the
                # original batch error — it has the clearer message.
                logger.error(
                    "Per-page fallback also failed: %s", fallback_err
                )
                raise batch_err from fallback_err
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

    def _per_page_fallback(
        self,
        preprocessed_pdf: Path,
        output_pdf: Path,
        config: OCRConfig,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Split the PDF into individual pages, OCR each one separately.

        When batch OCR fails because Tesseract crashes on one specific
        page (e.g. a page with a complex stamp, a rotated table, or a
        mix of print and handwriting), this fallback ensures the OTHER
        pages still get a text layer. The problematic page is left as
        a raster-only page with no invisible text — the user gets 3/4
        pages searchable instead of 0/4.

        This is significantly slower than batch OCR (no parallelism,
        each page goes through ocrmypdf.ocr independently) but it's a
        last resort, not the happy path.
        """
        import shutil
        import tempfile

        import fitz

        logger.info("Per-page fallback: splitting %s into individual pages", preprocessed_pdf)

        src = fitz.open(str(preprocessed_pdf))
        page_count = src.page_count
        work_dir = Path(tempfile.mkdtemp(prefix="ocr-perpage-"))

        page_pdfs: list[Path] = []
        page_results: list[Path | None] = []

        try:
            # 1. Split into single-page PDFs.
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

            # 2. OCR each page individually.
            for i, page_pdf in enumerate(page_pdfs):
                page_out = work_dir / f"page_{i + 1:04d}_ocr.pdf"
                page_opts = map_ocr_config(config, page_pdf, page_out)

                if progress_callback is not None:
                    try:
                        progress_callback(i, page_count, "ocr-fallback")
                    except Exception:  # noqa: BLE001
                        pass

                try:
                    run_ocrmypdf(page_opts)
                    page_results.append(page_out)
                    logger.info("Per-page fallback: page %d/%d OK", i + 1, page_count)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Per-page fallback: page %d/%d FAILED (%s); "
                        "using original raster page (no text layer)",
                        i + 1, page_count, exc,
                    )
                    # Use the original (un-OCR'd) page as fallback.
                    page_results.append(page_pdf)

            # 3. Merge results into a single output PDF.
            merged = fitz.open()
            try:
                for result_pdf in page_results:
                    if result_pdf is not None and result_pdf.exists():
                        with fitz.open(str(result_pdf)) as p:
                            merged.insert_pdf(p)
                merged.save(str(output_pdf))
            finally:
                merged.close()

            ok_count = sum(
                1 for r in page_results
                if r is not None and r.name.endswith("_ocr.pdf")
            )
            logger.info(
                "Per-page fallback: %d/%d pages OCR'd successfully, "
                "output at %s",
                ok_count, page_count, output_pdf,
            )

            if progress_callback is not None:
                try:
                    progress_callback(page_count, page_count, "ocr-fallback")
                except Exception:  # noqa: BLE001
                    pass

        finally:
            import contextlib

            with contextlib.suppress(Exception):
                src.close()
            shutil.rmtree(work_dir, ignore_errors=True)
