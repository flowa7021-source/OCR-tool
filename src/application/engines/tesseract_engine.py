"""Tesseract 5 back-end (via OCRmyPDF).

Processes each page INDIVIDUALLY through ``ocrmypdf.ocr`` so that a
Tesseract crash on one page does not kill the rest. This is the
primary design, not a fallback — production logs showed that batch
OCR on multi-page PDFs fails whenever a single page has a complex
element (stamp, rotated table, handwritten signature) that crashes
Tesseract's layout analysis.

When a page fails on the primary attempt, the engine retries that
specific page with progressively simpler settings (lower DPI raster,
Otsu binarisation, no preprocessing, more tolerant PSM) so that the
final output has a text layer on **every** page. Only if every retry
tier also fails for a page does the engine fall back to keeping the
raster.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
from src.shared.types import OCREngineKind, OptimizeLevel, PSM

logger = logging.getLogger(__name__)

# Retry-tier rasterisation DPI. Lower than the typical 300-600 the user
# configures — we deliberately trade resolution for layout-analysis
# stability. Tesseract's layout analyser crashes far less often on a
# 200 DPI grayscale page than on a 600 DPI multi-channel one.
_RETRY_DPI: int = 200

# Last-resort rasterisation DPI. Even cheaper to analyse; paired with
# PSM.SPARSE_TEXT this almost always returns *some* text layer, which
# is the whole point of this tier.
_LAST_RESORT_DPI: int = 150


class TesseractEngine(OCREngine):
    """Tesseract 5 via OCRmyPDF — page-by-page processing with per-page retry.

    Each page is split into a single-page PDF, OCR'd independently via
    ``ocrmypdf.ocr``, then merged back. A page that crashes Tesseract
    is retried with simpler settings (lower DPI, Otsu binarisation,
    single-block PSM) and finally with sparse-text PSM before giving
    up. In practice every retry tier is cheap compared to the primary
    run so total wall time is dominated by the happy path.
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

        Processing pipeline per page:

          1. Split the preprocessed PDF into N single-page PDFs.
          2. For each page: ``ocrmypdf.ocr`` with user settings.
             Success → searchable page.
          3. On failure: retry the page with simplified settings
             (lower DPI raster, Otsu binarisation, PSM=SINGLE_BLOCK,
             no optimization). Most layout-crash pages recover here.
          4. On further failure: last-resort retry with PSM=SPARSE_TEXT
             at 150 DPI — tolerant of practically any layout.
          5. Only if EVERY tier fails: keep the original raster page
             (no text layer on that one page).
          6. Merge all pages into the final output PDF.

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
        retry_count = 0

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

            # 2. OCR each page: primary → retry → last-resort.
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
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Page %d/%d primary OCR FAILED (%s: %s); "
                        "retrying with simplified settings",
                        i + 1, page_count,
                        type(exc).__name__, exc,
                    )

                recovered_pdf = self._retry_page_with_simpler_settings(
                    page_pdf=page_pdf,
                    work_dir=work_dir,
                    page_index=i + 1,
                    original_config=config,
                )
                if recovered_pdf is not None:
                    page_results.append(recovered_pdf)
                    ok_count += 1
                    retry_count += 1
                    logger.info(
                        "Page %d/%d recovered via simplified-settings retry",
                        i + 1, page_count,
                    )
                    continue

                last_resort_pdf = self._retry_page_last_resort(
                    page_pdf=page_pdf,
                    work_dir=work_dir,
                    page_index=i + 1,
                    original_config=config,
                )
                if last_resort_pdf is not None:
                    page_results.append(last_resort_pdf)
                    ok_count += 1
                    retry_count += 1
                    logger.info(
                        "Page %d/%d recovered via last-resort sparse-text retry",
                        i + 1, page_count,
                    )
                    continue

                logger.warning(
                    "Page %d/%d FAILED on every retry tier; keeping "
                    "original raster (no text layer on this page)",
                    i + 1, page_count,
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

            # Only raise if EVERY tier failed for EVERY page. A mixed
            # result (some pages recovered via retry, some raster-only)
            # still produces a usable output PDF and must not error.
            if ok_count == 0:
                raise OCRmyPDFError(
                    f"Ни одна из {page_count} страниц не была распознана "
                    "даже после автоматического повтора с упрощёнными "
                    "настройками. Tesseract не смог обработать этот "
                    "документ на любом уровне настроек. Попробуйте другой "
                    "профиль (quick_reliable) или уменьшите DPI в настройках."
                )

            if ok_count < page_count:
                logger.warning(
                    "%d/%d pages OCR'd (%d via retry), %d pages left as raster",
                    ok_count, page_count, retry_count, page_count - ok_count,
                )

            logger.info(
                "OCR complete: %d/%d pages with text layer (%d via retry), "
                "output=%s",
                ok_count, page_count, retry_count, output_pdf,
            )

        finally:
            with contextlib.suppress(Exception):
                src.close()
            shutil.rmtree(work_dir, ignore_errors=True)

        return [
            PageOCRResult(page_number=i + 1) for i in range(page_count)
        ]

    def _retry_page_with_simpler_settings(
        self,
        *,
        page_pdf: Path,
        work_dir: Path,
        page_index: int,
        original_config: OCRConfig,
    ) -> Path | None:
        """Second-tier retry: simpler raster + PSM=SINGLE_BLOCK + no optimize.

        When the user's configured settings crash Tesseract on a
        specific page, this strategy usually recovers it:

          * Re-rasterise the single-page PDF at 200 DPI grayscale
            (sidesteps colour-space quirks and keeps layout analysis
            cheap).
          * Rebuild the page PDF from the flat raster so OCRmyPDF
            sees a blank-text canvas with a single image — this
            effectively discards whatever preprocessing burnt the
            first attempt.
          * Override PSM to ``SINGLE_BLOCK``: the most forgiving
            layout for business documents, contracts, invoices.
          * Drop optimize to NONE so OCRmyPDF doesn't re-encode the
            output (optimizer pipelines occasionally trip on the
            simplified raster).

        Returns the path to the recovered OCR'd page PDF on success,
        or ``None`` if this tier also fails.
        """
        simplified_pdf = self._build_simplified_page_pdf(
            page_pdf=page_pdf,
            work_dir=work_dir,
            page_index=page_index,
            dpi=_RETRY_DPI,
            suffix="simpler_raw",
        )
        if simplified_pdf is None:
            return None

        retry_config = dataclasses.replace(
            original_config,
            psm=PSM.SINGLE_BLOCK,
            optimize_level=OptimizeLevel.NONE,
            skip_text=False,
            dpi=_RETRY_DPI,
        )

        page_out = work_dir / f"page_{page_index:04d}_simpler.pdf"
        opts = map_ocr_config(retry_config, simplified_pdf, page_out)
        try:
            run_ocrmypdf(opts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Page %d simplified-settings retry also failed (%s: %s)",
                page_index, type(exc).__name__, exc,
            )
            return None
        return page_out

    def _retry_page_last_resort(
        self,
        *,
        page_pdf: Path,
        work_dir: Path,
        page_index: int,
        original_config: OCRConfig,
    ) -> Path | None:
        """Last-resort retry: PSM=SPARSE_TEXT on a 150 DPI grayscale raster.

        PSM=SPARSE_TEXT (11) tells Tesseract to find text in any
        orientation without assuming structured layout — it rarely
        crashes on stamps, rotated tables or mixed handwriting that
        kill the ``AUTO`` / ``SINGLE_BLOCK`` analysers. Paired with
        a 150 DPI raster, this is the cheapest-to-analyse form the
        page can take; recognition quality is poorer than tier 2 but
        the *point* of this tier is to produce *some* text layer,
        not the best possible one.

        Returns the path to the recovered OCR'd page PDF on success,
        or ``None`` if even this tier fails.
        """
        simplified_pdf = self._build_simplified_page_pdf(
            page_pdf=page_pdf,
            work_dir=work_dir,
            page_index=page_index,
            dpi=_LAST_RESORT_DPI,
            suffix="lastresort_raw",
        )
        if simplified_pdf is None:
            return None

        retry_config = dataclasses.replace(
            original_config,
            psm=PSM.SPARSE_TEXT,
            optimize_level=OptimizeLevel.NONE,
            skip_text=False,
            dpi=_LAST_RESORT_DPI,
        )

        page_out = work_dir / f"page_{page_index:04d}_lastresort.pdf"
        opts = map_ocr_config(retry_config, simplified_pdf, page_out)
        try:
            run_ocrmypdf(opts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Page %d last-resort retry also failed (%s: %s)",
                page_index, type(exc).__name__, exc,
            )
            return None
        return page_out

    @staticmethod
    def _build_simplified_page_pdf(
        *,
        page_pdf: Path,
        work_dir: Path,
        page_index: int,
        dpi: int,
        suffix: str,
    ) -> Path | None:
        """Re-rasterise a page PDF at ``dpi`` grayscale and re-wrap as PDF.

        This discards whatever preprocessing the original pipeline
        applied and produces a clean flat raster — the shape Tesseract
        is least likely to crash on. The caller controls DPI so the
        same helper serves both the 200 DPI "simpler" tier and the
        150 DPI "last resort" tier.

        Returns the path to the freshly built single-page PDF on
        success, or ``None`` if PyMuPDF couldn't build the raster
        (extremely rare — almost always an I/O or OOM issue).
        """
        import fitz

        out_path = work_dir / f"page_{page_index:04d}_{suffix}.pdf"
        try:
            with fitz.open(str(page_pdf)) as src:
                page = src[0]
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            new_doc = fitz.open()
            try:
                # Page size in points = pixels * 72 / dpi
                pt_w = pix.width * 72.0 / dpi
                pt_h = pix.height * 72.0 / dpi
                new_page = new_doc.new_page(width=pt_w, height=pt_h)
                new_page.insert_image(new_page.rect, pixmap=pix)
                new_doc.save(str(out_path))
            finally:
                new_doc.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not rebuild page %d at %d DPI (%s): %s",
                page_index, dpi, suffix, exc,
            )
            return None
        return out_path
