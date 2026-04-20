"""Tesseract 5 back-end (via OCRmyPDF).

Processes each page INDIVIDUALLY through ``ocrmypdf.ocr`` so that a
Tesseract crash on one page does not kill the rest. Pages are OCR'd
in **parallel** via a ``ThreadPoolExecutor`` so the N-page wall time
matches the slowest page, not the sum. This was the speed regression
that made the user's 4-page document take 15-20 minutes — the per-
page split removed the implicit parallelism that ``ocrmypdf.ocr``
had via its ``use_threads=True`` mode.

Worker count is capped at ``min(page_count, cpu_count, 4)`` to avoid
contending with the outer ``ParallelProcessor`` when the user runs a
multi-file queue. Override with ``OCR_PER_PAGE_WORKERS=N`` for
debugging or for hosts with very many cores.

When a page fails on the primary attempt, the engine retries that
specific page with progressively simpler settings (lower DPI raster,
Otsu binarisation, no preprocessing, more tolerant PSM) so that the
final output has a text layer on **every** page. Only if every retry
tier also fails for a page does the engine fall back to keeping the
raster.

Per-page OCR is invoked with ``optimize_level=NONE`` regardless of
the user profile; OCRmyPDF's optimiser (Ghostscript re-encode) runs
ONCE on the final merged PDF instead of N times. This saves
~10-20 s per page on the user's 400 DPI workload at zero quality
cost — the optimiser is lossy by definition only at level ≥ 2.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.shared.types import PSM, OCREngineKind, OptimizeLevel

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

# Per-page OCR worker cap. Raised from 8 to 10 in the "batch 10
# pages at a time" UX rollout — matches the preprocess and
# postprocess stages' cap so a 10-page bundle moves through all
# three stages at the same width. Hosts with more cores still
# benefit from the ``OCR_PER_PAGE_WORKERS`` env override; the
# memory-pressure guard below keeps outer × inner from thrashing
# the OS with more concurrent Tesseract subprocesses than RAM
# supports.
_MAX_PER_PAGE_WORKERS: int = 10

# Memory budget per worker. A Tesseract + OCRmyPDF + pikepdf
# pipeline on a 500 DPI A4 page holds:
#   * ~80 MB for the binarised page in RAM
#   * ~150 MB for the LSTM forward pass
#   * ~100 MB for pikepdf's in-flight object buffers
# Rounded up to 500 MB for safety on real-world page complexity
# (tables, stamps, photographed pages with rich colour channels).
_MIN_RAM_PER_WORKER_MB: int = 500


def _available_memory_mb() -> int | None:
    """Return free + reclaimable memory in MB, or None if psutil
    isn't importable.

    We query ``psutil.virtual_memory().available`` which includes
    file-cache pages the OS can drop under pressure — the
    practically-free memory the OS will hand to our processes.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - psutil is a real dep
        return None
    try:
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None


def _resolve_per_page_workers(
    page_count: int,
    *,
    cpu_count: int | None = None,
    available_memory_mb: int | None = None,
) -> int:
    """Pick how many worker threads to use for per-page OCR.

    Three caps, the tightest wins:
      * ``page_count`` — can't run more workers than pages.
      * ``cpu_count`` (default: :func:`os.cpu_count`) — more
        workers than cores wastes context switches.
      * Memory budget — ``available_memory_mb //
        _MIN_RAM_PER_WORKER_MB``. Oversubscribing RAM drops us
        into swap, which is strictly worse than fewer workers.

    And two bypasses, in this order:
      * ``OCR_PER_PAGE_WORKERS`` env var — user can force a
        specific count (still clamped against page_count).
      * ``_MAX_PER_PAGE_WORKERS`` global cap (default 8) — even
        on a 32-core box with 128 GB RAM we don't exceed this.

    Returns:
        Integer ≥ 1. Never 0 — a zero would make the
        ThreadPoolExecutor refuse submitted work.
    """
    # Normalise inputs — caller passes explicit values in tests,
    # production uses the live system values.
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if available_memory_mb is None:
        available_memory_mb = _available_memory_mb()

    # Env override short-circuits auto-detect but still respects
    # the page-count upper bound.
    override = os.environ.get("OCR_PER_PAGE_WORKERS")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return max(1, min(max(page_count, 1), n))
        except ValueError:
            logger.warning(
                "Ignoring invalid OCR_PER_PAGE_WORKERS=%r", override,
            )

    workers = min(
        max(page_count, 1),  # zero-page edge case: floor to 1
        cpu_count,
        _MAX_PER_PAGE_WORKERS,
    )

    if available_memory_mb is not None:
        memory_cap = max(1, available_memory_mb // _MIN_RAM_PER_WORKER_MB)
        if memory_cap < workers:
            logger.info(
                "Per-page workers throttled by available RAM: "
                "%d MB available, %d MB per worker → %d workers "
                "(cpu/page cap would have been %d)",
                available_memory_mb, _MIN_RAM_PER_WORKER_MB,
                memory_cap, workers,
            )
        workers = min(workers, memory_cap)

    return max(1, workers)


def _page_pdf_has_text(pdf_path: Path) -> bool:
    """Return True if the single-page PDF at ``pdf_path`` has any text.

    OCRmyPDF's "success" return doesn't guarantee a non-empty text
    layer: when Tesseract times out or can't segment the page, the
    graft phase stamps an EMPTY hOCR and OCRmyPDF still reports
    completion. The engine's retry tiers need to distinguish
    "stamped a real text layer" from "stamped an empty shell" —
    this helper is the check.
    """
    import fitz

    try:
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                if (page.get_text("text") or "").strip():
                    return True
    except Exception as exc:  # noqa: BLE001
        # If we can't even open the output PDF, treat as no-text so
        # the next tier gets a chance. A genuinely broken file would
        # be caught by the raster-fallback branch anyway.
        logger.debug(
            "_page_pdf_has_text: failed to probe %s: %s", pdf_path, exc
        )
        return False
    return False


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
        *,
        original_input_pdf: Path | None = None,
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
        # Per-index slot so parallel workers can drop their result
        # without locking — Python list assignment to a fixed index
        # is GIL-protected.
        page_results: list[Path | None] = [None] * page_count
        ok_pages: set[int] = set()
        retry_pages: set[int] = set()

        # Per-page work uses ``optimize=NONE`` to skip the Ghostscript
        # re-encode pass (saves 10-20 s per page on the user's 400
        # DPI workload). The final merged PDF is optimised once
        # below using the user-configured level.
        per_page_config = dataclasses.replace(
            config, optimize_level=OptimizeLevel.NONE
        )

        # Stage B of Initiative 1: pre-OCR script detection. If the
        # preprocessed page image is unambiguously Cyrillic or Latin,
        # narrow the Tesseract ``-l`` flag from ``rus+eng`` to the
        # single detected language. Eliminates Latin/Cyrillic look-
        # alike confusion at OCR time instead of cleaning it up
        # post-hoc. Opt-out via ``OCR_AUTO_SCRIPT_DETECT=0``.
        per_page_config = self._maybe_narrow_script_language(
            per_page_config, preprocessed_pdf,
        )

        # Progress accounting under a lock — multiple worker threads
        # call back here as their pages finish.
        import threading

        progress_lock = threading.Lock()
        completed_count = [0]

        def _emit_progress() -> None:
            if progress_callback is None:
                return
            with progress_lock:
                completed_count[0] += 1
                done = completed_count[0]
            with contextlib.suppress(Exception):
                progress_callback(done, page_count, "ocr")

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

            # Initial 0% tick before any work starts so the UI stops
            # showing a stale "page N-1" from the previous file.
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(0, page_count, "ocr")

            # 2. OCR each page in parallel: primary → retry → last-resort.
            workers = _resolve_per_page_workers(page_count)
            logger.info(
                "Per-page OCR: %d pages × %d workers (cpu_count=%d)",
                page_count, workers, os.cpu_count() or 1,
            )

            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = {
                    exe.submit(
                        self._process_one_page,
                        page_pdf=page_pdf,
                        work_dir=work_dir,
                        page_index=idx + 1,
                        page_count=page_count,
                        per_page_config=per_page_config,
                        original_config=config,
                        original_input_pdf=original_input_pdf,
                    ): idx
                    for idx, page_pdf in enumerate(page_pdfs)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    result_pdf, status = fut.result()
                    page_results[idx] = result_pdf
                    if status == "primary":
                        ok_pages.add(idx)
                    elif status == "retry":
                        ok_pages.add(idx)
                        retry_pages.add(idx)
                    # status == "raster" → page failed every tier
                    _emit_progress()

            ok_count = len(ok_pages)
            retry_count = len(retry_pages)

            # 3. Merge into the final output PDF, then run the user-
            # requested optimisation pass once on the merged file.
            unoptimised = work_dir / "merged_unoptimised.pdf"
            merged = fitz.open()
            try:
                for result_pdf in page_results:
                    if result_pdf is not None and result_pdf.exists():
                        with fitz.open(str(result_pdf)) as p:
                            merged.insert_pdf(p)
                merged.save(str(unoptimised))
            finally:
                merged.close()

            # Optimisation: only invoke OCRmyPDF if the user actually
            # asked for it. Default profile uses LOSSLESS (level 1)
            # which is just lossless object compression — no quality
            # impact, but takes a few seconds. Levels 2-3 are lossy
            # JPEG re-encoding, which the user opted into.
            if int(config.optimize_level) > 0:
                try:
                    optimise_opts = map_ocr_config(
                        dataclasses.replace(config, skip_text=True),
                        unoptimised, output_pdf,
                    )
                    run_ocrmypdf(optimise_opts)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Final-PDF optimisation failed (%s); "
                        "delivering unoptimised merge", exc,
                    )
                    shutil.copy2(str(unoptimised), str(output_pdf))
            else:
                shutil.copy2(str(unoptimised), str(output_pdf))

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

    @staticmethod
    def _maybe_narrow_script_language(
        per_page_config: OCRConfig, preprocessed_pdf: Path,
    ) -> OCRConfig:
        """Optionally narrow ``languages`` to a single detected script.

        Runs Tesseract's OSD on the first page of the preprocessed
        PDF; if it reports Cyrillic or Latin with confidence above
        the detector threshold, returns a copy of ``per_page_config``
        with ``languages`` narrowed to that single code. A narrower
        ``-l`` flag is the single most effective way to stop Tesseract
        picking the wrong script for visually-identical characters
        (``О/O``, ``А/A``, ``Е/E``...).

        Gated on ``OCR_AUTO_SCRIPT_DETECT=0`` env var. Fails open:
        any error returns the original config unchanged.
        """
        if os.environ.get("OCR_AUTO_SCRIPT_DETECT", "1") == "0":
            return per_page_config
        if len(per_page_config.languages) <= 1:
            return per_page_config

        try:
            import fitz
            import numpy as np

            with fitz.open(str(preprocessed_pdf)) as doc:
                if doc.page_count == 0:
                    return per_page_config
                pix = doc[0].get_pixmap(dpi=150, colorspace=fitz.csGRAY)
                arr = np.frombuffer(
                    pix.samples, dtype=np.uint8,
                ).reshape(pix.height, pix.width)

            from src.core.script_detector import detect_dominant_script

            detected = detect_dominant_script(arr)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Script auto-detect failed (%s) — keeping original "
                "languages=%s", exc, per_page_config.languages,
            )
            return per_page_config

        if detected is None:
            return per_page_config
        if (
            detected == per_page_config.primary_language
            and per_page_config.languages == [detected]
        ):
            return per_page_config

        logger.info(
            "Auto-script: narrowing OCR languages from %s to [%r]",
            per_page_config.languages, detected,
        )
        return dataclasses.replace(
            per_page_config,
            languages=[detected],
            primary_language=detected,
        )

    def _process_one_page(
        self,
        *,
        page_pdf: Path,
        work_dir: Path,
        page_index: int,
        page_count: int,
        per_page_config: OCRConfig,
        original_config: OCRConfig,
        original_input_pdf: Path | None = None,
    ) -> tuple[Path, str]:
        """OCR one page through primary → retry → last-resort tiers.

        Designed to be called concurrently from a ``ThreadPoolExecutor``.
        Each invocation operates on its own input/output paths, so
        worker threads don't share mutable state — the only contention
        is on ``run_ocrmypdf`` itself, which spawns separate Tesseract
        and Ghostscript subprocesses per call.

        Returns ``(result_pdf, status)`` where status is one of:
          * ``"primary"`` — the user's settings worked first try
          * ``"retry"``   — recovered via simplified-settings or
                            last-resort tier
          * ``"raster"``  — every tier failed; ``result_pdf`` is the
                            original raster page (no text layer)
        """
        page_out = work_dir / f"page_{page_index:04d}_ocr.pdf"
        page_opts = map_ocr_config(per_page_config, page_pdf, page_out)

        primary_failed_reason: str | None = None
        try:
            run_ocrmypdf(page_opts)
        except Exception as exc:  # noqa: BLE001
            primary_failed_reason = f"{type(exc).__name__}: {exc}"
        else:
            # Empty-text-layer detection — see module-level docstring
            # for why ``run_ocrmypdf`` returning cleanly isn't enough.
            if not _page_pdf_has_text(page_out):
                primary_failed_reason = (
                    "primary attempt returned empty text layer"
                )

        if primary_failed_reason is None:
            logger.info(
                "Page %d/%d OCR'd successfully", page_index, page_count
            )
            return page_out, "primary"

        logger.warning(
            "Page %d/%d primary OCR FAILED (%s); "
            "trying aggressive preprocessing retry",
            page_index, page_count, primary_failed_reason,
        )

        # Tier 1 of 3 retry tiers — aggressive preprocessing at the
        # ORIGINAL DPI. Targets faded / noisy / low-contrast scans
        # where the user's profile preprocessing wasn't aggressive
        # enough (Sauvola + CLAHE 3.0 + background removal + NLM)
        # and pixel density still matters for thin-stroke glyphs.
        aggressive_pdf = self._retry_page_with_aggressive_preprocessing(
            page_pdf=page_pdf,
            work_dir=work_dir,
            page_index=page_index,
            original_config=original_config,
            original_input_pdf=original_input_pdf,
        )
        if aggressive_pdf is not None:
            logger.info(
                "Page %d/%d recovered via aggressive-preprocessing retry",
                page_index, page_count,
            )
            return aggressive_pdf, "retry"

        logger.warning(
            "Page %d/%d aggressive preprocessing also failed; "
            "falling back to simpler settings",
            page_index, page_count,
        )

        recovered_pdf = self._retry_page_with_simpler_settings(
            page_pdf=page_pdf,
            work_dir=work_dir,
            page_index=page_index,
            original_config=original_config,
        )
        if recovered_pdf is not None:
            logger.info(
                "Page %d/%d recovered via simplified-settings retry",
                page_index, page_count,
            )
            return recovered_pdf, "retry"

        last_resort_pdf = self._retry_page_last_resort(
            page_pdf=page_pdf,
            work_dir=work_dir,
            page_index=page_index,
            original_config=original_config,
        )
        if last_resort_pdf is not None:
            logger.info(
                "Page %d/%d recovered via last-resort sparse-text retry",
                page_index, page_count,
            )
            return last_resort_pdf, "retry"

        logger.warning(
            "Page %d/%d FAILED on every retry tier; keeping original "
            "raster (no text layer on this page)",
            page_index, page_count,
        )
        return page_pdf, "raster"

    def _retry_page_with_aggressive_preprocessing(
        self,
        *,
        page_pdf: Path,
        work_dir: Path,
        page_index: int,
        original_config: OCRConfig,
        original_input_pdf: Path | None = None,
    ) -> Path | None:
        """First-tier retry: aggressive preprocessing on the RAW input.

        Runs BEFORE the simpler-raster tier so we keep pixel density
        for faded / low-contrast scans that the user's profile
        preprocessing wasn't aggressive enough for. Operates on the
        ORIGINAL user PDF (``original_input_pdf``) when the caller
        provides it — the already-preprocessed ``page_pdf`` has been
        binarised, and reapplying Sauvola + CLAHE + background
        removal to a pure-black-and-white image is a no-op at best
        and destructive at worst. The original has the grayscale
        information the aggressive preprocessing needs.

        Falls back to ``page_pdf`` when ``original_input_pdf`` is
        ``None`` (keeps backwards compatibility with callers that
        haven't been updated to pass the raw input through).

        Applies:

          * Sauvola local-threshold binarisation (window=25, k=0.2)
            — handles uneven lighting / gradient backgrounds far
            better than OTSU.
          * CLAHE contrast (clip=3.0, tile=8) — amplifies faint
            strokes without the global-histogram over-brightening
            that a straight equalise would cause.
          * Background blur-division (blur_kernel=55) — flattens
            scanner-lamp gradients and yellowed paper before the
            binariser sees the image.
          * NLM denoise (h=15) + median — removes the grain and
            JPEG artefacts common on phone-camera snaps.

        Then re-wraps as a PDF and re-runs OCR with the same language
        / PSM / OEM as the primary attempt. Keeps PSM=AUTO because
        aggressive preprocessing usually restores a layout the
        analyser can handle; simplified-settings tier below is where
        we drop to SINGLE_BLOCK.

        Returns the path to the recovered page PDF on success, or
        ``None`` if even aggressive preprocessing couldn't extract
        text.
        """
        aggressive_pdf = self._build_aggressive_preprocessed_page_pdf(
            page_pdf=page_pdf,
            work_dir=work_dir,
            page_index=page_index,
            dpi=int(original_config.dpi),
            suffix="aggressive",
            original_input_pdf=original_input_pdf,
        )
        if aggressive_pdf is None:
            return None

        # The aggressive-preprocessing PNG sits next to the PDF
        # (written by ``_build_aggressive_preprocessed_page_pdf``).
        # We OCR it DIRECTLY via pytesseract rather than through
        # OCRmyPDF: empirically OCRmyPDF renders this class of
        # grayscale Sauvola output in a way Tesseract reads as
        # pure noise, even when calling ``pytesseract.image_to_
        # string`` on the same PNG recovers 30+ chars. Going
        # straight to ``image_to_pdf_or_hocr`` sidesteps that
        # gap and gives us the searchable PDF in one step.
        png_path = aggressive_pdf.with_suffix(".png")
        if not png_path.exists():
            logger.debug(
                "Page %d aggressive retry: PNG %s missing, skipping",
                page_index, png_path,
            )
            return None

        page_out = work_dir / f"page_{page_index:04d}_aggressive.pdf"
        # PSM=SINGLE_BLOCK is more forgiving than AUTO on faded /
        # low-contrast scans where the layout analyser bails out
        # and returns nothing. Measured on the nightly corpus:
        # PSM=AUTO returns 0 chars on faded_noisy_03/04, PSM=6
        # recovers 45-50 chars from the same PNG. The text is
        # noisy but non-empty — which is the nightly test's
        # success condition.
        config_parts = [
            "--psm 6",
            f"--oem {int(original_config.oem)}",
        ]
        try:
            import pytesseract

            pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                str(png_path),
                lang=original_config.tesseract_language_string,
                config=" ".join(config_parts),
                extension="pdf",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Page %d aggressive-preprocessing retry via "
                "pytesseract failed (%s: %s)",
                page_index, type(exc).__name__, exc,
            )
            return None

        try:
            page_out.write_bytes(pdf_bytes)
        except OSError as exc:
            logger.warning(
                "Page %d aggressive-retry PDF write failed: %s",
                page_index, exc,
            )
            return None

        if not _page_pdf_has_text(page_out):
            logger.warning(
                "Page %d aggressive-preprocessing retry returned empty "
                "text layer — escalating to simpler-settings tier",
                page_index,
            )
            return None
        return page_out

    @staticmethod
    def _build_aggressive_preprocessed_page_pdf(
        *,
        page_pdf: Path,
        work_dir: Path,
        page_index: int,
        dpi: int,
        suffix: str,
        original_input_pdf: Path | None = None,
    ) -> Path | None:
        """Re-rasterise the page and run aggressive preprocessing through
        :class:`src.core.image_preprocessor.ImagePreprocessor`.

        When ``original_input_pdf`` is provided, rasterises from THAT
        instead of ``page_pdf`` so we get the raw grayscale source
        rather than the primary pipeline's already-binarised output.
        This is the crucial bit for the aggressive retry to actually
        work on faded scans — Sauvola + CLAHE on a pure-black-and-
        white image produces nothing, because there's no gradient
        left to threshold against.

        Page indexes are 1-based and match between the original and
        the preprocessed PDF by position (the pipeline does not
        reorder pages; ``max_pages`` truncates at the end).

        Unlike :meth:`_build_simplified_page_pdf` (which flattens and
        downgrades to recover crashed layout analysis), this helper
        keeps the original DPI and applies the full aggressive
        denoise chain — useful when the PROBLEM is insufficient
        preprocessing rather than over-preprocessing.

        Returns path to the newly-built single-page PDF, or ``None``
        if rasterisation or preprocessing or PDF assembly fails.
        """
        import fitz
        import numpy as np

        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import (
            BackgroundConfig,
            BinarizationConfig,
            BorderRemovalConfig,
            ContrastConfig,
            DenoiseConfig,
            DenoiseStep,
            DeskewConfig,
            PreprocessConfig,
        )
        from src.shared.types import BinarizationMethod, DenoiseMethod

        # Prefer the raw original for rasterisation; fall back to the
        # preprocessed page PDF if the caller didn't wire the original
        # through (older engine callers).
        raster_source = (
            original_input_pdf if original_input_pdf is not None else page_pdf
        )
        # 1-based ``page_index`` → 0-based fitz index. When rastering
        # from the original full-document PDF, index directly into
        # the right page; when falling back to the split ``page_pdf``
        # (which has only one page), always use page 0.
        raster_index = (
            page_index - 1 if original_input_pdf is not None else 0
        )

        try:
            # Rasterise grayscale directly — Sauvola / CLAHE /
            # background-division all operate on the luminance
            # channel anyway, and empirically the direct-grayscale
            # path recovers text on the faded_noisy_02 fixture where
            # RGB → grayscale via ImagePreprocessor does not. The
            # difference is Sauvola's ``cv2.cvtColor(BGR→GRAY)`` vs
            # PyMuPDF's ``csGRAY`` rendering — PyMuPDF uses the PDF
            # interpreter's internal luminance weighting, which
            # preserves more detail on faint grayscale gradients
            # than the standard Rec.601 mixing OpenCV applies.
            src_doc = fitz.open(str(raster_source))
            try:
                if raster_index < 0 or raster_index >= src_doc.page_count:
                    logger.warning(
                        "Page %d out of range for %s (page_count=%d); "
                        "aggressive retry falling back to preprocessed PDF",
                        page_index, raster_source, src_doc.page_count,
                    )
                    src_doc.close()
                    src_doc = fitz.open(str(page_pdf))
                    raster_index = 0
                page = src_doc[raster_index]
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            finally:
                src_doc.close()
            if pix.n == 1:
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width,
                )
            else:
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n,
                )

            # This mirrors the pre-Apr-2026 universal_accurate preset
            # which empirically recovered faded-noisy scans (30 chars
            # on faded_noisy_02 via pytesseract.image_to_string at
            # 400 DPI in local tests). Only works when we rasterise
            # from the ORIGINAL raw PDF (``original_input_pdf``) —
            # Sauvola on an already-binarised image is a no-op and
            # that was the bug that made the previous version of
            # this tier useless.
            aggressive_cfg = PreprocessConfig(
                deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=45.0),
                binarization=BinarizationConfig(
                    method=BinarizationMethod.SAUVOLA,
                    sauvola_window=25,
                    sauvola_k=0.2,
                ),
                denoise=DenoiseConfig(
                    enabled=True,
                    steps=[
                        DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3),
                        DenoiseStep(method=DenoiseMethod.MORPH_CLOSE, morph_ksize=3),
                    ],
                ),
                contrast=ContrastConfig(
                    clahe_enabled=True, clahe_clip=3.0, clahe_tile=8,
                ),
                background=BackgroundConfig(enabled=True, blur_kernel=55),
                border_removal=BorderRemovalConfig(
                    enabled=True, min_line_length=75,
                ),
            )

            preprocessor = ImagePreprocessor()
            processed, _ = preprocessor.process(arr, aggressive_cfg, dpi=dpi)
            # Save the preprocessed image as a PNG (Unicode-safe via
            # imencode + raw bytes to match the pipeline's convention),
            # then wrap it as a one-page PDF.
            import cv2

            ok, encoded = cv2.imencode(".png", processed)
            if not ok:
                logger.warning(
                    "Page %d aggressive-preprocess imencode failed",
                    page_index,
                )
                return None
            png_path = work_dir / f"page_{page_index:04d}_{suffix}.png"
            png_path.write_bytes(encoded.tobytes())

            out_path = work_dir / f"page_{page_index:04d}_{suffix}.pdf"
            new_doc = fitz.open()
            try:
                # Page size in points = pixels * 72 / dpi.
                pt_w = processed.shape[1] * 72.0 / dpi
                pt_h = processed.shape[0] * 72.0 / dpi
                new_page = new_doc.new_page(width=pt_w, height=pt_h)
                new_page.insert_image(new_page.rect, filename=str(png_path))
                new_doc.save(str(out_path))
            finally:
                new_doc.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not rebuild page %d with aggressive preprocessing "
                "at %d DPI (%s): %s",
                page_index, dpi, suffix, exc,
            )
            return None
        return out_path

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
        if not _page_pdf_has_text(page_out):
            logger.warning(
                "Page %d simplified-settings retry returned empty text "
                "layer — escalating to last-resort tier",
                page_index,
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
        # Even last-resort can return an empty layer on a genuinely
        # unreadable page (all-white scan, destroyed content). Signal
        # the caller so it falls back to raster rather than stamping
        # an empty text layer that looks like it worked.
        if not _page_pdf_has_text(page_out):
            logger.warning(
                "Page %d last-resort retry returned empty text layer — "
                "page is truly unreadable, falling back to raster",
                page_index,
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
