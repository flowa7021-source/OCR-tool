"""End-to-end OCR pipeline.

Stages per job:
    1. Analyze the input PDF (page count, per-page text presence).
    2. For every page, rasterize via PyMuPDF and preprocess via
       :class:`ImagePreprocessor`. Save preprocessed PNGs to a temp workdir.
    3. Reassemble a "preprocessed" PDF from the PNGs with PyMuPDF.
    4. Run OCRmyPDF on the preprocessed PDF to produce the final searchable PDF.
    5. Extract per-page text from the OCR'd PDF and run
       :class:`TextPostprocessor` on it.
    6. Optionally compute per-page confidence via ``pytesseract.image_to_data``
       on the preprocessed image.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.application.ocrmypdf_integration import OCRmyPDFError
from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import (
    JobResult,
    OCRJobConfig,
    PageResult,
)
from src.infrastructure.file_utils import create_temp_workdir
from src.infrastructure.tesseract_wrapper import TesseractWrapper
from src.shared.types import JobStatus

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Base class for pipeline-level failures surfaced to the UI."""


class CorruptPdfError(PipelineError):
    """Raised when PyMuPDF cannot parse the input PDF."""


class EncryptedPdfError(PipelineError):
    """Raised when the input PDF requires a password we do not have."""


class EmptyPdfError(PipelineError):
    """Raised when the input PDF reports zero pages."""


ProgressCallback = Callable[[int, int, str], None]


class OCRPipeline:
    """Full OCR pipeline orchestrator.

    Attributes:
        preprocessor: Image preprocessing engine.
        postprocessor: Text post-processing engine (duck-typed:
            ``process(text, cfg) -> str``).
        tesseract: Tesseract wrapper used for binary/tessdata discovery.
        progress_callback: Optional ``(current, total, stage)`` callback.
        compute_confidence: If True, compute per-page mean confidence via
            :func:`pytesseract.image_to_data` (slower).
    """

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        postprocessor: object,
        tesseract: TesseractWrapper,
        progress_callback: ProgressCallback | None = None,
        compute_confidence: bool = True,
        autosave_interval_pages: int = 0,
        autosave_path: Path | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            preprocessor: :class:`ImagePreprocessor` instance.
            postprocessor: Object exposing ``process(text, cfg) -> str``.
            tesseract: :class:`TesseractWrapper` (already or lazily configured).
            progress_callback: Optional progress reporter.
            compute_confidence: Whether to compute per-page confidence.
            autosave_interval_pages: Dump the partial TXT every N OCR'd
                pages. ``0`` disables partial autosave. Useful for very long
                jobs: if the process crashes between autosaves the previous
                dump is still on disk.
            autosave_path: Destination for partial TXT dumps. Defaults to the
                job output path with the ``.partial.txt`` suffix.
        """
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.tesseract = tesseract
        self.progress_callback = progress_callback
        self.compute_confidence = compute_confidence
        self.autosave_interval_pages = max(0, int(autosave_interval_pages))
        self.autosave_path = autosave_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, job: OCRJobConfig) -> JobResult:
        """Execute the pipeline for a single job.

        Args:
            job: Fully populated job configuration.

        Returns:
            Aggregated :class:`JobResult`.
        """
        job_id = uuid.uuid4().hex
        input_path = job.input
        output_path = job.output

        logger.info(
            "Starting OCR job %s: input=%s output=%s profile=%s",
            job_id,
            input_path,
            output_path,
            job.profile.name,
        )

        started = time.time()
        result = JobResult(
            job_id=job_id,
            status=JobStatus.RUNNING,
            input_path=str(input_path),
            output_path=str(output_path),
        )

        workdir: Path | None = None
        try:
            self._ensure_tesseract_configured()

            workdir = create_temp_workdir(prefix="ocrjob_")
            logger.debug("Workdir for job %s: %s", job_id, workdir)

            # 1. Analyze
            page_infos = self._analyze_pdf(input_path)
            total_pages = len(page_infos)
            logger.info("Job %s: %d pages detected", job_id, total_pages)
            self._report(0, total_pages, "analyze")

            # 2. Preprocess pages -> PNGs
            page_results: list[PageResult] = []
            png_paths: list[Path] = []
            for idx, _info in enumerate(page_infos):
                page_num = idx + 1
                t0 = time.time()
                png_path = workdir / f"page_{page_num:05d}.png"
                try:
                    img = self._rasterize_page(
                        input_path, idx, dpi=job.profile.ocr.dpi
                    )
                    processed, angle = self.preprocessor.process(
                        img, job.profile.preprocess
                    )
                    self._save_png(processed, png_path)
                    png_paths.append(png_path)
                    pr = PageResult(
                        page_number=page_num,
                        text="",
                        skew_angle=float(angle),
                        processing_time_sec=time.time() - t0,
                    )
                    page_results.append(pr)
                except Exception as exc:  # noqa: BLE001 - capture, continue
                    logger.exception(
                        "Preprocess failure on page %d of %s", page_num, input_path
                    )
                    page_results.append(
                        PageResult(
                            page_number=page_num,
                            error=f"preprocess: {exc}",
                            processing_time_sec=time.time() - t0,
                        )
                    )
                self._report(page_num, total_pages, "preprocess")

            if not png_paths:
                raise RuntimeError("Не удалось подготовить ни одной страницы")

            # 3. Assemble preprocessed PDF
            preprocessed_pdf = workdir / "preprocessed.pdf"
            self._assemble_pdf(png_paths, preprocessed_pdf)
            self._report(total_pages, total_pages, "assemble")

            # 4. OCR — dispatch to the engine selected by profile.ocr.engine.
            from src.application.engines import get_engine
            from src.application.engines.base import EngineNotAvailableError

            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                engine = get_engine(job.profile.ocr.engine)
                engine_results = engine.run(
                    preprocessed_pdf=preprocessed_pdf,
                    output_pdf=output_path,
                    config=job.profile.ocr,
                    progress_callback=lambda c, t, s: self._report(
                        # Keep page-level progress monotonic across stages.
                        total_pages * c // max(1, t), total_pages, s
                    ),
                )
            except (OCRmyPDFError, EngineNotAvailableError, KeyError) as exc:
                result.status = JobStatus.FAILED
                result.error = str(exc)
                result.pages = page_results
                result.total_time_sec = time.time() - started
                logger.error("Job %s failed during OCR engine: %s", job_id, exc)
                return result

            # Engines may pre-populate text/word_boxes (e.g. GOT-OCR2);
            # for Tesseract these stubs stay empty and step 5 fills them
            # by reading the produced searchable PDF.
            if engine_results:
                for stub, page_result in zip(
                    engine_results, page_results, strict=False
                ):
                    if stub.text and not page_result.text:
                        page_result.text = stub.text
                    if stub.mean_confidence and not page_result.mean_confidence:
                        page_result.mean_confidence = stub.mean_confidence
            self._report(total_pages, total_pages, "ocr")

            # 5. Extract per-page text, postprocess
            self._extract_and_postprocess(
                output_path, page_results, job, png_paths
            )
            self._report(total_pages, total_pages, "postprocess")

            result.pages = page_results
            result.status = JobStatus.COMPLETED
            result.total_time_sec = time.time() - started
            logger.info(
                "Job %s completed in %.2fs (avg conf=%.1f)",
                job_id,
                result.total_time_sec,
                result.average_confidence,
            )
            return result

        except (CorruptPdfError, EncryptedPdfError, EmptyPdfError) as exc:
            logger.warning("Job %s aborted: %s", job_id, exc)
            result.status = JobStatus.FAILED
            result.error = str(exc)
            result.total_time_sec = time.time() - started
            return result
        except Exception as exc:  # noqa: BLE001 - top-level failure
            logger.exception("Job %s failed", job_id)
            result.status = JobStatus.FAILED
            result.error = str(exc)
            result.total_time_sec = time.time() - started
            return result
        finally:
            if workdir is not None:
                self._cleanup(workdir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_tesseract_configured(self) -> None:
        """Configure pytesseract if not already configured."""
        try:
            if not TesseractWrapper._configured:
                self.tesseract.configure_pytesseract()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tesseract configuration failed: %s", exc)

    def _report(self, current: int, total: int, stage: str) -> None:
        """Invoke the progress callback, swallowing any errors."""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(current, total, stage)
        except Exception:  # noqa: BLE001
            logger.debug("progress_callback raised", exc_info=True)

    def _analyze_pdf(self, pdf_path: Path) -> list[dict]:
        """Inspect ``pdf_path`` and return per-page metadata.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            List of per-page dicts ``{"page_number", "has_text"}``.

        Raises:
            EmptyPdfError: If the document reports zero pages.
            EncryptedPdfError: If the document is password-protected and we
                cannot authenticate with an empty password.
            CorruptPdfError: If PyMuPDF fails to parse the file at all.
        """
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001 — PyMuPDF raises a variety
            raise CorruptPdfError(
                f"Не удалось открыть PDF (повреждён или неверный формат): {pdf_path}"
            ) from exc

        try:
            # Encrypted PDFs: try empty password, otherwise bail out with a
            # typed error. OCRmyPDF can also decrypt but would fail silently
            # for a missing password.
            if getattr(doc, "needs_pass", False):
                ok = False
                try:
                    ok = bool(doc.authenticate(""))
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    raise EncryptedPdfError(
                        f"PDF защищён паролем и не может быть распознан без него: {pdf_path.name}"
                    )

            if doc.page_count == 0:
                raise EmptyPdfError(f"PDF не содержит страниц: {pdf_path.name}")

            infos: list[dict] = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = page.get_text("text") or ""
                infos.append(
                    {
                        "page_number": i + 1,
                        "has_text": bool(text.strip()),
                    }
                )
            return infos
        finally:
            doc.close()

    def _rasterize_page(
        self, pdf_path: Path, page_index: int, dpi: int
    ) -> np.ndarray:
        """Render one PDF page to a numpy array at the given DPI.

        Args:
            pdf_path: Input PDF path.
            page_index: 0-based page index.
            dpi: Target DPI for rasterization.

        Returns:
            HxWxC uint8 numpy array (RGB) or HxW for grayscale pixmaps.
        """
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            width, height, n = pix.width, pix.height, pix.n
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(height, width, n)
            # PyMuPDF returns RGB; OpenCV operations in preprocessor usually
            # accept any ordering, but we convert to BGR so contrast/denoise
            # steps that assume BGR behave as on typical OpenCV pipelines.
            if n == 3:
                import cv2

                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif n == 4:
                import cv2

                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif n == 1:
                arr = arr.reshape(height, width)
            return arr
        finally:
            doc.close()

    def _save_png(self, image: np.ndarray, path: Path) -> None:
        """Write a numpy image to PNG on disk."""
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"Не удалось сохранить PNG: {path}")

    def _assemble_pdf(self, png_paths: list[Path], output_pdf: Path) -> None:
        """Assemble a PDF from a list of PNGs (one page per image).

        Args:
            png_paths: Ordered list of PNG files.
            output_pdf: Output PDF path.
        """
        import fitz

        doc = fitz.open()
        try:
            for png_path in png_paths:
                # Probe image dimensions via a temporary pixmap.
                pix = fitz.Pixmap(str(png_path))
                try:
                    width = float(pix.width)
                    height = float(pix.height)
                finally:
                    pix = None  # noqa: F841 - release native resource

                page = doc.new_page(width=width, height=height)
                rect = page.rect
                page.insert_image(rect, filename=str(png_path))
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(output_pdf))
        finally:
            doc.close()

    def _extract_and_postprocess(
        self,
        ocrd_pdf: Path,
        page_results: list[PageResult],
        job: OCRJobConfig,
        png_paths: list[Path],
    ) -> None:
        """Populate ``page_results`` with OCR'd text and confidence.

        Args:
            ocrd_pdf: Path to the searchable PDF produced by OCRmyPDF.
            page_results: List to populate in-place (page_number already set).
            job: Original job config (for postprocess + OCR config).
            png_paths: Preprocessed PNGs used for optional confidence scan.
        """
        import fitz

        doc = fitz.open(str(ocrd_pdf))
        try:
            for idx, pr in enumerate(page_results):
                if pr.error is not None:
                    # Skip pages that failed preprocess, but still try to pull
                    # whatever text the OCR'd PDF may have for that page.
                    pass
                if idx >= doc.page_count:
                    continue
                try:
                    if pr.text:
                        # Engine (e.g. GOT-OCR2) already produced text —
                        # postprocess that, don't re-read from the PDF where
                        # the layout serialisation may differ.
                        pr.text = self._postprocess_text(
                            pr.text, job.profile.postprocess
                        )
                    else:
                        page = doc.load_page(idx)
                        raw_text = page.get_text("text") or ""
                        pr.text = self._postprocess_text(
                            raw_text, job.profile.postprocess
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to extract text for page %d: %s",
                        pr.page_number,
                        exc,
                    )
                    pr.error = (pr.error or "") + f"; extract: {exc}"

                # Periodic partial-result autosave.
                if (
                    self.autosave_interval_pages > 0
                    and (idx + 1) % self.autosave_interval_pages == 0
                ):
                    self._autosave_partial_txt(page_results[: idx + 1], job)
        finally:
            doc.close()

        # Only run pytesseract-based confidence scoring when the OCR engine
        # was Tesseract. Other engines (GOT-OCR 2.0, future TrOCR) populate
        # mean_confidence themselves; re-scoring with pytesseract would
        # overwrite that with a number derived from a different model.
        from src.shared.types import OCREngineKind

        if self.compute_confidence and job.profile.ocr.engine is OCREngineKind.TESSERACT:
            self._compute_confidences(page_results, job, png_paths)

    def _autosave_partial_txt(
        self, partial_pages: list[PageResult], job: OCRJobConfig
    ) -> None:
        """Dump a running TXT of what has been OCR'd so far.

        Best-effort: any I/O error is logged and swallowed so autosave never
        aborts a running job.
        """
        try:
            from src.shared.constants import UI_PAGE_NUM_FORMAT

            target = self.autosave_path or Path(
                str(job.output_path) + ".partial.txt"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            parts: list[str] = []
            for pr in partial_pages:
                parts.append(UI_PAGE_NUM_FORMAT.format(num=pr.page_number))
                parts.append("")
                if pr.error:
                    parts.append(f"[ERROR: {pr.error}]")
                else:
                    parts.append(pr.text or "")
                parts.append("")
            # Atomic write
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
            tmp.replace(target)
            logger.info(
                "Autosave: wrote %d partial pages to %s",
                len(partial_pages),
                target,
            )
        except OSError as exc:
            logger.warning("Partial autosave failed: %s", exc)

    def _postprocess_text(self, text: str, cfg) -> str:
        """Run :class:`TextPostprocessor` if available."""
        if self.postprocessor is None or not text:
            return text
        try:
            return self.postprocessor.process(text, cfg)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postprocess failed: %s", exc)
            return text

    def _compute_confidences(
        self,
        page_results: list[PageResult],
        job: OCRJobConfig,
        png_paths: list[Path],
    ) -> None:
        """Compute per-page confidence using ``pytesseract.image_to_data``."""
        try:
            import cv2
            import pytesseract
        except ImportError as exc:
            logger.warning("pytesseract unavailable, skipping confidence: %s", exc)
            return

        lang = job.profile.ocr.tesseract_language_string
        cfg_parts = [
            f"--psm {int(job.profile.ocr.psm)}",
            f"--oem {int(job.profile.ocr.oem)}",
        ]
        if job.profile.ocr.char_whitelist:
            cfg_parts.append(
                f"-c tessedit_char_whitelist={job.profile.ocr.char_whitelist}"
            )
        if job.profile.ocr.char_blacklist:
            cfg_parts.append(
                f"-c tessedit_char_blacklist={job.profile.ocr.char_blacklist}"
            )
        tess_cfg = " ".join(cfg_parts)

        threshold = float(job.profile.ocr.confidence_threshold)

        for pr, png_path in zip(page_results, png_paths, strict=False):
            if pr.error is not None:
                continue
            try:
                img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                data = pytesseract.image_to_data(
                    img,
                    lang=lang,
                    config=tess_cfg,
                    output_type=pytesseract.Output.DICT,
                )
                confidences: list[float] = []
                low_words: list[str] = []
                for conf, word in zip(
                    data.get("conf", []), data.get("text", []), strict=False
                ):
                    try:
                        c = float(conf)
                    except (TypeError, ValueError):
                        continue
                    if c < 0:
                        continue
                    if not word or not word.strip():
                        continue
                    confidences.append(c)
                    if c < threshold:
                        low_words.append(word)
                if confidences:
                    pr.mean_confidence = sum(confidences) / len(confidences)
                    pr.low_confidence_words = low_words
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Confidence computation failed for page %d: %s",
                    pr.page_number,
                    exc,
                )

    def _cleanup(self, workdir: Path) -> None:
        """Remove the temporary workdir, logging but not raising on error."""
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            logger.debug("Removed workdir: %s", workdir)
        except OSError as exc:  # pragma: no cover - platform-specific
            logger.warning("Failed to remove workdir %s: %s", workdir, exc)
