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

from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import (
    JobResult,
    OCRJobConfig,
    PageResult,
)
from src.core.text_postprocessor import TextPostprocessor
from src.infrastructure.file_utils import create_temp_workdir
from src.shared.types import JobStatus

logger = logging.getLogger(__name__)


def _resolve_stage_parallelism(
    *, env_var: str, fallback: int,
) -> int:
    """Return the worker cap for a pipeline stage.

    Checks ``os.environ[env_var]`` first; if present and a positive
    integer, returns it. Otherwise returns ``fallback``. Invalid
    values (non-integers, zero, negative) are logged and ignored so
    the stage falls back to the code default — safer than crashing
    the pipeline over a typo.

    Env vars, defaults (kept in sync with the docstrings on the
    call sites):

      * ``OCR_PREPROCESS_WORKERS`` — default 10
      * ``OCR_POSTPROCESS_WORKERS`` — default 10
      * per-page OCR workers are capped separately in
        :mod:`src.application.engines.easyocr_engine` via
        ``OCR_PER_PAGE_WORKERS`` (default 4 on CPU)
    """
    import os

    raw = os.environ.get(env_var)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using default %d",
            env_var, raw, fallback,
        )
        return fallback
    if value < 1:
        logger.warning(
            "Ignoring non-positive %s=%d; using default %d",
            env_var, value, fallback,
        )
        return fallback
    return value


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
        progress_callback: Optional ``(current, total, stage)`` callback.
        compute_confidence: If True, compute per-page mean confidence
            from the engine's word_boxes output (near-free).
    """

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        postprocessor: TextPostprocessor | None,
        progress_callback: ProgressCallback | None = None,
        compute_confidence: bool = True,
        autosave_interval_pages: int = 0,
        autosave_path: Path | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
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

        # Persistent cache short-circuit. If the same (input bytes,
        # profile hash) was processed before, copy the cached PDF and
        # rebuild the JobResult from the stored metadata — skipping
        # rasterisation, OCR, and post-processing entirely. This is the
        # single biggest win for the "tune a profile, re-run" workflow.
        #
        # Escape-hatch: env var ``OCR_DISABLE_CACHE=1`` отключает и
        # lookup, и последующий store. Нужно для тестов / отладки,
        # чтобы точно знать что видишь результат текущего кода, а не
        # удачный хит с прошлого прогона (классическая ошибка
        # "fix работает" когда на самом деле просто cache-hit).
        import os as _os
        cache_disabled = _os.environ.get("OCR_DISABLE_CACHE", "").lower() in (
            "1", "true", "yes",
        )
        if cache_disabled:
            logger.info(
                "OCR cache disabled via OCR_DISABLE_CACHE env var; "
                "skipping lookup for job %s", job_id,
            )
        cached = None if cache_disabled else self._try_cache_hit(
            input_path, job.profile, output_path, job_id,
        )
        if cached is not None:
            self._report(cached.page_count or 1, cached.page_count or 1, "cache-hit")
            # Profile may have flipped ``extract.enabled=True`` since
            # this cache entry was stored — run the parser on the
            # cached text so the user still gets structured output.
            self._maybe_run_parser(cached, job)
            return cached

        workdir: Path | None = None
        try:
            # 0. Pre-flight. Runs in ~1 second and fails fast if the
            #    selected engine is not actually usable — missing
            #    Tesseract binary or an incomplete tessdata directory.
            #    Without this check a 600 DPI / 4-page job used to
            #    spend ~30 seconds on
            #    preprocessing BEFORE discovering the engine was
            #    misconfigured. Now the user finds out immediately.
            engine_kind = job.profile.ocr.engine
            self._report(0, 1, "preflight")
            try:
                from src.application.engines import get_engine
                from src.application.engines.base import EngineNotAvailableError

                engine = get_engine(engine_kind)
                ok, msg = engine.is_available()
                if not ok:
                    logger.error(
                        "Job %s preflight FAILED: engine=%s not available: %s",
                        job_id, engine_kind, msg,
                    )
                    result.status = JobStatus.FAILED
                    result.error = msg
                    result.total_time_sec = time.time() - started
                    return result
                logger.info(
                    "Job %s preflight OK: engine=%s is_available", job_id, engine_kind
                )
            except (EngineNotAvailableError, KeyError) as exc:
                logger.error(
                    "Job %s preflight FAILED: %s", job_id, exc, exc_info=True
                )
                result.status = JobStatus.FAILED
                result.error = str(exc)
                result.total_time_sec = time.time() - started
                return result

            workdir = create_temp_workdir(prefix="ocrjob_")
            logger.info("Job %s workdir: %s", job_id, workdir)

            # 1. Analyze
            logger.info("Job %s stage=analyze: opening PDF", job_id)
            t_stage = time.time()
            page_infos = self._analyze_pdf(input_path)
            full_page_count = len(page_infos)
            max_pages = int(getattr(job.profile.ocr, "max_pages", 0) or 0)
            if max_pages > 0 and full_page_count > max_pages:
                logger.warning(
                    "Job %s: preview mode — processing first %d of %d pages "
                    "(profile has max_pages=%d). Set max_pages=0 in the "
                    "profile to disable the limit.",
                    job_id, max_pages, full_page_count, max_pages,
                )
                page_infos = page_infos[:max_pages]
            total_pages = len(page_infos)
            logger.info(
                "Job %s stage=analyze done in %.2fs: %d page(s) to process "
                "(document has %d total)",
                job_id, time.time() - t_stage, total_pages, full_page_count,
            )

            # Text-layer bypass. If skip_text=True AND every page
            # already carries a substantial digital text layer, skip
            # the entire rasterise/preprocess/OCR pipeline — copy the
            # input to the output (truncated to max_pages when set),
            # extract text straight from the font stream via PyMuPDF,
            # and run it through the profile's postprocessor. This is
            # the fast path for digital PDFs (exports from Word,
            # LaTeX, reports generated by back-office systems): what
            # used to be a minutes-long OCR run becomes a sub-second
            # file copy + text extract with zero accuracy loss (the
            # text is ground truth, not an OCR guess).
            #
            # Only bypasses when every page passes a substance check
            # (≥ 50 non-whitespace chars and ≥ 3 word-like tokens).
            # Short-title pages / decorative covers fall through to
            # the full pipeline so we don't silently skip real content.
            skip_text = getattr(job.profile.ocr, "skip_text", True)
            if skip_text:
                bypass_result = self._try_text_layer_bypass(
                    input_path=input_path,
                    output_path=output_path,
                    page_infos=page_infos,
                    job=job,
                    job_id=job_id,
                    started=started,
                )
                if bypass_result is not None:
                    # Digital PDFs (Word/LaTeX exports) skip OCR but
                    # still benefit from structured extraction —
                    # call the parser on the extracted text layer.
                    self._maybe_run_parser(bypass_result, job)
                    return bypass_result

            self._report(0, total_pages, "analyze")

            # 2. Preprocess pages -> PNGs (parallel across pages).
            logger.info(
                "Job %s stage=preprocess: rasterising + cleaning %d page(s) "
                "at %d DPI",
                job_id, total_pages,
                int(getattr(job.profile.ocr, "dpi", 300)),
            )
            t_stage = time.time()
            page_results, png_paths = self._preprocess_pages_parallel(
                input_path=input_path,
                workdir=workdir,
                page_count=total_pages,
                profile=job.profile,
            )
            logger.info(
                "Job %s stage=preprocess done in %.2fs: %d/%d page(s) ready",
                job_id, time.time() - t_stage, len(png_paths), total_pages,
            )

            if not png_paths:
                raise RuntimeError("Не удалось подготовить ни одной страницы")

            # 3. Assemble preprocessed PDF
            preprocessed_pdf = workdir / "preprocessed.pdf"
            logger.info(
                "Job %s stage=assemble: building %s from %d PNG(s)",
                job_id, preprocessed_pdf.name, len(png_paths),
            )
            t_stage = time.time()
            self._assemble_pdf(
                png_paths,
                preprocessed_pdf,
                dpi=int(getattr(job.profile.ocr, "dpi", 300) or 300),
            )
            logger.info(
                "Job %s stage=assemble done in %.2fs (preprocessed.pdf = %d bytes)",
                job_id, time.time() - t_stage,
                preprocessed_pdf.stat().st_size if preprocessed_pdf.exists() else -1,
            )
            self._report(total_pages, total_pages, "assemble")

            # 4. OCR — dispatch to the engine selected by profile.ocr.engine.
            from src.application.engines import get_engine

            engine_kind = job.profile.ocr.engine
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Job %s stage=ocr: engine=%s lang=%s dpi=%d",
                job_id, engine_kind,
                "+".join(job.profile.ocr.languages),
                int(job.profile.ocr.dpi),
            )
            t_stage = time.time()
            try:
                engine = get_engine(engine_kind)
                progress_fn = lambda c, t, s: self._report(  # noqa: E731
                    total_pages * c // max(1, t), total_pages, s
                )
                # Give the engine the RAW user PDF so retry tiers
                # that need to re-preprocess
                # (``_retry_page_with_aggressive_preprocessing``) can
                # rasterise the untouched source instead of the
                # already-binarised preprocessed PDF. Aggressive-on-
                # top-of-binary is a no-op / destructive combo — the
                # original is what aggressive preprocessing needs.
                # Engines / stubs that predate this kwarg fall back
                # to the older four-argument signature.
                try:
                    engine_results = engine.run(
                        preprocessed_pdf=preprocessed_pdf,
                        output_pdf=output_path,
                        config=job.profile.ocr,
                        progress_callback=progress_fn,
                        original_input_pdf=input_path,
                    )
                except TypeError as te:
                    if "original_input_pdf" not in str(te):
                        raise
                    logger.debug(
                        "Engine %s does not accept original_input_pdf "
                        "kwarg — falling back to legacy 4-arg call",
                        engine_kind,
                    )
                    engine_results = engine.run(
                        preprocessed_pdf=preprocessed_pdf,
                        output_pdf=output_path,
                        config=job.profile.ocr,
                        progress_callback=progress_fn,
                    )
            except (EngineNotAvailableError, KeyError, RuntimeError) as exc:
                result.status = JobStatus.FAILED
                result.error = str(exc)
                result.pages = page_results
                result.total_time_sec = time.time() - started
                logger.error(
                    "Job %s stage=ocr FAILED after %.2fs: %s",
                    job_id, time.time() - t_stage, exc, exc_info=True,
                )
                return result
            logger.info(
                "Job %s stage=ocr done in %.2fs; output %s (%d bytes)",
                job_id, time.time() - t_stage, output_path.name,
                output_path.stat().st_size if output_path.exists() else -1,
            )

            if engine_results:
                for stub, page_result in zip(
                    engine_results, page_results, strict=False
                ):
                    if stub.text and not page_result.text:
                        page_result.text = stub.text
                    if stub.mean_confidence and not page_result.mean_confidence:
                        page_result.mean_confidence = stub.mean_confidence
                    if stub.word_boxes:
                        page_result.word_boxes = stub.word_boxes
            self._report(total_pages, total_pages, "ocr")

            # 5. Extract per-page text, postprocess
            logger.info(
                "Job %s stage=postprocess: extracting text + applying "
                "post-filters to %d page(s)",
                job_id, len(page_results),
            )
            t_stage = time.time()
            self._extract_and_postprocess(
                output_path, page_results, job, png_paths
            )
            logger.info(
                "Job %s stage=postprocess done in %.2fs",
                job_id, time.time() - t_stage,
            )
            self._report(total_pages, total_pages, "postprocess")

            result.pages = page_results
            # Post-OCR structured-field extraction (ТН / УПД parser)
            # runs AFTER postprocess so it sees the same text the
            # user will see in the results panel. Populates
            # ``result.parsed`` when the profile asks for it; else
            # leaves it ``None``.
            self._maybe_run_parser(result, job)
            result.status = JobStatus.COMPLETED
            result.total_time_sec = time.time() - started

            # Detect "COMPLETED but nothing recognised" — surface a
            # clear warning so the user isn't left staring at an empty
            # searchable PDF wondering if the app is broken.
            all_empty = all(
                not (p.text or "").strip() for p in result.pages
            ) if result.pages else True
            if all_empty:
                logger.warning(
                    "Job %s COMPLETED but NO text was recognised on any "
                    "page. Likely causes: damaged scan, Tesseract timed "
                    "out, или preprocessing уничтожил глифы. Создайте "
                    "копию профиля и понизьте DPI / включите более "
                    "агрессивный denoise (CLAHE clip 4.0, NLM h=15).",
                    job_id,
                )
                result.error = (
                    "Документ обработан, но текст не был распознан "
                    "ни на одной странице. Создайте копию профиля "
                    "universal_accurate и снизьте DPI или включите "
                    "более агрессивный денойз."
                )

            # Wrong-profile hint — после удаления fast/slow профилей
            # (декабрь 2026) рекомендация переключиться на специальный
            # builtin больше не релевантна: единственный builtin
            # ``universal_accurate`` уже включает все best-of-all
            # настройки. Если confidence остался низким, проблема
            # либо в самом скане (низкое DPI / artefacts), либо в
            # пользовательской копии профиля с урезанным препроцессингом.
            # Предлагаем — но не предписываем — пересохранить копию
            # профиля или вернуться на builtin.
            if not all_empty and result.pages:
                avg_conf = result.average_confidence
                if 0.0 < avg_conf < 60.0:
                    current_profile = job.profile.name
                    if current_profile != "universal_accurate":
                        logger.warning(
                            "Job %s finished at %.1f%% mean confidence "
                            "with custom profile %r. Try the builtin "
                            "'universal_accurate' to compare — оно "
                            "включает Sauvola + полный postprocessing + "
                            "адаптивный confidence threshold.",
                            job_id, avg_conf, current_profile,
                        )
                        import contextlib

                        with contextlib.suppress(Exception):
                            self._report(
                                total_pages,
                                total_pages,
                                (
                                    f"profile_recommendation:{avg_conf:.0f}:"
                                    "universal_accurate"
                                ),
                            )
                    else:
                        logger.warning(
                            "Job %s finished at %.1f%% mean confidence on "
                            "the canonical builtin profile. Это указывает "
                            "на качество скана (low DPI / artefacts), а не "
                            "на конфигурацию OCR.",
                            job_id, avg_conf,
                        )

            logger.info(
                "Job %s COMPLETED in %.2fs (avg conf=%.1f, pages=%d, out=%s)",
                job_id,
                result.total_time_sec,
                result.average_confidence,
                len(result.pages),
                output_path,
            )
            self._try_cache_store(input_path, job.profile, output_path, result)
            return result

        except (CorruptPdfError, EncryptedPdfError, EmptyPdfError) as exc:
            logger.warning(
                "Job %s aborted (typed PDF error): %s", job_id, exc
            )
            result.status = JobStatus.FAILED
            result.error = str(exc)
            result.total_time_sec = time.time() - started
            return result
        except Exception as exc:  # noqa: BLE001 - top-level failure
            logger.exception("Job %s FAILED with unexpected error", job_id)
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

    def _try_cache_hit(
        self,
        input_path: Path,
        profile,  # noqa: ANN001
        output_path: Path,
        job_id: str,
    ) -> JobResult | None:
        """Return a pre-built JobResult if the cache has this (input, profile)."""
        try:
            from src.infrastructure import ocr_cache
        except ImportError:  # pragma: no cover
            return None
        try:
            hit = ocr_cache.lookup(input_path, profile)
        except Exception as exc:  # noqa: BLE001 — never let cache fail the job
            logger.debug("Cache lookup failed: %s", exc)
            return None
        if hit is None:
            return None
        cached_pdf, meta = hit
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_pdf, output_path)
        except OSError as exc:
            logger.warning("Could not materialise cached PDF: %s", exc)
            return None
        # Rebuild a JobResult from the stored dict. Any schema drift
        # (extra / missing keys) falls back to a miss.
        try:
            pages_data = meta.get("pages", [])
            pages = [
                PageResult(
                    page_number=int(p.get("page_number", i + 1)),
                    text=str(p.get("text", "")),
                    mean_confidence=float(p.get("mean_confidence", 0.0)),
                    low_confidence_words=list(p.get("low_confidence_words", [])),
                    processing_time_sec=float(p.get("processing_time_sec", 0.0)),
                    error=p.get("error"),
                    skew_angle=float(p.get("skew_angle", 0.0)),
                )
                for i, p in enumerate(pages_data)
            ]
            result = JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                input_path=str(input_path),
                output_path=str(output_path),
                pages=pages,
                total_time_sec=0.0,
                error=None,
            )
            logger.info(
                "Job %s: served from cache (%d pages, cached PDF=%s)",
                job_id, len(pages), cached_pdf,
            )
            return result
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("Cached metadata unusable: %s", exc)
            return None

    def _try_cache_store(
        self,
        input_path: Path,
        profile,  # noqa: ANN001
        output_path: Path,
        result: JobResult,
    ) -> None:
        """Best-effort write of a completed job into the OCR cache.

        Honours ``AppSettings.ocr_cache_max_mb``. A zero value disables
        caching outright (for users on tight disk budgets); any other
        value caps the total cache size at that many megabytes with
        LRU eviction.

        We **refuse to cache a result where every page came back with
        zero recognised characters** — that's almost always a broken
        configuration (wrong DPI reporting, missing tessdata, a
        corrupted preprocessed image) rather than a genuinely empty
        document, and caching it poisons every subsequent attempt on
        the same input: the cache short-circuits before we can fix
        the root cause, the user re-runs and gets the same "empty"
        output forever. Letting the empty result skip the cache means
        a fix to the underlying problem immediately takes effect on
        the next run.
        """
        try:
            from src.infrastructure import ocr_cache
            from src.infrastructure.config_storage import SettingsStorage

            try:
                settings = SettingsStorage().load()
                cap_mb = int(settings.ocr_cache_max_mb)
                min_conf = float(settings.ocr_cache_min_confidence)
            except Exception:  # noqa: BLE001
                cap_mb = 2048
                min_conf = 50.0
            if cap_mb <= 0:
                logger.debug("OCR cache disabled (max_mb=0) — skipping store")
                return

            # Refuse to cache a "recognized nothing" result. Every page
            # with no text AND no confidence indicates a pipeline
            # failure rather than a legitimately blank document.
            if result.pages and all(
                not (p.text or "").strip() and p.mean_confidence <= 0
                for p in result.pages
            ):
                logger.warning(
                    "Cache SKIPPED: every page is empty with zero "
                    "confidence — treating as a broken run rather than "
                    "a legitimate blank document. Not poisoning the "
                    "cache for subsequent attempts on the same input."
                )
                return

            # Low-confidence floor. Second poison-prevention guard after
            # the empty-pages check: catches a run that completed
            # structurally but produced mostly-garbage text — wrong
            # profile for the document, truly unreadable scan, or the
            # simplified-settings retry tier succeeded at 150 DPI with
            # compromised accuracy. Caching that result would make every
            # subsequent attempt on the same file short-circuit to the
            # same garbage before the user can try a better profile.
            # ``min_conf == 0.0`` disables the floor (legacy behaviour).
            if min_conf > 0.0 and result.pages:
                mean_conf = result.average_confidence
                if 0.0 < mean_conf < min_conf:
                    logger.warning(
                        "Cache SKIPPED: mean_confidence %.1f%% below "
                        "floor %.1f%% — likely wrong profile or damaged "
                        "scan. Not poisoning the cache; re-run with a "
                        "different profile to try again.",
                        mean_conf, min_conf,
                    )
                    return

            import os as _os
            if _os.environ.get("OCR_DISABLE_CACHE", "").lower() in (
                "1", "true", "yes",
            ):
                logger.debug("OCR cache disabled via env; skipping store")
                return

            ocr_cache.store(
                input_path,
                profile,
                output_pdf=output_path,
                job_result=result,
                max_bytes=cap_mb * 1024 * 1024,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache store failed: %s", exc)

    def _try_text_layer_bypass(
        self,
        *,
        input_path: Path,
        output_path: Path,
        page_infos: list[dict],
        job: OCRJobConfig,
        job_id: str,
        started: float,
    ) -> JobResult | None:
        """Fast path for digital PDFs — return a completed JobResult
        without running the OCR pipeline, or None if not applicable.

        Triggered when every page of the input PDF already carries a
        substantial digital text layer (≥ 50 non-whitespace chars and
        ≥ 3 word-like tokens). In that case there is nothing for OCR
        to add — the text in the PDF IS the answer, at 100 %
        confidence. Rasterising + preprocessing + re-OCR would be an
        expensive no-op with a real risk of degrading the output
        (re-OCR can produce slightly different characters than the
        original font stream, especially on unusual ligatures or
        Cyrillic punctuation).

        Args:
            input_path: Input PDF path.
            output_path: Where to write the result (a copy of input,
                optionally truncated to ``max_pages``).
            page_infos: Output of ``_analyze_pdf`` — we reuse the
                total-pages count and skip a second ``fitz.open`` for
                document metadata.
            job: Full job configuration.
            job_id: Short hex ID for logging.
            started: Wall-clock time when the job started, for the
                final ``total_time_sec`` field.

        Returns:
            Completed :class:`JobResult` on bypass, or ``None`` when
            the bypass does not apply (short pages, decorative-only
            content, extraction error, etc.).
        """
        import re
        import shutil

        import fitz

        if not page_infos:
            return None

        max_pages = int(getattr(job.profile.ocr, "max_pages", 0) or 0)
        effective_pages = page_infos
        if max_pages > 0 and len(effective_pages) > max_pages:
            effective_pages = effective_pages[:max_pages]
        total_pages = len(effective_pages)

        # Extract text for every candidate page and check substance.
        # Words ≥ 2 chars weeds out stray punctuation; 50 chars + 3
        # words per page weeds out decorative covers without
        # false-positives on real business-document pages.
        word_re = re.compile(r"\w{2,}", re.UNICODE)
        try:
            doc = fitz.open(str(input_path))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Text-layer bypass: fitz.open(%s) failed (%s) — "
                "falling through to full pipeline",
                input_path, exc,
            )
            return None

        try:
            texts: list[str] = []
            for idx in range(total_pages):
                try:
                    raw = doc.load_page(idx).get_text("text") or ""
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Text-layer bypass: get_text(page=%d) failed: %s",
                        idx + 1, exc,
                    )
                    return None
                stripped = raw.strip()
                if len(stripped) < 50:
                    return None
                if len(word_re.findall(stripped)) < 3:
                    return None
                texts.append(raw)
        finally:
            doc.close()

        logger.info(
            "Job %s: text-layer bypass — all %d page(s) carry a "
            "substantial digital text layer; skipping rasterise / "
            "preprocess / OCR. Copying input → output and extracting "
            "text via PyMuPDF.",
            job_id, total_pages,
        )
        self._report(0, total_pages, "text-layer-bypass")

        # Materialise the output. Byte-identical copy when no truncation
        # is needed; otherwise rebuild a truncated PDF via fitz.
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if max_pages > 0 and max_pages < len(page_infos):
                src_doc = fitz.open(str(input_path))
                try:
                    dst_doc = fitz.open()
                    try:
                        dst_doc.insert_pdf(
                            src_doc, from_page=0, to_page=max_pages - 1
                        )
                        dst_doc.save(str(output_path))
                    finally:
                        dst_doc.close()
                finally:
                    src_doc.close()
            else:
                shutil.copy2(str(input_path), str(output_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Text-layer bypass: failed to materialise output (%s) "
                "— falling through to full pipeline",
                exc,
            )
            return None

        # Build per-page results. 100 % confidence reflects that this is
        # ground-truth text from the digital PDF, not an OCR guess.
        pages: list[PageResult] = []
        for idx, raw in enumerate(texts):
            page_num = idx + 1
            t0 = time.time()
            final_text = self._postprocess_text(
                raw, job.profile.postprocess
            )
            pages.append(
                PageResult(
                    page_number=page_num,
                    text=final_text,
                    mean_confidence=100.0,
                    processing_time_sec=time.time() - t0,
                )
            )
            self._report(page_num, total_pages, "text-layer-bypass")

        result = JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            input_path=str(input_path),
            output_path=str(output_path),
            pages=pages,
            total_time_sec=time.time() - started,
        )
        logger.info(
            "Job %s: text-layer bypass COMPLETED in %.2fs, "
            "%d page(s), mean_confidence=100.0%%",
            job_id, result.total_time_sec, total_pages,
        )
        return result

    def _preprocess_pages_parallel(
        self,
        *,
        input_path: Path,
        workdir: Path,
        page_count: int,
        profile,  # noqa: ANN001 — ProfileData, imported via type-check cycle
    ) -> tuple[list[PageResult], list[Path]]:
        """Rasterise + preprocess ``page_count`` pages using a thread pool.

        Returns ``(page_results, png_paths)`` in page order. Errors on an
        individual page turn into a ``PageResult`` with ``error=...`` and
        that slot is skipped in ``png_paths`` — the caller raises a
        pipeline-level error only if ALL pages failed.
        """
        import concurrent.futures

        # Preprocess parallelism: default cap 10, overridable via the
        # ``OCR_PREPROCESS_WORKERS`` env var for tuning on constrained
        # hosts. 10 matches the common "10 pages at a time" UX ask
        # (batch processing of small bundles) without blowing past the
        # typical memory budget: one preprocessed page in flight holds
        # ~50 MB at 400 DPI, so 10 workers = ~500 MB peak, safe on
        # 8 GB hosts. Single-page jobs skip the thread overhead
        # entirely.
        default_cap = _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        )
        max_workers = (
            1 if page_count == 1 else min(default_cap, page_count)
        )

        slots: list[PageResult | None] = [None] * page_count
        png_slots: list[Path | None] = [None] * page_count

        # No per-word re-OCR rescues in the EasyOCR pipeline — the
        # detector+recognizer is end-to-end neural, so a pre-binarised
        # grayscale is unnecessary.
        want_gray = False

        def _worker(idx: int) -> None:
            page_num = idx + 1
            t0 = time.time()
            png_path = workdir / f"page_{page_num:05d}.png"
            try:
                img = self._rasterize_page(
                    input_path, idx, dpi=profile.ocr.dpi
                )
                if want_gray:
                    processed, gray, angle = self.preprocessor.process(
                        img, profile.preprocess, dpi=profile.ocr.dpi,
                        return_pre_binarization=True,
                    )
                    gray_path = workdir / f"page_{page_num:05d}_gray.png"
                    self._save_png(gray, gray_path)
                else:
                    processed, angle = self.preprocessor.process(
                        img, profile.preprocess, dpi=profile.ocr.dpi
                    )
                self._save_png(processed, png_path)
                slots[idx] = PageResult(
                    page_number=page_num,
                    text="",
                    skew_angle=float(angle),
                    processing_time_sec=time.time() - t0,
                )
                png_slots[idx] = png_path
            except Exception as exc:  # noqa: BLE001 - capture, continue
                logger.exception(
                    "Preprocess failure on page %d of %s", page_num, input_path
                )
                slots[idx] = PageResult(
                    page_number=page_num,
                    error=f"preprocess: {exc}",
                    processing_time_sec=time.time() - t0,
                )

        completed = 0
        if max_workers == 1:
            # Fast path: avoid ThreadPoolExecutor allocation noise for
            # single-page jobs or when running in an already-parallel
            # ProcessPoolExecutor worker.
            for idx in range(page_count):
                _worker(idx)
                completed += 1
                self._report(completed, page_count, "preprocess")
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="ocr-preproc",
            ) as pool:
                futures = [pool.submit(_worker, idx) for idx in range(page_count)]
                for fut in concurrent.futures.as_completed(futures):
                    # Propagate any unexpected exception (the worker
                    # already catches the expected ones and writes them
                    # into `slots[idx]`).
                    fut.result()
                    completed += 1
                    self._report(completed, page_count, "preprocess")

        page_results = [s for s in slots if s is not None]
        png_paths = [p for p in png_slots if p is not None]
        return page_results, png_paths

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
        """Write a numpy image to PNG on disk.

        Uses ``cv2.imencode`` + :meth:`Path.write_bytes` instead of the
        more obvious ``cv2.imwrite`` because the latter goes through
        ``fopen`` on Windows, which takes an ANSI-encoded path and
        silently fails for any character outside the active code page.
        In practice that means users whose Windows profile contains
        Cyrillic characters (e.g. ``C:\\Users\\Т.Н. 020\\...``) get
        ``Не удалось сохранить PNG: ...`` for every page, every job.
        Piping the encoded bytes through Python's own filesystem layer
        bypasses the issue — :meth:`Path.write_bytes` honours Unicode
        paths natively on every platform.
        """
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", image)
        if not ok or buf is None:
            raise RuntimeError(f"Не удалось закодировать PNG: {path}")
        path.write_bytes(buf.tobytes())

    def _assemble_pdf(
        self, png_paths: list[Path], output_pdf: Path, *, dpi: int = 300
    ) -> None:
        """Assemble a PDF from a list of PNGs (one page per image).

        PDF page dimensions are stored in **points** (1/72 inch), not
        pixels. A 300 DPI scan of an A4 page is ~2480x3508 pixels but
        the page must be 595x842 points (A4 in points) so downstream
        tooling — most importantly OCRmyPDF's rasterisation-for-Tesseract
        step — infers the correct DPI. If we use the pixel dimensions
        directly as points, the page claims to be 34×48 inches at
        72 DPI, and Tesseract's layout analysis decides the text is
        sub-glyph-size and silently recognises nothing.

        Convert from pixels to points using the DPI that was used to
        rasterise from the original PDF (``profile.ocr.dpi``).

        Args:
            png_paths: Ordered list of PNG files.
            output_pdf: Output PDF path.
            dpi: Rasterisation DPI used in :meth:`_rasterize_page` —
                determines the pixels→points conversion.
        """
        import fitz

        dpi_factor = 72.0 / float(dpi)

        doc = fitz.open()
        try:
            for png_path in png_paths:
                # Probe image dimensions via a temporary pixmap.
                pix = fitz.Pixmap(str(png_path))
                try:
                    pixel_width = int(pix.width)
                    pixel_height = int(pix.height)
                finally:
                    pix = None  # noqa: F841 - release native resource

                # Convert pixels → points so the embedded image is
                # reported at the correct DPI. OCRmyPDF uses page
                # dimensions + image dimensions to pick the DPI for
                # Tesseract, and ~72 DPI was producing empty hOCR on
                # synthetic English/Russian text because layout
                # analysis ignored the glyphs.
                page_width_points = pixel_width * dpi_factor
                page_height_points = pixel_height * dpi_factor

                page = doc.new_page(
                    width=page_width_points, height=page_height_points
                )
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
        import concurrent.futures

        import fitz

        # Pass 1: serial text extraction from the searchable PDF. Fitz
        # doc objects aren't safe to share across threads — we open the
        # document once on the main thread, pull the raw text per page
        # (cheap: milliseconds per page on a text-bearing PDF), and
        # close it before spinning up the parallel postprocess workers.
        raw_per_page: list[str | None] = [None] * len(page_results)
        doc = fitz.open(str(ocrd_pdf))
        try:
            for idx, pr in enumerate(page_results):
                if idx >= doc.page_count:
                    continue
                try:
                    if pr.text:
                        # Engine already produced text — keep that, do
                        # NOT re-read from the PDF where the layout
                        # serialisation may differ. The postprocess
                        # worker below will see ``pr.text`` as seed.
                        raw_per_page[idx] = pr.text
                    else:
                        page = doc.load_page(idx)
                        raw_per_page[idx] = page.get_text("text") or ""
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to extract text for page %d: %s",
                        pr.page_number, exc,
                    )
                    pr.error = (pr.error or "") + f"; extract: {exc}"
                    raw_per_page[idx] = None
        finally:
            doc.close()

        # Pass 2: parallel postprocess across pages. Postprocess is
        # pure-Python regex + string work; re.sub releases the GIL
        # during pattern execution so thread parallelism scales up to
        # ~10 pages even with the interpreter lock. ThreadPool is
        # cheaper than ProcessPool for this stage because we don't
        # have to serialise the page text across the process boundary.
        # Batch size matches the preprocess / OCR stage caps (10) so
        # a 10-page bundle moves through all three stages at the same
        # width, and autosave semantics are preserved by running the
        # periodic checkpoint between batches instead of per page.
        workers = _resolve_stage_parallelism(
            env_var="OCR_POSTPROCESS_WORKERS", fallback=10,
        )

        def _postprocess_one(idx: int) -> None:
            raw = raw_per_page[idx]
            if raw is None:
                return
            pr = page_results[idx]
            try:
                pr.text = self._postprocess_text(
                    raw, job.profile.postprocess,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to postprocess page %d: %s",
                    pr.page_number, exc,
                )
                pr.error = (pr.error or "") + f"; postprocess: {exc}"

        if len(page_results) == 1:
            _postprocess_one(0)
        else:
            batch_size = max(1, workers)
            for batch_start in range(0, len(page_results), batch_size):
                batch_indices = range(
                    batch_start,
                    min(batch_start + batch_size, len(page_results)),
                )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="ocr-postproc",
                ) as pool:
                    list(pool.map(_postprocess_one, batch_indices))
                # Periodic partial-result autosave between batches. The
                # old per-page cadence was ``(idx + 1) %
                # autosave_interval_pages == 0``; after parallelising we
                # approximate that by checkpointing on every batch
                # boundary that crosses an interval multiple.
                if self.autosave_interval_pages > 0:
                    completed = min(
                        batch_start + batch_size, len(page_results),
                    )
                    if (
                        completed % self.autosave_interval_pages == 0
                        or completed == len(page_results)
                    ):
                        self._autosave_partial_txt(
                            page_results[:completed], job,
                        )

        # The engine populates ``mean_confidence`` and ``word_boxes``
        # natively; ``_compute_confidences`` consumes those word_boxes
        # to apply adaptive threshold, drop-low-conf filtering, fuzzy
        # rescue and PDF text-layer redaction.
        if self.compute_confidence:
            self._compute_confidences(
                page_results, job, png_paths, output_pdf=ocrd_pdf,
            )

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
            return self.postprocessor.process(text, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postprocess failed: %s", exc)
            return text

    def _maybe_run_parser(self, result: JobResult, job: OCRJobConfig) -> None:
        """Invoke the post-OCR structured-field parser, if requested.

        Populates ``result.parsed`` in place. This method is the SINGLE
        entry point to the ТН / УПД parser from the pipeline — it is
        called from each of the three completion paths (cache hit,
        text-layer bypass, full OCR) so every successful job gets the
        same extraction treatment regardless of how the OCR phase
        completed.

        Never raises. Any exception — including the orchestrator's
        own import error if ``src.application.parsers.tn_orchestrator``
        can't be loaded — is caught here and logged. An OCR job that
        succeeded on the OCR side must complete with ``status=COMPLETED``
        even when the parser layer is broken; ``parsed`` simply stays
        ``None`` and the UI falls back to showing the raw OCR text.
        """
        try:
            from src.application.parsers.tn_orchestrator import (
                extract_from_pages,
            )
        except ImportError as exc:
            logger.warning(
                "Post-OCR extraction unavailable (orchestrator import "
                "failed: %s); skipping",
                exc,
            )
            return
        try:
            # Batch context propagation (idea #3 top-10): pipeline
            # может работать в batch-mode через ParallelProcessor
            # или CLI folder-batch. Если caller подготовил общий
            # BatchContext и передал его через job.batch_ctx,
            # орchestrator использует его для cross-doc learning.
            # Single-doc runs — batch_ctx=None, no-op.
            batch_ctx = getattr(job, "batch_ctx", None)
            result.parsed = extract_from_pages(
                pages=result.pages,
                config=job.profile.extract,
                source_path=Path(result.input_path),
                batch_ctx=batch_ctx,
            )
        except Exception as exc:  # noqa: BLE001 — belt-and-braces
            logger.exception(
                "Pipeline parser hook raised despite orchestrator catch: %s",
                exc,
            )
            result.parsed = None

    def _compute_confidences(
        self,
        page_results: list[PageResult],
        job: OCRJobConfig,
        png_paths: list[Path],
        *,
        output_pdf: Path | None = None,
    ) -> None:
        """Apply confidence-based filtering on engine-produced word_boxes.

        The OCR engine has already populated each PageResult with
        ``word_boxes: [(x, y, w, h, conf_0_100, text)]`` and
        ``mean_confidence``. We synthesise a TSV-compatible dict so
        existing helpers (confidence_filter, pdf_text_filter,
        tn_parser.layout_anchor) keep working unchanged.
        """
        from src.core.confidence_filter import (
            detect_handwritten_blocks,
            reconstruct_text_from_tsv,
        )

        threshold = float(job.profile.ocr.confidence_threshold)
        ocr = job.profile.ocr
        tsv_per_page: list[dict] = []
        image_sizes_px: list[tuple[int, int]] = []

        for pr, png_path in zip(page_results, png_paths, strict=False):
            data = _word_boxes_to_tsv(pr.word_boxes)
            tsv_per_page.append(data)
            try:
                import cv2
                raw = np.frombuffer(png_path.read_bytes(), dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw.size else None
                w_px = int(img.shape[1]) if img is not None else 0
                h_px = int(img.shape[0]) if img is not None else 0
            except Exception:  # noqa: BLE001
                w_px = h_px = 0
            image_sizes_px.append((w_px, h_px))

            if pr.error or not pr.word_boxes:
                continue

            confidences = [w[4] for w in pr.word_boxes if w[4] >= 0]
            if not confidences:
                continue

            page_mean = sum(confidences) / len(confidences)
            effective = threshold
            if getattr(ocr, "adaptive_confidence_threshold", False):
                if page_mean >= 90.0:
                    effective = min(threshold, 40.0)
                elif page_mean < 70.0:
                    effective = max(threshold, 70.0)

            low_words = [w[5] for w in pr.word_boxes if w[4] < effective]
            pr.mean_confidence = page_mean
            pr.low_confidence_words = low_words
            pr.tsv_data = data
            pr.page_width_px = w_px
            pr.raster_path = str(png_path) if png_path.exists() else None

            if getattr(ocr, "user_words_fuzzy_rescue", False):
                self._apply_user_words_rescue(pr, data)

            if ocr.drop_low_conf_words:
                hw_blocks: set[tuple[int, int]] = set()
                if getattr(
                    job.profile.postprocess,
                    "mark_suspect_handwritten_blocks", False,
                ):
                    hw_blocks = detect_handwritten_blocks(
                        data, max_mean_confidence=40.0, min_words=3,
                    )
                filtered = reconstruct_text_from_tsv(
                    data,
                    min_confidence=effective,
                    soft_rescue=ocr.soft_rescue_dropped_words,
                    handwritten_blocks=hw_blocks,
                )
                if filtered.strip():
                    pr.text = self._postprocess_text(
                        filtered, job.profile.postprocess,
                    )
                    kept = [c for c in confidences if c >= effective]
                    if kept:
                        pr.mean_confidence = sum(kept) / len(kept)

        if (
            ocr.drop_low_conf_words
            and output_pdf is not None
            and output_pdf.exists()
            and tsv_per_page
        ):
            try:
                from src.core.pdf_text_filter import filter_pdf_text_layer
                redacted = filter_pdf_text_layer(
                    output_pdf,
                    tsv_per_page=tsv_per_page,
                    image_sizes_px=image_sizes_px,
                    min_confidence=threshold,
                )
                logger.info(
                    "PDF text-layer filter: redacted %d region(s) on %s",
                    redacted, output_pdf,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF text-layer filter failed: %s", exc)

    def _apply_user_words_rescue(
        self, pr: PageResult, data: dict,
    ) -> None:
        """Fuzzy-correct mid-confidence words against the bundled lexicon."""
        try:
            from src.core.user_words_rescue import (
                load_user_words_catalog,
                user_words_fuzzy_rescue,
            )
            from src.shared.constants import RESOURCES_DIR
            catalog = load_user_words_catalog(RESOURCES_DIR / "ru_lexicon.txt")
        except Exception as exc:  # noqa: BLE001
            logger.debug("user-words rescue unavailable: %s", exc)
            return
        if catalog is None:
            return
        texts = data.get("text", [])
        confs = data.get("conf", [])
        for i in range(len(texts)):
            word = texts[i] if isinstance(texts[i], str) else ""
            if not word.strip():
                continue
            try:
                c = float(confs[i])
            except (TypeError, ValueError, IndexError):
                continue
            new_word, new_conf = user_words_fuzzy_rescue(word, c, catalog)
            if new_word != word or new_conf != c:
                texts[i] = new_word
                confs[i] = str(new_conf)


    def _cleanup(self, workdir: Path) -> None:
        """Remove the temporary workdir, logging but not raising on error."""
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            logger.debug("Removed workdir: %s", workdir)
        except OSError as exc:  # pragma: no cover - platform-specific
            logger.warning("Failed to remove workdir %s: %s", workdir, exc)


def _word_boxes_to_tsv(
    word_boxes: list[tuple[float, float, float, float, float, str]],
) -> dict:
    """Build a TSV-compatible dict from engine word_boxes.

    All words land in block_num=1, line_num=1 (EasyOCR has no layout-
    block analysis); word_num increments. ``level=5`` marks word rows.
    """
    n = len(word_boxes)
    return {
        "level": [5] * n,
        "page_num": [1] * n,
        "block_num": [1] * n,
        "par_num": [1] * n,
        "line_num": [1] * n,
        "word_num": list(range(1, n + 1)),
        "left": [int(w[0]) for w in word_boxes],
        "top": [int(w[1]) for w in word_boxes],
        "width": [int(w[2]) for w in word_boxes],
        "height": [int(w[3]) for w in word_boxes],
        "conf": [str(w[4]) for w in word_boxes],
        "text": [w[5] for w in word_boxes],
    }
