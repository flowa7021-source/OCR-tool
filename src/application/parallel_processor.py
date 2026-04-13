"""Multi-process OCR job runner.

The pipeline itself is not picklable (it references OpenCV / PyMuPDF handles
lazily imported inside worker methods, and the Tesseract wrapper holds
process-local caches). We therefore serialize just the
:class:`~src.core.models.OCRJobConfig` to a ``dict`` and let each worker
process rebuild its own pipeline.

Progress is reported at job granularity only when running through a process
pool; per-page progress is not bridged back to the main process to keep the
worker simple and robust.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.core.models import JobResult, OCRJobConfig, PageResult, ProfileData
from src.shared.constants import DEFAULT_PARALLEL_WORKERS
from src.shared.types import JobStatus

logger = logging.getLogger(__name__)


OnProgress = Callable[[str, int, int], None]
OnComplete = Callable[[JobResult], None]
OnError = Callable[[BaseException], None]


# ---------------------------------------------------------------------------
# Serialization helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------


def job_to_dict(job: OCRJobConfig) -> dict[str, Any]:
    """Serialize an :class:`OCRJobConfig` to a JSON-friendly dict.

    Profile serialization reuses :meth:`ProfileData.to_dict` which already
    handles enum coercion; remaining scalar fields are passed as-is.

    Args:
        job: Job to serialize.

    Returns:
        Pickle- and JSON-safe dictionary.
    """
    return {
        "input_path": job.input_path,
        "output_path": job.output_path,
        "profile": job.profile.to_dict(),
        "export_formats": list(job.export_formats),
        "priority": job.priority,
    }


def job_from_dict(data: dict[str, Any]) -> OCRJobConfig:
    """Reconstruct an :class:`OCRJobConfig` produced by :func:`job_to_dict`."""
    return OCRJobConfig(
        input_path=str(data["input_path"]),
        output_path=str(data["output_path"]),
        profile=ProfileData.from_dict(data["profile"]),
        export_formats=list(data.get("export_formats", ["pdf"])),
        priority=int(data.get("priority", 0)),
    )


def job_result_to_dict(result: JobResult) -> dict[str, Any]:
    """Serialize a :class:`JobResult` to a plain dict."""
    return {
        "job_id": result.job_id,
        "status": result.status.value,
        "input_path": result.input_path,
        "output_path": result.output_path,
        "pages": [asdict(p) for p in result.pages],
        "total_time_sec": result.total_time_sec,
        "error": result.error,
    }


def job_result_from_dict(data: dict[str, Any]) -> JobResult:
    """Inverse of :func:`job_result_to_dict`."""
    pages = [PageResult(**p) for p in data.get("pages", [])]
    return JobResult(
        job_id=str(data["job_id"]),
        status=JobStatus(data["status"]),
        input_path=str(data["input_path"]),
        output_path=str(data["output_path"]),
        pages=pages,
        total_time_sec=float(data.get("total_time_sec", 0.0)),
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# Worker entry-point
# ---------------------------------------------------------------------------


def _worker_run_job(job_dict: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker. Runs one pipeline start-to-finish.

    Args:
        job_dict: Dict produced by :func:`job_to_dict`.

    Returns:
        Dict produced by :func:`job_result_to_dict`.
    """
    # Set up logging inside the worker — root logger has no handlers in fresh
    # processes. We keep this minimal; the host app can reconfigure as needed.
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    worker_logger = logging.getLogger(__name__ + ".worker")

    try:
        # Local imports keep top-level module lightweight for pickling.
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.text_postprocessor import TextPostprocessor
        from src.infrastructure.tesseract_wrapper import TesseractWrapper

        job = job_from_dict(job_dict)
        worker_logger.info("Worker picked up job for %s", job.input_path)

        tess = TesseractWrapper()
        try:
            tess.configure_pytesseract()
        except Exception as exc:  # noqa: BLE001
            worker_logger.warning("configure_pytesseract failed: %s", exc)

        preprocessor = ImagePreprocessor()
        postprocessor = TextPostprocessor()
        pipeline = OCRPipeline(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            tesseract=tess,
        )

        result = pipeline.run(job)
        return job_result_to_dict(result)
    except Exception as exc:  # noqa: BLE001 - always return a dict
        worker_logger.exception("Worker crashed")
        return {
            "job_id": "",
            "status": JobStatus.FAILED.value,
            "input_path": str(job_dict.get("input_path", "")),
            "output_path": str(job_dict.get("output_path", "")),
            "pages": [],
            "total_time_sec": 0.0,
            "error": f"worker: {exc}",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ParallelProcessor:
    """Submit OCR jobs to a :class:`ProcessPoolExecutor`.

    Attributes:
        max_workers: Maximum concurrent worker processes.
    """

    def __init__(self, max_workers: int = DEFAULT_PARALLEL_WORKERS) -> None:
        """Create a processor with ``max_workers`` workers.

        Args:
            max_workers: Number of worker processes.
        """
        self.max_workers = max(1, int(max_workers))
        self._executor: ProcessPoolExecutor | None = None

    # -- lifecycle ---------------------------------------------------------

    def _ensure_executor(self) -> ProcessPoolExecutor:
        """Lazily instantiate the process pool."""
        if self._executor is None:
            logger.info(
                "Starting ProcessPoolExecutor with %d workers", self.max_workers
            )
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool.

        Args:
            wait: Whether to block until running jobs finish.
        """
        if self._executor is not None:
            logger.info("Shutting down ParallelProcessor (wait=%s)", wait)
            self._executor.shutdown(wait=wait)
            self._executor = None

    # -- submission --------------------------------------------------------

    def submit(
        self,
        job: OCRJobConfig,
        on_progress: OnProgress | None = None,
        on_complete: OnComplete | None = None,
        on_error: OnError | None = None,
    ) -> Future:
        """Submit one job to the pool.

        Args:
            job: Fully populated job config.
            on_progress: Optional ``(input_path, current, total)`` callback.
                Fires once at submission (0/1) and once on completion (1/1).
                Per-page progress is NOT bridged across processes.
            on_complete: Callback receiving the :class:`JobResult`.
            on_error: Callback receiving any exception raised by the worker.

        Returns:
            The underlying :class:`concurrent.futures.Future`.
        """
        executor = self._ensure_executor()

        if on_progress is not None:
            try:
                on_progress(job.input_path, 0, 1)
            except Exception:  # noqa: BLE001
                logger.debug("on_progress(start) raised", exc_info=True)

        future = executor.submit(_worker_run_job, job_to_dict(job))

        def _done(fut: Future) -> None:
            try:
                result_dict = fut.result()
            except BaseException as exc:  # noqa: BLE001
                logger.exception("Worker future raised")
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:  # noqa: BLE001
                        logger.debug("on_error raised", exc_info=True)
                return
            try:
                result = job_result_from_dict(result_dict)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to deserialize JobResult")
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:  # noqa: BLE001
                        logger.debug("on_error raised", exc_info=True)
                return

            if on_progress is not None:
                try:
                    on_progress(job.input_path, 1, 1)
                except Exception:  # noqa: BLE001
                    logger.debug("on_progress(end) raised", exc_info=True)
            if on_complete is not None:
                try:
                    on_complete(result)
                except Exception:  # noqa: BLE001
                    logger.debug("on_complete raised", exc_info=True)

        future.add_done_callback(_done)
        return future

    def submit_all(self, jobs: list[OCRJobConfig]) -> list[Future]:
        """Submit a list of jobs. Returns a list of futures in the same order."""
        return [self.submit(job) for job in jobs]

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ParallelProcessor":
        self._ensure_executor()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def is_pdf_path(path: Path) -> bool:
    """Return True if ``path`` has a ``.pdf`` suffix (case-insensitive)."""
    return path.suffix.lower() == ".pdf"
