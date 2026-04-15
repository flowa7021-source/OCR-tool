"""Multi-process OCR job runner.

The pipeline itself is not picklable (it references OpenCV / PyMuPDF handles
lazily imported inside worker methods, and the Tesseract wrapper holds
process-local caches). We therefore serialize just the
:class:`~src.core.models.OCRJobConfig` to a ``dict`` and let each worker
process rebuild its own pipeline.

Per-page progress is bridged back to the main process through a shared
:class:`multiprocessing.Queue`. Workers emit tagged progress events; a
:class:`ProgressBridge` running in a background thread on the main side
drains them and fans out to per-job callbacks.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import queue as queue_mod
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

from src.core.models import JobResult, OCRJobConfig, PageResult, ProfileData
from src.shared.constants import DEFAULT_PARALLEL_WORKERS
from src.shared.types import JobStatus

logger = logging.getLogger(__name__)


# Callback signatures
# on_progress(job_id, current, total, stage)
OnProgress = Callable[[str, int, int, str], None]
OnComplete = Callable[[JobResult], None]
OnError = Callable[[BaseException], None]


# Sentinel marking the end of progress events for a job
_PROGRESS_DONE = "__done__"


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


def _prewarm_worker() -> str:
    """No-op worker that imports everything a real job will need.

    On Windows ``ProcessPoolExecutor`` uses ``spawn``, which means each
    worker freshly imports the whole Python environment on first use.
    For a PyInstaller-frozen build that includes torch + transformers
    that can be 3-8 seconds on disk I/O alone — during which the GUI
    thread sits waiting for ``future.result()``.

    Calling :func:`ParallelProcessor.prewarm` from a background thread
    forces a worker to start + import eagerly right after the main
    window appears, so the user's first click of "Start OCR" returns
    to the event loop immediately.
    """
    import src.application.pipeline  # noqa: F401

    return "ready"


def _worker_run_job(
    job_dict: dict[str, Any],
    progress_queue: multiprocessing.Queue | None = None,
    tracking_id: str = "",
) -> dict[str, Any]:
    """Process-pool worker. Runs one pipeline start-to-finish.

    Args:
        job_dict: Dict produced by :func:`job_to_dict`.
        progress_queue: Optional cross-process queue for progress events.
            Each event is a tuple ``(tracking_id, current, total, stage)``.
            A final ``(tracking_id, _PROGRESS_DONE, 0, 0, "")``-shaped message
            is NOT emitted by this worker — the bridge infers completion from
            the future's done callback.
        tracking_id: Opaque identifier echoed back in every progress event so
            the bridge can route updates to the correct listener.

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

        def _progress(current: int, total: int, stage: str) -> None:
            if progress_queue is None:
                return
            # Queue full or closed — don't let progress reporting crash the job
            with contextlib.suppress(Exception):
                progress_queue.put_nowait((tracking_id, int(current), int(total), str(stage)))

        # Read autosave setting (best-effort; defaults to 0 if unavailable).
        autosave_interval = 0
        try:
            from src.infrastructure.config_storage import SettingsStorage

            autosave_interval = int(SettingsStorage().load().autosave_interval_pages)
        except Exception:  # noqa: BLE001
            autosave_interval = 0

        pipeline = OCRPipeline(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            tesseract=tess,
            progress_callback=_progress if progress_queue is not None else None,
            autosave_interval_pages=autosave_interval,
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

    Progress events emitted by worker processes are drained by a background
    thread (``_progress_thread``) and dispatched to per-job callbacks registered
    at submission time.

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

        # Cross-process progress plumbing. A Manager gives us a proxy Queue
        # that survives across pool workers created on Windows (spawn).
        self._manager: multiprocessing.managers.SyncManager | None = None
        self._progress_queue: multiprocessing.Queue | None = None
        self._progress_thread: threading.Thread | None = None
        self._progress_stop = threading.Event()
        self._progress_lock = threading.Lock()
        # tracking_id -> (on_progress, job_id_for_callback)
        self._progress_listeners: dict[str, tuple[OnProgress | None, str]] = {}
        # job_id -> Future (for cancellation)
        self._futures: dict[str, Future] = {}

    # -- lifecycle ---------------------------------------------------------

    def _ensure_executor(self) -> ProcessPoolExecutor:
        """Lazily instantiate the process pool and start the progress bridge.

        ``max_tasks_per_child=10`` (Python 3.11+) recycles each worker
        after 10 jobs. Long-running sessions used to accumulate gigabytes
        of per-worker memory — torch caches, PyMuPDF mmaps, OCRmyPDF
        temp-files holding onto file-descriptor caches — because the
        default executor keeps workers alive forever. Recycling after
        10 jobs caps per-worker RSS at a stable plateau without enough
        churn to dominate the spawn-start cost.
        """
        if self._executor is None:
            logger.info(
                "Starting ProcessPoolExecutor with %d workers (recycle=10)",
                self.max_workers,
            )
            # max_tasks_per_child was added in 3.11; fall back if we
            # ever get run on an older interpreter.
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    max_tasks_per_child=10,
                )
            except TypeError:
                logger.debug(
                    "max_tasks_per_child unsupported on this Python; "
                    "workers will not recycle"
                )
                self._executor = ProcessPoolExecutor(
                    max_workers=self.max_workers
                )
            self._start_progress_bridge()
        return self._executor

    def prewarm(self) -> None:
        """Spin up the pool + a dummy job so the user's first submit is fast.

        Must be called off the GUI thread — constructing
        :class:`multiprocessing.Manager` spawns a subprocess which can
        block the caller for ~1 s on Windows. Safe to call multiple
        times; subsequent calls are a no-op once the executor is up.
        """
        try:
            executor = self._ensure_executor()
            # Submit a cheap no-op that imports the heavy modules so
            # Windows spawn-start cost is paid *now*, off the GUI thread.
            # We don't care about the future; the pool is keyed by state
            # stored on ``self``.
            executor.submit(_prewarm_worker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ParallelProcessor prewarm failed: %s", exc)

    def _start_progress_bridge(self) -> None:
        """Start the manager, queue, and drain thread if not already running."""
        if self._progress_thread is not None and self._progress_thread.is_alive():
            return
        try:
            self._manager = multiprocessing.Manager()
            self._progress_queue = self._manager.Queue()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not start progress bridge: %s", exc)
            self._progress_queue = None
            return
        self._progress_stop.clear()
        self._progress_thread = threading.Thread(
            target=self._drain_progress, name="ocr-progress-drain", daemon=True
        )
        self._progress_thread.start()

    def _drain_progress(self) -> None:
        """Background thread: dequeue progress events and dispatch to listeners."""
        q = self._progress_queue
        if q is None:
            return
        while not self._progress_stop.is_set():
            try:
                event = q.get(timeout=0.25)
            except queue_mod.Empty:
                continue
            except (EOFError, OSError):
                break
            if not event:
                continue
            try:
                tracking_id, current, total, stage = event
            except Exception:  # noqa: BLE001
                logger.debug("Malformed progress event: %r", event)
                continue
            with self._progress_lock:
                listener = self._progress_listeners.get(tracking_id)
            if listener is None:
                continue
            on_progress, job_id = listener
            if on_progress is None:
                continue
            try:
                on_progress(job_id, int(current), int(total), str(stage))
            except Exception:  # noqa: BLE001
                logger.debug("on_progress dispatch raised", exc_info=True)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool and progress bridge.

        Args:
            wait: Whether to block until running jobs finish.
        """
        if self._executor is not None:
            logger.info("Shutting down ParallelProcessor (wait=%s)", wait)
            self._executor.shutdown(wait=wait)
            self._executor = None
        # Stop drain thread
        self._progress_stop.set()
        if self._progress_thread is not None:
            self._progress_thread.join(timeout=2.0)
            self._progress_thread = None
        if self._manager is not None:
            with contextlib.suppress(Exception):
                self._manager.shutdown()
            self._manager = None
        self._progress_queue = None
        with self._progress_lock:
            self._progress_listeners.clear()

    # -- submission --------------------------------------------------------

    def submit(
        self,
        job: OCRJobConfig,
        on_progress: OnProgress | None = None,
        on_complete: OnComplete | None = None,
        on_error: OnError | None = None,
        job_id: str | None = None,
    ) -> Future:
        """Submit one job to the pool.

        Args:
            job: Fully populated job config.
            on_progress: Optional ``(job_id, current, total, stage)`` callback.
                Fires for every per-page progress event emitted by the pipeline
                inside the worker, marshaled through a manager Queue and a
                background drain thread. Callers MUST assume it executes on a
                non-main, non-UI thread.
            on_complete: Callback receiving the final :class:`JobResult`.
            on_error: Callback receiving any exception raised by the worker.
            job_id: Opaque identifier passed through to ``on_progress``.
                Defaults to a random UUID if not provided.

        Returns:
            The underlying :class:`concurrent.futures.Future`.
        """
        executor = self._ensure_executor()

        tracking_id = uuid.uuid4().hex
        exposed_id = job_id or tracking_id

        with self._progress_lock:
            self._progress_listeners[tracking_id] = (on_progress, exposed_id)

        future = executor.submit(
            _worker_run_job, job_to_dict(job), self._progress_queue, tracking_id
        )
        with self._progress_lock:
            self._futures[exposed_id] = future

        def _done(fut: Future) -> None:
            # Drop listener so the drain thread doesn't accumulate stale refs
            with self._progress_lock:
                self._progress_listeners.pop(tracking_id, None)
                self._futures.pop(exposed_id, None)

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

    # -- cancellation ------------------------------------------------------

    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a job by ``job_id``.

        This only works for futures that have not yet started executing
        (Python's ProcessPoolExecutor does not support killing running tasks
        through Future.cancel). Callers should treat this as best-effort.

        Returns:
            ``True`` if the future transitioned to cancelled, ``False`` if it
            was already running/finished or not found.
        """
        with self._progress_lock:
            future = self._futures.get(job_id)
        if future is None:
            return False
        cancelled = future.cancel()
        if cancelled:
            logger.info("Cancelled pending job %s", job_id)
            with self._progress_lock:
                self._futures.pop(job_id, None)
        else:
            logger.info("Could not cancel job %s (already running)", job_id)
        return cancelled

    def cancel_all(self) -> int:
        """Attempt to cancel every pending job. Returns count actually cancelled."""
        with self._progress_lock:
            futures = list(self._futures.items())
        n = 0
        for job_id, fut in futures:
            if fut.cancel():
                n += 1
                with self._progress_lock:
                    self._futures.pop(job_id, None)
        logger.info("Cancel-all removed %d pending job(s)", n)
        return n

    def active_job_count(self) -> int:
        """Number of submitted jobs that are not yet resolved."""
        with self._progress_lock:
            return len(self._futures)

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> ParallelProcessor:
        self._ensure_executor()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


