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


def _resolve_logs_dir() -> str | None:
    """Return the path where worker log files should land.

    Matches ``src/shared/constants.LOGS_DIR`` without importing it — that
    module pulls in constants which pull in enum / dataclass machinery,
    making the early-worker setup heavier than necessary. If neither
    ``OCRSTUDIO_LOGS_DIR`` nor ``LOCALAPPDATA`` / ``APPDATA`` are set
    (very unusual), returns None and the caller silently skips file
    logging.
    """
    import os

    explicit = os.environ.get("OCRSTUDIO_LOGS_DIR")
    if explicit:
        return explicit
    appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if appdata:
        return os.path.join(appdata, "OCRStudio", "logs")
    return None


def _setup_worker_logging() -> logging.Logger:
    """Configure per-worker logging to a file inside the app's logs dir.

    PyInstaller ``--windowed`` builds redirect stderr to ``NUL`` on
    Windows, so the default ``StreamHandler`` attached by the previous
    version of this function wrote into a black hole. That left us
    with no trace of what the worker actually did — every failure
    surfaced to the host as a bare ``BrokenProcessPool``.

    File logging is the only channel that survives a ``--windowed``
    build. We write one file per worker process (``worker-<pid>.log``)
    alongside the host's ``ocr-studio.log``, keyed by PID so parallel
    workers don't step on each other.

    Level is controlled by ``OCRSTUDIO_LOG_LEVEL`` (default INFO). Set
    to ``DEBUG`` when diagnosing an issue to capture per-page timings.
    """
    import os

    level_name = os.environ.get("OCRSTUDIO_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    # Fresh subprocess should have no handlers; clear defensively so repeated
    # fork/spawn inside tests doesn't stack them.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )

    # Keep the stream handler as a fallback — useful when running under
    # `python -m` where stderr is live. Harmless under --windowed.
    try:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    except Exception:  # noqa: BLE001
        pass

    # File handler — primary diagnostic channel.
    logs_dir = _resolve_logs_dir()
    if logs_dir:
        try:
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(logs_dir, f"worker-{os.getpid()}.log")
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as exc:  # noqa: BLE001
            # Can't log the failure to file (we just failed to open it),
            # but stderr may still work in dev builds.
            logging.getLogger(__name__).warning(
                "Could not open worker log file: %s", exc
            )

    return logging.getLogger(__name__ + ".worker")


def _enable_worker_faulthandler(worker_logger: logging.Logger) -> None:
    """Install ``faulthandler`` so native crashes leave a readable trace.

    Without this a segfault inside OCRmyPDF / PyMuPDF / Tesseract kills
    the worker process with no Python traceback at all. ``faulthandler``
    registers a signal handler that dumps all thread stacks when a
    fatal signal fires (SIGSEGV, SIGFPE, SIGABRT, SIGILL on Windows).
    """
    try:
        import faulthandler
        import os

        logs_dir = _resolve_logs_dir()
        if not logs_dir:
            return
        os.makedirs(logs_dir, exist_ok=True)
        crash_path = os.path.join(logs_dir, f"worker-crash-{os.getpid()}.log")
        # crash_file MUST stay open for the lifetime of the worker —
        # faulthandler writes to it if the process segfaults. A
        # context manager or explicit close() would defeat the whole
        # point. But if ``faulthandler.enable()`` itself raises, the
        # handle would leak; catch that specific case and close
        # before re-logging.
        crash_file = open(crash_path, "w", encoding="utf-8")  # noqa: SIM115
        try:
            faulthandler.enable(crash_file)
        except Exception:
            crash_file.close()
            raise
        worker_logger.debug("faulthandler enabled, crash log at %s", crash_path)
    except Exception as exc:  # noqa: BLE001
        worker_logger.debug("Could not enable faulthandler: %s", exc)


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
    import os
    import time

    worker_logger = _setup_worker_logging()
    _enable_worker_faulthandler(worker_logger)

    # Silence the brief console windows Tesseract/Ghostscript flash
    # on Windows. Must land before OCRmyPDF spawns its first child.
    # No-op on POSIX and on already-patched processes.
    try:
        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        install_windows_console_hide()
    except Exception as exc:  # noqa: BLE001 — hygiene is best-effort
        worker_logger.debug("subprocess_hygiene unavailable: %s", exc)

    pid = os.getpid()
    input_path = job_dict.get("input_path", "?")
    output_path = job_dict.get("output_path", "?")

    worker_logger.info("=" * 72)
    worker_logger.info(
        "Worker PID=%d picked up job: input=%s  output=%s",
        pid, input_path, output_path,
    )
    worker_logger.info("tracking_id=%s", tracking_id)

    # Track the current stage so a mid-pipeline exception can name it.
    current_stage = "startup"

    try:
        current_stage = "imports"
        worker_logger.info("[1/6] Importing pipeline modules…")
        t_imports = time.time()
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.text_postprocessor import TextPostprocessor
        from src.infrastructure.tesseract_wrapper import TesseractWrapper

        worker_logger.info(
            "[1/6] Imports OK in %.2fs", time.time() - t_imports
        )

        current_stage = "deserialize_job"
        job = job_from_dict(job_dict)
        worker_logger.info(
            "[2/6] Job deserialized: profile=%s engine=%s dpi=%s langs=%s",
            job.profile.name,
            getattr(job.profile.ocr, "engine", "?"),
            getattr(job.profile.ocr, "dpi", "?"),
            getattr(job.profile.ocr, "tesseract_language_string", "?"),
        )

        current_stage = "configure_hf_runtime"
        try:
            from src.infrastructure.hf_runtime import configure_huggingface_runtime

            configure_huggingface_runtime()
            worker_logger.info(
                "[3/6] HF runtime configured: HF_HOME=%s offline=%s",
                os.environ.get("HF_HOME", "<default>"),
                os.environ.get("HF_HUB_OFFLINE", "0"),
            )
        except Exception as exc:  # noqa: BLE001
            worker_logger.warning(
                "configure_huggingface_runtime failed (continuing): %s", exc
            )

        current_stage = "register_external_tools"
        worker_logger.info(
            "[3a/6] Registering bundled external binaries on PATH…"
        )
        try:
            from src.infrastructure.external_tools import (
                ensure_on_path,
                verify_required_for_ocrmypdf,
            )

            resolved = ensure_on_path()
            for name, path in resolved.items():
                worker_logger.info(
                    "[3a/6]   %s -> %s", name, path or "NOT FOUND",
                )
            missing = verify_required_for_ocrmypdf()
            if missing:
                worker_logger.error(
                    "[3a/6] Required external tools missing: %s — OCRmyPDF "
                    "will fail. Pipeline will short-circuit with a clear "
                    "error.",
                    ", ".join(missing),
                )
        except Exception as exc:  # noqa: BLE001
            worker_logger.warning(
                "[3a/6] ensure_on_path raised (continuing): %s",
                exc, exc_info=True,
            )

        current_stage = "configure_tesseract"
        worker_logger.info("[3b/6] Configuring Tesseract…")
        tess = TesseractWrapper()
        try:
            tess.configure_pytesseract()
            worker_logger.info(
                "[3b/6] Tesseract OK: bin=%s tessdata=%s",
                getattr(tess, "_binary_path", "?"),
                getattr(tess, "_tessdata_path", "?"),
            )
        except Exception as exc:  # noqa: BLE001
            worker_logger.warning(
                "[3b/6] configure_pytesseract failed: %s", exc, exc_info=True
            )

        current_stage = "build_pipeline"
        worker_logger.info("[4/6] Building pipeline (preprocess + postprocess)…")
        preprocessor = ImagePreprocessor()
        postprocessor = TextPostprocessor()

        def _progress(current: int, total: int, stage: str) -> None:
            # Fan out to the host bridge AND log locally so we have a
            # per-stage timeline even if the host never receives the event.
            worker_logger.info(
                "progress: stage=%s  %d / %d", stage, current, total
            )
            if progress_queue is None:
                return
            with contextlib.suppress(Exception):
                progress_queue.put_nowait(
                    (tracking_id, int(current), int(total), str(stage))
                )

        autosave_interval = 0
        try:
            from src.infrastructure.config_storage import SettingsStorage

            autosave_interval = int(SettingsStorage().load().autosave_interval_pages)
        except Exception as exc:  # noqa: BLE001
            worker_logger.debug("Could not load autosave_interval: %s", exc)

        pipeline = OCRPipeline(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            tesseract=tess,
            progress_callback=_progress if progress_queue is not None else None,
            autosave_interval_pages=autosave_interval,
        )

        current_stage = "pipeline.run"
        worker_logger.info(
            "[5/6] Starting pipeline.run(job) — this performs analyze → "
            "preprocess → assemble → OCR → postprocess"
        )
        t_run = time.time()
        result = pipeline.run(job)
        worker_logger.info(
            "[5/6] pipeline.run finished in %.2fs  status=%s  pages=%d  "
            "avg_conf=%.1f  error=%s",
            time.time() - t_run,
            result.status.value,
            len(result.pages),
            result.average_confidence,
            result.error or "<none>",
        )

        current_stage = "serialize_result"
        worker_logger.info("[6/6] Serializing result for host handoff…")
        result_dict = job_result_to_dict(result)
        worker_logger.info(
            "[6/6] Done. Returning to host. Output file: %s", output_path
        )
        return result_dict

    except Exception as exc:  # noqa: BLE001 - always return a dict
        worker_logger.exception(
            "Worker crashed at stage=%s: %s", current_stage, exc
        )
        return {
            "job_id": "",
            "status": JobStatus.FAILED.value,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "pages": [],
            "total_time_sec": 0.0,
            "error": f"worker@{current_stage}: {exc}",
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

    def __init__(
        self,
        max_workers: int = DEFAULT_PARALLEL_WORKERS,
        mp_context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        """Create a processor with ``max_workers`` workers.

        Args:
            max_workers: Number of worker processes.
            mp_context: Optional :mod:`multiprocessing` context to pass
                to :class:`ProcessPoolExecutor` (e.g.
                ``multiprocessing.get_context("spawn")``). ``None``
                uses the platform default (``fork`` on Linux, ``spawn``
                on Windows). Exposed so test suites can force ``spawn``
                on Linux for Windows-parity.
        """
        self.max_workers = max(1, int(max_workers))
        self._mp_context = mp_context
        self._executor: ProcessPoolExecutor | None = None

        # Cross-process progress plumbing. A Manager gives us a proxy Queue
        # that survives across pool workers created on Windows (spawn).
        self._manager: multiprocessing.managers.SyncManager | None = None
        self._progress_queue: multiprocessing.Queue | None = None
        self._progress_thread: threading.Thread | None = None
        self._progress_stop = threading.Event()
        self._progress_lock = threading.Lock()
        # Serialises executor + progress-bridge creation. Without this,
        # a near-simultaneous ``prewarm`` (QThreadPool worker) and
        # ``submit`` (GUI thread) both observe ``self._executor is None``
        # and each construct their own ProcessPoolExecutor +
        # multiprocessing.Manager — leaving orphan subprocesses, two
        # progress queues, and in the worst case workers reporting into
        # a queue nobody drains (→ "Start OCR, nothing happens").
        self._executor_lock = threading.Lock()
        # tracking_id -> (on_progress, job_id_for_callback)
        self._progress_listeners: dict[str, tuple[OnProgress | None, str]] = {}
        # job_id -> Future (for cancellation)
        self._futures: dict[str, Future] = {}

    # -- lifecycle ---------------------------------------------------------

    def _is_executor_broken(self) -> bool:
        """Return True if a worker died and the pool is unusable.

        ``concurrent.futures.process.ProcessPoolExecutor`` flips an
        internal ``_broken`` flag when any worker terminates abruptly
        (segfault, OOM kill, missing DLL, recursive freeze_support
        spawn). After that, every subsequent ``.submit()`` raises
        ``BrokenProcessPool`` — which surfaces in our UI as
        "A child process terminated abruptly, the process pool is not
        usable anymore". We detect this defensively so the next
        ``_ensure_executor`` call rebuilds the pool instead of locking
        the user out of OCR for the rest of the session.
        """
        ex = self._executor
        if ex is None:
            return False
        broken = getattr(ex, "_broken", None)
        return bool(broken)

    def _ensure_executor(self) -> ProcessPoolExecutor:
        """Lazily instantiate the process pool and start the progress bridge.

        Self-healing: if the previous executor was poisoned by a worker
        crash (see ``_is_executor_broken``) it is shut down and replaced
        with a fresh one. This makes the "submit, child dies, every
        future call fails forever" scenario at most a single-job loss
        instead of a session-wide hang.

        ``max_tasks_per_child=10`` (Python 3.11+) recycles each worker
        after 10 jobs. Long-running sessions used to accumulate gigabytes
        of per-worker memory — torch caches, PyMuPDF mmaps, OCRmyPDF
        temp-files holding onto file-descriptor caches — because the
        default executor keeps workers alive forever. Recycling after
        10 jobs caps per-worker RSS at a stable plateau without enough
        churn to dominate the spawn-start cost.
        """
        with self._executor_lock:
            if self._executor is not None and self._is_executor_broken():
                logger.warning(
                    "ProcessPoolExecutor was broken by a worker crash — "
                    "shutting it down and creating a fresh pool."
                )
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Broken-pool shutdown raised: %s", exc)
                self._executor = None
            if self._executor is None:
                logger.info(
                    "Starting ProcessPoolExecutor with %d workers (recycle=10)",
                    self.max_workers,
                )
                # max_tasks_per_child was added in 3.11; fall back if we
                # ever get run on an older interpreter.
                executor_kwargs: dict[str, object] = {
                    "max_workers": self.max_workers,
                }
                if self._mp_context is not None:
                    executor_kwargs["mp_context"] = self._mp_context
                try:
                    self._executor = ProcessPoolExecutor(
                        **executor_kwargs,
                        max_tasks_per_child=10,
                    )
                except TypeError:
                    logger.debug(
                        "max_tasks_per_child unsupported on this Python; "
                        "workers will not recycle"
                    )
                    self._executor = ProcessPoolExecutor(**executor_kwargs)
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
        # Counter for transient OSErrors (e.g., Windows Defender briefly
        # locking the manager pipe). We keep draining across a handful
        # of these — exiting the drain thread for good would silently
        # freeze the progress bar for the rest of the session, since
        # ``_start_progress_bridge`` only re-spawns the thread when the
        # executor is rebuilt. True pipe death (``EOFError``) is still
        # terminal because retry there just spins.
        transient_os_errors = 0
        while not self._progress_stop.is_set():
            try:
                event = q.get(timeout=0.25)
            except queue_mod.Empty:
                transient_os_errors = 0
                continue
            except EOFError:
                # Manager pipe closed — the Manager subprocess died or
                # was shut down. No point retrying.
                logger.info("progress bridge: queue EOF — drain thread exiting")
                break
            except OSError as exc:
                transient_os_errors += 1
                if transient_os_errors >= 10:
                    logger.warning(
                        "progress bridge: too many transient OSError "
                        "from queue.get (%d); giving up: %s",
                        transient_os_errors, exc,
                    )
                    break
                logger.debug(
                    "progress bridge: transient OSError (#%d), "
                    "continuing: %s",
                    transient_os_errors, exc,
                )
                continue
            transient_os_errors = 0
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

        # ``BrokenProcessPool`` at submit time is a known race: the
        # previous job's worker can die after ``_ensure_executor``
        # already returned a "fine" executor but before the pool's
        # internal manager thread has flipped ``_broken``. The check in
        # ``_ensure_executor`` therefore misses the freshly-broken
        # state, and ``executor.submit`` below throws. Catch it once,
        # force a rebuild, and retry — ``_ensure_executor`` will spin
        # up a fresh pool the second time around.
        from concurrent.futures.process import BrokenProcessPool

        def _drop_listener() -> None:
            """Remove the listener we pre-registered on line above."""
            with self._progress_lock:
                self._progress_listeners.pop(tracking_id, None)

        try:
            future = executor.submit(
                _worker_run_job, job_to_dict(job), self._progress_queue, tracking_id
            )
        except BrokenProcessPool:
            logger.warning(
                "BrokenProcessPool at submit — rebuilding pool and retrying once"
            )
            with self._executor_lock:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Broken-pool shutdown raised: %s", exc)
                self._executor = None
            try:
                executor = self._ensure_executor()
                future = executor.submit(
                    _worker_run_job,
                    job_to_dict(job),
                    self._progress_queue,
                    tracking_id,
                )
            except BaseException:
                # Retry also failed (broken pool on fresh rebuild, or an
                # unrelated exception). Drop the listener we registered
                # above before propagating — otherwise it leaks in the
                # ``_progress_listeners`` dict forever, since no
                # ``_done`` callback will ever fire to clean it up.
                _drop_listener()
                raise
        except BaseException:
            # Non-BrokenProcessPool submit failure (e.g. pickling error
            # for an OCRJobConfig with an exotic field). Same leak
            # hazard — drop the pre-registered listener before the
            # exception propagates to the caller.
            _drop_listener()
            raise

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


