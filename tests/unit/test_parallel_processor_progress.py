"""Tests for the progress-bridge plumbing in ParallelProcessor.

The bridge converts cross-process ``(tracking_id, current, total, stage)``
events into per-listener callbacks. We exercise it directly without spinning
up real worker processes — that keeps the test portable and fast.
"""

from __future__ import annotations

import time

from src.application.parallel_processor import ParallelProcessor


def test_drain_thread_dispatches_to_listener() -> None:
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]
    assert pp._progress_queue is not None
    events: list[tuple[str, int, int, str]] = []

    def cb(job_id: str, current: int, total: int, stage: str) -> None:
        events.append((job_id, current, total, stage))

    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._progress_listeners["tid-1"] = (cb, "job-42")  # type: ignore[attr-defined]

    # Emit a few events as a worker would
    pp._progress_queue.put(("tid-1", 1, 10, "preprocess"))
    pp._progress_queue.put(("tid-1", 5, 10, "ocr"))
    pp._progress_queue.put(("tid-1", 10, 10, "done"))

    # Wait up to 2s for the drain thread to process
    deadline = time.time() + 2.0
    while len(events) < 3 and time.time() < deadline:
        time.sleep(0.05)

    pp.shutdown(wait=False)

    assert len(events) == 3
    assert events[0] == ("job-42", 1, 10, "preprocess")
    assert events[-1] == ("job-42", 10, 10, "done")


def test_unknown_tracking_id_is_ignored() -> None:
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]
    assert pp._progress_queue is not None
    received: list[object] = []

    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._progress_listeners["known"] = (  # type: ignore[attr-defined]
            lambda *a: received.append(a),
            "j",
        )

    pp._progress_queue.put(("unknown", 1, 1, "x"))
    pp._progress_queue.put(("known", 2, 2, "y"))

    deadline = time.time() + 2.0
    while not received and time.time() < deadline:
        time.sleep(0.05)

    pp.shutdown(wait=False)
    assert received == [("j", 2, 2, "y")]


def test_shutdown_is_idempotent() -> None:
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]
    pp.shutdown()
    pp.shutdown()  # second call must not raise


def test_setup_worker_logging_writes_file(tmp_path, monkeypatch) -> None:
    """Worker setup must create a per-PID log file in OCRSTUDIO_LOGS_DIR.

    Regression guard for the ``--windowed`` diagnostic gap: PyInstaller
    windowed builds close stderr, so the previous ``StreamHandler``-only
    setup silently dropped every worker log line. Without a file
    handler we had no way to tell where a job died.
    """
    import logging
    import os

    from src.application.parallel_processor import _setup_worker_logging

    monkeypatch.setenv("OCRSTUDIO_LOGS_DIR", str(tmp_path))

    worker_logger = _setup_worker_logging()
    worker_logger.info("hello from worker %d", os.getpid())

    # Flush + close so the file is readable on Windows.
    import contextlib as _ctx

    for h in list(logging.getLogger().handlers):
        with _ctx.suppress(Exception):
            h.flush()

    log_path = tmp_path / f"worker-{os.getpid()}.log"
    assert log_path.exists(), f"worker log file not created at {log_path}"
    content = log_path.read_text(encoding="utf-8")
    assert "hello from worker" in content, (
        "worker log file exists but doesn't contain the test message"
    )

    # Clean up handlers so the file can be unlinked on Windows and the
    # next test starts from a clean root logger.
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
        with _ctx.suppress(Exception):
            h.close()


def test_listener_exception_does_not_crash_drain_thread() -> None:
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]
    assert pp._progress_queue is not None

    calls: list[int] = []

    def good(job_id: str, current: int, total: int, stage: str) -> None:
        calls.append(current)

    def bad(*_args: object) -> None:
        raise RuntimeError("boom")

    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._progress_listeners["bad"] = (bad, "b")  # type: ignore[attr-defined]
        pp._progress_listeners["good"] = (good, "g")  # type: ignore[attr-defined]

    pp._progress_queue.put(("bad", 1, 1, "x"))
    pp._progress_queue.put(("good", 7, 10, "ok"))

    deadline = time.time() + 2.0
    while not calls and time.time() < deadline:
        time.sleep(0.05)

    pp.shutdown(wait=False)
    assert calls == [7]


def test_submit_retries_once_on_broken_process_pool() -> None:
    """If ``executor.submit`` throws BrokenProcessPool (worker died
    after _ensure_executor's broken-check), the pool must self-heal
    and the job must be submitted against the fresh executor.
    """
    from concurrent.futures.process import BrokenProcessPool
    from unittest.mock import MagicMock

    from src.core.models import OCRJobConfig, ProfileData

    pp = ParallelProcessor(max_workers=1)

    submit_calls: list[object] = []

    class _Future:
        def add_done_callback(self, _):
            pass

    class _FirstExecutor:
        def submit(self, *_a, **_kw):
            submit_calls.append("first")
            raise BrokenProcessPool("worker died")

        def shutdown(self, *_a, **_kw):
            pass

    class _SecondExecutor:
        def submit(self, *_a, **_kw):
            submit_calls.append("second")
            return _Future()

        def shutdown(self, *_a, **_kw):
            pass

    executors = iter([_FirstExecutor(), _SecondExecutor()])

    def _fake_ensure():
        pp._executor = next(executors)  # type: ignore[attr-defined]
        return pp._executor  # type: ignore[attr-defined]

    pp._ensure_executor = _fake_ensure  # type: ignore[assignment]
    pp._start_progress_bridge = lambda: None  # type: ignore[assignment]

    job = OCRJobConfig(
        input_path="x.pdf",
        output_path="y.pdf",
        profile=ProfileData(name="test"),
    )
    future = pp.submit(job, on_progress=MagicMock(), on_complete=MagicMock())

    assert submit_calls == ["first", "second"], (
        f"Expected one retry after BrokenProcessPool, got: {submit_calls}"
    )
    assert future is not None


def test_concurrent_ensure_executor_creates_only_one_pool() -> None:
    """prewarm (QThreadPool) and submit (GUI thread) race on the first
    click of Start OCR. Without a lock they each construct their own
    ProcessPoolExecutor + Manager, leaving one orphaned and workers
    potentially reporting into a queue nobody drains.
    """
    import threading

    pp = ParallelProcessor(max_workers=1)
    created: list[object] = []

    # Stub the heavy constructors so the test doesn't spawn real subprocesses.
    from src.application import parallel_processor as pp_mod

    class _FakeExecutor:
        def __init__(self, *a, **kw):
            created.append(self)
            # Simulate the slow Windows Manager spawn so the race window is real.
            time.sleep(0.1)

        def submit(self, *a, **kw):
            class _F:
                def add_done_callback(self, _):
                    pass

                def result(self):
                    return {}

            return _F()

        def shutdown(self, *a, **kw):
            pass

    with (
        _patch(pp_mod, "ProcessPoolExecutor", _FakeExecutor),
        _patch_method(pp, "_start_progress_bridge", lambda: None),
    ):
        threads = [threading.Thread(target=pp._ensure_executor) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(created) == 1, (
        f"Expected exactly one ProcessPoolExecutor, got {len(created)} — "
        "race condition in _ensure_executor regressed"
    )


# ---- tiny helpers used only by the race test -----------------------------


import contextlib  # noqa: E402


@contextlib.contextmanager
def _patch(owner, name, value):
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


@contextlib.contextmanager
def _patch_method(instance, name, value):
    original = getattr(instance, name)
    setattr(instance, name, value)
    try:
        yield
    finally:
        setattr(instance, name, original)


# ---------------------------------------------------------------------------
# Regression: drain thread must survive transient OSErrors
# ---------------------------------------------------------------------------


def test_drain_thread_survives_transient_oserror() -> None:
    """Windows Defender briefly locks the Manager pipe → OSError from
    ``queue.get``. The previous code broke out of the drain loop on the
    first such error, silently freezing the progress bar for the rest
    of the session because the loop only gets re-spawned when the
    executor is rebuilt. Confirm that a one-off OSError is tolerated
    and events after it are still dispatched.
    """
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]
    assert pp._progress_queue is not None

    events: list[tuple[int, int, str]] = []

    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._progress_listeners["tid"] = (  # type: ignore[attr-defined]
            lambda jid, c, t, s: events.append((c, t, s)),
            "jid",
        )

    # Monkey-patch the queue's ``get`` to throw one OSError, then
    # behave normally. We leave the OSError as the FIRST result and
    # ensure subsequent calls return the real events.
    original_get = pp._progress_queue.get  # type: ignore[attr-defined]
    injected: list[bool] = [False]

    def flaky_get(*args, **kwargs):
        if not injected[0]:
            injected[0] = True
            raise OSError("simulated antivirus transient lock on pipe")
        return original_get(*args, **kwargs)

    pp._progress_queue.get = flaky_get  # type: ignore[attr-defined,assignment]

    pp._progress_queue.put(("tid", 1, 1, "ocr"))

    deadline = time.time() + 3.0
    while not events and time.time() < deadline:
        time.sleep(0.05)

    pp.shutdown(wait=False)

    assert injected[0], "flaky_get was never invoked — test is defective"
    assert events == [(1, 1, "ocr")], (
        f"Event lost after transient OSError — drain thread died "
        f"prematurely. Got: {events!r}"
    )


# ---------------------------------------------------------------------------
# Regression: listener dict must not leak when submit fails
# ---------------------------------------------------------------------------


def test_listener_dropped_when_broken_pool_retry_also_fails() -> None:
    """If the first submit raises BrokenProcessPool AND the retry
    submit also raises (say, another BrokenProcessPool on the fresh
    pool, or a pickling error), the listener we pre-registered must
    be cleaned up. Otherwise it accumulates in ``_progress_listeners``
    for every failed submit, a slow memory leak over long sessions.
    """
    from concurrent.futures.process import BrokenProcessPool
    from unittest.mock import MagicMock

    from src.core.models import OCRJobConfig, ProfileData

    pp = ParallelProcessor(max_workers=1)

    class _AlwaysBrokenExecutor:
        def submit(self, *_a, **_kw):
            raise BrokenProcessPool("worker died")

        def shutdown(self, *_a, **_kw):
            pass

    def _fake_ensure():
        pp._executor = _AlwaysBrokenExecutor()  # type: ignore[attr-defined]
        return pp._executor  # type: ignore[attr-defined]

    pp._ensure_executor = _fake_ensure  # type: ignore[assignment]
    pp._start_progress_bridge = lambda: None  # type: ignore[assignment]

    job = OCRJobConfig(
        input_path="x.pdf",
        output_path="y.pdf",
        profile=ProfileData(name="test"),
    )

    import pytest

    # Both attempts raise → exception propagates.
    with pytest.raises(BrokenProcessPool):
        pp.submit(job, on_progress=MagicMock(), on_complete=MagicMock())

    # Listener must NOT still be dangling in the dict.
    with pp._progress_lock:  # type: ignore[attr-defined]
        dangling = dict(pp._progress_listeners)  # type: ignore[attr-defined]
    assert dangling == {}, (
        f"listener leaked after double-failed submit: {dangling!r}"
    )


def test_listener_dropped_when_initial_submit_raises_unrelated_error() -> None:
    """Non-BrokenProcessPool failures (e.g., pickle error on a weird
    job argument) also must not leak the listener."""
    from unittest.mock import MagicMock

    from src.core.models import OCRJobConfig, ProfileData

    pp = ParallelProcessor(max_workers=1)

    class _PickleFailingExecutor:
        def submit(self, *_a, **_kw):
            raise RuntimeError("cannot pickle custom object")

        def shutdown(self, *_a, **_kw):
            pass

    def _fake_ensure():
        pp._executor = _PickleFailingExecutor()  # type: ignore[attr-defined]
        return pp._executor  # type: ignore[attr-defined]

    pp._ensure_executor = _fake_ensure  # type: ignore[assignment]
    pp._start_progress_bridge = lambda: None  # type: ignore[assignment]

    job = OCRJobConfig(
        input_path="x.pdf",
        output_path="y.pdf",
        profile=ProfileData(name="test"),
    )

    import pytest

    with pytest.raises(RuntimeError, match="cannot pickle"):
        pp.submit(job, on_progress=MagicMock(), on_complete=MagicMock())

    with pp._progress_lock:  # type: ignore[attr-defined]
        dangling = dict(pp._progress_listeners)  # type: ignore[attr-defined]
    assert dangling == {}
