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
