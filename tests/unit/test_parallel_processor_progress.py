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
