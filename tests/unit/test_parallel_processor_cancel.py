"""Tests for ParallelProcessor cancellation APIs."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock

from src.application.parallel_processor import ParallelProcessor


def test_cancel_unknown_id_returns_false() -> None:
    pp = ParallelProcessor(max_workers=1)
    assert pp.cancel_job("nope") is False
    pp.shutdown(wait=False)


def test_cancel_registers_and_clears_future() -> None:
    """Simulate a stored future and verify cancel_job flow."""
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]

    fake_future = MagicMock(spec=Future)
    fake_future.cancel.return_value = True

    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._futures["abc"] = fake_future  # type: ignore[attr-defined]

    assert pp.active_job_count() == 1
    assert pp.cancel_job("abc") is True
    fake_future.cancel.assert_called_once()
    assert pp.active_job_count() == 0
    pp.shutdown(wait=False)


def test_cancel_all_removes_every_pending() -> None:
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]

    futures = {}
    for idx in range(3):
        f = MagicMock(spec=Future)
        f.cancel.return_value = True
        futures[f"j-{idx}"] = f
    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._futures.update(futures)  # type: ignore[attr-defined]

    assert pp.cancel_all() == 3
    assert pp.active_job_count() == 0
    pp.shutdown(wait=False)


def test_cancel_all_skips_uncancellable_but_keeps_them_reported() -> None:
    pp = ParallelProcessor(max_workers=1)
    pp._start_progress_bridge()  # type: ignore[attr-defined]

    cancellable = MagicMock(spec=Future)
    cancellable.cancel.return_value = True
    running = MagicMock(spec=Future)
    running.cancel.return_value = False

    with pp._progress_lock:  # type: ignore[attr-defined]
        pp._futures["a"] = cancellable  # type: ignore[attr-defined]
        pp._futures["b"] = running  # type: ignore[attr-defined]

    assert pp.cancel_all() == 1
    # The uncancellable future remains tracked
    assert pp.active_job_count() == 1
    pp.shutdown(wait=False)
