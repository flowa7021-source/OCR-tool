"""Tests for :mod:`src.application.queue_manager`."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from src.application.queue_manager import (
    EVENT_ADDED,
    EVENT_REMOVED,
    EVENT_UPDATED,
    QueueManager,
)
from src.core.models import OCRJobConfig, ProfileData, QueueItem
from src.shared.types import JobStatus


def _make_item(name: str = "file.pdf") -> QueueItem:
    cfg = OCRJobConfig(
        input_path=f"/tmp/{name}",
        output_path=f"/tmp/out_{name}",
        profile=ProfileData(name="t"),
    )
    return QueueItem(config=cfg)


def test_add_and_list() -> None:
    q = QueueManager()
    a, b = _make_item("a.pdf"), _make_item("b.pdf")
    q.add(a)
    q.add(b)
    items = q.list_items()
    assert [it.job_id for it in items] == [a.job_id, b.job_id]


def test_remove() -> None:
    q = QueueManager()
    item = _make_item()
    q.add(item)
    assert q.remove(item.job_id) is True
    assert q.list_items() == []
    assert q.remove("ghost") is False


def test_update_status_persists() -> None:
    q = QueueManager()
    item = _make_item()
    q.add(item)
    q.update_status(item.job_id, JobStatus.RUNNING)
    got = q.get(item.job_id)
    assert got is not None
    assert got.status == JobStatus.RUNNING

    q.update_status(item.job_id, JobStatus.FAILED, error="boom")
    got = q.get(item.job_id)
    assert got is not None
    assert got.status == JobStatus.FAILED
    assert got.error_message == "boom"


def test_update_progress_persists() -> None:
    q = QueueManager()
    item = _make_item()
    q.add(item)
    q.update_progress(item.job_id, 3, 10)
    got = q.get(item.job_id)
    assert got is not None
    assert got.progress_current == 3
    assert got.progress_total == 10


def test_move_up_and_down() -> None:
    q = QueueManager()
    a, b, c = _make_item("a.pdf"), _make_item("b.pdf"), _make_item("c.pdf")
    q.add(a)
    q.add(b)
    q.add(c)

    assert q.move_up(c.job_id) is True
    assert [it.job_id for it in q.list_items()] == [a.job_id, c.job_id, b.job_id]

    assert q.move_down(a.job_id) is True
    assert [it.job_id for it in q.list_items()] == [c.job_id, a.job_id, b.job_id]

    # Top/bottom edge cases.
    first = q.list_items()[0]
    last = q.list_items()[-1]
    assert q.move_up(first.job_id) is False
    assert q.move_down(last.job_id) is False
    assert q.move_up("ghost") is False
    assert q.move_down("ghost") is False


def test_next_pending_flips_to_running() -> None:
    q = QueueManager()
    a, b = _make_item("a.pdf"), _make_item("b.pdf")
    q.add(a)
    q.add(b)

    picked = q.next_pending()
    assert picked is not None
    assert picked.job_id == a.job_id
    assert picked.status == JobStatus.RUNNING

    # Queue now reflects the new status.
    stored = q.get(a.job_id)
    assert stored is not None
    assert stored.status == JobStatus.RUNNING

    # Second call picks the next pending item.
    picked2 = q.next_pending()
    assert picked2 is not None
    assert picked2.job_id == b.job_id

    # No more pending → None.
    assert q.next_pending() is None


def test_clear_completed_removes_terminals() -> None:
    q = QueueManager()
    a, b, c, d = (_make_item(f"{n}.pdf") for n in "abcd")
    for it in (a, b, c, d):
        q.add(it)

    q.update_status(a.job_id, JobStatus.COMPLETED)
    q.update_status(b.job_id, JobStatus.FAILED)
    q.update_status(c.job_id, JobStatus.CANCELLED)
    # d stays PENDING

    removed = q.clear_completed()
    assert removed == 3
    remaining = [it.job_id for it in q.list_items()]
    assert remaining == [d.job_id]


def test_subscribe_notifications_on_add_update_remove() -> None:
    q = QueueManager()
    events: list[tuple[str, str]] = []

    def listener(item: QueueItem, event: str) -> None:
        events.append((event, item.job_id))

    q.subscribe(listener)

    item = _make_item()
    q.add(item)
    q.update_status(item.job_id, JobStatus.RUNNING)
    q.update_progress(item.job_id, 1, 2)
    q.remove(item.job_id)

    kinds = [e[0] for e in events]
    assert kinds[0] == EVENT_ADDED
    assert EVENT_UPDATED in kinds
    assert kinds[-1] == EVENT_REMOVED
    # All events refer to our item.
    assert all(jid == item.job_id for _, jid in events)


def test_thread_safety_concurrent_add_remove() -> None:
    """Smoke: 4 threads hammering add/remove must not raise or corrupt state."""
    q = QueueManager()
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            created: list[str] = []
            for i in range(50):
                item = _make_item(f"w{n}_{i}.pdf")
                q.add(item)
                created.append(item.job_id)
            for jid in created:
                q.remove(jid)
        except Exception as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # All items were removed, so final state is empty.
    assert q.list_items() == []
