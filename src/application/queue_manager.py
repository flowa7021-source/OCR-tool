"""Thread-safe, in-memory job queue.

The :class:`QueueManager` orders :class:`~src.core.models.QueueItem` instances
in FIFO order and exposes atomic transitions for status/progress updates. It
is consumed by the UI (to render the queue table) and the
:class:`~src.application.parallel_processor.ParallelProcessor` (to pull the
next pending job).
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Callable

from src.core.models import QueueItem
from src.shared.types import JobStatus

logger = logging.getLogger(__name__)


QueueListener = Callable[[QueueItem, str], None]

EVENT_ADDED: str = "added"
EVENT_UPDATED: str = "updated"
EVENT_REMOVED: str = "removed"


class QueueManager:
    """In-memory job queue with observer notifications.

    Thread safety:
        All public methods acquire an internal :class:`threading.Lock`.
        Listener callbacks are invoked AFTER the lock is released so that
        re-entrant calls from listeners cannot deadlock.
    """

    def __init__(self) -> None:
        """Create a new, empty queue."""
        self._items: list[QueueItem] = []
        self._lock = threading.Lock()
        self._listeners: list[QueueListener] = []

    # ------------------------------------------------------------------
    # Observer wiring
    # ------------------------------------------------------------------

    def subscribe(self, listener: QueueListener) -> None:
        """Register ``listener`` to receive queue events.

        Args:
            listener: Callable ``(item, event)`` invoked for every change.
        """
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: QueueListener) -> None:
        """Remove a previously registered listener, if present."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify(self, item: QueueItem, event: str) -> None:
        """Dispatch ``event`` to all listeners (called OUTSIDE the lock)."""
        for listener in list(self._listeners):
            try:
                listener(item, event)
            except Exception:  # noqa: BLE001 - listeners must not break the queue
                logger.exception("Queue listener raised")

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, item: QueueItem) -> None:
        """Append ``item`` to the queue.

        Args:
            item: Item to enqueue.
        """
        with self._lock:
            self._items.append(item)
            snapshot = copy.copy(item)
        logger.info("Queue: added %s (%s)", item.job_id, item.file_name)
        self._notify(snapshot, EVENT_ADDED)

    def remove(self, job_id: str) -> bool:
        """Remove the item with the given id.

        Args:
            job_id: Identifier of the queue item.

        Returns:
            True if an item was removed, False otherwise.
        """
        removed: QueueItem | None = None
        with self._lock:
            for idx, it in enumerate(self._items):
                if it.job_id == job_id:
                    removed = self._items.pop(idx)
                    break
        if removed is None:
            return False
        logger.info("Queue: removed %s", job_id)
        self._notify(copy.copy(removed), EVENT_REMOVED)
        return True

    def list_items(self) -> list[QueueItem]:
        """Return a shallow copy of the queue (safe to iterate outside lock)."""
        with self._lock:
            return [copy.copy(it) for it in self._items]

    def get(self, job_id: str) -> QueueItem | None:
        """Return a copy of the item with ``job_id`` or None if missing."""
        with self._lock:
            for it in self._items:
                if it.job_id == job_id:
                    return copy.copy(it)
        return None

    def update_status(
        self, job_id: str, status: JobStatus, error: str = ""
    ) -> None:
        """Transition an item to a new status.

        Args:
            job_id: Identifier of the item.
            status: New status value.
            error: Optional error message (stored on the item when set).
        """
        snapshot: QueueItem | None = None
        with self._lock:
            for it in self._items:
                if it.job_id == job_id:
                    it.status = status
                    if error:
                        it.error_message = error
                    snapshot = copy.copy(it)
                    break
        if snapshot is not None:
            self._notify(snapshot, EVENT_UPDATED)

    def update_progress(self, job_id: str, current: int, total: int) -> None:
        """Update the progress counters of a queue item."""
        snapshot: QueueItem | None = None
        with self._lock:
            for it in self._items:
                if it.job_id == job_id:
                    it.progress_current = int(current)
                    it.progress_total = int(total)
                    snapshot = copy.copy(it)
                    break
        if snapshot is not None:
            self._notify(snapshot, EVENT_UPDATED)

    def move_up(self, job_id: str) -> bool:
        """Swap the item with its predecessor.

        Returns:
            True if the move happened, False if the item is missing or already
            at the top.
        """
        snapshot: QueueItem | None = None
        with self._lock:
            for idx, it in enumerate(self._items):
                if it.job_id == job_id:
                    if idx == 0:
                        return False
                    self._items[idx - 1], self._items[idx] = (
                        self._items[idx],
                        self._items[idx - 1],
                    )
                    snapshot = copy.copy(self._items[idx - 1])
                    break
            else:
                return False
        if snapshot is not None:
            self._notify(snapshot, EVENT_UPDATED)
        return True

    def move_down(self, job_id: str) -> bool:
        """Swap the item with its successor (mirror of :meth:`move_up`)."""
        snapshot: QueueItem | None = None
        with self._lock:
            for idx, it in enumerate(self._items):
                if it.job_id == job_id:
                    if idx >= len(self._items) - 1:
                        return False
                    self._items[idx + 1], self._items[idx] = (
                        self._items[idx],
                        self._items[idx + 1],
                    )
                    snapshot = copy.copy(self._items[idx + 1])
                    break
            else:
                return False
        if snapshot is not None:
            self._notify(snapshot, EVENT_UPDATED)
        return True

    def clear_completed(self) -> int:
        """Remove COMPLETED, FAILED and CANCELLED items.

        Returns:
            Number of items removed.
        """
        removed: list[QueueItem] = []
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        with self._lock:
            keep: list[QueueItem] = []
            for it in self._items:
                if it.status in terminal:
                    removed.append(it)
                else:
                    keep.append(it)
            self._items = keep
        for it in removed:
            self._notify(copy.copy(it), EVENT_REMOVED)
        return len(removed)

    def next_pending(self) -> QueueItem | None:
        """Atomically pick the next PENDING item and flip it to RUNNING.

        Returns:
            A copy of the updated item, or None if there is no pending work.
        """
        snapshot: QueueItem | None = None
        with self._lock:
            for it in self._items:
                if it.status == JobStatus.PENDING:
                    it.status = JobStatus.RUNNING
                    snapshot = copy.copy(it)
                    break
        if snapshot is not None:
            self._notify(snapshot, EVENT_UPDATED)
        return snapshot

    def pause(self, job_id: str) -> None:
        """Mark an item as PAUSED."""
        self.update_status(job_id, JobStatus.PAUSED)

    def resume(self, job_id: str) -> None:
        """Re-mark a previously paused item as PENDING."""
        self.update_status(job_id, JobStatus.PENDING)

    def cancel(self, job_id: str) -> None:
        """Mark an item as CANCELLED."""
        self.update_status(job_id, JobStatus.CANCELLED)
