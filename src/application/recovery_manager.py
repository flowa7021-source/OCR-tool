"""Persist in-flight job state to disk so the app can recover after a crash.

Each active queue item is serialized as a JSON file in
``%LOCALAPPDATA%/OCRStudio/recovery/``. The file is removed when the job
reaches a terminal state (COMPLETED / CANCELLED / FAILED-with-no-retry).

On startup the UI can call :meth:`RecoveryManager.list_pending` to present the
user with a "resume?" dialog. This is a best-effort facility: there is no
attempt to continue from the exact failed page — the entire job is simply
re-queued with its original config.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

from src.application.parallel_processor import job_from_dict, job_to_dict
from src.core.models import QueueItem
from src.shared.constants import RECOVERY_DIR
from src.shared.types import JobStatus

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Best-effort crash-recovery snapshot for queued jobs."""

    def __init__(self, recovery_dir: Path | None = None) -> None:
        """Create a manager using ``recovery_dir`` (defaults to RECOVERY_DIR)."""
        self._dir = recovery_dir or RECOVERY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ api
    def snapshot(self, item: QueueItem) -> None:
        """Atomically write ``item`` to disk. Safe to call repeatedly."""
        if item.config is None:
            return
        payload = {
            "job_id": item.job_id,
            "status": item.status.value,
            "progress_current": item.progress_current,
            "progress_total": item.progress_total,
            "config": job_to_dict(item.config),
        }
        target = self._path_for(item.job_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError as exc:
            logger.warning("Recovery snapshot failed for %s: %s", item.job_id, exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def remove(self, job_id: str) -> None:
        """Delete the snapshot for ``job_id`` if present."""
        try:
            self._path_for(job_id).unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not remove recovery for %s: %s", job_id, exc)

    def list_pending(self) -> list[QueueItem]:
        """Return pending items recovered from disk.

        Terminal-state snapshots (COMPLETED) are not returned; they are also
        deleted as a side effect, so we self-heal from stale files.
        """
        items: list[QueueItem] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                status = JobStatus(data.get("status", JobStatus.PENDING.value))
                if status is JobStatus.COMPLETED:
                    # Clean up stale successful snapshots
                    path.unlink(missing_ok=True)
                    continue
                cfg_dict = data.get("config")
                if not isinstance(cfg_dict, dict):
                    path.unlink(missing_ok=True)
                    continue
                cfg = job_from_dict(cfg_dict)
                item = QueueItem(
                    job_id=str(data.get("job_id", path.stem)),
                    config=cfg,
                    status=JobStatus.PENDING,  # restart as pending
                    progress_current=0,
                    progress_total=int(data.get("progress_total", 0)),
                )
                items.append(item)
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Ignoring malformed recovery file %s: %s", path, exc)
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        return items

    def clear_all(self) -> int:
        """Remove every recovery file. Returns count of files deleted."""
        count = 0
        for path in self._dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        return count

    # -------------------------------------------------------------- internal
    def _path_for(self, job_id: str) -> Path:
        # Sanitize job_id for use as a filename
        safe = "".join(c for c in job_id if c.isalnum() or c in ("-", "_")) or "job"
        return self._dir / f"{safe}.json"


__all__ = ["RecoveryManager"]
