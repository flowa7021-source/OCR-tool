"""Tests for the crash-recovery snapshot manager."""

from __future__ import annotations

from pathlib import Path

from src.application.recovery_manager import RecoveryManager
from src.core.models import OCRJobConfig, ProfileData, QueueItem
from src.shared.types import JobStatus


def _make_item(tmp_path: Path, job_id: str = "abc123") -> QueueItem:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")  # minimal stub; RecoveryManager doesn't open it
    cfg = OCRJobConfig(
        input_path=str(pdf),
        output_path=str(tmp_path / "doc_ocr.pdf"),
        profile=ProfileData(name="default"),
    )
    return QueueItem(job_id=job_id, config=cfg, progress_total=5, progress_current=2)


def test_snapshot_creates_json(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    rm.snapshot(_make_item(tmp_path))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_snapshot_is_atomic(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    rm.snapshot(_make_item(tmp_path))
    # No leftover .tmp file
    assert not list(tmp_path.glob("*.tmp"))


def test_roundtrip_list_pending(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    rm.snapshot(_make_item(tmp_path, job_id="one"))
    rm.snapshot(_make_item(tmp_path, job_id="two"))
    pending = rm.list_pending()
    assert len(pending) == 2
    ids = sorted(p.job_id for p in pending)
    assert ids == ["one", "two"]
    # Restored items should be in PENDING state regardless of what was saved
    for item in pending:
        assert item.status is JobStatus.PENDING
        assert item.config is not None


def test_completed_snapshots_are_purged(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    item = _make_item(tmp_path)
    item.status = JobStatus.COMPLETED
    rm.snapshot(item)
    pending = rm.list_pending()
    assert pending == []
    # The stale file was also removed
    assert not list(tmp_path.glob("*.json"))


def test_remove(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    item = _make_item(tmp_path, job_id="remove_me")
    rm.snapshot(item)
    rm.remove("remove_me")
    assert rm.list_pending() == []


def test_clear_all(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    rm.snapshot(_make_item(tmp_path, job_id="a"))
    rm.snapshot(_make_item(tmp_path, job_id="b"))
    rm.snapshot(_make_item(tmp_path, job_id="c"))
    assert rm.clear_all() == 3
    assert rm.list_pending() == []


def test_malformed_json_is_ignored(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    pending = rm.list_pending()
    assert pending == []
    # The bad file is cleaned up
    assert not bad.exists()


def test_sanitized_filename(tmp_path: Path) -> None:
    rm = RecoveryManager(tmp_path)
    item = _make_item(tmp_path, job_id="../../evil/id")
    rm.snapshot(item)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    # Filename was sanitized — no path separators left
    assert "/" not in files[0].name
    assert "\\" not in files[0].name
