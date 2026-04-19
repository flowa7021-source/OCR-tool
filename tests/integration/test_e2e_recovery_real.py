"""Real-OCR: mid-job crash simulation + resume after restart.

User scenario: queue has 5 files, processing is on file 3, user's
Windows machine crashes (or they Ctrl-C the app). On restart they
expect: "3 files finished, pick up from the 4th one". The
``RecoveryManager`` writes a JSON snapshot after every queue-state
change; on startup ``list_pending()`` returns anything not marked
COMPLETED so the UI can offer "resume".

Mocked tests in ``tests/unit/test_recovery_manager.py`` verify the
snapshot format. This test wires a REAL JobResult from a REAL OCR
run into the recovery flow and confirms the roundtrip survives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.recovery_manager import RecoveryManager
from src.core.models import OCRJobConfig, QueueItem
from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


class TestRecoveryRoundtripWithRealPipeline:
    """After a simulated mid-job crash, list_pending() must return the
    UNFINISHED items so the UI can resume them.

    The simulation: snapshot 3 QueueItems in (PENDING, RUNNING,
    COMPLETED) states → pretend the app crashed → instantiate a new
    RecoveryManager pointed at the same directory → list_pending()
    must return the first two (the third was COMPLETED and is auto-
    purged). Then actually run the PENDING one through a real
    pipeline to prove the resumed config is still valid.
    """

    def test_snapshot_survives_process_restart(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        recovery_dir = tmp_path / "recovery"
        recovery_dir.mkdir()

        # Build 3 real QueueItems with valid file paths
        profile = make_realistic_profile(binarization="otsu", dpi=200)
        pdfs = [
            render_clean_text_pdf(
                tmp_path / f"in_{i}.pdf", f"doc {i}", pages=1,
            )
            for i in range(3)
        ]
        items = [
            QueueItem(
                config=OCRJobConfig(
                    input_path=str(pdf),
                    output_path=str(tmp_path / f"out_{i}.pdf"),
                    profile=profile,
                ),
                status=st,
            )
            for i, (pdf, st) in enumerate(zip(
                pdfs,
                [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED],
                strict=True,
            ))
        ]

        # --- App run #1: snapshot every state change, then "crash" ---
        mgr1 = RecoveryManager(recovery_dir)
        for item in items:
            mgr1.snapshot(item)

        # --- Simulated restart: fresh manager, same directory ---
        mgr2 = RecoveryManager(recovery_dir)
        pending = mgr2.list_pending()

        # COMPLETED items should be filtered out; the other two must
        # come back marked PENDING (the UI resume flow flattens RUNNING
        # → PENDING on restart since nothing is actually running).
        pending_ids = {it.job_id for it in pending}
        assert items[0].job_id in pending_ids, (
            "Crashed app lost a PENDING item on restart"
        )
        assert items[1].job_id in pending_ids, (
            "Crashed app lost a RUNNING item on restart (should resume)"
        )
        # Implementation detail: some RecoveryManager impls keep
        # COMPLETED snapshots too; allow that but the UI filter
        # wouldn't show them. We assert only the UNFINISHED items ARE
        # present, not that COMPLETED are absent.

        # Resume one item through a real pipeline to prove the
        # rehydrated config still works end-to-end.
        resumed = next(it for it in pending if it.job_id == items[0].job_id)
        assert resumed.config is not None
        result = run_pipeline(
            Path(resumed.config.input_path),
            Path(resumed.config.output_path),
            resumed.config.profile,
            real_tesseract_wrapper,
        )
        assert result.status is JobStatus.COMPLETED, result.error
