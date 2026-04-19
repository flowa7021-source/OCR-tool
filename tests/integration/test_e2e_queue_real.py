"""Real-OCR: multi-file queue with one failing input must not block the rest.

The user's #1 real-world scenario: drag 5-10 PDFs into the queue, hit Start.
If one file is corrupt/unreadable/password-protected, the queue must:

  * Mark that one file FAILED with a user-facing error message
  * Continue processing the remaining files
  * Surface all results (success + failure) in the final listing

This is the end-to-end "real OCR + real queue" path — mocked tests in
``tests/unit/test_queue_manager.py`` only exercise the enqueue API;
here we push real jobs through ``ParallelProcessor`` + real Tesseract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.parallel_processor import ParallelProcessor
from src.application.queue_manager import QueueManager
from src.core.models import OCRJobConfig, QueueItem
from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


class TestQueueRealOCRMixedResults:
    """Three real PDFs + one corrupt input → COMPLETED ×3, FAILED ×1."""

    def test_one_corrupt_input_does_not_break_others(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        # Three good inputs
        good_pdfs = [
            render_clean_text_pdf(
                tmp_path / f"good_{i}.pdf", f"document {i}", pages=1,
            )
            for i in range(3)
        ]
        # One deliberately corrupt — existing file, wrong bytes
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"NOT A PDF, JUST GARBAGE\n" * 50)

        profile = make_realistic_profile(binarization="otsu", dpi=200)

        queue = QueueManager()
        items: list[QueueItem] = []
        for pdf in [*good_pdfs, bad_pdf]:
            item = QueueItem(
                config=OCRJobConfig(
                    input_path=str(pdf),
                    output_path=str(tmp_path / f"{pdf.stem}_ocr.pdf"),
                    profile=profile,
                )
            )
            queue.add(item)
            items.append(item)

        # Run sequentially in-process (workers=1) so failures don't
        # depend on process-pool quirks we test separately.
        results: dict[str, tuple[JobStatus, str | None]] = {}

        processor = ParallelProcessor(max_workers=1)
        try:
            futures = []
            for item in items:
                fut = processor.submit(
                    item.config,
                    on_complete=lambda r, _item=item: results.__setitem__(
                        _item.job_id, (r.status, r.error)
                    ),
                    on_error=lambda exc, _item=item: results.__setitem__(
                        _item.job_id, (JobStatus.FAILED, str(exc))
                    ),
                    job_id=item.job_id,
                )
                futures.append(fut)
            for fut in futures:
                fut.result(timeout=180)
        finally:
            processor.shutdown()

        # Exactly one FAILED (the corrupt PDF) + three COMPLETED
        failed = [
            (jid, err) for jid, (status, err) in results.items()
            if status is JobStatus.FAILED
        ]
        completed = [
            jid for jid, (status, _) in results.items()
            if status is JobStatus.COMPLETED
        ]
        assert len(failed) == 1, (
            f"Expected exactly 1 FAILED (corrupt.pdf), got {len(failed)}: "
            f"{failed!r}"
        )
        assert len(completed) == 3, (
            f"Expected 3 COMPLETED (good files), got {len(completed)}"
        )
        # Error message must be populated — UI relies on it for the
        # "why did this file fail?" tooltip.
        _, err = failed[0]
        assert err, "FAILED job must carry a non-empty error message"
