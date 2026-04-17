"""End-to-end tests using the production ProcessPoolExecutor path.

The GUI submits jobs via ``ParallelProcessor.submit`` which spawns a
worker subprocess, pickles the job config, runs the full pipeline
inside the worker, and streams progress back via
``multiprocessing.Queue``. Every other test file calls
``OCRPipeline.run`` **directly** on the main thread — which skips:

    * Pickle round-trip of ``OCRJobConfig`` / ``ProfileData``
    * Windows spawn-mode worker startup + imports
    * Environment inheritance (``PATH``, ``TESSDATA_PREFIX``, etc.)
    * IPC via ``multiprocessing.Queue``
    * Worker log file creation
    * ``max_tasks_per_child=10`` worker recycling

Those are exactly the layers where the most opaque production bugs
live — a pickle error or a worker that can't find Tesseract surfaces
as ``BrokenProcessPool`` with no traceback, and the current suite
wouldn't catch it. These tests run the **real** multi-process path
against **real** Tesseract so regressions in that layer show up
immediately.

Costs ~30 s per test (spawn + import + real OCR). Skipped cleanly
when tesseract + gs aren't on PATH.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.integration._real_ocr_helpers import (
    render_clean_text_pdf,
    requires_real_ocr,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quick_profile():
    """A ProfileData suitable for quick subprocess E2E.

    We want the ``quick_reliable`` shape (300 DPI + OTSU + CLAHE) but
    constructed in code so the test doesn't depend on ProfileStorage
    seeding — the subprocess worker will build its own storage from
    the bundled profiles anyway.
    """
    from src.core.models import (
        BinarizationConfig,
        ContrastConfig,
        DenoiseConfig,
        DeskewConfig,
        OCRConfig,
        PostprocessConfig,
        PreprocessConfig,
        ProfileData,
    )
    from src.shared.types import (
        OEM,
        PSM,
        BinarizationMethod,
        OCREngineKind,
        OptimizeLevel,
    )

    pre = PreprocessConfig(
        deskew=DeskewConfig(enabled=True, auto_detect=True),
        binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
        denoise=DenoiseConfig(enabled=False),
        contrast=ContrastConfig(clahe_enabled=True, clahe_clip=2.0),
    )
    ocr = OCRConfig(
        engine=OCREngineKind.TESSERACT,
        languages=["eng"],
        primary_language="eng",
        psm=PSM.AUTO,
        oem=OEM.LSTM_ONLY,
        dpi=200,
        optimize_level=OptimizeLevel.NONE,
        skip_text=False,
        tesseract_timeout=120,
    )
    return ProfileData(
        name="pp-e2e",
        ocr=ocr,
        preprocess=pre,
        postprocess=PostprocessConfig(),
    )


class TestParallelProcessorSubprocessE2E:
    """Submit a real OCR job through ProcessPoolExecutor. Spawn mode
    pickles the full config, re-imports every module in the worker,
    and runs the pipeline there. Progress events travel back via
    ``multiprocessing.Queue``."""

    def test_single_job_through_real_worker(self, tmp_path: Path) -> None:
        from src.application.parallel_processor import ParallelProcessor
        from src.core.models import OCRJobConfig
        from src.shared.types import JobStatus

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="SUBPROCESS WORKER"
        )
        output_pdf = tmp_path / "out.pdf"

        pp = ParallelProcessor(max_workers=1)
        try:
            progress_events: list[tuple[str, int, int, str]] = []

            def on_progress(
                job_id: str, current: int, total: int, stage: str
            ) -> None:
                progress_events.append((job_id, current, total, stage))

            future = pp.submit(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_quick_profile(),
                ),
                on_progress=on_progress,
            )
            # Block up to 2 minutes for the real worker to finish.
            result_dict = future.result(timeout=120)
        finally:
            pp.shutdown(wait=True)

        assert result_dict["status"] == JobStatus.COMPLETED.value, (
            f"Job FAILED: {result_dict.get('error')!r}"
        )
        assert output_pdf.exists(), "Subprocess worker didn't write output"
        # Progress events made it through the IPC bridge. At minimum
        # we saw preprocess + ocr stages.
        stages = {s for _, _, _, s in progress_events}
        assert "preprocess" in stages, (
            f"Progress bridge lost preprocess events. Got stages: {stages}"
        )
        assert "ocr" in stages

        # Recognised text landed on disk.
        import fitz
        with fitz.open(str(output_pdf)) as doc:
            text = doc.load_page(0).get_text("text") or ""
        assert any(
            w in text.upper() for w in ("SUBPROCESS", "WORKER")
        ), f"output PDF lacks recognised text: {text!r}"

    def test_two_concurrent_jobs_both_complete(self, tmp_path: Path) -> None:
        """The user's real queue often has multiple files. Two jobs
        submitted in parallel must both land their outputs without
        stomping on each other's progress routing or temp dirs."""
        from src.application.parallel_processor import ParallelProcessor
        from src.core.models import OCRJobConfig
        from src.shared.types import JobStatus

        in1 = render_clean_text_pdf(tmp_path / "a.pdf", text="FIRST")
        in2 = render_clean_text_pdf(tmp_path / "b.pdf", text="SECOND")
        out1 = tmp_path / "a_ocr.pdf"
        out2 = tmp_path / "b_ocr.pdf"

        pp = ParallelProcessor(max_workers=2)
        events_by_job: dict[str, list[str]] = {}

        def on_progress(job_id, current, total, stage):
            events_by_job.setdefault(job_id, []).append(stage)

        try:
            f1 = pp.submit(
                OCRJobConfig(
                    input_path=str(in1),
                    output_path=str(out1),
                    profile=_quick_profile(),
                ),
                on_progress=on_progress,
                job_id="job-1",
            )
            f2 = pp.submit(
                OCRJobConfig(
                    input_path=str(in2),
                    output_path=str(out2),
                    profile=_quick_profile(),
                ),
                on_progress=on_progress,
                job_id="job-2",
            )
            r1 = f1.result(timeout=120)
            r2 = f2.result(timeout=120)
        finally:
            pp.shutdown(wait=True)

        assert r1["status"] == JobStatus.COMPLETED.value, r1.get("error")
        assert r2["status"] == JobStatus.COMPLETED.value, r2.get("error")
        assert out1.exists() and out2.exists()
        # Each job's progress routed to its own listener via tracking_id.
        assert "job-1" in events_by_job, (
            f"job-1 progress lost. Seen: {list(events_by_job.keys())}"
        )
        assert "job-2" in events_by_job

    def test_on_complete_callback_fires_with_real_job_result(
        self, tmp_path: Path
    ) -> None:
        """The GUI relies on ``on_complete(result)`` to update its UI
        after a job lands. That callback runs on the ``_done`` thread
        in the main process — a pickling regression in JobResult would
        break this silently."""
        from src.application.parallel_processor import ParallelProcessor
        from src.core.models import JobResult, OCRJobConfig

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="CALLBACK LAND"
        )
        output_pdf = tmp_path / "out.pdf"

        received: list[JobResult] = []

        def on_complete(r: JobResult) -> None:
            received.append(r)

        pp = ParallelProcessor(max_workers=1)
        try:
            future = pp.submit(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_quick_profile(),
                ),
                on_complete=on_complete,
            )
            future.result(timeout=120)
            # ``on_complete`` runs on the _done callback thread —
            # give it a moment.
            deadline = time.time() + 5
            while not received and time.time() < deadline:
                time.sleep(0.1)
        finally:
            pp.shutdown(wait=True)

        assert received, (
            "on_complete never fired — either the callback thread is "
            "broken or JobResult failed to serialise back from the worker"
        )
        result = received[0]
        assert isinstance(result, JobResult)
        assert result.pages, "JobResult came back but with no pages"


# ---------------------------------------------------------------------------
# Windows-parity: force ``spawn`` start method on Linux
# ---------------------------------------------------------------------------


class TestSpawnModeE2E:
    """Force ``multiprocessing.get_context("spawn")`` on Linux so the
    worker goes through the same code path as Windows production:

      * Full re-import of every module (no fork-inherited state).
      * ``OCRJobConfig`` / ``ProfileData`` must survive a real pickle
        round-trip — any ``__slot__`` / lambda / local-class would
        crash on ``PickleError``.
      * ``os.environ`` is inherited BUT class-level caches
        (``TesseractWrapper._binary_path``) are NOT. The worker must
        rediscover Tesseract from scratch.

    This is exactly the code path that surfaces the "works on Linux
    CI, crashes on Windows" class of bug. If this test is green, the
    same job submission would succeed on Windows — barring
    platform-specific filesystem/encoding differences that are
    tested separately in ``test_e2e_user_workflow.py``.

    ~20 s per test because spawn re-imports everything from scratch.
    """

    def test_single_job_with_spawn_context(self, tmp_path: Path) -> None:
        """Full real-OCR job through a SPAWN-mode worker. The tightest
        possible parity with Windows production."""
        import multiprocessing

        from src.application.parallel_processor import ParallelProcessor
        from src.core.models import OCRJobConfig
        from src.shared.types import JobStatus

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="SPAWN MODE"
        )
        output_pdf = tmp_path / "out.pdf"

        spawn_ctx = multiprocessing.get_context("spawn")
        pp = ParallelProcessor(max_workers=1, mp_context=spawn_ctx)
        try:
            progress_events: list[tuple[str, int, int, str]] = []

            def on_progress(
                job_id: str, current: int, total: int, stage: str
            ) -> None:
                progress_events.append((job_id, current, total, stage))

            future = pp.submit(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_quick_profile(),
                ),
                on_progress=on_progress,
            )
            result_dict = future.result(timeout=120)
        finally:
            pp.shutdown(wait=True)

        assert result_dict["status"] == JobStatus.COMPLETED.value, (
            f"Spawn-mode job FAILED: {result_dict.get('error')!r}"
        )
        assert output_pdf.exists()

        # Progress events crossed the IPC bridge — spawn mode uses
        # a NEW multiprocessing.Queue, not a fork-inherited one.
        stages = {s for _, _, _, s in progress_events}
        assert "ocr" in stages, (
            f"No OCR progress in spawn mode. Stages: {stages}"
        )

        # Real text landed.
        import fitz

        with fitz.open(str(output_pdf)) as doc:
            text = doc.load_page(0).get_text("text") or ""
        assert any(
            w in text.upper() for w in ("SPAWN", "MODE")
        ), f"output PDF lacks text: {text!r}"

    def test_spawn_context_concurrent_jobs(self, tmp_path: Path) -> None:
        """Two concurrent jobs in spawn mode — worker processes share
        nothing. Each must independently discover Tesseract from
        PATH, reconstruct the pipeline, and produce output."""
        import multiprocessing

        from src.application.parallel_processor import ParallelProcessor
        from src.core.models import OCRJobConfig
        from src.shared.types import JobStatus

        in1 = render_clean_text_pdf(tmp_path / "a.pdf", text="ALPHA")
        in2 = render_clean_text_pdf(tmp_path / "b.pdf", text="BRAVO")
        out1 = tmp_path / "a_ocr.pdf"
        out2 = tmp_path / "b_ocr.pdf"

        spawn_ctx = multiprocessing.get_context("spawn")
        pp = ParallelProcessor(max_workers=2, mp_context=spawn_ctx)
        try:
            f1 = pp.submit(
                OCRJobConfig(
                    input_path=str(in1),
                    output_path=str(out1),
                    profile=_quick_profile(),
                ),
            )
            f2 = pp.submit(
                OCRJobConfig(
                    input_path=str(in2),
                    output_path=str(out2),
                    profile=_quick_profile(),
                ),
            )
            r1 = f1.result(timeout=120)
            r2 = f2.result(timeout=120)
        finally:
            pp.shutdown(wait=True)

        assert r1["status"] == JobStatus.COMPLETED.value, r1.get("error")
        assert r2["status"] == JobStatus.COMPLETED.value, r2.get("error")
        assert out1.exists() and out2.exists()
