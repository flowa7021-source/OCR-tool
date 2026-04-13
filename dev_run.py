"""Quick launcher for UI development without real OCR dependencies.

This script stubs out the Tesseract / OCRmyPDF pipeline so the full UI can be
exercised on any machine (including Linux CI) without installing the OCR
toolchain. Useful for reviewing layout, theming, and panel behavior.

Usage:
    python dev_run.py                 # launch with an empty workspace
    python dev_run.py path/to.pdf     # open the given PDF on startup

Notes:
    * Submitted jobs instantly "complete" with placeholder PageResult text.
    * No real Tesseract binary is invoked.
    * Recovery snapshots are still written to RECOVERY_DIR.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _install_stubs() -> None:
    """Replace ParallelProcessor and TesseractWrapper with no-op dev stubs."""
    from src.application import parallel_processor as pp_module
    from src.core.models import JobResult, PageResult
    from src.infrastructure import tesseract_wrapper as tw_module
    from src.shared.types import JobStatus

    class _DevParallelProcessor:
        """Fake processor: reports progress + completion on the main thread."""

        def __init__(self, max_workers: int = 1) -> None:
            self.max_workers = max_workers

        def submit(
            self,
            job: Any,
            on_progress: Any = None,
            on_complete: Any = None,
            on_error: Any = None,
            job_id: str = "dev",
        ) -> Any:
            # Emit a couple of fake progress ticks
            total = 3
            for current in range(1, total + 1):
                if on_progress is not None:
                    on_progress(job_id, current, total, f"stage-{current}")
                time.sleep(0.02)
            result = JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                input_path=str(job.input_path),
                output_path=str(job.output_path),
                pages=[
                    PageResult(
                        page_number=i + 1,
                        text=f"[DEV STUB] Страница {i + 1} — OCR не выполнялся.",
                        mean_confidence=50.0 + i * 10,
                    )
                    for i in range(total)
                ],
                total_time_sec=0.1,
            )
            if on_complete is not None:
                on_complete(result)

            class _Future:
                def done(self) -> bool:
                    return True

            return _Future()

        def submit_all(self, jobs: list[Any]) -> list[Any]:
            return [self.submit(j) for j in jobs]

        def shutdown(self, wait: bool = True) -> None:  # noqa: ARG002
            pass

    class _DevTesseract:
        def verify(self) -> tuple[bool, str]:
            return True, "DEV STUB — Tesseract not actually checked"

        def configure_pytesseract(self) -> None:
            pass

        def find_tesseract_binary(self) -> Path:
            return Path("/usr/bin/true")

        def find_tessdata_dir(self) -> Path:
            return Path("/tmp")

        def available_languages(self) -> list[str]:
            return ["rus", "eng"]

    pp_module.ParallelProcessor = _DevParallelProcessor  # type: ignore[assignment]
    tw_module.TesseractWrapper = _DevTesseract  # type: ignore[assignment]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _install_stubs()

    from src.app import create_application
    from PySide6.QtCore import QTimer

    app, window = create_application(sys.argv)
    window.show()

    # If a PDF path was passed on the CLI, open it automatically
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        pdf = Path(args[0]).expanduser().resolve()
        if pdf.exists():
            QTimer.singleShot(100, lambda: window._open_pdf(pdf))  # type: ignore[attr-defined]
        else:
            logger.warning("PDF not found: %s", pdf)

    QTimer.singleShot(50, window.prompt_recovery)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
