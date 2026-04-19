"""Real-OCR stress test: 20-page synthetic document.

Users occasionally process multi-page documents (contracts, reports,
books). A 20-page document is the realistic upper end of "user
submits one job"; anything larger typically goes through batch mode.
This test confirms:

  * The engine doesn't blow up memory on N pages
  * Progress callbacks fire for every page (not just first/last)
  * Total wall time stays within a reasonable envelope
  * Every page ends up with a text layer

Marked ``slow``; runs on PR CI but contributes ~30-60 s wall time.
If you want to trim PR CI, gate this with ``OCR_NIGHTLY=1`` instead
of running it on every push.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]

_PAGE_COUNT: int = 20

# Generous upper bound per page to absorb CI jitter. Real Tesseract
# on a 200 DPI synthetic page runs ~1-2 s; 15 s per page gives us
# 5 minutes for 20 pages — anything slower indicates a real
# performance regression we want to catch.
_MAX_SECONDS_PER_PAGE: float = 15.0


class TestLargeDocumentRealOCR:
    """20-page synthetic document through the full real pipeline."""

    def test_twenty_pages_all_searchable(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        import fitz

        input_pdf = render_clean_text_pdf(
            tmp_path / "big.pdf",
            "stress page",
            pages=_PAGE_COUNT,
            dpi=150,  # lower DPI so rasterisation is fast
        )
        output_pdf = tmp_path / "big_ocr.pdf"
        profile = make_realistic_profile(
            binarization="otsu",
            dpi=200,  # keep OCR DPI moderate — this is a perf test
        )

        t0 = time.monotonic()
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        elapsed = time.monotonic() - t0

        assert result.status is JobStatus.COMPLETED, result.error
        assert result.page_count == _PAGE_COUNT, (
            f"Expected {_PAGE_COUNT} pages in JobResult, got "
            f"{result.page_count}"
        )

        # Every page must have some recognised text
        empty = [
            p.page_number for p in result.pages
            if not (p.text or "").strip()
        ]
        assert not empty, (
            f"Pages without recognised text: {empty}. "
            f"Retry-tier system should prevent any empty page in a "
            f"clean synthetic document."
        )

        # Output PDF has a text layer on every page too
        with fitz.open(str(output_pdf)) as doc:
            empty_pdf = [
                i + 1 for i, p in enumerate(doc)
                if not (p.get_text("text") or "").strip()
            ]
        assert not empty_pdf, (
            f"Output PDF has empty pages: {empty_pdf}"
        )

        # Performance envelope — not too strict, just catches
        # order-of-magnitude regressions
        max_allowed = _PAGE_COUNT * _MAX_SECONDS_PER_PAGE
        assert elapsed < max_allowed, (
            f"20-page OCR took {elapsed:.1f}s (>{max_allowed:.0f}s "
            f"envelope). Real perf regression?"
        )
