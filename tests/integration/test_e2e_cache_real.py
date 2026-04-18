"""Real-OCR: persistent cache HIT on second run of the same (input, profile).

Users re-run the same document many times while tweaking profiles.
The cache is the UX difference between "OCR takes 30 s every time"
and "OCR takes 30 s once, 50 ms thereafter". Mocked unit tests in
``tests/unit/test_caches.py`` verify the cache data structure; these
tests verify the cache ACTUALLY short-circuits the pipeline on a
second run with the same input and the same profile.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.infrastructure import ocr_cache
from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


class TestCacheHitOnSecondRun:
    """Second run with same input + same profile must be dramatically faster.

    The pipeline's own cache layer calls ``ocr_cache.lookup`` before
    launching Tesseract. If it hits, the output PDF is copied from
    cache and JobResult is rehydrated from ``result.json`` — zero
    OCR work done.
    """

    def test_second_run_is_cache_hit(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Point cache at a clean per-test dir. The autouse
        # _isolated_ocr_cache fixture in _real_ocr_helpers already does
        # this, but we want to be explicit about what we're measuring.
        cache_dir = tmp_path / "ocr-cache-under-test"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "src.infrastructure.ocr_cache.OCR_CACHE_DIR", cache_dir
        )

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", "cache test content", pages=1,
        )
        profile = make_realistic_profile(binarization="otsu", dpi=200)

        # First run — uncached, real OCR, slow
        out1 = tmp_path / "out1.pdf"
        t0 = time.monotonic()
        r1 = run_pipeline(
            input_pdf, out1, profile, real_tesseract_wrapper
        )
        elapsed_first = time.monotonic() - t0
        assert r1.status is JobStatus.COMPLETED, r1.error

        # Cache must now hold the entry
        hit = ocr_cache.lookup(input_pdf, profile, cache_root=cache_dir)
        assert hit is not None, (
            "Pipeline completed but cache.lookup() returned None — "
            "pipeline isn't writing to the cache, or the key differs "
            "between write and read"
        )

        # Second run — should be a cache HIT, MUCH faster
        out2 = tmp_path / "out2.pdf"
        t0 = time.monotonic()
        r2 = run_pipeline(
            input_pdf, out2, profile, real_tesseract_wrapper
        )
        elapsed_second = time.monotonic() - t0
        assert r2.status is JobStatus.COMPLETED, r2.error
        assert out2.exists()

        # Cache hit should be at least 5× faster than the real OCR run.
        # Real-OCR baseline is ~3-10 s; cache hit is <300 ms. Use a
        # conservative ratio so CI jitter doesn't flake the test.
        assert elapsed_second < elapsed_first / 5, (
            f"Second run not significantly faster — first={elapsed_first:.2f}s, "
            f"second={elapsed_second:.2f}s. Cache HIT isn't short-circuiting "
            f"the pipeline."
        )


class TestCacheMissOnProfileChange:
    """Same input + different profile = cache MISS.

    The cache key includes the profile hash, so tweaking the DPI or
    binarisation method invalidates the entry. Without this, users
    would see stale OCR results when they experiment with profiles.
    """

    def test_profile_change_triggers_cache_miss(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "src.infrastructure.ocr_cache.OCR_CACHE_DIR", cache_dir
        )

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", "cache miss test", pages=1,
        )
        p_otsu = make_realistic_profile(
            name="p_otsu", binarization="otsu", dpi=200,
        )
        p_adaptive = make_realistic_profile(
            name="p_adaptive", binarization="adaptive_gaussian", dpi=200,
        )

        # Prime the cache with p_otsu
        r1 = run_pipeline(
            input_pdf, tmp_path / "out_otsu.pdf",
            p_otsu, real_tesseract_wrapper,
        )
        assert r1.status is JobStatus.COMPLETED

        # p_adaptive must NOT hit the p_otsu entry
        miss = ocr_cache.lookup(
            input_pdf, p_adaptive, cache_root=cache_dir
        )
        assert miss is None, (
            "Different profile produced the same cache key — "
            "profile hash isn't part of the key"
        )

        # And a fresh p_otsu lookup DOES hit
        hit = ocr_cache.lookup(
            input_pdf, p_otsu, cache_root=cache_dir
        )
        assert hit is not None, (
            "Same profile produced different cache key across runs"
        )
