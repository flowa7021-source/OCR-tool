"""Tests for ``_resolve_per_page_workers`` — Initiative 3.

The per-page parallelism knob balances three pressures:
  1. Page count — can't run more workers than pages.
  2. CPU cores — more workers than cores = thrashing.
  3. Available memory — each worker needs ~500 MB headroom for
     the binarised page, Tesseract's LSTM, and OCRmyPDF's
     in-flight buffers. Oversubscribing RAM makes the OS swap
     and tanks throughput worse than fewer workers would.

The function must also respect the ``OCR_PER_PAGE_WORKERS`` env
override (for debugging / many-core hosts) and NEVER return < 1.

Tests use dependency injection — ``cpu_count`` and
``available_memory_mb`` are overridable so tests don't depend on
whatever runner happens to be executing them.
"""

from __future__ import annotations

import os
import unittest.mock

import pytest


@pytest.fixture
def resolver():
    """Import lazily — module may not yet expose the new signature."""
    from src.application.engines.tesseract_engine import (
        _resolve_per_page_workers,
    )

    return _resolve_per_page_workers


class TestCpuCountCap:
    """Worker count is bounded above by ``min(page_count, cpu_count,
    _MAX_PER_PAGE_WORKERS)``."""

    def test_ten_pages_eight_cpus_ample_ram_returns_eight(
        self, resolver,
    ) -> None:
        # _MAX_PER_PAGE_WORKERS is 8 after Initiative 3.
        assert resolver(
            page_count=10, cpu_count=8, available_memory_mb=32_000,
        ) == 8

    def test_three_pages_eight_cpus_returns_three(self, resolver) -> None:
        # Page count is the binding constraint.
        assert resolver(
            page_count=3, cpu_count=8, available_memory_mb=32_000,
        ) == 3

    def test_ten_pages_four_cpus_returns_four(self, resolver) -> None:
        # CPU count is the binding constraint.
        assert resolver(
            page_count=10, cpu_count=4, available_memory_mb=32_000,
        ) == 4

    def test_max_cap_is_eight_even_on_many_core_cpu(
        self, resolver,
    ) -> None:
        # 32-core workstation — we still cap at 8 to avoid
        # OCRmyPDF's internal thread pool colliding with ours.
        assert resolver(
            page_count=50, cpu_count=32, available_memory_mb=64_000,
        ) == 8


class TestMemoryPressure:
    """Memory budget caps the worker count below what CPU alone
    would allow."""

    def test_tight_ram_throttles_workers(self, resolver) -> None:
        # 1 GB available, 500 MB per worker → 2 workers.
        assert resolver(
            page_count=10, cpu_count=8, available_memory_mb=1024,
        ) == 2

    def test_medium_ram_throttles_to_six(self, resolver) -> None:
        # 3 GB available, 500 MB per worker → 6 workers.
        assert resolver(
            page_count=10, cpu_count=8, available_memory_mb=3072,
        ) == 6

    def test_extremely_low_ram_falls_back_to_one(
        self, resolver,
    ) -> None:
        # < 500 MB free — we still return 1 (must never return 0
        # or the pool refuses jobs).
        assert resolver(
            page_count=10, cpu_count=8, available_memory_mb=256,
        ) == 1

    def test_unknown_memory_uses_cpu_cap(self, resolver) -> None:
        # available_memory_mb=None (caller couldn't query psutil) —
        # fall back to the pre-Initiative-3 behaviour: cpu-bound
        # only.
        assert resolver(
            page_count=10, cpu_count=8, available_memory_mb=None,
        ) == 8


class TestEnvOverride:
    """``OCR_PER_PAGE_WORKERS`` env var forces a specific count
    (still clamped against page_count)."""

    def test_env_override_respected(self, resolver) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"OCR_PER_PAGE_WORKERS": "12"},
        ):
            # Override takes precedence over cpu+memory caps, but
            # page count still wins (can't have 12 workers on 10
            # pages).
            assert resolver(
                page_count=10, cpu_count=8, available_memory_mb=32_000,
            ) == 10

    def test_env_override_below_default_respected(
        self, resolver,
    ) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"OCR_PER_PAGE_WORKERS": "2"},
        ):
            assert resolver(
                page_count=10, cpu_count=8, available_memory_mb=32_000,
            ) == 2

    def test_invalid_env_ignored(self, resolver) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"OCR_PER_PAGE_WORKERS": "not-a-number"},
        ):
            # Garbage input → fall back to auto-detection.
            assert resolver(
                page_count=10, cpu_count=8, available_memory_mb=32_000,
            ) == 8

    def test_zero_or_negative_env_ignored(self, resolver) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"OCR_PER_PAGE_WORKERS": "0"},
        ):
            assert resolver(
                page_count=10, cpu_count=8, available_memory_mb=32_000,
            ) == 8


class TestFloor:
    """The function must never return 0 even if all inputs suggest it."""

    def test_returns_at_least_one(self, resolver) -> None:
        assert resolver(
            page_count=1, cpu_count=1, available_memory_mb=100,
        ) == 1

    def test_zero_pages_returns_one(self, resolver) -> None:
        # Defensive: caller should never pass 0 but we return
        # something sane anyway.
        assert resolver(
            page_count=0, cpu_count=8, available_memory_mb=32_000,
        ) == 1
