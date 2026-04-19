"""Tests for the stage-parallelism resolver + worker caps.

Three stages run pages in parallel:
  * preprocess (``OCR_PREPROCESS_WORKERS``, default 10)
  * per-page OCR (``OCR_PER_PAGE_WORKERS``, default 10)
  * postprocess (``OCR_POSTPROCESS_WORKERS``, default 10)

The caps are bumped from 4/8/serial to 10/10/10 in the "batch 10
pages at a time" rollout. These tests pin the defaults and the env
var overrides so a future refactor can't silently regress the
parallelism budget.
"""

from __future__ import annotations

import pytest

from src.application.pipeline import _resolve_stage_parallelism


class TestResolveStageParallelism:
    def test_returns_fallback_when_env_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OCR_PREPROCESS_WORKERS", raising=False)
        assert _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        ) == 10

    def test_env_override_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_PREPROCESS_WORKERS", "4")
        assert _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        ) == 4

    def test_invalid_env_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_PREPROCESS_WORKERS", "not-a-number")
        assert _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        ) == 10

    def test_zero_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_PREPROCESS_WORKERS", "0")
        assert _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        ) == 10

    def test_negative_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OCR_PREPROCESS_WORKERS", "-5")
        assert _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        ) == 10

    def test_large_value_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Large values are accepted — the code that uses the
        # resolver (per-stage worker pools) applies its own caps
        # like the RAM-based memory budget. Don't double-cap here.
        monkeypatch.setenv("OCR_PREPROCESS_WORKERS", "64")
        assert _resolve_stage_parallelism(
            env_var="OCR_PREPROCESS_WORKERS", fallback=10,
        ) == 64


class TestPerPageWorkerCap:
    def test_cap_is_ten(self) -> None:
        """The Tesseract per-page OCR cap matches the preprocess /
        postprocess stages so a batch of 10 pages progresses through
        every stage at the same width."""
        from src.application.engines.tesseract_engine import (
            _MAX_PER_PAGE_WORKERS,
        )

        assert _MAX_PER_PAGE_WORKERS == 10
