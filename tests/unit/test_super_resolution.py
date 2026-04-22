"""Tests for super-resolution upscaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.super_resolution import (
    LOW_DPI_THRESHOLD,
    TARGET_DPI,
    upscale_for_ocr,
)


def _synth_page(h: int = 100, w: int = 200) -> np.ndarray:
    img = np.full((h, w), 240, dtype=np.uint8)
    # Fake "text" — some random dark pixels.
    rng = np.random.default_rng(seed=42)
    ys = rng.integers(10, h - 10, size=500)
    xs = rng.integers(10, w - 10, size=500)
    img[ys, xs] = 20
    return img


class TestSkipLogic:
    def test_skipped_when_source_dpi_unknown(self):
        img = _synth_page()
        out, stats = upscale_for_ocr(img, source_dpi=None)
        assert stats.applied is False
        np.testing.assert_array_equal(out, img)

    def test_skipped_when_source_already_high(self):
        img = _synth_page()
        out, stats = upscale_for_ocr(img, source_dpi=300)
        assert stats.applied is False
        assert stats.input_dpi == 300
        np.testing.assert_array_equal(out, img)

    def test_skipped_at_threshold(self):
        img = _synth_page()
        out, stats = upscale_for_ocr(img, source_dpi=LOW_DPI_THRESHOLD)
        assert stats.applied is False

    def test_force_bypasses_skip(self):
        img = _synth_page()
        out, stats = upscale_for_ocr(img, source_dpi=300, force=True)
        assert stats.applied is True


class TestBicubicSharpen:
    def test_applied_for_low_dpi(self):
        img = _synth_page(h=100, w=200)
        out, stats = upscale_for_ocr(img, source_dpi=100)
        assert stats.applied is True
        assert stats.mode == "bicubic_sharpen"
        # Scale = 300 / 100 = 3 → output at least 2× bigger each dim.
        assert out.shape[0] > img.shape[0] * 2
        assert out.shape[1] > img.shape[1] * 2
        assert out.dtype == np.uint8

    def test_elapsed_ms_reported(self):
        img = _synth_page(h=50, w=100)
        _, stats = upscale_for_ocr(img, source_dpi=120)
        assert stats.elapsed_ms > 0
        assert stats.elapsed_ms < 5000  # should be very fast

    def test_effective_dpi_reported(self):
        img = _synth_page()
        _, stats = upscale_for_ocr(img, source_dpi=150, target_dpi=300)
        assert stats.output_dpi >= 250  # approximately 2× input


class TestDnnFallback:
    def test_missing_model_falls_back(self, tmp_path: Path):
        img = _synth_page()
        # Empty model dir — DNN mode should fall back to bicubic.
        out, stats = upscale_for_ocr(
            img, source_dpi=100, mode="dnn_edsr", model_dir=tmp_path,
        )
        assert stats.applied is True
        assert "fallback" in stats.mode
        assert out.shape[0] > img.shape[0]


class TestInvalidMode:
    def test_unknown_mode_raises(self):
        img = _synth_page()
        with pytest.raises(ValueError, match="Unknown"):
            upscale_for_ocr(img, source_dpi=100, mode="unknown")


class TestPreservesPixelRange:
    def test_output_in_uint8_range(self):
        img = _synth_page()
        out, _ = upscale_for_ocr(img, source_dpi=100)
        assert out.min() >= 0
        assert out.max() <= 255
        assert out.dtype == np.uint8


class TestTargetDpi:
    def test_default_target_is_sensible(self):
        assert TARGET_DPI == 300
        assert LOW_DPI_THRESHOLD < TARGET_DPI
