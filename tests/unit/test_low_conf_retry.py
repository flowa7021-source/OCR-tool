"""Tests for the low-confidence retry module."""

from __future__ import annotations

import numpy as np

from src.application.low_conf_retry import (
    RetryStats,
    retry_low_confidence,
)


def _make_page(h: int = 100, w: int = 200, fill: int = 255) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _bbox(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class TestRetryThreshold:
    def test_high_conf_not_retried(self):
        page = _make_page()
        raw = [(_bbox(10, 10, 50, 30), "ООО", 0.95)]

        def never_called(arr):
            raise AssertionError("recogniser called on high-conf word")

        out, stats = retry_low_confidence(
            page, raw, never_called, threshold=0.5,
        )
        assert len(out) == 1
        assert out[0] == raw[0]
        assert stats.attempted == 0

    def test_low_conf_retried_and_improved(self):
        page = _make_page()
        raw = [(_bbox(10, 10, 50, 30), "O00", 0.3)]
        calls = []

        def recogniser(arr):
            calls.append(arr.shape)
            # Every variant yields the same high-conf correction
            return [([[0, 0], [10, 0], [10, 5], [0, 5]], "ООО", 0.8)]

        out, stats = retry_low_confidence(
            page, raw, recogniser, threshold=0.5,
        )
        assert len(out) == 1
        assert out[0][1] == "ООО"
        assert out[0][2] == 0.8
        assert stats.attempted == 1
        assert stats.improved == 1
        assert len(calls) >= 1  # at least one variant was tried

    def test_low_conf_retry_no_improvement_keeps_original(self):
        page = _make_page()
        original = (_bbox(10, 10, 50, 30), "original", 0.3)
        raw = [original]

        def recogniser(arr):
            # Every variant returns something WORSE
            return [([[0, 0], [10, 0], [10, 5], [0, 5]], "worse", 0.1)]

        out, stats = retry_low_confidence(
            page, raw, recogniser, threshold=0.5,
        )
        assert out[0] == original
        assert stats.attempted == 1
        assert stats.unchanged == 1

    def test_tiny_crop_skipped(self):
        page = _make_page()
        # Bbox at page corner so padding can't extend it; 3×3 → final
        # crop is only 7×7 after clipping, below the 8-px floor.
        raw = [(_bbox(0, 0, 3, 3), "x", 0.3)]

        def recogniser(arr):
            raise AssertionError("should not be called on tiny crop")

        out, stats = retry_low_confidence(page, raw, recogniser)
        assert out[0] == raw[0]
        assert stats.failed == 1

    def test_min_improvement_threshold(self):
        page = _make_page()
        raw = [(_bbox(10, 10, 50, 30), "text", 0.4)]

        def recogniser(arr):
            # Slight improvement, below default min_improvement=0.05
            return [([[0, 0], [10, 0], [10, 5], [0, 5]], "text", 0.42)]

        out, stats = retry_low_confidence(
            page, raw, recogniser,
            threshold=0.5, min_improvement=0.05,
        )
        # Should keep original since improvement < min_improvement
        assert out[0] == raw[0]
        assert stats.unchanged == 1

    def test_stats_serialisation(self):
        s = RetryStats(attempted=5, improved=2, unchanged=2, failed=1)
        d = s.to_dict()
        assert d == {
            "attempted": 5, "improved": 2, "unchanged": 2, "failed": 1,
        }

    def test_multi_box_variant_concatenates(self):
        """If a variant causes CRAFT to split one crop into two, we
        concatenate left-to-right and pick max confidence.
        """
        page = _make_page()
        raw = [(_bbox(10, 10, 100, 30), "Москва улицаа", 0.3)]

        def recogniser(arr):
            # Variant split into two boxes
            return [
                ([[60, 0], [90, 0], [90, 10], [60, 10]], "улица", 0.85),
                ([[0, 0], [30, 0], [30, 10], [0, 10]], "Москва", 0.80),
            ]

        out, stats = retry_low_confidence(page, raw, recogniser)
        assert stats.improved == 1
        # Left-to-right: Москва (x=0) before улица (x=60)
        assert out[0][1] == "Москва улица"
        assert out[0][2] == 0.85  # max of the two
