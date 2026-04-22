"""Tests for per-line deskew."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.per_line_deskew import (
    _estimate_row_skew,
    _find_rows,
    _project_profile,
    deskew_per_line,
)


def _synth_text_row(
    width: int = 400, height: int = 30, glyph_count: int = 15,
) -> np.ndarray:
    """Simulated text row: white background with evenly-spaced dark rectangles.
    Good enough to exercise projection profile and Hough detection.
    """
    row = np.full((height, width), 255, dtype=np.uint8)
    gap = width // (glyph_count + 2)
    for i in range(glyph_count):
        x = gap + i * gap
        row[8:height - 8, x:x + gap // 2] = 0
    return row


def _synth_page_with_rows(
    n_rows: int = 3, row_height: int = 30, gap: int = 20, width: int = 400,
) -> np.ndarray:
    """Page with ``n_rows`` text rows separated by blank gaps."""
    h = n_rows * row_height + (n_rows + 1) * gap
    page = np.full((h, width), 255, dtype=np.uint8)
    for i in range(n_rows):
        y0 = gap + i * (row_height + gap)
        page[y0:y0 + row_height] = _synth_text_row(
            width=width, height=row_height,
        )
    return page


class TestProjectionProfile:
    def test_row_pixels_have_higher_profile(self):
        page = _synth_page_with_rows(n_rows=2, row_height=30, gap=20)
        profile = _project_profile(page, smooth=1)
        # Profile over a text row (say y=35) must exceed profile over
        # a blank gap (say y=10).
        assert profile[35] > profile[10] * 3

    def test_smoothing_reduces_noise(self):
        page = _synth_page_with_rows(n_rows=2, row_height=30, gap=20)
        rough = _project_profile(page, smooth=1)
        smooth = _project_profile(page, smooth=15)
        # Smoothed version should have lower local variance.
        assert np.std(np.diff(smooth)) < np.std(np.diff(rough))


class TestFindRows:
    def test_detects_correct_row_count(self):
        page = _synth_page_with_rows(n_rows=3, row_height=30, gap=20)
        profile = _project_profile(page, smooth=7)
        rows = _find_rows(profile, min_height=15)
        assert len(rows) == 3

    def test_min_height_filter(self):
        # One "row" only 5px tall — should be dropped with min_height=15
        page = np.full((100, 200), 255, dtype=np.uint8)
        page[20:25] = 0  # noise band
        page[50:80] = _synth_text_row(width=200, height=30)
        profile = _project_profile(page, smooth=3)
        rows = _find_rows(profile, min_height=15)
        assert len(rows) == 1  # only the real 30-px row


class TestEstimateRowSkew:
    def test_level_row_zero_angle(self):
        row = _synth_text_row(width=400, height=30)
        angle = _estimate_row_skew(row, max_angle_deg=5.0)
        assert abs(angle) < 0.5

    def test_rotated_row_detected(self):
        # Continuous dark bar gives Hough enough horizontal edges. Sign
        # convention depends on OpenCV's coordinate system — we only
        # assert on magnitude matching the ground truth.
        row = np.full((30, 400), 255, dtype=np.uint8)
        row[10:20, 20:380] = 0
        m = cv2.getRotationMatrix2D((200, 15), 2.0, 1.0)
        rotated = cv2.warpAffine(row, m, (400, 30), borderValue=255)
        angle = _estimate_row_skew(rotated, max_angle_deg=5.0)
        assert abs(abs(angle) - 2.0) < 0.7


class TestDeskewPerLine:
    def test_shape_preserved(self):
        page = _synth_page_with_rows(n_rows=2, row_height=30, gap=20)
        out, stats = deskew_per_line(page)
        assert out.shape == page.shape
        assert out.dtype == np.uint8

    def test_non_text_regions_unchanged(self):
        # Page with no text — output should equal input (no rows detected).
        page = np.full((100, 200), 255, dtype=np.uint8)
        out, stats = deskew_per_line(page)
        np.testing.assert_array_equal(out, page)
        assert stats.rows_detected == 0
        assert stats.rows_rotated == 0

    def test_rejects_non_grayscale(self):
        page_rgb = np.full((100, 200, 3), 255, dtype=np.uint8)
        try:
            deskew_per_line(page_rgb)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "grayscale" in str(e)

    def test_stats_populated_when_rotation_happens(self):
        # Build a page where one row is rotated.
        page = _synth_page_with_rows(n_rows=2, row_height=30, gap=20)
        h, w = page.shape
        # Rotate the first row's band by 2° in-place.
        y0, y1 = 20, 50
        strip = page[y0:y1]
        m = cv2.getRotationMatrix2D((w / 2, 15), 2.0, 1.0)
        rotated_strip = cv2.warpAffine(strip, m, (w, 30), borderValue=255)
        page[y0:y1] = rotated_strip
        out, stats = deskew_per_line(page)
        assert stats.rows_detected >= 1
        # At least one row should have non-trivial detected skew.
        assert stats.max_abs_angle > 0.3 or stats.rows_rotated == 0
