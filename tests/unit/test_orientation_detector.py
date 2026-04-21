"""Unit tests for :mod:`src.core.orientation_detector`.

The detector is a no-op under the EasyOCR engine: it returns 0 (the
"already upright" signal) for any usable image and ``None`` for
inputs that aren't analysable arrays. Tests pin both branches plus
the legacy ``_parse_osd_output`` helper that some downstream
debug-tooling still calls with hand-built OSD strings.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.core.orientation_detector import (
    MIN_ORIENTATION_CONFIDENCE,
    _parse_osd_output,
    detect_orientation,
)


class TestParseOsdOutput:
    def test_rotate_and_confidence_extracted(self) -> None:
        out = (
            "Page number: 0\n"
            "Orientation in degrees: 270\n"
            "Rotate: 90\n"
            "Orientation confidence: 3.45\n"
        )
        assert _parse_osd_output(out) == (90, 3.45)

    def test_zero_rotation_is_valid(self) -> None:
        out = "Rotate: 0\nOrientation confidence: 5.20\n"
        assert _parse_osd_output(out) == (0, 5.20)

    def test_missing_rotate_returns_none(self) -> None:
        assert _parse_osd_output("Orientation confidence: 3.0\n") is None

    def test_missing_confidence_returns_none(self) -> None:
        assert _parse_osd_output("Rotate: 90\n") is None

    def test_malformed_numbers_return_none(self) -> None:
        assert _parse_osd_output(
            "Rotate: abc\nOrientation confidence: 3.0\n"
        ) is None

    def test_negative_rotation_normalised(self) -> None:
        result = _parse_osd_output(
            "Rotate: -90\nOrientation confidence: 2.0\n"
        )
        assert result is not None
        assert result[0] == 270

    def test_off_grid_rotation_snapped(self) -> None:
        result = _parse_osd_output(
            "Rotate: 88\nOrientation confidence: 2.0\n"
        )
        assert result is not None
        assert result[0] == 90


class TestDetectOrientation:
    def test_returns_none_on_tiny_image(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert detect_orientation(img) is None

    def test_returns_none_on_invalid_input(self) -> None:
        assert detect_orientation("not an array") is None
        assert detect_orientation(None) is None

    def test_returns_zero_on_usable_image(self) -> None:
        # No-op under the EasyOCR engine — returns 0 ("upright").
        img = np.zeros((200, 200), dtype=np.uint8)
        assert detect_orientation(img) == 0

    def test_min_confidence_default_matches_constant(self) -> None:
        assert pytest.approx(1.0) == MIN_ORIENTATION_CONFIDENCE


class TestPreprocessorAutoRotateIntegration:
    """End-to-end check that ImagePreprocessor handles the rotation hint."""

    def test_apply_auto_rotate_90cw(self) -> None:
        import cv2

        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import AutoRotateConfig

        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        cfg = AutoRotateConfig(enabled=True, min_confidence=1.0)

        preprocessor = ImagePreprocessor()
        with patch(
            "src.core.image_preprocessor.detect_orientation",
            return_value=90,
        ):
            rotated = preprocessor._apply_auto_rotate(img, cfg)

        expected = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        assert rotated.shape == expected.shape
        assert np.array_equal(rotated, expected)

    def test_apply_auto_rotate_no_rotation_needed(self) -> None:
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import AutoRotateConfig

        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        cfg = AutoRotateConfig(enabled=True, min_confidence=1.0)

        preprocessor = ImagePreprocessor()
        with patch(
            "src.core.image_preprocessor.detect_orientation",
            return_value=0,
        ):
            result = preprocessor._apply_auto_rotate(img, cfg)

        assert np.array_equal(result, img)

    def test_apply_auto_rotate_low_confidence(self) -> None:
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import AutoRotateConfig

        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        cfg = AutoRotateConfig(enabled=True, min_confidence=1.0)

        preprocessor = ImagePreprocessor()
        with patch(
            "src.core.image_preprocessor.detect_orientation",
            return_value=None,
        ):
            result = preprocessor._apply_auto_rotate(img, cfg)

        assert np.array_equal(result, img)
