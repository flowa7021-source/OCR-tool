"""Unit tests for :mod:`src.core.orientation_detector`."""

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
            "Script: Cyrillic\n"
            "Script confidence: 2.10\n"
        )
        assert _parse_osd_output(out) == (90, 3.45)

    def test_zero_rotation_is_valid(self) -> None:
        out = (
            "Rotate: 0\n"
            "Orientation confidence: 5.20\n"
        )
        assert _parse_osd_output(out) == (0, 5.20)

    def test_missing_rotate_returns_none(self) -> None:
        out = "Orientation confidence: 3.0\n"
        assert _parse_osd_output(out) is None

    def test_missing_confidence_returns_none(self) -> None:
        out = "Rotate: 90\n"
        assert _parse_osd_output(out) is None

    def test_malformed_numbers_return_none(self) -> None:
        out = "Rotate: abc\nOrientation confidence: 3.0\n"
        assert _parse_osd_output(out) is None

    def test_negative_rotation_normalised(self) -> None:
        # Tesseract has been observed emitting e.g. ``-90`` on some
        # inputs; we normalise modulo 360 so the caller always sees
        # a value in {0, 90, 180, 270}.
        out = "Rotate: -90\nOrientation confidence: 2.0\n"
        result = _parse_osd_output(out)
        assert result is not None
        assert result[0] == 270

    def test_off_grid_rotation_snapped(self) -> None:
        # OSD only ever means 90° increments; a measurement of 88°
        # is noise and snaps to 90°.
        out = "Rotate: 88\nOrientation confidence: 2.0\n"
        result = _parse_osd_output(out)
        assert result is not None
        assert result[0] == 90


class TestDetectOrientation:
    def test_returns_none_on_tiny_image(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert detect_orientation(img) is None

    def test_returns_none_on_invalid_input(self) -> None:
        assert detect_orientation("not an array") is None
        assert detect_orientation(None) is None

    def test_returns_rotation_on_confident_detection(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        fake_osd = (
            "Rotate: 90\nOrientation confidence: 3.0\nScript: Latin\n"
        )
        with patch(
            "src.core.orientation_detector.pytesseract"
        ) as mock_py:
            mock_py.image_to_osd.return_value = fake_osd
            assert detect_orientation(img) == 90

    def test_returns_zero_when_already_upright(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        fake_osd = "Rotate: 0\nOrientation confidence: 5.0\n"
        with patch(
            "src.core.orientation_detector.pytesseract"
        ) as mock_py:
            mock_py.image_to_osd.return_value = fake_osd
            assert detect_orientation(img) == 0

    def test_returns_none_below_min_confidence(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        fake_osd = "Rotate: 180\nOrientation confidence: 0.4\n"
        with patch(
            "src.core.orientation_detector.pytesseract"
        ) as mock_py:
            mock_py.image_to_osd.return_value = fake_osd
            # Default floor (1.0) exceeds the reported 0.4 → None.
            assert detect_orientation(img) is None

    def test_respects_caller_threshold(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        fake_osd = "Rotate: 180\nOrientation confidence: 1.5\n"
        with patch(
            "src.core.orientation_detector.pytesseract"
        ) as mock_py:
            mock_py.image_to_osd.return_value = fake_osd
            # Caller lowers the threshold → accepts.
            assert detect_orientation(img, min_confidence=1.0) == 180
            # Caller raises it → rejects.
            assert detect_orientation(img, min_confidence=2.0) is None

    def test_osd_error_returns_none(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        with patch(
            "src.core.orientation_detector.pytesseract"
        ) as mock_py:
            mock_py.image_to_osd.side_effect = RuntimeError("too few glyphs")
            assert detect_orientation(img) is None

    def test_min_confidence_default_matches_constant(self) -> None:
        # Guard against accidental drift between the public constant
        # (used in AutoRotateConfig defaults) and the internal value.
        assert pytest.approx(1.0) == MIN_ORIENTATION_CONFIDENCE


class TestPreprocessorAutoRotateIntegration:
    """End-to-end check that ImagePreprocessor applies the rotation."""

    def test_apply_auto_rotate_90cw(self) -> None:
        import cv2

        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import AutoRotateConfig

        # 200x300 image so we can tell portrait from landscape after
        # the rotation fires.
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        cfg = AutoRotateConfig(enabled=True, min_confidence=1.0)

        preprocessor = ImagePreprocessor()
        with patch(
            "src.core.image_preprocessor.detect_orientation",
            return_value=90,
        ):
            rotated = preprocessor._apply_auto_rotate(img, cfg)

        # 90° clockwise turns a (H=200, W=300) into (H=300, W=200)
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

        # 0° means "already upright" — image returned unchanged.
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

        # None means OSD wasn't confident enough — leave alone.
        assert np.array_equal(result, img)
