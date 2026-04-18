"""Tests for :mod:`src.core.script_detector` — TDD: tests written BEFORE
implementation to pin down expected behaviour.

Mocks ``pytesseract.image_to_osd`` so the suite runs without a
real Tesseract binary.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def script_detector_module():
    from src.core import script_detector

    return script_detector


def _fake_osd(script: str, script_conf: float) -> str:
    return (
        "Page number: 0\n"
        "Orientation in degrees: 0\n"
        "Rotate: 0\n"
        "Orientation confidence: 10.00\n"
        f"Script: {script}\n"
        f"Script confidence: {script_conf:.2f}"
    )


def _blank_image(h: int = 500, w: int = 500):
    return np.full((h, w), 255, dtype=np.uint8)


class TestPublicAPI:
    def test_exports_detector_class(self, script_detector_module) -> None:
        assert hasattr(script_detector_module, "ScriptDetector")

    def test_exports_detect_function(self, script_detector_module) -> None:
        assert callable(script_detector_module.detect_dominant_script)

    def test_exports_constants(self, script_detector_module) -> None:
        assert hasattr(script_detector_module, "MIN_SCRIPT_CONFIDENCE")


class TestCyrillicDetection:
    def test_high_confidence_cyrillic_returns_rus(
        self, script_detector_module,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd("Cyrillic", 5.0),
        ):
            assert script_detector_module.detect_dominant_script(
                _blank_image()
            ) == "rus"


class TestLatinDetection:
    def test_high_confidence_latin_returns_eng(
        self, script_detector_module,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd("Latin", 5.0),
        ):
            assert script_detector_module.detect_dominant_script(
                _blank_image()
            ) == "eng"


class TestLowConfidenceFallback:
    def test_low_confidence_returns_none(
        self, script_detector_module,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd("Cyrillic", 0.3),
        ):
            assert (
                script_detector_module.detect_dominant_script(_blank_image())
                is None
            )

    def test_confidence_exactly_at_threshold_accepts(
        self, script_detector_module,
    ) -> None:
        threshold = script_detector_module.MIN_SCRIPT_CONFIDENCE
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd("Latin", threshold),
        ):
            assert script_detector_module.detect_dominant_script(
                _blank_image()
            ) == "eng"


class TestUnrecognisedScript:
    @pytest.mark.parametrize("script", ["Han", "Arabic", "Devanagari", ""])
    def test_unknown_script_returns_none(
        self, script_detector_module, script: str,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd(script, 5.0),
        ):
            assert (
                script_detector_module.detect_dominant_script(_blank_image())
                is None
            )


class TestTesseractErrorHandling:
    def test_tesseract_error_returns_none(
        self, script_detector_module,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            side_effect=RuntimeError("osd failed"),
        ):
            assert (
                script_detector_module.detect_dominant_script(_blank_image())
                is None
            )

    def test_malformed_osd_text_returns_none(
        self, script_detector_module,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value="Page number: 0\nTruncated output",
        ):
            assert (
                script_detector_module.detect_dominant_script(_blank_image())
                is None
            )


class TestEmptyImage:
    def test_too_small_image_returns_none(
        self, script_detector_module,
    ) -> None:
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
        ) as mock_osd:
            result = script_detector_module.detect_dominant_script(
                _blank_image(h=10, w=10),
            )
        assert result is None
        mock_osd.assert_not_called()


class TestScriptDetectorClass:
    def test_default_detector_uses_module_threshold(
        self, script_detector_module,
    ) -> None:
        detector = script_detector_module.ScriptDetector()
        assert detector.min_confidence == (
            script_detector_module.MIN_SCRIPT_CONFIDENCE
        )

    def test_custom_threshold_respected(
        self, script_detector_module,
    ) -> None:
        detector = script_detector_module.ScriptDetector(min_confidence=2.5)
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd("Cyrillic", 2.0),
        ):
            assert detector.detect(_blank_image()) is None
        with patch(
            "src.core.script_detector.pytesseract.image_to_osd",
            return_value=_fake_osd("Cyrillic", 3.0),
        ):
            assert detector.detect(_blank_image()) == "rus"
