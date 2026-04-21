"""Tests for :mod:`src.core.script_detector`.

The detector is a no-op under the EasyOCR engine — it returns
``None`` ("no preference") for every usable image and for
non-analysable inputs alike. The tests pin that contract plus the
public API surface that downstream code still imports.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def script_detector_module():
    from src.core import script_detector

    return script_detector


def _blank_image(h: int = 500, w: int = 500):
    return np.full((h, w), 255, dtype=np.uint8)


class TestPublicAPI:
    def test_exports_detector_class(self, script_detector_module) -> None:
        assert hasattr(script_detector_module, "ScriptDetector")

    def test_exports_detect_function(self, script_detector_module) -> None:
        assert callable(script_detector_module.detect_dominant_script)

    def test_exports_constants(self, script_detector_module) -> None:
        assert hasattr(script_detector_module, "MIN_SCRIPT_CONFIDENCE")


class TestDetectDominantScript:
    def test_returns_none_for_usable_image(self, script_detector_module) -> None:
        # No-op under EasyOCR — never narrows the language hint.
        assert script_detector_module.detect_dominant_script(
            _blank_image()
        ) is None

    def test_returns_none_for_tiny_image(self, script_detector_module) -> None:
        assert script_detector_module.detect_dominant_script(
            _blank_image(h=10, w=10)
        ) is None

    def test_returns_none_for_invalid_input(
        self, script_detector_module
    ) -> None:
        assert script_detector_module.detect_dominant_script("nope") is None
        assert script_detector_module.detect_dominant_script(None) is None


class TestScriptDetectorClass:
    def test_default_detector_uses_module_threshold(
        self, script_detector_module,
    ) -> None:
        detector = script_detector_module.ScriptDetector()
        assert detector.min_confidence == (
            script_detector_module.MIN_SCRIPT_CONFIDENCE
        )

    def test_custom_threshold_stored(self, script_detector_module) -> None:
        detector = script_detector_module.ScriptDetector(min_confidence=2.5)
        assert detector.min_confidence == 2.5

    def test_detect_returns_none(self, script_detector_module) -> None:
        detector = script_detector_module.ScriptDetector()
        assert detector.detect(_blank_image()) is None
